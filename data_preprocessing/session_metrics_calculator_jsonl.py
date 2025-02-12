"""
Session Metrics Calculator JSONL Script

This script processes system app transition data that already has day identifiers and computes 
session-level metrics such as total session duration, main app usage duration, and app usage patterns.
It also extracts additional temporal features (e.g., time of day, day of week) and produces a cleaned 
JSONL output combining session summaries with individual screen text records.

Usage:
    Process a specific participant:
       python session_metrics_calculator_jsonl.py --participant 1234

    Process all participants:
       python session_metrics_calculator_jsonl.py --all
"""

from datetime import datetime
import json
from collections import Counter
from typing import List, Dict, Any
import sys
import os
import re
import argparse

INVISIBLE_CHARS = {
    '\u200b': 'zero-width space',
    '\u200c': 'zero-width non-joiner',
    '\u200d': 'zero-width joiner',
    '\u200e': 'left-to-right mark',
    '\u200f': 'right-to-left mark',
    '\ufeff': 'zero-width no-break space (BOM)',
    '\u2060': 'word joiner',
    '\u2061': 'function application',
    '\u2062': 'invisible times',
    '\u2063': 'invisible separator',
    '\u2064': 'invisible plus'
}

def format_duration(total_seconds):
    # Round total seconds first
    total_seconds = round(total_seconds)
    
    # Calculate components
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    # Build duration string piece by piece
    duration_parts = []
    if hours > 0:
        duration_parts.append(f"{hours}h")
    if minutes > 0 or (hours > 0 and seconds > 0):
        duration_parts.append(f"{minutes}m")
    duration_parts.append(f"{seconds}s")
        
    return "".join(duration_parts)

def format_datetime_for_filename(datetime_str):
    dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
    return dt.strftime("%Y%m%d_%H%M%S")

def get_dates_from_data(data):
    sorted_data = sorted(data, key=lambda x: x["start_datetime"])
    start_date = sorted_data[0]["start_datetime"]
    end_date = sorted_data[-1]["end_datetime"]
    return start_date, end_date

def get_time_of_day_category(datetime_str):
    hour = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f").hour
    if 5 <= hour < 11:  # 5:00 AM - 10:59 AM (6 hours)
        return "morning"
    elif 11 <= hour < 17:  # 11:00 AM - 4:59 PM (6 hours)
        return "afternoon"
    elif 17 <= hour < 23:  # 5:00 PM - 10:59 PM (6 hours)
        return "evening"
    else:  # 11:00 PM - 4:59 AM (6 hours)
        return "late_night"

def get_day_of_week_category(datetime_str):
    day = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f").weekday()
    days = {
        0: "monday",
        1: "tuesday", 
        2: "wednesday",
        3: "thursday",
        4: "friday",
        5: "saturday",
        6: "sunday"
    }
    return days[day]

def is_weekend(datetime_str):
    day = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f").weekday()
    return day >= 5

def calculate_session_metrics(session_data: List[dict]) -> Dict[str, Any]:
    """Calculate core session metrics that indicate user goals"""
    
    # Get session start and end times
    sorted_data = sorted(session_data, key=lambda x: x["start_datetime"])
    session_start = sorted_data[0]["start_datetime"]
    session_end = sorted_data[-1]["end_datetime"]
    
    # 1. App Usage Pattern
    app_sequences = []
    main_app_durations = {}
    
    # Track non-system app usage
    for entry in session_data:
        app = entry["application_name"]
        duration = entry["duration_seconds"]
        
        if not entry["is_system_app"]:
            app_sequences.append({
                "app": app,
                "duration": duration
            })
            
            # Accumulate durations
            if app not in main_app_durations:
                main_app_durations[app] = 0
            main_app_durations[app] += duration
    
    # Get all non-system app usage time
    main_usage_duration = sum(main_app_durations.values())
    
    # Get top apps for the summary
    top_3_apps = sorted(main_app_durations.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Calculate session characteristics
    total_duration = sum(entry["duration_seconds"] for entry in session_data)
    
    return {
        "session_metrics": {
            "start_time": session_start,
            "end_time": session_end,
            "total_duration": total_duration,
            "main_usage_duration": main_usage_duration,
            "main_usage_ratio": main_usage_duration / total_duration if total_duration > 0 else 0
        },
        "app_metrics": {
            "primary_apps": [
                {
                    "name": app,
                    "duration": duration,
                    "usage_ratio": duration / main_usage_duration if main_usage_duration > 0 else 0
                }
                for app, duration in top_3_apps
            ],
            "app_sequence": app_sequences
        }
    }

def process_sessions(data: List[dict]) -> Dict[str, Any]:
    """Process session data and calculate metrics"""
    sessions = {}
    
    for entry in data:
        session_id = entry["session_id"]
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(entry)
    
    output = {}
    for session_id, session_data in sessions.items():
        session_data.sort(key=lambda x: x["start_datetime"])
        metrics = calculate_session_metrics(session_data)
        
        output[f"session_{session_id}"] = {
            "time_range": {
                "start": metrics["session_metrics"]["start_time"],
                "end": metrics["session_metrics"]["end_time"]
            },
            "duration": {
                "total": format_duration(metrics["session_metrics"]["total_duration"]),
                "main_usage": format_duration(metrics["session_metrics"]["main_usage_duration"]),
                "main_usage_ratio": round(metrics["session_metrics"]["main_usage_ratio"], 2)
            },
            "apps": {
                "primary_apps": [
                    {
                        "name": app["name"],
                        "duration": format_duration(app["duration"]),
                        "usage_ratio": round(app["usage_ratio"], 2)
                    }
                    for app in metrics["app_metrics"]["primary_apps"][:3]
                ]
            }
        }
    
    return output

def simplify_app_sequence(app_sequences):
    """Simplify app sequence by combining consecutive duplicates"""
    if not app_sequences:
        return []
        
    simplified = []
    current_app = app_sequences[0]
    current_duration = 0
    
    for seq in app_sequences:
        if seq["app"] == current_app["app"]:
            current_duration += seq["duration"]
        else:
            simplified.append({
                "app": current_app["app"],
                "duration": current_duration
            })
            current_app = seq
            current_duration = seq["duration"]
    
    simplified.append({
        "app": current_app["app"],
        "duration": current_duration
    })
    
    return simplified

def combine_metrics_with_records(data: List[dict]) -> List[dict]:
    """Combine session metrics with original records and output as JSONL format"""
    metrics_by_session = process_sessions(data)
    
    sessions = {}
    session_day_ids = {}  # Store day_id for each session
    for entry in data:
        session_id = entry["session_id"]
        if session_id not in sessions:
            sessions[session_id] = []
            session_day_ids[session_id] = entry.get('day_id')
        record = entry.copy()
        record.pop('session_id', None)
        record.pop('utc_offset', None)
        record.pop('day_id', None)
        sessions[session_id].append(record)
    
    combined_data = []
    for session_id, records in sessions.items():
        metrics_key = f"session_{session_id}"
        if metrics_key in metrics_by_session:
            combined_entry = {
                "session_id": session_id,
                "day_id": str(session_day_ids[session_id]),
                "overview": metrics_by_session[metrics_key],
                "screen_text_logs": records
            }
            combined_data.append(combined_entry)
    
    return combined_data

def process_participant_data(participant_id: str) -> None:
    """Process data for a specific participant"""
    input_path = f"step1_data/{participant_id}/system_app_transition_removed_2sec_with_day_id.jsonl"
    output_path = f"step1_data/{participant_id}/clean_input.jsonl"
    
    try:
        data = []
        pattern = '[' + ''.join(INVISIBLE_CHARS.keys()) + ']'
        
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if 'text' in record:
                        record['text'] = re.sub(pattern, '', record['text'])
                    data.append(record)
        
        combined_data = combine_metrics_with_records(data)
        
        with open(output_path, 'w') as f:
            for entry in combined_data:
                f.write(json.dumps(entry) + '\n')
                
        print(f"Processed participant {participant_id}")
        
    except FileNotFoundError:
        print(f"Error: Could not find data file for participant {participant_id}")
    except Exception as e:
        print(f"Error processing participant {participant_id}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="Calculate session metrics for a specific participant or all participants"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-p', '--participant', type=str, help="Participant ID to process")
    group.add_argument('--all', action='store_true', help="Process all participants in step1_data")
    args = parser.parse_args()
    
    base_dir = "step1_data"
    if args.participant:
        process_participant_data(args.participant)
    else:
        # Default to all participants if --all is set or no argument is provided.
        try:
            participants = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if not participants:
                print(f"Error: Could not find any participant directories in {base_dir}")
                return
            for participant_id in participants:
                process_participant_data(participant_id)
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

# python session_metrics_calculator_jsonl.py --participant 1234
# python session_metrics_calculator_jsonl.py --all