import os
import json
import pandas as pd
from datetime import datetime

def merge_by_session(input_file='step1_data/clean_df.json'):
    # Read input data
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Initialize list to store session data
    session_data = []
    
    # Group by session_id
    grouped = df.groupby('session_id')
    
    for session_id, session_group in grouped:
        # Combine all text segments from the session
        all_text_segments = []
        for text in session_group['text']:
            segments = [seg.strip() for seg in text.split('||')]
            all_text_segments.extend(segments)
        
        # Calculate total session duration
        start_time = pd.to_datetime(session_group['start_datetime'].min())
        end_time = pd.to_datetime(session_group['end_datetime'].max())
        total_duration = (end_time - start_time).total_seconds()
        
        # Count unique applications
        num_unique_apps = len(session_group['application_name'].unique())
        num_active_periods = len(session_group['active_period_id'].unique())
        
        session_info = {
            'session_id': session_id,
            'text_segments': all_text_segments,
            'metadata': {
                'start_datetime': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_datetime': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_seconds': total_duration,
                'num_applications': num_unique_apps,
                'num_active_periods': num_active_periods
            }
        }
        
        session_data.append(session_info)
    
    # Create output directory if it doesn't exist
    if not os.path.exists("step2_data"):
        os.makedirs("step2_data")
    
    # Save to JSON file
    with open(os.path.join("step2_data", "text_by_session.json"), 'w') as file:
        json.dump(session_data, file, indent=4)
    
    return session_data

if __name__ == "__main__":
    session_data = merge_by_session()
    print(f"Processed {len(session_data)} sessions")