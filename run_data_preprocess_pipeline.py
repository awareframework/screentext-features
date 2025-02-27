#!/usr/bin/env python
"""Master Pipeline Script for Data Preprocessing.

This script orchestrates the data preprocessing pipeline for either a single participant
or all participants. In full parallel mode, each participant's entire pipeline is run 
concurrently. The steps for a participant-specific processing are:

    1. Generate app package pairs (participant-specific).
    2. Clean screentext data.
    3. Generate filtered system app transition files.
    4. Add day IDs.
    5. Calculate session metrics.

Usage:
    For one participant:
        $ python run_data_preprocess_pipeline.py --participant <participant_id> [--timezone <timezone>] [--utc]

    For processing all participants fully in parallel:
        $ python run_data_preprocess_pipeline.py --all [--timezone <timezone>] [--utc] [--workers <num>]

Notes:
    Ensure that directories (e.g., participant_data, step1_data) exist in the parent directory.
    Also ensure that all required scripts support participant-specific operation (i.e. using -p).
    If a script (e.g., generate_system_app_transition_filtered_files.py) does not currently 
    support a participant-specific flag, you will need to modify it accordingly.
"""

import subprocess
import sys
import os
import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_command(cmd_list, exit_on_error=True):
    """Run a system command as a subprocess.
    
    Prints the command, executes it, and either exits on error
    or raises the exception if exit_on_error is False.
    """
    print("Running command: " + " ".join(cmd_list))
    try:
        subprocess.run(cmd_list, check=True)
        print("Finished successfully: " + " ".join(cmd_list) + "\n")
    except subprocess.CalledProcessError as e:
        print("Error while executing: " + " ".join(cmd_list))
        if exit_on_error:
            sys.exit(e.returncode)
        else:
            raise e


def process_full_participant_pipeline(participant, timezone):
    """
    Process the entire preprocessing pipeline for a single participant.
    
    For full parallelization, each participant's pipeline is executed sequentially
    within a separate thread:
      1. Generate app package pairs for the participant.
      2. Clean screentext data for the participant.
      3. Generate filtered system app transition files 
         (assumed to support a participant flag; modify if needed).
      4. Add day IDs.
      5. Calculate session metrics.
    """
    print(f"Started processing participant {participant}")
    try:
        # Step 1: Generate app package pairs for the participant.
        run_command(["python", "data_preprocessing/generate_app_package_pair.py", "-p", participant],
                    exit_on_error=False)
        # Step 2: Clean screentext data for the participant.
        run_command(["python", "data_preprocessing/clean_screentext_jsonl.py", "-p", participant,
                     "--timezone", timezone], exit_on_error=False)
        # Step 3: Generate filtered system app transition files.
        # (Ensure this script supports participant-specific processing using a flag like -p)
        run_command(["python", "data_preprocessing/generate_system_app_transition_filtered_files.py", "-p", participant,
                     "--base_dir", "step1_data", "--mode", "generate", "--threshold", "2.0"],
                    exit_on_error=False)
        # Step 4: Add day IDs.
        run_command(["python", "data_preprocessing/add_day_id.py", "step1_data", "-p", participant],
                    exit_on_error=False)
        # Step 5: Calculate session metrics.
        run_command(["python", "data_preprocessing/session_metrics_calculator_jsonl.py", "-p", participant],
                    exit_on_error=False)
        print(f"Completed processing participant {participant}")
        return True, participant
    except Exception as e:
        print(f"Error processing participant {participant}: {e}")
        return False, participant


def main():
    """Execute the data preprocessing pipeline."""
    # Calculate default workers as 75% of available CPU cores
    default_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
    
    parser = argparse.ArgumentParser(
        description="Run data preprocessing pipeline for one or all participants."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--participant', '-p', type=str,
                        help="Participant ID to process (e.g., 1234)")
    group.add_argument('--all', action='store_true',
                        help="Process data for all participants with full parallel participant-specific processing")
    parser.add_argument('--timezone', type=str, default="Australia/Melbourne",
                        help="Timezone for timestamp conversion (default: Australia/Melbourne)")
    parser.add_argument('--utc', action='store_true',
                        help="If set, overrides timezone with UTC")
    parser.add_argument('--workers', type=int, default=default_workers,
                        help=f"Number of worker threads for processing participants in parallel (default: {default_workers}, 75% of CPU cores)")
    args = parser.parse_args()

    if args.utc:
        args.timezone = "UTC"

    current_dir = os.getcwd()
    print(f"Working directory: {current_dir}\n")

    if args.participant:
        # In single participant mode, run the pipeline sequentially.
        run_command(["python", "data_preprocessing/generate_app_package_pair.py", "-p", args.participant])
        run_command(["python", "data_preprocessing/clean_screentext_jsonl.py", "-p", args.participant,
                     "--timezone", args.timezone])
        run_command(["python", "data_preprocessing/generate_system_app_transition_filtered_files.py", "-p", args.participant,
                     "--base_dir", "step1_data", "--mode", "generate", "--threshold", "2.0"])
        run_command(["python", "data_preprocessing/add_day_id.py", "step1_data", "-p", args.participant])
        run_command(["python", "data_preprocessing/session_metrics_calculator_jsonl.py", "-p", args.participant])
    else:
        # Fully parallel processing: run the complete pipeline for every participant concurrently.
        participant_dir = "participant_data"
        if not os.path.exists(participant_dir):
            print("Error: participant_data directory does not exist.")
            sys.exit(1)

        participants = [d for d in os.listdir(participant_dir)
                        if os.path.isdir(os.path.join(participant_dir, d))]

        if not participants:
            print("No participants found in the participant_data directory.")
            sys.exit(1)

        print(f"Found {len(participants)} participants. Processing them in parallel using {args.workers} worker threads.\n")
        
        successes = 0
        failures = 0
        total = len(participants)
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_participant = {
                executor.submit(process_full_participant_pipeline, participant, args.timezone): participant
                for participant in participants
            }
            for future in as_completed(future_to_participant):
                participant = future_to_participant[future]
                try:
                    success, pid = future.result()
                    if success:
                        successes += 1
                        print(f"Completed participant {pid} ({successes + failures}/{total})")
                    else:
                        failures += 1
                        print(f"Failed participant {pid} ({successes + failures}/{total})")
                except Exception as exc:
                    failures += 1
                    print(f"Exception processing participant {participant}: {exc} ({successes + failures}/{total})")
        
        print("\nProcessing Complete!")
        print(f"Successfully processed: {successes} participants")
        print(f"Failed to process: {failures} participants")
        

if __name__ == '__main__':
    main()
