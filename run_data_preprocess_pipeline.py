#!/usr/bin/env python
"""Master Pipeline Script for Data Preprocessing.

This script orchestrates the entire data preprocessing pipeline for either a single
participant (using their participant ID) or for all participants. The pipeline consists
of the following steps:

    1. Generate app package pairs: data_preprocessing/generate_app_package_pair.py
    2. Clean screentext data: data_preprocessing/clean_screentext_jsonl.py
    3. Generate filtered system app transition files:
       data_preprocessing/generate_system_app_transition_filtered_files.py
    4. Add day IDs: data_preprocessing/add_day_id.py
    5. Calculate session metrics: data_preprocessing/session_metrics_calculator_jsonl.py

Usage:
    For one participant:
        $ python run_data_preprocess_pipeline.py --participant <participant_id>

    For processing all participants:
        $ python run_data_preprocess_pipeline.py --all

Notes:
    Ensure that the directories (e.g., participant_data, step1_data, resources) exist in the
    parent directory and that all required scripts are present in the 'data_preprocessing'
    folder. This pipeline should be executed from the parent directory.
"""

import subprocess
import sys
import os
import argparse


def run_command(cmd_list):
    """Run a system command as a subprocess.

    This function executes the provided command (as a list of strings), prints the command
    being executed, and terminates the script if the command fails.

    Args:
        cmd_list (List[str]): A list of strings representing the command and its arguments.

    Raises:
        SystemExit: If the command returns a non-zero exit status.
    """
    print("Running command: " + " ".join(cmd_list))
    try:
        subprocess.run(cmd_list, check=True)
        print("Finished successfully: " + " ".join(cmd_list) + "\n")
    except subprocess.CalledProcessError as e:
        print("Error while executing: " + " ".join(cmd_list))
        sys.exit(e.returncode)


def main():
    """Execute the data preprocessing pipeline.

    This function parses command-line arguments to determine whether to process a single
    participant or all participants. It then executes the preprocessing steps in order:

      1. Generate app package pairs.
      2. Clean screentext data.
      3. Generate filtered system app transition files.
      4. For each participant:
         a. Add day IDs.
         b. Calculate session metrics.

    Raises:
        SystemExit: If the required directories are missing or if no participants are found.
    """
    parser = argparse.ArgumentParser(
        description="Run data preprocessing pipeline for one or all participants."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--participant', '-p', type=str,
        help="Participant ID to process (e.g., 1234)"
    )
    group.add_argument(
        '--all', action='store_true',
        help="Process data for all participants (based on directories in 'participant_data')"
    )
    args = parser.parse_args()

    # Display the current working directory; expected to be the parent directory.
    current_dir = os.getcwd()
    print(f"Working directory: {current_dir}\n")

    # Execute global preprocessing steps.
    if args.participant:
        # Process a single participant by passing the participant ID to the scripts.
        run_command(["python", "data_preprocessing/generate_app_package_pair.py", "-p", args.participant])
        run_command(["python", "data_preprocessing/clean_screentext_jsonl.py", "-p", args.participant])
        run_command([
            "python", "data_preprocessing/generate_system_app_transition_filtered_files.py",
            "--base_dir", "step1_data", "--mode", "generate", "--threshold", "2.0"
        ])
        participants = [args.participant]
    else:
        # Process all participants; assume scripts can handle processing all when no participant is specified.
        run_command(["python", "data_preprocessing/generate_app_package_pair.py"])
        run_command(["python", "data_preprocessing/clean_screentext_jsonl.py"])
        run_command([
            "python", "data_preprocessing/generate_system_app_transition_filtered_files.py",
            "--base_dir", "step1_data", "--mode", "generate", "--threshold", "2.0"
        ])
        participant_dir = "participant_data"
        if not os.path.exists(participant_dir):
            print("Error: participant_data directory does not exist.")
            sys.exit(1)
        # Assume each directory in 'participant_data' is a participant ID.
        participants = [
            d for d in os.listdir(participant_dir)
            if os.path.isdir(os.path.join(participant_dir, d))
        ]
        if not participants:
            print("No participants found in the participant_data directory.")
            sys.exit(1)

    # Execute participant-specific preprocessing steps.
    for participant in participants:
        print(f"Processing participant {participant}")
        run_command(["python", "data_preprocessing/add_day_id.py", "step1_data", "-p", participant])
        run_command(["python", "data_preprocessing/session_metrics_calculator_jsonl.py", "-p", participant])


if __name__ == '__main__':
    main()
