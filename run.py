"""
Extract and save linguistic features from session data files.

This script processes session data from participant folders located in a base input directory,
extracts linguistic features, and stores them in a base output directory. The processing can be 
restricted to a single participant (using --participant or -p) or applied to every participant folder
within the base input directory.

Usage:
    python run.py [--base_input_dir STEP1_DATA] [--base_output_dir STEP2_DATA] [--participant PARTICIPANT]
                  [--input_filename clean_input.jsonl] [--output_filename linguistic_features.csv]
"""

import argparse
import os
from get_features import save_features
#select screentext, clean it, and extract linguistic features then save the features to a file
def main():
    """
    Main function to parse command-line arguments and process linguistic features.

    This function uses command-line arguments to determine the base input and output directories,
    the participant folder (if specified), the input filename, and the output filename. If a participant
    is specified, only that participant's data is processed; otherwise, all participant folders under the
    base input directory are processed.

    Args:
        None. The function relies on argparse for input parameters.

    Raises:
        OSError: If there is an error creating a required output directory.
    """
    parser = argparse.ArgumentParser(
        description="Extract and save linguistic features from session data files."
    )
    parser.add_argument('--base_input_dir', type=str, default='step1_data',
                        help='Base input directory containing participant folders (default: step1_data)')
    parser.add_argument('--base_output_dir', type=str, default='step2_data',
                        help='Base output directory for storing extracted features (default: step2_data)')
    parser.add_argument('--participant', '-p', type=str, default=None,
                        help='Participant folder name to process. If not specified, process all participants.')
    parser.add_argument('--input_filename', type=str, default='clean_input.jsonl',
                        help='Input file name inside each participant folder (default: clean_input.jsonl)')
    parser.add_argument('--output_filename', type=str, default='linguistic_features.csv',
                        help=('Output file name for the extracted features (default: linguistic_features.csv). '
                              'The file extension determines the output format: CSV, JSON, or JSONL.'))
    args = parser.parse_args()

    if args.participant:
        input_file = os.path.join(args.base_input_dir, args.participant, args.input_filename)
        output_dir = os.path.join(args.base_output_dir, args.participant)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, args.output_filename)
        save_features(input_file, output_file)
        print(f"Processed participant: {args.participant} -> {output_file}")
    else:
        # Process every folder under the base input directory
        for participant in os.listdir(args.base_input_dir):
            participant_dir = os.path.join(args.base_input_dir, participant)
            if os.path.isdir(participant_dir):
                input_file = os.path.join(participant_dir, args.input_filename)
                output_dir = os.path.join(args.base_output_dir, participant)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, args.output_filename)
                save_features(input_file, output_file)
                print(f"Processed participant: {participant} -> {output_file}")

if __name__ == "__main__":
    main()