# screentext-features

This repository processes screen text data to extract linguistic features. The workflow consists of two main steps:

1. Preprocess screen text data using `run_data_preprocess_pipeline.py`
2. Generate linguistic features using `run.py`

## Prerequisites

- Python 3.6+
- Required Python packages (listed in requirements.txt)
- Raw participant data including applications_foreground, screen, and screentext in JSONL format placed in the `participant_data` directory

## Installation

If using conda environment:
```bash
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt
```

If using pip:
```bash
pip install -r requirements.txt
```

## Directory Structure

Before running the pipeline, ensure you have the following directory structure:
```
.
├── participant_data/       # Raw participant data (in JSONL format)
│   ├── participant1/       # Each participant has their own folder
│   ├── participant2/
│   └── ...
├── step1_data/             # Will contain preprocessed data (created automatically)
├── step2_data/             # Will contain feature data (created automatically)
└── data_preprocessing/     # Preprocessing scripts
```

## Data Requirements

- The pipeline assumes all raw participant data is stored in the `participant_data` directory
- Each participant should have their own subfolder (e.g., `participant_data/participant1/`)
- Raw data files must be in JSONL (JSON Lines) format
- The preprocessing pipeline will read these JSONL files and convert them into the required format for feature extraction

## Step 1: Data Preprocessing

The first step is to preprocess the raw screen text data using `run_data_preprocess_pipeline.py`. This script performs multiple preprocessing steps:

1. Generate app package pairs
2. Clean screentext data
3. Generate filtered system app transition files
4. Add day IDs
5. Calculate session metrics

### Usage

For processing a single participant:
```bash
python run_data_preprocess_pipeline.py --participant <participant_id> [--timezone <timezone>]
```

For processing all participants in parallel:
```bash
python run_data_preprocess_pipeline.py --all [--timezone <timezone>] [--workers <num>]
```

### Options
- `--participant`, `-p`: Participant ID to process (e.g., 1234)
- `--all`: Process data for all participants in parallel
- `--timezone`: Timezone for timestamp conversion (default: Australia/Melbourne)
- `--utc`: If set, overrides timezone with UTC
- `--workers`: Number of worker threads for parallel processing (default: 48)

## Step 2: Feature Extraction

After preprocessing, extract linguistic features using `run.py`. This script reads the cleaned data from `step1_data` and saves features to `step2_data`.

### Usage

For processing a single participant:
```bash
python run.py --participant <participant_id>
```

For processing all participants:
```bash
python run.py
```

### Options
- `--base_input_dir`: Base directory containing preprocessed data (default: step1_data)
- `--base_output_dir`: Base directory for storing extracted features (default: step2_data)
- `--participant`, `-p`: Participant folder to process (if not specified, processes all)
- `--input_filename`: Input file name inside each participant folder (default: clean_input.jsonl)
- `--output_filename`: Output file name for extracted features (default: linguistic_features.csv)

## Complete Example Workflow

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Preprocess data for all participants
python run_data_preprocess_pipeline.py --all

# Step 2: Extract features for all participants
python run.py

# Alternatively, process a single participant
python run_data_preprocess_pipeline.py --participant 1234
python run.py --participant 1234
```

## Output

After running both steps:
- `step1_data/` will contain cleaned and preprocessed data for each participant
- `step2_data/` will contain CSV files with linguistic features for each participant

## Troubleshooting

- If you encounter memory issues during parallel processing, reduce the number of workers using the `--workers` option
- Ensure that the `participant_data` directory exists and contains participant folders before running Step 1
- Check that preprocessing completed successfully before running the feature extraction step

### Memory Limitations

The feature extraction process (Step 2) can be memory-intensive, especially when processing large input files:

- **Segmentation Faults**: If you encounter segmentation faults, it's likely due to memory limitations when processing very large text files. The code now includes safeguards to handle large files by:
  - Processing text in smaller chunks
  - Limiting the amount of text analyzed for memory-intensive operations (NER, POS tagging)
  - Adding robust error handling to prevent crashes

- **RAM Dependency**: The maximum file size that can be processed depends on your system's available RAM:
  - 8GB RAM systems: May struggle with files larger than ~10MB
  - 16GB RAM systems: Should handle files up to ~30MB
  - 32GB+ RAM systems: Can process larger files more efficiently

- **Adjusting Memory Usage**: You can modify the following constants in `get_features.py` to adjust memory usage based on your system capabilities:
  - `MAX_TEXT_CHUNK_SIZE`: Controls the maximum text size processed at once
  - `MAX_TOKENS_FOR_INTENSIVE_ANALYSIS`: Limits token count for NLP operations

- **Environment Considerations**: Always run the script within the proper conda environment:
  ```bash
  conda activate scrtxt
  python run.py
  ```

If you continue to experience memory issues with very large files, consider preprocessing the input files to split them into smaller chunks before running the feature extraction.
