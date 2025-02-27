# screentext-features

This repository processes screen text data to extract linguistic features. The workflow consists of two main steps:

1. **Preprocess screen text data** using `run_data_preprocess_pipeline.py`
2. **Generate linguistic features** using `run.py`

## Prerequisites

- Python 3.6+
- Required Python packages (listed in requirements.txt)
- Conda environment (recommended: use the provided `scrtxt` environment)

## Required Data Tables

Each participant folder **must** contain the following data tables in JSONL format:

| Table Name | Description | Required |
|------------|-------------|----------|
| **applications_foreground.jsonl** | Information about applications running in the foreground | ✅ |
| **screen.jsonl** | Screen state information (on/off events) | ✅ |
| **screentext.jsonl** | Text extracted from screens | ✅ |

## Installation

If using conda environment (recommended):
```bash
conda activate scrtxt
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
│   │   ├── applications_foreground.jsonl  # Required table
│   │   ├── screen.jsonl                   # Required table
│   │   └── screentext.jsonl               # Required table
│   ├── participant2/
│   └── ...
├── step1_data/             # Will contain preprocessed data (created automatically)
├── step2_data/             # Will contain feature data (created automatically)
└── data_preprocessing/     # Preprocessing scripts
```

## Data Requirements

Important requirements for your data:

- All raw participant data must be stored in the `participant_data` directory
- Each participant must have their own subfolder (e.g., `participant_data/participant1/`)
- **Each participant folder must contain all required data tables**:
  - `applications_foreground.jsonl`
  - `screen.jsonl`
  - `screentext.jsonl`
- All data files must be in JSONL (JSON Lines) format
- The preprocessing pipeline will read these JSONL files and convert them into the required format for feature extraction

## Processing Workflow

### Step 1: Data Preprocessing

The first step is to preprocess the raw screen text data using `run_data_preprocess_pipeline.py`. This script performs multiple preprocessing steps:

1. Generate app package pairs
2. Clean screentext data
3. Generate filtered system app transition files
4. Add day IDs
5. Calculate session metrics

#### Usage

For processing a single participant:
```bash
conda activate scrtxt
python run_data_preprocess_pipeline.py --participant <participant_id> [--timezone <timezone>]
```

For processing all participants in parallel:
```bash
conda activate scrtxt
python run_data_preprocess_pipeline.py --all [--timezone <timezone>] [--workers <num>]
```

#### Options
- `--participant`, `-p`: Participant ID to process (e.g., 1234)
- `--all`: Process data for all participants in parallel
- `--timezone`: Timezone for timestamp conversion (default: Australia/Melbourne)
- `--utc`: If set, overrides timezone with UTC
- `--workers`: Number of worker threads for parallel processing (default: 48)

### Step 2: Feature Extraction

After preprocessing, extract linguistic features using `run.py`. This script reads the cleaned data from `step1_data` and saves features to `step2_data`.

#### Usage

For processing a single participant:
```bash
conda activate scrtxt
python run.py --participant <participant_id>
```

For processing all participants:
```bash
conda activate scrtxt
python run.py
```

#### Options
- `--base_input_dir`: Base directory containing preprocessed data (default: step1_data)
- `--base_output_dir`: Base directory for storing extracted features (default: step2_data)
- `--participant`, `-p`: Participant folder to process (if not specified, processes all)
- `--input_filename`: Input file name inside each participant folder (default: clean_input.jsonl)
- `--output_filename`: Output file name for extracted features (default: linguistic_features.csv)

## Complete Example Workflow

```bash
# Activate the conda environment
conda activate scrtxt

# Install dependencies (if not already installed)
while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt

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

### Common Issues

- **Missing Data Tables**: Ensure each participant folder contains all three required data tables:
  - `applications_foreground.jsonl`
  - `screen.jsonl`
  - `screentext.jsonl`
- **Environment Issues**: Always run scripts in the proper conda environment: `conda activate scrtxt`
- **Parallel Processing**: If encountering memory issues during parallel processing, reduce the number of workers using the `--workers` option
- **Processing Order**: Ensure preprocessing (Step 1) completed successfully before running feature extraction (Step 2)

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

If you continue to experience memory issues with very large files, consider preprocessing the input files to split them into smaller chunks before running the feature extraction.
