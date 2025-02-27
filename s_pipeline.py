[1mdiff --git a/README.md b/README.md[m
[1mindex 421432e..adec129 100644[m
[1m--- a/README.md[m
[1m+++ b/README.md[m
[36m@@ -95,7 +95,7 @@[m [mpython run_data_preprocess_pipeline.py --all [--timezone <timezone>] [--workers[m
 - `--all`: Process data for all participants in parallel[m
 - `--timezone`: Timezone for timestamp conversion (default: Australia/Melbourne)[m
 - `--utc`: If set, overrides timezone with UTC[m
[31m-- `--workers`: Number of worker threads for parallel processing (default: 48)[m
[32m+[m[32m- `--workers`: Number of worker threads for parallel processing (default: 75% of available CPU cores)[m
 [m
 ### Step 2: Feature Extraction[m
 [m
[1mdiff --git a/run_data_preprocess_pipeline.py b/run_data_preprocess_pipeline.py[m
[1mindex 84ec054..23f02a8 100644[m
[1m--- a/run_data_preprocess_pipeline.py[m
[1m+++ b/run_data_preprocess_pipeline.py[m
[36m@@ -29,6 +29,7 @@[m [mimport subprocess[m
 import sys[m
 import os[m
 import argparse[m
[32m+[m[32mimport multiprocessing[m
 from concurrent.futures import ThreadPoolExecutor, as_completed[m
 [m
 [m
[36m@@ -91,6 +92,9 @@[m [mdef process_full_participant_pipeline(participant, timezone):[m
 [m
 def main():[m
     """Execute the data preprocessing pipeline."""[m
[32m+[m[32m    # Calculate default workers as 75% of available CPU cores[m
[32m+[m[32m    default_workers = max(1, int(multiprocessing.cpu_count() * 0.75))[m
[32m+[m[41m    [m
     parser = argparse.ArgumentParser([m
         description="Run data preprocessing pipeline for one or all participants."[m
     )[m
[36m@@ -103,8 +107,8 @@[m [mdef main():[m
                         help="Timezone for timestamp conversion (default: Australia/Melbourne)")[m
     parser.add_argument('--utc', action='store_true',[m
                         help="If set, overrides timezone with UTC")[m
[31m-    parser.add_argument('--workers', type=int, default=48,[m
[31m-                        help="Number of worker threads for processing participants in parallel (default: 48)")[m
[32m+[m[32m    parser.add_argument('--workers', type=int, default=default_workers,[m
[32m+[m[32m                        help=f"Number of worker threads for processing participants in parallel (default: {default_workers}, 75% of CPU cores)")[m
     args = parser.parse_args()[m
 [m
     if args.utc:[m
