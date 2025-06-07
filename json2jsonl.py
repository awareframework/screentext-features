#!/usr/bin/env python3
"""
JSON to JSONL Converter
Converts all JSON files in a folder to JSONL (JSON Lines) format.
"""
import json
import os
import argparse
import glob
from pathlib import Path

def convert_json_to_jsonl(json_file_path, output_dir=None):
    """
    Convert a JSON file to JSONL format.
    
    Args:
        json_file_path (str): Path to the JSON file
        output_dir (str, optional): Output directory. If None, uses same directory as input file.
    
    Returns:
        str: Path to the created JSONL file
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Determine output path
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / (Path(json_file_path).stem + '.jsonl')
        else:
            output_file = Path(json_file_path).with_suffix('.jsonl')
        
        # Write JSONL file
        with open(output_file, 'w', encoding='utf-8') as f:
            if isinstance(data, list):
                # If JSON contains an array, write each item as a separate line
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
            else:
                # If JSON contains a single object, write it as one line
                json.dump(data, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✓ Converted: {json_file_path} -> {output_file}")
        return str(output_file)
        
    except json.JSONDecodeError as e:
        print(f"✗ Error parsing JSON file {json_file_path}: {e}")
        return None
    except Exception as e:
        print(f"✗ Error processing {json_file_path}: {e}")
        return None

def convert_folder_json_to_jsonl(input_folder, output_folder=None, recursive=False):
    """
    Convert all JSON files in a folder to JSONL format.
    
    Args:
        input_folder (str): Input folder path
        output_folder (str, optional): Output folder path
        recursive (bool): Whether to search recursively in subdirectories
    
    Returns:
        tuple: (successful_conversions, failed_conversions)
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"✗ Input folder does not exist: {input_folder}")
        return 0, 0
    
    # Find all JSON files
    if recursive:
        json_files = list(input_path.rglob('*.json'))
    else:
        json_files = list(input_path.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {input_folder}")
        return 0, 0
    
    print(f"Found {len(json_files)} JSON file(s) to convert...")
    
    successful = 0
    failed = 0
    
    for json_file in json_files:
        if output_folder:
            # Maintain relative directory structure in output
            if recursive:
                relative_path = json_file.relative_to(input_path)
                output_dir = Path(output_folder) / relative_path.parent
            else:
                output_dir = output_folder
        else:
            output_dir = None
        
        result = convert_json_to_jsonl(str(json_file), output_dir)
        if result:
            successful += 1
        else:
            failed += 1
    
    return successful, failed

def main():
    parser = argparse.ArgumentParser(
        description='Convert JSON files to JSONL format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python json2jsonl.py /path/to/json/folder
  python json2jsonl.py /path/to/json/folder -o /path/to/output/folder
  python json2jsonl.py /path/to/json/folder -r -o /path/to/output/folder
        '''
    )
    
    parser.add_argument('input_folder', 
                       help='Input folder containing JSON files')
    parser.add_argument('-o', '--output', 
                       help='Output folder (default: same as input folder)')
    parser.add_argument('-r', '--recursive', 
                       action='store_true',
                       help='Search recursively in subdirectories')
    
    args = parser.parse_args()
    
    print(f"Converting JSON files from: {args.input_folder}")
    if args.output:
        print(f"Output folder: {args.output}")
    if args.recursive:
        print("Searching recursively in subdirectories")
    print("-" * 50)
    
    successful, failed = convert_folder_json_to_jsonl(
        args.input_folder, 
        args.output, 
        args.recursive
    )
    
    print("-" * 50)
    print(f"Conversion complete!")
    print(f"✓ Successfully converted: {successful} files")
    if failed > 0:
        print(f"✗ Failed to convert: {failed} files")

if __name__ == '__main__':
    main()
