#!/usr/bin/env python3
"""
BATCH CONVERTER - Process all text files in data/ folder
=========================================================
Automatically finds and converts all .txt files to datasets.

Usage:
    python batch_convert.py                    # Process all files in data/
    python batch_convert.py --input custom/    # Process files in custom folder
"""

import argparse
import json
import os
from pathlib import Path
from convert import convert_text_to_dataset

def batch_convert(
    input_dir: str = "data",
    output_file: str = "dataset.json",
    system_prompt: str = "You are a helpful assistant.",
):
    """Convert all text files in a directory to a single dataset."""
    
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Creating {input_dir}/ folder...")
        input_path.mkdir(parents=True)
        print(f"  Put your .txt files in the {input_dir}/ folder and run again.")
        return
    
    # Find all text files
    txt_files = list(input_path.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {input_dir}/")
        print(f"  Put your .txt files there and run again.")
        return
    
    print("=" * 50)
    print("  BATCH CONVERTER")
    print("=" * 50)
    print(f"\nFound {len(txt_files)} text files in {input_dir}/\n")
    
    all_examples = []
    
    for txt_file in txt_files:
        print(f"Processing: {txt_file.name}")
        
        # Convert each file
        temp_output = f"_temp_{txt_file.stem}.json"
        
        try:
            examples = convert_text_to_dataset(
                str(txt_file),
                temp_output,
                format_type="auto",
                system_prompt=system_prompt,
            )
            all_examples.extend(examples)
            
            # Clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)
                
        except Exception as e:
            print(f"  Error: {e}")
        
        print()
    
    # Save combined dataset
    print("=" * 50)
    print(f"Total examples: {len(all_examples)}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to: {output_file}")
    print("=" * 50)
    print(f"\nReady to train!")
    print(f"  python train.py --model MODEL_NAME --dataset {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Batch convert all text files to dataset")
    parser.add_argument("--input", "-i", default="data", help="Input folder (default: data/)")
    parser.add_argument("--output", "-o", default="dataset.json", help="Output file (default: dataset.json)")
    parser.add_argument("--system", "-s", default="You are a helpful assistant.", help="System prompt")
    
    args = parser.parse_args()
    
    batch_convert(args.input, args.output, args.system)

if __name__ == "__main__":
    main()
