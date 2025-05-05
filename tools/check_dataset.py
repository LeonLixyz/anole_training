#!/usr/bin/env python3

import os
import json
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

def normalize_dataset(input_path, output_path):
    """
    Check and normalize a dataset to ensure it's compatible with the model.
    - Ensures images are loaded correctly
    - Ensures image tags are properly formatted
    - Reports statistics about the dataset
    """
    # Load the dataset
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Statistics
    total_examples = 0
    examples_with_input_images = 0
    examples_with_label_images = 0
    missing_images = 0
    fixed_tags = 0
    
    # Process each split
    splits = data.keys() if isinstance(data, dict) else ['train']
    
    for split in splits:
        examples = data[split] if isinstance(data, dict) else data
        total_examples += len(examples)
        
        print(f"Processing {split} split ({len(examples)} examples)...")
        
        for i, example in tqdm(enumerate(examples), total=len(examples)):
            # Check input images
            if 'input_img_paths' in example and example['input_img_paths']:
                examples_with_input_images += 1
                
                # Check each image path
                for j, img_path in enumerate(example['input_img_paths']):
                    if not os.path.exists(img_path):
                        print(f"Warning: Input image path not found: {img_path}")
                        missing_images += 1
                
                # Make sure input_text has exactly one <image> tag for simplicity
                if "<image>" not in example['input_text']:
                    example['input_text'] += " <image>"
                    fixed_tags += 1
                elif example['input_text'].count("<image>") > 1:
                    # Replace multiple image tags with just one at the end
                    example['input_text'] = example['input_text'].replace("<image>", "")
                    example['input_text'] += " <image>"
                    fixed_tags += 1
            
            # Check label images
            if 'label_img_paths' in example and example['label_img_paths']:
                examples_with_label_images += 1
                
                # Check each image path
                for j, img_path in enumerate(example['label_img_paths']):
                    if not os.path.exists(img_path):
                        print(f"Warning: Label image path not found: {img_path}")
                        missing_images += 1
                
                # Make sure label_text has exactly one <image> tag
                if "<image>" not in example['label_text']:
                    example['label_text'] += " <image>"
                    fixed_tags += 1
                elif example['label_text'].count("<image>") > 1:
                    # Replace multiple image tags with just one at the end
                    example['label_text'] = example['label_text'].replace("<image>", "")
                    example['label_text'] += " <image>"
                    fixed_tags += 1
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total examples: {total_examples}")
    print(f"Examples with input images: {examples_with_input_images}")
    print(f"Examples with label images: {examples_with_label_images}")
    print(f"Missing images: {missing_images}")
    print(f"Fixed image tags: {fixed_tags}")
    
    # Save the normalized dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nNormalized dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check and normalize a dataset for compatibility")
    parser.add_argument("--input", required=True, help="Path to the input dataset JSON file")
    parser.add_argument("--output", required=True, help="Path to save the normalized dataset")
    
    args = parser.parse_args()
    normalize_dataset(args.input, args.output) 