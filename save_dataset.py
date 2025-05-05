#!/usr/bin/env python
import os
import json
import torch
import base64
from io import BytesIO
from PIL import Image

from datasets import load_dataset

def image_to_base64(img):
    """Convert PIL Image to base64 string for JSON serialization"""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def save_interleaved_maze_dataset():
    """Load and save the interleaved maze dataset that's used in training"""
    
    print("Loading interleaved_maze dataset...")
    data = load_dataset(
        "utils/processed_data_wrapper/interleaved_maze.py", 
        tasks=['simulation'], 
        modes=['single_step_visualization', 'action_reasoning'], 
        data_dir='data_samples',
        trust_remote_code=True
    )
    
    print(f"Dataset loaded with splits: {list(data.keys())}")
    for split in data.keys():
        print(f"{split} split size: {len(data[split])}")
    
    # Convert and save examples
    output = {}
    for split in data.keys():
        examples = []
        for i, example in enumerate(data[split]):
            # Convert images to base64 for JSON serialization
            input_images_b64 = []
            for img in example['input_imgs']:
                if hasattr(img, 'convert'):  # Check if it's a PIL Image
                    input_images_b64.append(image_to_base64(img))
                    
            label_images_b64 = []
            for img in example['label_imgs']:
                if hasattr(img, 'convert'):  # Check if it's a PIL Image
                    label_images_b64.append(image_to_base64(img))
            
            # Create serializable example
            example_dict = {
                "idx": example.get('idx', i),
                "input_text": example.get('input_text', ''),
                "input_img_paths": example.get('input_img_paths', []),
                "input_images_base64": input_images_b64,
                "label_text": example.get('label_text', ''),
                "label_img_paths": example.get('label_img_paths', []),
                "label_images_base64": label_images_b64,
                "task": example.get('task', ''),
                "train_task": example.get('train_task', '')
            }
            
            # Add other fields
            for key in example:
                if key not in example_dict and key not in ['input_imgs', 'label_imgs']:
                    example_dict[key] = example[key]
            
            examples.append(example_dict)
            
            # Print progress
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(data[split])} examples in {split} split")
                
        output[split] = examples
    
    # Save to file
    os.makedirs('saved_datasets', exist_ok=True)
    output_file = 'saved_datasets/interleaved_maze_dataset.json'
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Dataset saved to {output_file}")
    print(f"Total examples saved: {sum(len(examples) for examples in output.values())}")

if __name__ == "__main__":
    save_interleaved_maze_dataset() 