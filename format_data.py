#!/usr/bin/env python
import os
import json
import base64
import argparse
import re
from io import BytesIO
from PIL import Image
from datasets import load_dataset, concatenate_datasets

def base64_to_image(base64_str):
    """Convert base64 string to PIL Image."""
    if not base64_str.startswith("data:"):
        # Add prefix if not already there
        base64_str = f"data:image/jpeg;base64,{base64_str}"
    
    # Extract the base64 data
    header, encoded = base64_str.split(",", 1)
    
    # Decode and convert to PIL Image
    image_data = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return image

def format_dataset(dataset, output_dir="formatted_data"):
    """Format the dataset into the interleaved text-image format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each problem image
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    formatted_data = []
    
    for idx, item in enumerate(dataset):
        question = item.get("question", "")
        reasoning = item.get("reasoning", "")
        answer = item.get("answer", "")
        
        print(f"Processing example {idx+1}/{len(dataset)}")
        
        # Process problem images
        problem_images = []
        problem_image_paths = []
        
        max_problem_idx = 0
        for key in item.keys():
            match = re.match(r"problem_image_(\d+)(_base64)?", key)
            if match:
                idx_num = int(match.group(1))
                if idx_num > max_problem_idx:
                    max_problem_idx = idx_num
        
        # Process each possible problem image
        for i in range(1, max_problem_idx + 1):  # Assuming up to 4 problem images
            base64_key = f"problem_image_{i}_base64"
            img_key = f"problem_image_{i}"
            
            if base64_key in item and item[base64_key]:
                try:
                    # Convert base64 to image
                    image = base64_to_image(item[base64_key])
                    
                    # Save image to disk
                    image_path = os.path.join(image_dir, f"problem_{idx}_{i}.jpg")
                    image.save(image_path)
                    
                    problem_images.append(image)
                    problem_image_paths.append(image_path)
                except Exception as e:
                    print(f"Error processing image {base64_key}: {e}")
            elif img_key in item and item[img_key] is not None:
                try:
                    if hasattr(item[img_key], 'size'):  # It's already a PIL Image
                        problem_images.append(item[img_key])
                        image_path = os.path.join(image_dir, f"problem_{idx}_{i}.jpg")
                        item[img_key].save(image_path)
                        problem_image_paths.append(image_path)
                except Exception as e:
                    print(f"Error processing image {img_key}: {e}")
        
        # Process reasoning images
        reasoning_images = []
        reasoning_image_paths = []
        
        max_reasoning_idx = 0
        for key in item.keys():
            match = re.match(r"reasoning_image_(\d+)(_base64)?", key)
            if match:
                idx_num = int(match.group(1))
                if idx_num > max_reasoning_idx:
                    max_reasoning_idx = idx_num
        
        # Process each possible reasoning image
        for i in range(1, max_reasoning_idx + 1):  # Assuming up to 4 reasoning images
            base64_key = f"reasoning_image_{i}_base64"
            img_key = f"reasoning_image_{i}"
            
            if base64_key in item and item[base64_key]:
                try:
                    # Convert base64 to image
                    image = base64_to_image(item[base64_key])
                    
                    # Save image to disk
                    image_path = os.path.join(image_dir, f"reasoning_{idx}_{i}.jpg")
                    image.save(image_path)
                    
                    reasoning_images.append(image)
                    reasoning_image_paths.append(image_path)
                except Exception as e:
                    print(f"Error processing image {base64_key}: {e}")
            elif img_key in item and item[img_key] is not None:
                try:
                    if hasattr(item[img_key], 'size'):  # It's already a PIL Image
                        reasoning_images.append(item[img_key])
                        image_path = os.path.join(image_dir, f"reasoning_{idx}_{i}.jpg")
                        item[img_key].save(image_path)
                        reasoning_image_paths.append(image_path)
                except Exception as e:
                    print(f"Error processing image {img_key}: {e}")
        
        # Replace image placeholders in the problem statement
        modified_question = question
        for i in range(1, len(problem_images) + 1):
            placeholder = f"<image_start>[problem_image_{i}]<image_end>"
            if placeholder in modified_question:
                modified_question = modified_question.replace(placeholder, "<image>")
        
        # Replace image placeholders in the reasoning 
        modified_reasoning = reasoning
        for i in range(1, len(reasoning_images) + 1):
            placeholder = f"<image_start>[reasoning_image_{i}]<image_end>"
            if placeholder in modified_reasoning:
                modified_reasoning = modified_reasoning.replace(placeholder, "<image>")
        
        # Create the full reasoning trace with all THOUGHTs and FINAL ANSWER
        full_reasoning = modified_reasoning
        if answer:
            if not full_reasoning.endswith(answer) and "FINAL ANSWER" not in full_reasoning:
                full_reasoning += f"\n\nFINAL ANSWER: {answer}"
        
        # Create a single example with question as input and full reasoning as output
        formatted_example = {
            "input_text": f"QUESTION:\n{modified_question}",
            "input_imgs": problem_images,
            "input_img_paths": problem_image_paths,
            "label_text": full_reasoning,
            "label_imgs": reasoning_images,  # Include reasoning images
            "label_img_paths": reasoning_image_paths,  # Include paths to reasoning images
            "task": "reasoning",
            "train_task": "interleaved_reasoning"
        }
        formatted_data.append(formatted_example)
    
    # Save the formatted data
    json_output_path = os.path.join(output_dir, "formatted_data.json")
    
    # Convert to serializable format for JSON
    serializable_data = []
    for item in formatted_data:
        serializable_item = {
            "input_text": item["input_text"],
            "input_img_paths": item["input_img_paths"],
            "label_text": item["label_text"],
            "label_img_paths": item["label_img_paths"],
            "task": item["task"],
            "train_task": item["train_task"]
        }
        serializable_data.append(serializable_item)
    
    # Wrap the list in a dictionary with 'train' key
    serializable_dict = {
        "train": serializable_data
    }
    
    with open(json_output_path, 'w') as f:
        json.dump(serializable_dict, f, indent=2)
    
    print(f"Formatted data saved to {json_output_path}")
    print(f"Total examples created: {len(formatted_data)}")
    
    return formatted_data

def main():
    parser = argparse.ArgumentParser(description="Format dataset for training")
    parser.add_argument("--dataset_names", type=str, nargs="+", 
                        default=[
                                #  "vlm-reasoning-cot/ARC-AGI", 
                                #  "vlm-reasoning-cot/visual_jigsaw",
                                 "vlm-reasoning-cot/graph",
                                #  "vlm-reasoning-cot/Mazes",
                                #  "vlm-reasoning-cot/Physics",
                                #  "vlm-reasoning-cot/Tetris",
                                #  "vlm-reasoning-cot/MATH_geometry",
                                #  "vlm-reasoning-cot/chem",
                                #  "vlm-reasoning-cot/robot_planning",
                                #  "vlm-reasoning-cot/checker",
                                #  "vlm-reasoning-cot/connect_4",
                                #  "vlm-reasoning-cot/cipher",
                                #  "vlm-reasoning-cot/embodied_cot",
                                #  "vlm-reasoning-cot/visual_search",
                                 ],
                        help="List of HuggingFace dataset names or paths to process sequentially")
    parser.add_argument("--output_dir", type=str, default="formatted_data",
                       help="Base output directory for formatted data")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode (processes only a few examples)")
    args = parser.parse_args()
    
    # Create the base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # To collect all formatted data from all datasets
    all_formatted_data = []
    
    # Process each dataset sequentially
    for dataset_name in args.dataset_names:
        print(f"\n{'='*50}")
        print(f"PROCESSING DATASET: {dataset_name}")
        print(f"{'='*50}\n")
        
        # Create dataset-specific output directory (just for images)
        dataset_output_dir = os.path.join(args.output_dir, dataset_name.split('/')[-1])
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        try:
            print(f"Loading dataset {dataset_name}...")
            
            # Try to load the dataset with default configuration
            try:
                dataset = load_dataset(dataset_name, trust_remote_code=True)
                print("Dataset loaded with default configuration")
            except Exception as config_error:
                # Check if the error is about missing config
                if "Config name is missing" in str(config_error):
                    # Extract available configs from the error message
                    configs_match = re.search(r"Please pick one among the available configs: \[(.*?)\]", str(config_error))
                    if configs_match:
                        configs_str = configs_match.group(1)
                        configs = [c.strip("'\"") for c in configs_str.split(", ")]
                        print(f"Dataset has multiple configurations: {configs}")
                        
                        # Load and combine all chunks
                        chunks = []
                        for config in configs:
                            print(f"Loading configuration: {config}")
                            chunk_dataset = load_dataset(dataset_name, config, trust_remote_code=True)
                            chunks.append(chunk_dataset['train'])
                        
                        # Combine all chunks into one dataset
                        combined_train = concatenate_datasets(chunks)
                        dataset = {'train': combined_train}
                    else:
                        raise config_error
                else:
                    raise config_error
            
            print("Dataset loaded with the following splits:", dataset.keys())
            print(f"Train split size: {len(dataset['train'])}")
            
            print("First example feature keys:", list(dataset['train'][0].keys()))
            
            # In debug mode, only process a few examples
            if args.debug:
                print("Debug mode enabled, processing only 5 examples")
                dataset['train'] = dataset['train'].select(range(min(5, len(dataset['train']))))
            
            print(f"Formatting dataset {dataset_name}...")
            # Process dataset and get formatted data
            formatted_data = format_dataset(dataset['train'], dataset_output_dir)
            
            # Add dataset name to each example for tracking purposes
            for example in formatted_data:
                example["dataset_source"] = dataset_name
            
            # Add this dataset's formatted data to the full collection
            all_formatted_data.extend(formatted_data)
            
            print(f"Done with {dataset_name}!")
            print(f"Added {len(formatted_data)} examples")
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            print(f"Skipping to next dataset...")
            continue
    
    # Save the combined formatted data to a single JSON file
    combined_json_path = os.path.join(args.output_dir, "train_dataset.json")
    
    # Convert to serializable format for JSON
    serializable_data = []
    for item in all_formatted_data:
        serializable_item = {
            "input_text": item["input_text"],
            "input_img_paths": item["input_img_paths"],
            "label_text": item["label_text"],
            "label_img_paths": item["label_img_paths"],
            "task": item["task"],
            "train_task": item["train_task"],
            "dataset_source": item.get("dataset_source", "unknown")
        }
        serializable_data.append(serializable_item)
    
    # Wrap the list in a dictionary with 'train' key
    serializable_dict = {
        "train": serializable_data
    }
    
    with open(combined_json_path, 'w') as f:
        json.dump(serializable_dict, f, indent=2)
    
    print("\nAll datasets processed!")
    print(f"Total examples collected: {len(all_formatted_data)}")
    print(f"Combined data saved to {combined_json_path}")

if __name__ == "__main__":
    main() 