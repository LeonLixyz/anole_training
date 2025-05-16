import os
from datasets import load_dataset, concatenate_datasets
import re
import json
import os
import shutil
from multiprocessing import Pool
from functools import partial
from huggingface_hub import HfApi
from huggingface_hub import constants
print(constants.HF_HUB_CACHE)


# Define temporary directory for processing
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_processing")

# Natural sort so that placeholders like reasoning_image_10 come after reasoning_image_2
def _natural_sort_key(s: str):
    """Split the string into digit and non-digit components for natural sorting."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def generate_chunk_names(config):
    """Generate chunk names based on configuration."""
    num_chunks = config['num_chunks']
    
    if "Zebra-CoT" in config['original_dataset']:
        return [f"2D Visual Reasoning-{config['subdataset_name']}-chunk_{i:04d}" for i in range(1, num_chunks + 1)]
    
    return [f'chunk_{i:04d}' for i in range(1, num_chunks + 1)]

def process_dataset(config):
    """Process a single dataset configuration."""
    try:
        print(f"Processing {config['original_dataset']} -> {config['hf_repo_id']}/{config['subdataset_name']}")
        
        process_tmp_dir = os.path.join(TMP_DIR, f"process_{os.getpid()}")
        os.makedirs(process_tmp_dir, exist_ok=True)
        
        os.environ["HF_DATASETS_CACHE"] = os.path.join(process_tmp_dir, "datasets_cache")
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(process_tmp_dir, "hub_cache")
        
        if config['num_chunks'] > 0:
            chunk_names = generate_chunk_names(config)
            
            print(f"Using chunk names: {chunk_names[:3]}... (total: {len(chunk_names)})")
            
            # Download all chunks
            datasets = [load_dataset(config['original_dataset'], name=chunk, cache_dir=os.path.join(process_tmp_dir, "raw_datasets")) for chunk in chunk_names]
            
            # Merge all splits for each chunk
            merged_ds = {}
            for split in datasets[0].keys():
                merged_ds[split] = concatenate_datasets([ds[split] for ds in datasets])
        else:
            # For datasets without chunks, load directly
            print(f"Loading single dataset without chunks: {config['original_dataset']}")
            merged_ds = load_dataset(config['original_dataset'], cache_dir=os.path.join(process_tmp_dir, "raw_datasets"))
        
        # Drop base64 fields and sort columns in natural order
        for split in merged_ds:
            # Identify columns to keep (non-base64)
            columns_to_keep = [col for col in merged_ds[split].column_names if not col.endswith('_base64')]
            
            # Sort columns using natural sort
            columns_to_keep = sorted(columns_to_keep, key=_natural_sort_key)
            
            # Select only the non-base64 columns in the naturally sorted order
            merged_ds[split] = merged_ds[split].select_columns(columns_to_keep)
            
            # Rename columns as requested
            column_mapping = {
                'question': 'Question',
                'reasoning': 'Text Reasoning Trace',
                'answer': 'Final Answer'
            }
            
            # Apply renaming for columns that exist
            rename_dict = {old: new for old, new in column_mapping.items() if old in merged_ds[split].column_names}
            if rename_dict:
                merged_ds[split] = merged_ds[split].rename_columns(rename_dict)
            
            # Reorder columns to put Question first, followed by Text Reasoning Trace and Final Answer
            new_column_order = []
            
            # Add Question first if it exists
            if 'Question' in merged_ds[split].column_names:
                new_column_order.append('Question')
            
            # Add Text Reasoning Trace next if it exists
            if 'Text Reasoning Trace' in merged_ds[split].column_names:
                new_column_order.append('Text Reasoning Trace')
            
            # Add Final Answer next if it exists
            if 'Final Answer' in merged_ds[split].column_names:
                new_column_order.append('Final Answer')
            
            # Add problem images
            problem_images = sorted([col for col in merged_ds[split].column_names if col.startswith('problem_image_')], 
                                   key=_natural_sort_key)
            new_column_order.extend(problem_images)
            
            # Add reasoning images
            reasoning_images = sorted([col for col in merged_ds[split].column_names if col.startswith('reasoning_image_')],
                                     key=_natural_sort_key)
            new_column_order.extend(reasoning_images)
            
            # Add any remaining columns
            remaining_cols = [col for col in merged_ds[split].column_names 
                             if col not in new_column_order]
            new_column_order.extend(remaining_cols)
            
            # Set the new column order
            merged_ds[split] = merged_ds[split].select_columns(new_column_order)
        
        print(f"Dataset processed, uploading to {config['hf_repo_id']}")
        
        # Create a temporary processed directory for this dataset
        processed_dir = os.path.join(process_tmp_dir, f"processed_{config['subdataset_name'].replace(' ', '_')}")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Upload the processed dataset to Huggingface
        for split, dataset in merged_ds.items():
            # Save the dataset locally first to avoid memory issues
            dataset_path = os.path.join(processed_dir, split)
            dataset.save_to_disk(dataset_path)
            
            # Upload from the saved location
            dataset.push_to_hub(
                repo_id=config['hf_repo_id'],
                config_name=config['subdataset_name'],
                split=split
            )
        
        # Clean up temporary files for this dataset
        print(f"Cleaning up temporary files for {config['subdataset_name']}")
        shutil.rmtree(process_tmp_dir, ignore_errors=True)
        
        print(f"Successfully uploaded {config['subdataset_name']} to {config['hf_repo_id']}")
        return config['subdataset_name']
    except Exception as e:
        print(f"Error processing {config['subdataset_name']}: {str(e)}")
        return f"{config['subdataset_name']} (FAILED)"

def main():
    # Load dataset configurations
    config_path = "dataset_config.json"
    with open(config_path, 'r') as f:
        configs = json.load(f)
    
    print(f"Found {len(configs)} dataset configurations to process")
    
    # Process only 4 datasets at a time to avoid disk space issues
    num_processes = 4
    print(f"Processing with {num_processes} parallel workers")
    
    # Process datasets in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_dataset, configs)
    
    # Clean up all temporary files
    print("Cleaning up all temporary files")
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    
    print(f"All datasets processed: {results}")

if __name__ == "__main__":
    main()


