import os
import torch
import json

from datasets import load_dataset, concatenate_datasets, DatasetDict

from utils.tokenized_dataset import AnoleTokenizedDataset
from utils.interleaved_tokenized_dataset import InterleaveAnoleTokenizedDataset

import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

def load_data(dataset, data_dir, custom_dataset_path=None):
    data_list = []
    if 'interleaved_maze' in dataset:
        data = load_dataset(
            "utils/processed_data_wrapper/interleaved_maze.py", 
            tasks=['simulation'], 
            modes=['single_step_visualization', 'action_reasoning'], 
            data_dir=data_dir,
            trust_remote_code=True
        )
        print(f"Interleaved Maze: {len(data['train'])}")
        data_list.append(data)
    if 'geometry_reasoning' in dataset:
        # Use the custom geometry reasoning data wrapper
        data = load_dataset(
            "utils/processed_data_wrapper/geometry_reasoning.py", 
            tasks=['reasoning'], 
            modes=['interleaved_reasoning'], 
            data_dir=data_dir,
            dataset_path=custom_dataset_path,
            trust_remote_code=True,
            download_mode="force_redownload"
        )
        print(f"data dir: {data_dir}")
        print(f"custom_dataset_path: {custom_dataset_path}")
        print(f"Geometry Reasoning: {len(data['train'])}")
        data_list.append(data)
    if 'frozenlake' in dataset:
        data = load_dataset(
            "utils/processed_data_wrapper/frozenlake.py", 
            tasks=['simulation'], 
            modes=['single_step_visualization', 'action_reasoning'], 
            data_dir=data_dir,
            trust_remote_code=True
        )
        print(f"FrozenLake: {len(data['train'])}")
        data_list.append(data)
    if 'reasoning_trace' in dataset:
        # Print data directory for debugging
        print(f"Data directory path: {os.path.abspath(data_dir)}")
        
        data = load_dataset(
            "utils/processed_data_wrapper/reasoning_trace.py", 
            tasks=['reasoning'], 
            modes=['interleaved_reasoning'], 
            data_dir=data_dir,
            dataset_path=custom_dataset_path,
            trust_remote_code=True
        )
        print(f"Reasoning Trace: {len(data['train'])}")
        data_list.append(data)
    if 'custom' in dataset and custom_dataset_path:
        # Load custom dataset from HuggingFace or local path
        try:
            if os.path.exists(custom_dataset_path):
                # Local file or directory
                if custom_dataset_path.endswith('.json'):
                    with open(custom_dataset_path, 'r') as f:
                        custom_data = json.load(f)
                    
                    # Check if the data already has a 'train' key
                    if isinstance(custom_data, dict) and 'train' in custom_data:
                        # Create Dataset from the 'train' data
                        train_dataset = datasets.Dataset.from_list(custom_data['train'])
                        data = datasets.DatasetDict({'train': train_dataset})
                    else:
                        # Create Dataset from the whole JSON
                        dataset_obj = datasets.Dataset.from_list(custom_data)
                        data = datasets.DatasetDict({'train': dataset_obj})
                else:
                    # Assuming it's a directory with the dataset files
                    data = load_dataset(custom_dataset_path, trust_remote_code=True)
            else:
                # HuggingFace dataset
                print(f"Loading HuggingFace dataset: {custom_dataset_path}")
                data = load_dataset(custom_dataset_path, trust_remote_code=True)
            
            print(f"Custom Dataset: {len(data['train'])}")
            
            # Debug: print the dataset structure
            print("\n===== DATASET FEATURES =====")
            print(f"Dataset features: {data['train'].features}")
            print("\n===== DATASET SAMPLE =====")
            sample = data['train'][0]
            print(f"Sample keys: {list(sample.keys())}")
            for key, value in sample.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    print(f"{key}: {value}")
                elif hasattr(value, 'shape'):
                    print(f"{key}: Array with shape {value.shape}")
                else:
                    print(f"{key}: {type(value)}")
            
            # If this is a geometry/reasoning dataset, we need to modify it to match our expected format
            # Check for typical fields like 'question', 'reasoning', etc.
            if 'question' in sample or 'reasoning' in sample:
                print("\n===== TRANSFORMING DATASET FORMAT =====")
                # Create a dataset with our expected format
                new_data = []
                for item in data['train']:
                    # Check for and process any images
                    input_imgs = []
                    if 'problem_image_1' in item and item['problem_image_1'] is not None:
                        input_imgs.append(item['problem_image_1'])
                    
                    # Create formatted entry
                    formatted_item = {
                        'input_text': item.get('question', ''),
                        'label_text': item.get('answer', ''),
                        'input_imgs': input_imgs,
                        'label_imgs': [],
                        'task': 'reasoning',
                        'train_task': 'interleaved_reasoning',
                        'input_img_paths': ['problem_image_1'] if input_imgs else [],
                        'label_img_paths': [],
                    }
                    new_data.append(formatted_item)
                
                # Create a new dataset with the transformed data
                formatted_dataset = datasets.Dataset.from_list(new_data)
                data = datasets.DatasetDict({'train': formatted_dataset})
                print(f"Transformed dataset size: {len(data['train'])}")
                
                # Show transformed sample
                print("\n===== TRANSFORMED SAMPLE =====")
                sample = data['train'][0]
                print(f"Sample keys: {list(sample.keys())}")
                for key, value in sample.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        print(f"{key}: {value}")
                    elif hasattr(value, 'shape'):
                        print(f"{key}: Array with shape {value.shape}")
                    else:
                        print(f"{key}: {type(value)}")

            data_list.append(data)
        except Exception as e:
            print(f"Error loading custom dataset: {e}")
            import traceback
            traceback.print_exc()

    if not data_list:
        raise ValueError(f"No datasets loaded for: {dataset}")

    concatenate_data = dict()
    for k in data_list[0].keys():
        if k in ['train']:
            concatenate_data[k] = concatenate_datasets([i[k] for i in data_list])
        else:
            concatenate_data[k] = concatenate_datasets([i[k].shuffle(seed=42).select(range(min(800, len(i[k])))) for i in data_list])
    return concatenate_data

def tokenize_dataset(train_split, eval_split, test_split, model, processor, **kwargs):
    tokenized_data = dict()

    data_name = kwargs.pop("data_name")
    
    # Add code to analyze and print raw text lengths
    if train_split:
        input_lengths = [len(example.get('input_text', '')) for example in train_split]
        label_lengths = [len(example.get('label_text', '')) for example in train_split]
        
        print(f"Raw text statistics for training data:")
        print(f"Input text - Min: {min(input_lengths)}, Max: {max(input_lengths)}, Avg: {sum(input_lengths)/len(input_lengths):.2f}")
        print(f"Label text - Min: {min(label_lengths)}, Max: {max(label_lengths)}, Avg: {sum(label_lengths)/len(label_lengths):.2f}")

    max_source_length = 2048
    # print(f"Max source length: {max_source_length}")

    max_target_length = 4096 + 1024
    # print(f"Max target length: {max_target_length}")

    if not kwargs["interleave"]:
        tokenized_dataset_type = AnoleTokenizedDataset
    else:
        tokenized_dataset_type = InterleaveAnoleTokenizedDataset

    if train_split:
        tokenized_train = tokenized_dataset_type(
            dataset=train_split,
            split='train',
            model=model,
            processor=processor,
            input_max_length=max_source_length, 
            label_max_length=max_target_length,
            **kwargs
        )
        tokenized_data['train'] = tokenized_train
    if eval_split:
        tokenized_eval = tokenized_dataset_type(
            dataset=eval_split,
            split='eval',
            model=model,
            processor=processor,
            input_max_length=max_source_length,
            label_max_length=max_target_length,
            **kwargs
        )
        tokenized_data['eval'] = tokenized_eval
    if test_split:
        tokenized_test = tokenized_dataset_type(
            dataset=test_split,
            split='test',
            model=model,
            processor=processor,
            input_max_length=max_source_length,
            label_max_length=max_target_length,
            **kwargs
        )
        tokenized_data['test'] = tokenized_test
    return tokenized_data, max_source_length, max_target_length


def get_image_token_num(model, processor, resolution):
    if hasattr(processor, 'image_seq_length'):
        return processor.image_seq_length
    elif hasattr(model, get_image_token_num):
        return model.get_image_token_num(resolution=resolution)
    else:
        raise NotImplementedError("Either model should have the get_image_token_num method or processor should have the iamge_seq_length property. ")