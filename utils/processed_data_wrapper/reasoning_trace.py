# coding=utf-8
import json
import os
import datasets
from PIL import Image
import base64
import io

_DESCRIPTION = """
Reasoning trace dataset with interleaved text and images.
"""

_CITATION = """
"""

class ReasoningTraceConfig(datasets.BuilderConfig):
    """BuilderConfig for Reasoning Trace Dataset."""

    def __init__(self, tasks, modes, data_dir, dataset_path=None, **kwargs):
        """BuilderConfig for Reasoning Trace.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ReasoningTraceConfig, self).__init__(**kwargs)
        self.tasks = tasks
        self.modes = modes
        self.data_dir = data_dir
        self.dataset_path = dataset_path


class ReasoningTrace(datasets.GeneratorBasedBuilder):
    """Reasoning Trace dataset."""

    BUILDER_CONFIG_CLASS = ReasoningTraceConfig
    BUILDER_CONFIGS = [
        ReasoningTraceConfig(
            name="reasoning_trace",
            version=datasets.Version("0.0.0"),
            description=_DESCRIPTION,
            tasks=["reasoning"],
            modes=["interleaved_reasoning"],
            data_dir="data_samples",
            dataset_path=None
        )
    ]

    DEFAULT_CONFIG_NAME = "reasoning_trace"

    def _info(self):
        features = datasets.Features(
            {
                'idx': datasets.Value('int32'),
                "input_text": datasets.Value("string"),
                "input_imgs": datasets.Sequence(datasets.Image()),
                "label_text": datasets.Value("string"),
                "label_imgs": datasets.Sequence(datasets.Image()),
                "label_img_paths": datasets.Sequence(datasets.Value("string")),
                "input_img_paths": datasets.Sequence(datasets.Value("string")),
                'task': datasets.Value('string'),
                'train_task': datasets.Value("string")
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = self.config.data_dir
        dataset_path = self.config.dataset_path

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": data_dir,
                    "dataset_path": dataset_path,
                    "split": "train"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split('dev'),
                gen_kwargs={
                    "data_dir": data_dir,
                    "dataset_path": dataset_path,
                    "split": "dev" if "dev" in self.config.modes else "train"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split('test'),
                gen_kwargs={
                    "data_dir": data_dir,
                    "dataset_path": dataset_path,
                    "split": "test" if "test" in self.config.modes else "train"
                },
            ),
        ]

    def _generate_examples(self, data_dir, dataset_path, split):
        """Generate examples from the Reasoning Trace dataset."""
        # Load the HuggingFace dataset
        if dataset_path:
            dataset = datasets.load_dataset(dataset_path, split=split)
        else:
            # Try to load from local files
            dataset_file = os.path.join(data_dir, f"{split}_reasoning_trace.json")
            if os.path.exists(dataset_file):
                with open(dataset_file, 'r') as f:
                    raw_data = json.load(f)
                dataset = datasets.Dataset.from_dict(raw_data)
            else:
                # Fallback - create an empty dataset if no data available
                return iter([])
        
        data_idx = 0
        for idx, item in enumerate(dataset):
            # Process the example to create interleaved format
            examples = process_reasoning_trace(item, data_dir, idx)
            
            for example in examples:
                yield data_idx, {
                    'idx': data_idx,
                    "input_text": example['input_text'],
                    "input_imgs": example["input_imgs"],
                    "label_text": example['label_text'],
                    "label_imgs": example['label_imgs'],
                    "label_img_paths": example['label_img_paths'],
                    "input_img_paths": example['input_img_paths'],
                    "task": example['task'],
                    "train_task": example['train_task']
                }
                data_idx += 1


def process_reasoning_trace(item, data_dir, example_idx):
    """Process a single reasoning trace example into multiple interleaved examples."""
    examples = []
    
    # Extract data from the item
    question = item.get('question', '')
    reasoning = item.get('reasoning', '')
    answer = item.get('answer', '')
    source_folder = item.get('source_folder', '')
    
    # 1. Process problem images
    problem_images = []
    problem_image_paths = []
    
    for i in range(1, 5):  # Assuming up to 4 problem images
        img_key = f'problem_image_{i}'
        img_base64_key = f'problem_image_{i}_base64'
        
        if img_key in item and item[img_key] is not None:
            if isinstance(item[img_key], (str, bytes)):
                # It's a path or raw image data
                img_path = os.path.join(data_dir, item[img_key]) if isinstance(item[img_key], str) else None
                try:
                    if img_path and os.path.exists(img_path):
                        img = Image.open(img_path).convert("RGB")
                        problem_images.append(img)
                        problem_image_paths.append(img_path)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            else:
                # It's already a PIL Image
                problem_images.append(item[img_key])
                problem_image_paths.append(f"problem_image_{i}_{example_idx}")
        
        # Try base64 format if available
        elif img_base64_key in item and item[img_base64_key]:
            try:
                img_data = base64.b64decode(item[img_base64_key])
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                problem_images.append(img)
                problem_image_paths.append(f"problem_image_{i}_{example_idx}_base64")
            except Exception as e:
                print(f"Error decoding base64 image: {e}")
    
    # 2. Process reasoning images
    reasoning_images = []
    reasoning_image_paths = []
    
    for i in range(1, 10):  # Assuming up to 9 reasoning images
        img_key = f'reasoning_image_{i}'
        img_base64_key = f'reasoning_image_{i}_base64'
        
        if img_key in item and item[img_key] is not None:
            if isinstance(item[img_key], (str, bytes)):
                img_path = os.path.join(data_dir, item[img_key]) if isinstance(item[img_key], str) else None
                try:
                    if img_path and os.path.exists(img_path):
                        img = Image.open(img_path).convert("RGB")
                        reasoning_images.append(img)
                        reasoning_image_paths.append(img_path)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            else:
                reasoning_images.append(item[img_key])
                reasoning_image_paths.append(f"reasoning_image_{i}_{example_idx}")
        
        # Try base64 format if available
        elif img_base64_key in item and item[img_base64_key]:
            try:
                img_data = base64.b64decode(item[img_base64_key])
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                reasoning_images.append(img)
                reasoning_image_paths.append(f"reasoning_image_{i}_{example_idx}_base64")
            except Exception as e:
                print(f"Error decoding base64 image: {e}")
    
    # 3. Create interleaved examples
    
    # First example: Problem statement with problem images -> First thought
    if reasoning and problem_images:
        thoughts = reasoning.split("THOUGHT ")
        
        # Extract initial thoughts that come before any images
        initial_thoughts = []
        for i, thought in enumerate(thoughts):
            if i == 0 and not thought.strip():
                continue  # Skip empty first split
            
            if "[reasoning_image_" in thought:
                break
            initial_thoughts.append(f"THOUGHT {thought}" if i > 0 else thought)
        
        initial_reasoning = "\n".join(initial_thoughts).strip()
        
        if initial_reasoning:
            # Create the first example
            modified_question = question
            for i, img_path in enumerate(problem_image_paths):
                placeholder = f"[problem_image_{i+1}]"
                if placeholder in modified_question:
                    modified_question = modified_question.replace(placeholder, "<image>")
            
            input_text = f"QUESTION:\n{modified_question}\n\nREASONING TRACE:"
            label_text = initial_reasoning
            
            examples.append({
                "task": "reasoning",
                "train_task": "interleaved_reasoning",
                "input_text": input_text,
                "input_imgs": problem_images,
                "input_img_paths": problem_image_paths,
                "label_text": label_text,
                "label_imgs": [],
                "label_img_paths": []
            })
    
    # Process remaining thoughts with reasoning images
    if reasoning_images:
        # Parse the reasoning to identify where images should be placed
        thoughts = reasoning.split("THOUGHT ")
        current_reasoning = []
        current_images = []
        current_image_paths = []
        
        for i, thought in enumerate(thoughts):
            if i == 0 and not thought.strip():
                continue  # Skip empty first split
                
            thought_text = f"THOUGHT {thought}" if i > 0 else thought
            
            # Check if this thought contains image placeholders
            if "[reasoning_image_" in thought_text:
                # Extract image index from placeholder
                for j in range(1, len(reasoning_images) + 1):
                    img_placeholder = f"[reasoning_image_{j}]"
                    if img_placeholder in thought_text:
                        # Split at this point to create a new example
                        parts = thought_text.split(img_placeholder)
                        
                        # Add first part to current reasoning
                        current_reasoning.append(parts[0].strip())
                        
                        # Create example with current state
                        if current_reasoning:
                            current_input = "\n".join(current_reasoning)
                            
                            # Previous reasoning becomes input, image becomes output
                            examples.append({
                                "task": "reasoning",
                                "train_task": "interleaved_reasoning",
                                "input_text": current_input,
                                "input_imgs": current_images.copy(),
                                "input_img_paths": current_image_paths.copy(),
                                "label_text": "<image>",
                                "label_imgs": [reasoning_images[j-1]],
                                "label_img_paths": [reasoning_image_paths[j-1]]
                            })
                            
                            # Add image to current state for next example
                            current_images.append(reasoning_images[j-1])
                            current_image_paths.append(reasoning_image_paths[j-1])
                            
                            # Continue with remaining text
                            current_reasoning = [parts[1].strip()] if len(parts) > 1 else []
                        
                        # For multiple images in one thought
                        if len(parts) > 1:
                            thought_text = parts[1]
            else:
                # No images in this thought, just add it to current reasoning
                current_reasoning.append(thought_text)
    
    # Final example: All previous reasoning + Final answer
    if current_reasoning and current_images:
        final_input = "\n".join(current_reasoning)
        
        examples.append({
            "task": "reasoning",
            "train_task": "interleaved_reasoning",
            "input_text": final_input,
            "input_imgs": current_images,
            "input_img_paths": current_image_paths,
            "label_text": f"FINAL ANSWER:\n{answer}",
            "label_imgs": [],
            "label_img_paths": []
        })
    
    return examples 