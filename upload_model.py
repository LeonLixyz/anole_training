import os
import sys
import shutil
from pathlib import Path
import json
from huggingface_hub import HfApi, create_repo, upload_folder
from getpass import getpass

# Paths
MERGED_MODEL_PATH = "./merged_model"
LORA_PATH = "/workspace/anole_training/outputs/geometry_reasoning_run/geometry_reasoning_runimage_seq_len-1024-train-anole-hyper-train1val1lr0.0002-geometry_reasoning-prompt_anole-42/checkpoint-160"
UPLOAD_DIR = "./hf_upload"

# Your HF username and desired repo name
REPO_NAME = "YOUR_USERNAME/anole-lora-merged"  # Change this!

# Create upload directory
upload_dir = Path(UPLOAD_DIR)
upload_dir.mkdir(parents=True, exist_ok=True)

# Copy all files from the merged model directory
src_dir = Path(MERGED_MODEL_PATH)
for file in src_dir.glob("*"):
    if file.is_file():
        shutil.copy(file, upload_dir / file.name)
        print(f"Copied {file.name}")

# Create model card (README.md)
model_card = """---
pipeline_tag: text-generation
language: en
license: cc-by-nc-4.0
tags:
  - anole
  - chameleon
  - multimodal
  - lora
  - text-to-image
---

# Anole LoRA Merged Model

This is a merged model that combines the [Anole 7B](https://huggingface.co/GAIR/Anole-7b-v0.1) base model with a LoRA adapter for improved physics reasoning.

## Model Details

- **Base Model:** GAIR/Anole-7b-v0.1
- **LoRA Parameters:**
  - r: 8
  - alpha: 16
  - Target Modules: q_proj, k_proj, v_proj, o_proj

## Usage

This model can be used for multimodal generation tasks like the base Anole model, but with enhanced capabilities from the LoRA fine-tuning.
"""

with open(upload_dir / "README.md", "w") as f:
    f.write(model_card)

# Get HF token
token = getpass("Enter your Hugging Face token: ")

# Upload to Hugging Face
api = HfApi(token=token)

# Create the repository
create_repo(repo_id=REPO_NAME, token=token, exist_ok=True)
print(f"Repository {REPO_NAME} is ready")

# Upload the files
upload_folder(
    folder_path=UPLOAD_DIR,
    repo_id=REPO_NAME,
    token=token
)
print(f"Successfully uploaded model to https://huggingface.co/{REPO_NAME}")