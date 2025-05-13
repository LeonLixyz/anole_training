#!/usr/bin/env python
import os
import argparse
import shutil
import importlib.util
import sys
import torch
import json
import hashlib
import pathlib
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer
from huggingface_hub import login, HfApi

def _sha256(fname, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(fname, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def produce_checkpoint_files(model, out_dir):
    """
    Produce consolidated format files compatible with Chameleon loader:
    - consolidated.pth
    - config.json
    - params.json
    - consolidate_params.json
    - checklist.chk
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. consolidated.pth (single-file pickled state-dict with correct structure)
    cons_file = out_dir / "consolidated.pth"
    
    # Create the correct structure expected by Chameleon loader
    # The loader expects a dict with a 'model' key
    state_dict = model.state_dict()
    save_dict = {"model": state_dict}
    
    torch.save(save_dict, cons_file)
    print(f"Saved consolidated weights to {cons_file}")

    # 2. config.json (model config)
    cfg_file = out_dir / "config.json"
    if hasattr(model, 'config'):
        with cfg_file.open("w") as f:
            if hasattr(model.config, 'to_dict'):
                json.dump(model.config.to_dict(), f, indent=2)
            else:
                json.dump(model.config, f, indent=2)
    else:
        with cfg_file.open("w") as f:
            json.dump({}, f)

    # 3. params.json (tensor names & shapes)
    pjson = {
        k: list(v.shape) for k, v in state_dict.items()
    }
    with (out_dir / "params.json").open("w") as f:
        json.dump(pjson, f, indent=2)

    # 4. consolidate_params.json (manifest)
    with (out_dir / "consolidate_params.json").open("w") as f:
        json.dump(
            {
                "model_file": cons_file.name,
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "sha256": _sha256(cons_file),
            },
            f,
            indent=2,
        )

    # 5. checklist.chk (empty marker file)
    (out_dir / "checklist.chk").touch()

    print("✅ Created consolidated checkpoint files compatible with Chameleon loader")

def merge_and_upload(
    adapter_path,
    output_path,
    repo_name,
    cached_model_path=None,
    private=False,
    token=None,
    readme_path=None,
    verify=True
):
    # Login to Hugging Face
    if token:
        login(token=token)
    elif os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])
    else:
        login()
    
    # Load adapter config 
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = peft_config.base_model_name_or_path
    print(f"Base model: {base_model_name}")
    
    # Determine model path to use
    model_path = cached_model_path or base_model_name
    print(f"Using model from: {model_path}")
    
    # Check if model_utils is in the path
    if os.path.exists("model_utils"):
        print("Found model_utils directory, adding to path...")
        sys.path.append(os.getcwd())
    
    # For custom models like Anole, we need to use the correct class
    try:
        # Try to import the custom module
        from model_utils.wrapped_visualizer import AnoleforConditionalGeneration
        print("Successfully imported AnoleforConditionalGeneration")
        
        # Load base model with the custom class
        print("Loading base model...")
        base_model = AnoleforConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
        )
    except ImportError as e:
        print(f"Warning: Could not import custom model class: {e}")
        print("Trying with generic AutoModel...")
        
        from transformers import AutoModel
        print("Loading base model with AutoModel...")
        base_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
        )
    
    # Load tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load adapter weights
    print("Loading adapter weights...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Verify adapter weights if requested
    if verify:
        print("Verifying adapter weights...")
        # Check that all target modules in peft_config are present in the model
        target_modules = peft_config.target_modules
        adapter_name = list(model.peft_config.keys())[0]  # Default adapter name
        
        # Check that adapter modules match expected configuration
        num_adapter_modules = 0
        
        # Safer way to check LoRA modules
        for name, module in model.named_modules():
            if any(target in name for target in target_modules):
                # This is a target module
                if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
                    num_adapter_modules += 1
        
        print(f"Found {num_adapter_modules} adapter modules")
        print(f"Expected target modules: {target_modules}")
        
        # Validate that adapter modules are properly attached
        if num_adapter_modules == 0:
            print("Warning: No adapter modules found in the model. This might indicate a problem with adapter loading.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                raise ValueError("Aborting due to adapter verification failure")
        
        print("Adapter verification complete")
    
    # Merge adapter with base model
    print("Merging weights...")
    merged_model = model.merge_and_unload()
    
    # Create both formats
    
    # 1. First save in standard HuggingFace format (safetensors)
    print(f"Saving model in HuggingFace format to {output_path}...")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    base_tokenizer.save_pretrained(output_path)
    
    # Copy processor config and any other necessary files
    for file in ["processor_config.json", "preprocessor_config.json"]:
        if os.path.exists(os.path.join(adapter_path, file)):
            shutil.copy(
                os.path.join(adapter_path, file),
                os.path.join(output_path, file)
            )
    
    # Copy README if provided
    if readme_path and os.path.exists(readme_path):
        shutil.copy(readme_path, os.path.join(output_path, "README.md"))
    
    # 2. Create consolidated checkpoint files in Chameleon-compatible format
    print("Creating consolidated checkpoint format...")
    produce_checkpoint_files(merged_model, output_path)
    
    print("Model successfully merged and saved in both formats!")
    
    # Upload to Hugging Face
    print(f"Uploading model to {repo_name}...")
    api = HfApi()
    
    # Create repository if it doesn't exist
    api.create_repo(repo_name, private=private, exist_ok=True)
    
    # Upload all files
    api.upload_folder(
        folder_path=output_path,
        repo_id=repo_name,
        commit_message="Upload merged model in both safetensors and consolidated formats"
    )
    
    print(f"Model successfully uploaded to https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter weights with base model and upload to Hugging Face")
    parser.add_argument("--adapter_path", type=str, 
                        default="outputs/geometry_reasoning_run/geometry_reasoning_runimage_seq_len-1024-train-anole-hyper-train1val1lr3e-05-geometry_reasoning-prompt_anole-42",
                        help="Path to the adapter model")
    parser.add_argument("--output_path", type=str, default="merged_model",
                        help="Path where to save the merged model")
    parser.add_argument("--repo_name", type=str, required=True,
                        help="Repository name in format 'username/model-name'")
    parser.add_argument("--cached_model_path", type=str, 
                        default="/root/.cache/huggingface/hub/models--leloy--Anole-7b-v0.1-hf/snapshots/96df52301e844d8a624a13953051ead4c008343b",
                        help="Path to the cached model")
    parser.add_argument("--private", action="store_true",
                        help="Make the repository private")
    parser.add_argument("--token", type=str,
                        help="Hugging Face token (if not using environment variable)")
    parser.add_argument("--readme_path", type=str, default="merged_model_README.md",
                        help="Path to README file to include in the repository")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip adapter verification")
    
    args = parser.parse_args()
    merge_and_upload(
        args.adapter_path,
        args.output_path,
        args.repo_name,
        args.cached_model_path,
        args.private,
        args.token,
        args.readme_path,
        not args.no_verify
    ) 
