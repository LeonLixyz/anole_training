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

def convert_hf_to_chameleon(state_dict):
    """
    Convert Hugging Face model format to Chameleon format
    """
    chameleon_dict = {}
    
    # Mapping rules
    for key, value in state_dict.items():
        # Skip VQ model components
        if 'vqmodel' in key:
            continue
            
        # Embedding layer
        if key == 'model.embed_tokens.weight':
            chameleon_dict['tok_embeddings.weight'] = value
        
        # Match layers.X patterns
        elif 'model.layers.' in key:
            # Extract layer number
            parts = key.split('.')
            layer_idx = parts[2]
            
            # Attention component mappings
            if 'self_attn.q_proj.weight' in key:
                # Store for later concatenation
                q_weight = value
                chameleon_key = f'layers.{layer_idx}.attention.q_weight_temp'
                chameleon_dict[chameleon_key] = q_weight
            
            elif 'self_attn.k_proj.weight' in key:
                # Store for later concatenation
                k_weight = value
                chameleon_key = f'layers.{layer_idx}.attention.k_weight_temp'
                chameleon_dict[chameleon_key] = k_weight
            
            elif 'self_attn.v_proj.weight' in key:
                # Store for later concatenation
                v_weight = value
                chameleon_key = f'layers.{layer_idx}.attention.v_weight_temp'
                chameleon_dict[chameleon_key] = v_weight
            
            elif 'self_attn.o_proj.weight' in key:
                chameleon_key = f'layers.{layer_idx}.attention.wo.weight'
                chameleon_dict[chameleon_key] = value
            
            # Normalization layers
            elif 'self_attn.q_norm.weight' in key:
                chameleon_key = f'layers.{layer_idx}.attention.q_normalization.weight'
                chameleon_dict[chameleon_key] = value
            
            elif 'self_attn.q_norm.bias' in key:
                chameleon_key = f'layers.{layer_idx}.attention.q_normalization.bias'
                chameleon_dict[chameleon_key] = value
            
            elif 'self_attn.k_norm.weight' in key:
                chameleon_key = f'layers.{layer_idx}.attention.k_normalization.weight'
                chameleon_dict[chameleon_key] = value
            
            elif 'self_attn.k_norm.bias' in key:
                chameleon_key = f'layers.{layer_idx}.attention.k_normalization.bias'
                chameleon_dict[chameleon_key] = value
            
            # Feed forward components
            elif 'mlp.gate_proj.weight' in key:
                gate_weight = value
                chameleon_key = f'layers.{layer_idx}.feed_forward.gate_weight_temp'
                chameleon_dict[chameleon_key] = gate_weight
            
            elif 'mlp.up_proj.weight' in key:
                up_weight = value
                chameleon_key = f'layers.{layer_idx}.feed_forward.up_weight_temp'
                chameleon_dict[chameleon_key] = up_weight
            
            elif 'mlp.down_proj.weight' in key:
                chameleon_key = f'layers.{layer_idx}.feed_forward.w2.weight'
                chameleon_dict[chameleon_key] = value
            
            # Layer norms
            elif 'input_layernorm.weight' in key:
                chameleon_key = f'layers.{layer_idx}.attention_norm.weight'
                chameleon_dict[chameleon_key] = value
            
            elif 'post_attention_layernorm.weight' in key:
                chameleon_key = f'layers.{layer_idx}.ffn_norm.weight'
                chameleon_dict[chameleon_key] = value
        
        # Model final norm and output
        elif key == 'model.norm.weight':
            chameleon_dict['norm.weight'] = value
        
        elif key == 'lm_head.weight':
            chameleon_dict['output.weight'] = value
    
    # Process all layers to combine Q, K, V weights and feed forward
    for layer_idx in range(32):  # Assuming up to 32 layers
        # Check if we have all three weights
        q_key = f'layers.{layer_idx}.attention.q_weight_temp'
        k_key = f'layers.{layer_idx}.attention.k_weight_temp'
        v_key = f'layers.{layer_idx}.attention.v_weight_temp'
        
        if q_key in chameleon_dict and k_key in chameleon_dict and v_key in chameleon_dict:
            # Concatenate QKV
            q_weight = chameleon_dict.pop(q_key)
            k_weight = chameleon_dict.pop(k_key)
            v_weight = chameleon_dict.pop(v_key)
            
            # Concatenate the weights along appropriate dimension
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            chameleon_dict[f'layers.{layer_idx}.attention.wqkv.weight'] = qkv_weight
        
        # Check if we have both feed forward weights
        gate_key = f'layers.{layer_idx}.feed_forward.gate_weight_temp'
        up_key = f'layers.{layer_idx}.feed_forward.up_weight_temp'
        
        if gate_key in chameleon_dict and up_key in chameleon_dict:
            # Concatenate gate and up projections
            gate_weight = chameleon_dict.pop(gate_key)
            up_weight = chameleon_dict.pop(up_key)
            
            # Concatenate the weights
            w13_weight = torch.cat([gate_weight, up_weight], dim=0)
            chameleon_dict[f'layers.{layer_idx}.feed_forward.w13.weight'] = w13_weight
    
    return chameleon_dict

def produce_checkpoint_files(model, out_dir):
    """
    Produce only consolidated.pth file with transformer backbone only
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get state dict
    state_dict = model.state_dict()
    
    print(f"Original state dict size: {len(state_dict)} keys")
    
    # Convert HF format to Chameleon format
    print("Converting weights from HF format to Chameleon format...")
    chameleon_dict = convert_hf_to_chameleon(state_dict)
    
    print(f"Converted state dict size: {len(chameleon_dict)} keys")
    print(f"Sample keys: {list(chameleon_dict.keys())[:5]}")
    
    # Create the output dictionary with the expected format
    save_dict = {"model": chameleon_dict}
    
    # Convert to half precision
    for k, v in save_dict["model"].items():
        if v.dtype == torch.float32:
            save_dict["model"][k] = v.half()
    
    # Save consolidated.pth file
    cons_file = out_dir / "consolidated.pth"
    torch.save(save_dict, cons_file)
    print(f"Saved consolidated weights to {cons_file} in half precision")

def print_model_structure(model):
    """
    Print model structure to show components
    """
    print("\n=== MODEL STRUCTURE ===")
    # Print top-level attributes
    print("Top-level components:")
    for name, child in model.named_children():
        print(f"  - {name}")
    
    # Print detailed structure
    print("\nDetailed structure (first few levels):")
    for name, _ in model.named_modules():
        if name.count('.') <= 2:  # Limit depth to make output manageable
            print(f"  - {name}")
    
    # Print sample state dict keys
    print("\nSample state dict keys:")
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    for key in keys[:10]:  # Show first 10 keys
        print(f"  - {key}")

def print_lora_components(model):
    """
    Print LoRA components to check what was trained
    """
    print("\n=== LORA COMPONENTS ===")
    lora_modules = []
    
    # Get adapter name
    adapter_name = list(model.peft_config.keys())[0] if hasattr(model, 'peft_config') else None
    if not adapter_name:
        print("No LoRA adapter found")
        return
    
    # Check model for LoRA modules
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
            lora_modules.append(name)
    
    print(f"Found {len(lora_modules)} LoRA modules:")
    for module in lora_modules:
        print(f"  - {module}")
    
    # Check if any VQ model components were trained
    vq_lora = [m for m in lora_modules if 'vq_model' in m]
    if vq_lora:
        print(f"\nWARNING: Found {len(vq_lora)} LoRA modules in VQ model:")
        for module in vq_lora:
            print(f"  - {module}")
    else:
        print("\nNo LoRA modules found in VQ model components - good!")

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
    
    # Print model structure to show components
    print_model_structure(base_model)
    
    # Load tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load adapter weights
    print("Loading adapter weights...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Print LoRA components
    print_lora_components(model)
    
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
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save checkpoint in Chameleon-compatible format
    print("Creating consolidated checkpoint in half precision (fp16)...")
    produce_checkpoint_files(merged_model, output_path)
    
    print("Model successfully merged and saved!")
    
    # Upload to Hugging Face
    print(f"Uploading model to {repo_name}...")
    api = HfApi()
    
    # Create repository if it doesn't exist
    api.create_repo(repo_name, private=private, exist_ok=True)
    
    # Upload only the consolidated.pth file
    api.upload_file(
        path_or_fileobj=os.path.join(output_path, "consolidated.pth"),
        path_in_repo="consolidated.pth",
        repo_id=repo_name,
        commit_message="Upload merged model (transformer backbone only, fp16)"
    )
    
    # Create params.json and consolidate_params.json for Chameleon compatibility
    params = {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 32,
        "vocab_size": 65536,
        "multiple_of": 256,
        "ffn_dim_multiplier": 1.0,
        "norm_eps": 1e-5,
        "model": {
            "rope_theta": 10000.0,
            "qk_normalization": True,
            "swin_norm": False
        }
    }
    
    consolidate_params = {
        "checkpoint_format": "consolidated.pth",
    }
    
    # Save parameter files
    with open(os.path.join(output_path, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
    
    with open(os.path.join(output_path, "consolidate_params.json"), "w") as f:
        json.dump(consolidate_params, f, indent=2)
    
    # Upload parameter files
    api.upload_file(
        path_or_fileobj=os.path.join(output_path, "params.json"),
        path_in_repo="params.json",
        repo_id=repo_name,
        commit_message="Upload model parameters"
    )
    
    api.upload_file(
        path_or_fileobj=os.path.join(output_path, "consolidate_params.json"),
        path_in_repo="consolidate_params.json",
        repo_id=repo_name,
        commit_message="Upload consolidation parameters"
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