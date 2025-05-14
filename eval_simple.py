import os
import torch
import argparse
import numpy as np
from PIL import Image
from utils.load_model import load_model
from utils.postprocess_logits_utils import split_token_sequence

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate text and images with Anole model")
    parser.add_argument("--model_path", type=str, default="x", help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="generation_results", help="Directory to save outputs")
    parser.add_argument("--prompt", type=str, default="Draw interleaved text and image and show me how to cook an egg", 
                       help="Text prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Maximum number of new tokens to generate")
    parser.add_argument("--image_seq_length", type=int, default=1024, help="Image sequence length parameter")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading model from {args.model_path}")
    
    # Setup model args with ALL required attributes
    model_args = type('Args', (), {
        "model": "anole",
        "decoder_type": "anole",
        "image_seq_length": args.image_seq_length,
        "model_ckpt": args.model_path,
        "input_format": "anole",
        "no_perceptual_loss": True,
        # Additional required attributes
        "do_train": False,
        "do_eval": True,
        "do_predict": False,
        "data": ["custom"],
        "data_dir": "data_samples",
        "custom_dataset_path": None,
        "note": "",
        "load_last_checkpoint": False,
        "cfg_path": "cfg",
        "patience": 5,
        "output": "outputs",
        "report_to": "none",
        "cache_dir": None,
        "seed": 42,
        "local_rank": 0,
        "toy": False,
        "train_bz": None,
        "val_bz": None,
        "grad_acc": None
    })

    # Load the model and processor
    print("Loading model and processor...")
    model_processor = load_model(model_args)
    model = model_processor['model'].to("cuda")
    processor = model_processor["processor"]
    print("Model loaded successfully")

    # Process input
    print(f"Generating for prompt: '{args.prompt}'")
    inputs = processor(
        text=args.prompt,
        return_tensors="pt"
    ).to("cuda")

    # Generate
    print("Generating response...")
    with torch.no_grad():
        # First create a generation parameters dict
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": 1,
            "do_sample": True,
            "temperature": args.temperature,
            "multimodal_generation_mode": "interleaved-text-image"
        }
        
        # Then call generate with both sets of parameters
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **gen_kwargs
        )

    # Post-process output to extract text and images
    print("Processing results...")
    generated_results = split_token_sequence(
        tokens=outputs, 
        image_seq_length=model.image_token_num,
        boi=model.config.boi_token_id, 
        eoi=model.config.eoi_token_id,
        max_length=outputs.shape[-1],
        pad_token_id=model.config.pad_token_id
    )

    # Save generated text
    text_tokens = generated_results['texts'][0]
    generated_text = processor.decode(text_tokens, skip_special_tokens=False)
    print("\nGenerated text:")
    print("==============")
    print(generated_text)
    print("==============\n")
    
    with open(f"{args.output_dir}/generated_text.txt", "w") as f:
        f.write(generated_text)
    print(f"Saved text to {args.output_dir}/generated_text.txt")

    # Save generated images
    if generated_results["images"]:
        print(f"Found {len(generated_results['images'])} images in generation")
        image_tokens = torch.cat(generated_results["images"], dim=0).to(model.device)
        decoded_images = model.decode_image_tokens(image_tokens)
        processed_images = processor.postprocess_pixel_values(decoded_images)
        
        for i, img_tensor in enumerate(processed_images):
            img_array = np.transpose(img_tensor.cpu().numpy(), (1, 2, 0))
            img = Image.fromarray((img_array * 255).astype(np.uint8))
            img_path = f"{args.output_dir}/generated_image_{i}.png"
            img.save(img_path)
            print(f"Saved image {i} to {img_path}")
    else:
        print("No images generated")

if __name__ == "__main__":
    main()