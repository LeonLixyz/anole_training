import json
import random

# Load the dataset
with open('formatted_data/all_datasets.json', 'r') as f:
    data = json.load(f)

# Select a subset of the training data (e.g., 5 samples)
num_samples = 8
selected_data = random.sample(data['train'], num_samples)

# Modify each selected sample by appending text for image generation
for item in selected_data:
    item['input_text'] = item['input_text'] + " Generate Three images to solve the problem."
    # You can also add a key to specify that images should be generated
    item['generate_images'] = True
    
    # Optionally, you could initialize predicted_sketch_paths to an empty list
    # instead of null to indicate that sketches should be generated
    item['predicted_sketch_paths'] = []

# Save the modified subset to a new file
with open('formatted_data/selected_data_with_image_gen.json', 'w') as f:
    json.dump({"train": selected_data}, f, indent=2)

# Print confirmation
print(f"Selected {num_samples} samples and added image generation instruction")
print("Saved to selected_data_with_image_gen.json")

# If you want to use this directly for inference instead of saving:
def prepare_for_inference(selected_data):
    inference_data = []
    for item in selected_data:
        inference_item = {
            "text": item['input_text'],
            "input_img_paths": item['input_img_paths'],
            # Add any other required fields for your inference pipeline
        }
        inference_data.append(inference_item)
    return inference_data

# inference_ready_data = prepare_for_inference(selected_data)