# split_data.py
import json
import os
import sys

def split_data(input_file, output_dir, chunk_size):
    # Load the original data file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Get all examples from all splits
    all_examples = []
    for split_name, examples in data.items():
        all_examples.extend(examples)
    
    print(f"Loaded {len(all_examples)} examples total")
    
    # Split data into chunks
    chunks = []
    for i in range(0, len(all_examples), chunk_size):
        chunk = all_examples[i:i+chunk_size]
        chunks.append(chunk)
    
    # Create JSON file for each chunk
    chunk_files = []
    for i, chunk in enumerate(chunks):
        # Create data structure with the same format
        chunk_data = {
            "train": chunk,
            "validation": chunk,
            "test": chunk
        }
        
        # Save to file
        chunk_filename = os.path.join(output_dir, f"chunk_{i+1}_of_{len(chunks)}.json")
        with open(chunk_filename, 'w') as f:
            json.dump(chunk_data, f, indent=2)
        
        chunk_files.append(chunk_filename)
        print(f"Created chunk {i+1}/{len(chunks)} with {len(chunk)} examples: {chunk_filename}")
    
    return chunk_files, len(chunks)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python split_data.py <input_file> <output_dir> <chunk_size>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    chunk_size = int(sys.argv[3])
    
    _, total_chunks = split_data(input_file, output_dir, chunk_size)
    print(total_chunks)  # Output the total number of chunks