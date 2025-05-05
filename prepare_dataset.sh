#!/bin/bash

# Script to prepare the dataset for training
# Usage: ./prepare_dataset.sh

# Create necessary directories
mkdir -p formatted_data
mkdir -p tools

# Check if the dataset exists
if [ ! -f "formatted_data/formatted_data.json" ]; then
  echo "Error: formatted_data/formatted_data.json not found"
  exit 1
fi

# Make sure the check_dataset.py script is executable
chmod +x tools/check_dataset.py

# Normalize the dataset to ensure compatibility
echo "Normalizing dataset for compatibility with Anole model..."
python tools/check_dataset.py \
  --input formatted_data/formatted_data.json \
  --output formatted_data/formatted_data_normalized.json

# Update the training script to use the normalized dataset
sed -i 's/formatted_data.json/formatted_data_normalized.json/g' run_train.sh

echo "Dataset prepared successfully!"
echo "You can now run: ./run_train.sh" 