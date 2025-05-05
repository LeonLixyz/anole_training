#!/bin/bash

# Script for formatting the dataset into interleaved text-image format
# Usage: ./format_data.sh

# Configuration values
DATA_DIR="formatted_data"  # Directory where formatted data will be stored
ORIGINAL_DATASET="vlm-reasoning-cot/vlm_reasoning_geometry_auxlines"  # HuggingFace dataset path
DEBUG_MODE=false  # Set to true to process only 5 examples for testing

# Create data directory if it doesn't exist
mkdir -p $DATA_DIR

echo "Starting data formatting..."
echo "Dataset: $ORIGINAL_DATASET"
echo "Output directory: $DATA_DIR"

# Run the formatting script
if [ "$DEBUG_MODE" = true ]; then
    echo "Debug mode ON - processing only 5 examples"
    python format_data.py --dataset_name $ORIGINAL_DATASET --output_dir $DATA_DIR --debug
else
    python format_data.py --dataset_name $ORIGINAL_DATASET --output_dir $DATA_DIR
fi

# Check if formatting was successful
if [ ! -f "$DATA_DIR/formatted_data.json" ]; then
    echo "Error: Data formatting failed. Formatted data not found."
    exit 1
fi

echo "Formatting completed successfully!"
echo "Output:"
echo "- Formatted JSON: $DATA_DIR/formatted_data.json"
echo "- Images directory: $DATA_DIR/images/"

# Display statistics about the formatted data
echo "Number of formatted examples: $(grep -o '"input_text"' $DATA_DIR/formatted_data.json | wc -l)"
echo "Number of images saved: $(ls $DATA_DIR/images | wc -l)" 