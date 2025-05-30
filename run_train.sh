#!/bin/bash

# Script for distributed training of the anole model
# Usage: ./run_train.sh
rm -rf ~/.cache/huggingface/datasets/geometry_reasoning*
rm -rf outputs/anole_train

# Configuration values
NODE_NUM=1
GPU_NUM_PER_NODE=1
RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29500
OUTPUT_PATH="outputs/anole_train"
NOTE="anole_train"
DATASET="geometry_reasoning" 
DATA_DIR="/home/jovyan/workspace/anole_training/formatted_data"  # Point to our formatted data directory
FORMATTED_DATA_PATH="/home/jovyan/workspace/anole_training/formatted_data/train_dataset.json"  # Path to our formatted JSON

# Create data directory if it doesn't exist
mkdir -p $DATA_DIR

# Debug: List environment and directories
echo "Current directory: $(pwd)"
echo "Data directory: $(realpath $DATA_DIR)"
echo "Python path: $(which python)"
ls -la $DATA_DIR

# Execute the training command
torchrun --nnodes $NODE_NUM \
    --nproc_per_node $GPU_NUM_PER_NODE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train.py \
    --model anole \
    --data $DATASET \
    --data_dir $DATA_DIR \
    --decoder_type anole \
    --image_seq_length 1024 \
    --input_format anole \
    --output $OUTPUT_PATH \
    --note $NOTE \
    --report_to "wandb" \
    --do_train \
    --custom_dataset_path "$FORMATTED_DATA_PATH" \
    --save_dataset \
    --train_bz 1 \
    --val_bz 1 \
    --grad_acc 8 
