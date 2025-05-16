#!/bin/bash

# Clear Hugging Face datasets cache for geometry_reasoning
rm -rf ~/.cache/huggingface/datasets/geometry_reasoning*

# Script for running inference with the anole model
# Usage: ./run_eval.sh

# Configuration values
NODE_NUM=1
GPU_NUM_PER_NODE=8  # Adjust based on available GPUs
RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29500
OUTPUT_PATH="outputs/vlm_reasoning_eval"
NOTE="vlm_reasoning_eval"
DATASET="geometry_reasoning" 
DATA_DIR="/workspace/anole_training/formatted_data"
FORMATTED_DATA_PATH="/workspace/anole_training/formatted_data/test_dataset.json"
MODEL_CHECKPOINT="/workspace/anole_training/lora-128"
# # Path to your trained model checkpoint
# MODEL_CHECKPOINT="x"
# Path to your trained model checkpoint

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_PATH

# Execute the evaluation command
torchrun --nnodes $NODE_NUM \
    --nproc_per_node $GPU_NUM_PER_NODE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    eval.py \
    --model anole \
    --data $DATASET \
    --data_dir $DATA_DIR \
    --decoder_type anole \
    --image_seq_length 1024 \
    --input_format anole \
    --output $OUTPUT_PATH \
    --note $NOTE \
    --report_to none \
    --do_eval \
    --model_ckpt $MODEL_CHECKPOINT \
    --custom_dataset_path "$FORMATTED_DATA_PATH" \
    --val_bz 1