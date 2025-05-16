#!/bin/bash

# Configuration
CHUNK_SIZE=8
OUTPUT_DIR="/workspace/anole_training/formatted_data"
MODEL_CHECKPOINT="/workspace/anole_training/lora-128"
MAX_PARALLEL_JOBS=4

# Get total chunks count
TOTAL_CHUNKS=$(ls "$OUTPUT_DIR"/chunk_*_of_*.json | wc -l)
echo "Processing $TOTAL_CHUNKS chunks"

# Run evaluation on each chunk
RUNNING=0

for CHUNK_FILE in $(ls "$OUTPUT_DIR"/chunk_*_of_*.json | sort -V); do
    CHUNK_ID=$(basename "$CHUNK_FILE" | sed -E 's/chunk_([0-9]+)_of_.*/\1/')
    
    echo "Starting evaluation for chunk $CHUNK_ID of $TOTAL_CHUNKS"
    
    # Set custom output path for this chunk
    OUTPUT_PATH="outputs/vlm_reasoning_eval_chunk_${CHUNK_ID}_of_${TOTAL_CHUNKS}"
    MASTER_PORT=$((29500 + $CHUNK_ID - 1))
    
    # Run evaluation for this chunk
    torchrun --nnodes 1 \
        --nproc_per_node 8 \
        --node_rank 0 \
        --master_addr "localhost" \
        --master_port $MASTER_PORT \
        eval.py \
        --model anole \
        --data geometry_reasoning \
        --data_dir "/workspace/anole_training/formatted_data" \
        --decoder_type anole \
        --image_seq_length 1024 \
        --input_format anole \
        --output $OUTPUT_PATH \
        --note "vlm_reasoning_eval_chunk_${CHUNK_ID}" \
        --report_to none \
        --do_eval \
        --model_ckpt $MODEL_CHECKPOINT \
        --custom_dataset_path "$CHUNK_FILE" \
        --val_bz 1 &
    
    # Count running jobs
    ((RUNNING++))
    
    # If max parallel jobs reached, wait for one to finish
    if [ $RUNNING -ge $MAX_PARALLEL_JOBS ]; then
        wait -n
        ((RUNNING--))
    fi
    
    # Small delay to prevent resource contention
    sleep 2
done

# Wait for all remaining jobs to finish
wait

echo "All evaluation jobs completed!"