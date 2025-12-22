#!/bin/bash
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 1. Configuration
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="controlnet_output"

# 2. Launch Training
# Note: mixed_precision="no" is required for Mac MPS stability
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir=dataset_controlnet \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "$(pwd)/dataset_controlnet/conditioning_images/1.png" \
 --validation_prompt "official art of a pokemon, ken sugimori style, white background" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --max_train_steps=500 \
 --checkpointing_steps=250 \
 --mixed_precision="no" \
 --report_to="tensorboard"