#!/bin/bash

torchrun \
    --nproc_per_node="7" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_vllm.py \
    --deepspeed local_scripts/zero3_offload.json \
    --output_dir /lpai/output/models \
    --model_name_or_path "/lpai/models/qwen2-5-vl-3b-instruct/25-02-18-1" \
    --dataset_name parquet \
    --dataset_config train-00000-of-00001.parquet
    --max_prompt_length 2048 \
    --num_generations 8 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 1000000 \
    --save_total_limit 30 \
    --num_train_epochs 1 \
    --run_name Qwen2_5-VL-3B-GRPO-8k \
    >> train.log 2>&1
