export LD_LIBRARY_PATH=/opt/nvidia/nsight-compute/2023.1.0/host/linux-desktop-glibc_2_11_3-x64/Mesa:$LD_LIBRARY_PATH
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $PET_NNODES \
    --node_rank $PET_NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


torchrun --nproc_per_node="7" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    /src/open_r1/grpo4_2_5_3b_vllm.py \
    --deepspeed /local_scripts/zero3_offload.json \
    --output_dir /results \
    --model_name_or_path Qwen2_5_VL_3B_Instruct \
    --dataset_name /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/shaowenqi-shaowenqi/mengfanqing/open-r1-multimodal/k12data_20k_valid_fix_geo170k \
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
    --run_name Qwen2-VL-7B-GRPO-8k \
    >> train.log 2>&1
