#!/bin/bash
OUTPUT='/projects/ksun3/dwu25/trained_models/apr_rm/'
mkdir -p $OUTPUT

accelerate launch --config_file accelerate_config.yaml train_bash.py \
    --stage rm \
    --do_train \
    --model_name_or_path codellama/CodeLlama-7b-hf \
    --adapter_name_or_path $OUTPUT \
    --dataset apr_rm \
    --dataset_dir /projects/ksun3/dwu25/apr_datasets_processing/coconut/data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --lora_rank 32 \
    --output_dir $OUTPUT \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-6 \
    --num_train_epochs 5.0 \
    --plot_loss \
    --quantization_bit 4 \
    &> $OUTPUT/training.log
