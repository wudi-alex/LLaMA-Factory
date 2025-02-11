#!/bin/bash
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=ksun
#SBATCH --job-name=apr_rm
#SBATCH --output=/projects/ksun3/%u/sbatch_log/%x-%N-%j.out
#SBATCH --error=/projects/ksun3/%u/sbatch_log/%x-%N-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:4
#SBATCH --ntasks-per-node=20
#SBATCH --mem=200G
#SBATCH --export=ALL
#SBATCH --time=2-00:00:00

set echo
umask 0022

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10

source 	~/Anaconda/etc/profile.d/conda.sh
conda activate llama_factory

OUTPUT='/projects/ksun3/dwu25/trained_models/apr_rm/'

echo "start"

accelerate launch --config_file accelerate_ds_config.yaml train_bash.py \
    --stage rm \
    --do_train \
    --model_name_or_path codellama/CodeLlama-7b-hf \
    --dataset apr_rm \
    --dataset_dir /projects/ksun3/dwu25/apr_datasets_processing/coconut/data \
    --template vanilla \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --lora_rank 8 \
    --output_dir $OUTPUT/output \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-6 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --quantization_bit 4 \
