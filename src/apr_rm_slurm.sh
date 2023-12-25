#!/bin/bash

#SBATCH --partition=gpuq                    # need to set 'gpuq' or 'contrib-gpuq'  partition
#SBATCH --qos=gpu                          # need to select 'gpu' QOS or other relvant QOS
#SBATCH --job-name=apr_rm
#SBATCH --output=/projects/ksun3/dwu25/sbatch_log/apr_rm/%x-%N-%j.out   # Output file
#SBATCH --error=/projects/ksun3/dwu25/sbatch_log/apr_rm/%x-%N-%j.err    # Error file
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100.80gb:4               # up to 8; only request what you need
#SBATCH --cpus-per-task=20                   # up to 48 per node
#SBATCH --mem-per-cpu=1000M
#SBATCH --export=ALL
#SBATCH --time=2-00:00:00                   # set to 2hr; please choose carefully

set echo
umask 0022

module load gnu10

source 	~/Anaconda/etc/profile.d/conda.sh
conda activate llama_factory

OUTPUT='/projects/ksun3/dwu25/trained_models/apr_rm/'

echo "start"

python train_bash.py

~/Anaconda/envs/llama_factory/bin/accelerate launch --config_file accelerate_ds_config.yaml train_bash.py \
    --stage rm \
    --do_train \
    --model_name_or_path codellama/CodeLlama-7b-hf \
    --dataset apr_rm \
    --dataset_dir /projects/ksun3/dwu25/apr_datasets_processing/coconut/data \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --lora_rank 32 \
    --output_dir $OUTPUT/output \
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
