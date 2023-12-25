#!/bin/bash
#SBATCH --partition=gpuq                    # the DGX only belongs in the 'gpu'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QoS
#SBATCH --job-name=python-gpu
## Separate output and error messages into 2 files.
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 # up to 128;
#SBATCH --gres=gpu:A100.80gb:1              # up to 8; only request what you need
#SBATCH --mem-per-cpu=3500M                 # memory per CORE; total memory is 1 TB (1,000,000 MB)
#SBATCH --export=ALL
#SBATCH --time=0-01:00:00                   # set to 1hr; please choose carefully

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

module load gnu10
module load python

source 	~/Anaconda/etc/profile.d/conda.sh
conda activate llama_factory

accelerate launch --config_file accelerate_ds_config.yaml train_bash.py \
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