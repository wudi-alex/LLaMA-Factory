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

python myscript.py