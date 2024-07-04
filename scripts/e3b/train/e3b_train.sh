#!/bin/bash
#SBATCH --job-name=e3b_train           									# Job name
#SBATCH --partition=											# Specify the partition or queue name
#SBATCH --gres=gpu:2                   									# GPU
#SBATCH --constraint="gpu_A40_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_P100_16G|gpu_RTX5000_16G"
#SBATCH --mem=48G                      									# Memory per node (in GB)
#SBATCH --time=24:00:00                									# Time limit (hh:mm:ss)
#SBATCH --output=e3b_11.out         									# Standard output file
#SBATCH --error=e3b_11.err          									# Standard error file
#SBATCH --exclude=
#SBATCH --account=

#export PYTHONPATH='/homes'

source /etc/profile.d/modules.sh

# Load any necessary modules (e.g., for Python, R, or other software)
#module unload cuda
#module load cuda/11.8
module unload gcc
module load gcc/9.5.0

source activate 
cd /homes
srun python -u run.py --exp-config configs/model_configs/impact_pixel/ppo_impact_pixel_training_e3b.yaml --run-type train

# Change to the directory where your code is located
# cd /path/to/your/code

# Your actual job commands go here
# For example, running a Python script
# python my_script.py

# Or running an executable
# ./my_executable

# When your job is done, you can optionally include cleanup commands here

echo "Job completed on $(date)"
