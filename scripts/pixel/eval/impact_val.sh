#!/bin/bash
#SBATCH --job-name=impact_val               # Job name
#SBATCH --partition=all_usr_prod			# Specify the partition or queue name
#SBATCH --gres=gpu:1                        # GPU
#SBATCH --mem=16G                        # Memory per node (in GB)
#SBATCH --time=24:00:00                 # Time limit (hh:mm:ss)
#SBATCH --output=impact_val.out             # Standard output file
#SBATCH --error=impact_val.err              # Standard error file
#SBATCH --account=tesi_dborghi

#export PYTHONPATH='/homes/dborghi/projects/thesis_exploration2'

source /etc/profile.d/modules.sh

# Load any necessary modules (e.g., for Python, R, or other software)
#module unload cuda
#module load cuda/11.7
module unload gcc
module load gcc/9.5.0

source activate thesis2
cd /homes/dborghi/projects/thesis_exploration2

#Validation
srun python -u run.py --exp-config configs/model_configs/impact_pixel/ppo_impact_pixel_mp3d_val_noisy.yaml --run-type eval

# Change to the directory where your code is located
# cd /path/to/your/code

# Your actual job commands go here
# For example, running a Python script
# python my_script.py

# Or running an executable
# ./my_executable

# When your job is done, you can optionally include cleanup commands here

echo "Job completed on $(date)"