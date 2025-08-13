#!/bin/bash
#SBATCH --job-name="SSDA"
#SBATCH --mail-user=felipecadarchamone@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output="log_%j.out" # out file name
#SBATCH --error="log_%j.err" # error file name
#SBATCH --signal=USR1@60

### Use this for a 1x A100 node
##SBATCH --time=20:00:00
##SBATCH --account=xab@a100
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu_p5
##SBATCH -C a100
##SBATCH --ntasks=1 # nbr of MPI tasks (= nbr of GPU)
##SBATCH --ntasks-per-node=1 # nbr of task per node

### Use this for a 1x H100 node
#SBATCH --time=20:00:00
#SBATCH --account=xab@h100
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --ntasks=1 # nbr of MPI tasks (= nbr of GPU)
#SBATCH --ntasks-per-node=1 # nbr of task per node

# Use this for a 1x V100 32G node
##SBATCH --time=20:00:00
##SBATCH --gres=gpu:1
##SBATCH -C v100-32g
##SBATCH --account xab@v100
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1

echo '-------------------------------------'
echo "Start : $0"
echo '-------------------------------------'
echo "Job id : $SLURM_JOB_ID"
echo "Job name : $SLURM_JOB_NAME"
echo "Job node list : $SLURM_JOB_NODELIST"
echo '--------------------------------------'

module purge

if [[ $SLURM_JOB_PARTITION == "gpu_p5" ]]; then
    module load arch/a100
elif [[ $SLURM_JOB_PARTITION == "gpu_p6" ]]; then
    module load arch/h100
fi

module load cuda/12.6.3
module load miniforge/24.11.3
conda deactivate

if [[ $SLURM_JOB_PARTITION == "gpu_p5" ]]; then
    ENVNAME="modetect-a100"
elif [[ $SLURM_JOB_PARTITION == "gpu_p6" ]]; then
    ENVNAME="modetect-h100"
fi

# check if conda environment exists
conda activate "$ENVNAME"

# print a message to indicate the environment is activated
CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
echo "Activated conda environment: $CURRENT_ENV"

# assert that the environment is activated
if [[ "$CURRENT_ENV" != "$ENVNAME" ]]; then
    echo "Failed to activate conda environment: $ENVNAME"
    exit 1
fi

DATASET_PATH=$SCRATCH/h5_scannet

echo "Dataset path: $DATASET_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Running training script..."

python src/modetect/train.py --data_path $DATASET_PATH \
    --n_agents 6 \
    --nkps 1024 \
    --batch_size 4

