#!/bin/bash
#SBATCH --job-name="Not a Matcher"
#SBATCH --account=xab@a100
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu_p5
#SBATCH -C a100
#SBATCH --output="G.out" # out file name
#SBATCH --error="G.err" # error file name
#SBATCH --mail-user=felipecadarchamone@gmail.com
#SBATCH --mail-type=ALL

# activate conda env
module load cpuarch/amd
module load python/3.9.12

conda activate lightglue

nvidia-smi > gpu.txt
python -m gluefactory.train aliked+lightglue_simulation --conf gluefactory/configs/aliked+lightglue_simulation.yaml data.batch_size=128

