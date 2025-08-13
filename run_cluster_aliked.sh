#!/bin/bash
#SBATCH --job-name="Sim Glue"
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00

##SBATCH --gres=gpu:8
##SBATCH --partition=gpu_p5
##SBATCH -C a100
##SBATCH -account xab@a100

#SBATCH --gres=gpu:4
#SBATCH -C v100-32g
#SBATCH --account xab@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10

#SBATCH --output="G.out" # out file name
#SBATCH --error="G.err" # error file name
#SBATCH --mail-user=felipecadarchamone@gmail.com
#SBATCH --mail-type=ALL

# activate conda env
#module load cpuarch/amd
module load python/3.9.12

conda activate lightglue

nvidia-smi > gpu.txt
python -m gluefactory.train aliked+lightglue_simulation --distributed --conf gluefactory/configs/aliked+lightglue_simulation.yaml data.batch_size=128 data.num_workers=1

