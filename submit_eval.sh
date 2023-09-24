#!/bin/bash
#SBATCH --job-name=eval
#SBATCH -p dgx_preemptable_q
#SBATCH -N1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH -A nlp_lab
#SBATCH -t 24:00:00
#SBATCH -o sub_outputs/slurm%j.out
#SBATCH --mem=128GB


module reset
module load gcc/8.2.0
source /home/zhiyangx/miniconda3/etc/profile.d/conda.sh
conda activate lavis
cd /projects/nlp_lab/zhiyang/phd4_projects/VL-Instruct


model_name=$1
checkpoint=$2
python evaluate.py --model_name $model_name --checkpoint $checkpoint
conda activate mata-instruct
python compute_performance.py /projects/nlp_lab/zhiyang/phd4_projects/VL-Instruct/checkpoints/$model_name/$checkpoint