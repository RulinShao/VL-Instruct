#!/bin/bash
#SBATCH --job-name=train
#SBATCH -p dgx_normal_q
#SBATCH -N1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH -A nlp_lab
#SBATCH -t 48:00:00
#SBATCH -o sub_outputs/slurm%j.out
#SBATCH --mem=512GB

module reset
module load gcc/8.2.0
source /home/zhiyangx/miniconda3/etc/profile.d/conda.sh
conda activate lavis
cd /projects/nlp_lab/zhiyang/phd4_projects/VL-Instruct

export TRANSFORMERS_CACHE=/projects/nlp_lab/zhiyang/.cache/

model_type=$1
data_type=$2
output_dir=$3
python -m torch.distributed.run --nproc_per_node=8 train_doremi.py --model_type $model_type --train_qformer --output_dir $output_dir --max_txt_len 512 --train_data_type $data_type