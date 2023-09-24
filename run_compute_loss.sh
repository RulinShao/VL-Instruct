model_type=$1
ckpt_path=$2

python -m torch.distributed.run --nproc_per_node=1 compute_loss.py --model_type $model_type  --pretrained $ckpt_path --doremi