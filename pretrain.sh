#!/bin/bash
#SBATCH --nodes=2
#SBATCH --account=punim1671
#SBATCH --partition=gpgpu
#SBATCH --qos=gpgpumse
#SBATCH --gpus-per-node=2
#SBATCH --time=120:00:00

module load anaconda3/2021.11
eval "$(conda shell.bash hook)"
conda activate mae

# python -m torch.distributed.launch --nproc_per_node=1 submitit_finetune.py --ngpus 2 --nodes 1
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes 2 --node_rank=0 --master_addr="192.168.1.1"
#            --master_port=1234 main_pretrain.py --batch_size 32  --num_workers 2 --epochs 100 --data_path /data/scratch/projects/punim1671/FireRisk/

# python -m torch.distributed.launch --nproc_per_node=1 main_pretrain.py --batch_size 16  --num_workers 2 --epochs 100 --data_path /data/scratch/projects/punim1671/FireRisk/ --output_dir /data/scratch/projects/punim1671/FireRisk/pretrain/  --log_dir /data/scratch/projects/punim1671/FireRisk/pretrain/

python -m torch.distributed.launch --master_addr='10.0.3.29' --master_port=9901 --nproc_per_node=2 --nnodes=2 main_pretrain.py \
--batch_size 16  --num_workers 4 \
--model mae_vit_base_patch16 \
--epochs 100 \
--data_path /data/scratch/projects/punim1671/FireRisk/ \
--output_dir /data/projects/punim1671/Experiments/pretrain/ \
--log_dir /data/projects/punim1671/Experiments/pretrain/ \
--world_size 4