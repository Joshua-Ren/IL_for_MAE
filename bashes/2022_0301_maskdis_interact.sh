#!/bin/bash
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH --time=36:00:00
#SBATCH --job-name=nil_inter
#SBATCH --output=./logs/test.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:4

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.8 cuda/11.0 cudnn/8.0_cuda-11.1

source /home/sg955/egg-env/bin/activate

cd /home/sg955/GitWS/IL_for_MAE/

python -m torch.distributed.launch --nproc_per_node=4 --master_port 1086 main_interact.py --proj_name Interact_MAE \
--run_name int400_maskdis49_int95_senlr --batch_size 384 --epochs 90 --de_epochs 5 \
--en_ckp Distill-IN1K/mae_vit_base_patch16_smallde/cos_mask/checkpoint-49.pth