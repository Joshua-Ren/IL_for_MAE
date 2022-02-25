#!/bin/bash
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH --time=36:00:00
#SBATCH --job-name=IN1K-Distill
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

srun python -m torch.distributed.launch --nproc_per_node=4 --master_port 1086 main_distill.py --proj_name Distill-IN1K \
--run_name cos_distill_novalid --batch_size 128 --dataset imagenet --epochs 50 --dis_ratio 1.0 --dist_loss cosine \
--teach_ckp Interact_MAE/mae_vit_base_patch16_smallde/offi_4GPU_smallDE400/checkpoint-399.pth