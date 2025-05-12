wandb_group='triton'
yaml_config='config/Model.yaml'
config='Triton' 
batch_size=16
run_num=$(date "+%Y%m%d-%H%M%S")
multi_steps_finetune=1
finetune_max_epochs=100

TRAIN_DIR=$(dirname $(realpath train.py))

export MASTER_ADDR=30.207.96.179  # ip of main node
export MASTER_PORT=31310
export WORLD_SIZE=16
export NODE_RANK=0 

source ~/.bashrc
conda activate triton
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_TOPO_AFFINITY=0
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export NCCL_DEBUG=WARN
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=2400
export NCCL_SOCKET_IFNAME=bond1

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $TRAIN_DIR/train.py \
  --yaml_config=$yaml_config --config=$config --run_num=$run_num --batch_size=$batch_size --multi_steps_finetune=$multi_steps_finetune --finetune_max_epochs=$finetune_max_epochs \
  >> ./logs/${config}_${wandb_group}_rank0_${SLURM_JOB_ID}_${run_num}.log 2>&1 &

ssh root@30.207.98.112 "
source ~/.bashrc; \
conda activate triton; \

export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_TOPO_AFFINITY=0
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export NCCL_DEBUG=WARN
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=2400
export NCCL_SOCKET_IFNAME=bond1

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; \
export MASTER_ADDR=$MASTER_ADDR; export MASTER_PORT=$MASTER_PORT; export WORLD_SIZE=16; export NODE_RANK=1; \
nohup torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT $TRAIN_DIR/train.py \
  --yaml_config=$yaml_config --config=$config --run_num=$run_num --batch_size=$batch_size --multi_steps_finetune=$multi_steps_finetune --finetune_max_epochs=$finetune_max_epochs \
>> $TRAIN_DIR/logs/${config}_${wandb_group}_rank1_${SLURM_JOB_ID}_${run_num}.log 2>&1 &"