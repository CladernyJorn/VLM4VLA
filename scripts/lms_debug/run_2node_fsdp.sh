
set -eo pipefail

# NCCL configuration for DLC multi-node training - optimized for performance
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_MIN_NCHANNELS=4
export NCCL_NET_PLUGIN=none
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USER=whoami
export PRODUCT=1


# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO



# 项目路径
export workspace=/mnt/zjk/jianke_z/VLM4VLA
cd $workspace

# setup distributed training args for 2 nodes
GPUS_PER_NODE=8
WORKER_NUM=2 # number of distributed workers (2 nodes)

# DLC environment variables - these will be automatically set by DLC platform
NODE_ID=$RANK
MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=$MASTER_PORT

echo "Node ID: $NODE_ID"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World Size: $WORLD_SIZE"

# FSDP configuration file path
fsdp_config_file="$workspace/configs/fsdp_config.yaml"

subfix=`date "+%H-%M"`

echo "RUNNING:"
echo accelerate launch \
    --config_file $fsdp_config_file \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $RANK \
    --num_processes $((${WORLD_SIZE}*8)) \
    --num_machines $WORLD_SIZE \
    main.py \
    --exp_name ${subfix} \
    ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $WORLD_SIZE

accelerate launch \
    --config_file $fsdp_config_file \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $RANK \
    --num_processes $((${WORLD_SIZE}*8)) \
    --num_machines $WORLD_SIZE \
    main.py \
    --exp_name ${subfix} \
    ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $WORLD_SIZE

