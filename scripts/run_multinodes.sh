
set -eo pipefail

##########NCCL configuration for DLC multi-node training - optimized for performance#######
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
export USER=${whoami}
export PRODUCT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO

###########训练启动参数配置###########
# setup distributed training args for 2 nodes
# DLC environment variables - these will be automatically set by DLC platform
GPUS_PER_NODE=${NPROC_PER_NODE:-1}       #节点显卡数
WORLD_SIZE=${WORLD_SIZE:-1}             #节点数
# NODE_ID=${RANK:-0}                      #节点编号
NODE_ID=$RANK
# MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}   #master节点ip
# MASTER_PORT=${MASTER_PORT:-29500}       #master节点端口
MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=$MASTER_PORT

echo "Node ID: $NODE_ID"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "Gpus Per Node: $GPUS_PER_NODE"

# # convert deepspeed checkpoint first
# if [ $NODE_ID == "0" ]; then
#   echo "---------- Converting deepspeed checkpoint to fp32. ----------"
#   python3 tools/convert_deepspeed_to_fp32.py ${@:1}
# fi

subfix=`date "+%H-%M"`
echo "RUNNING:"
echo torchrun \
    --nnodes $WORLD_SIZE \
    --node_rank $NODE_ID \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    ./Examples/QwenVLAHF/main.py \
    --exp_name ${subfix} \
    ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $WORLD_SIZE

torchrun \
    --nnodes $WORLD_SIZE \
    --node_rank $NODE_ID \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    main.py \
    --exp_name ${subfix} \
    ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $WORLD_SIZE
# bash scripts/run.sh configs/oxe_training/finetune_pi0_paligemma-3b_bridge.json # pi0
# bash scripts/run.sh configs/realdualarm_training/finetune_qwen3vl-4b_x2w_0928.json
