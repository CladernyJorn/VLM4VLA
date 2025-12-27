
CUR_DIR=$(cd $(dirname $0); pwd)
# sudo chmod 777 -R ./

port=6042

echo "master port: ${port}"

set -x
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=16
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0

# setup distributed training args
GPUS_PER_NODE=4
WORKER_NUM=1 # number of distributed workers

NODE_ID=0
METIS_WORKER_0_HOST=127.0.0.1

# convert deepspeed checkpoint first
if [ $NODE_ID == "0" ]; then
  echo "---------- Converting deepspeed checkpoint to fp32. ----------"
  python3 tools/convert_deepspeed_to_fp32.py ${@:1}
fi

subfix=`date "+%H-%M"`

echo "RUNNING:"
echo torchrun \
    --nnodes $WORKER_NUM \
    --node_rank $NODE_ID \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $METIS_WORKER_0_HOST \
    --master_port $port \
    main.py \
    --exp_name ${subfix} \
    ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $WORKER_NUM

torchrun \
    --nnodes $WORKER_NUM \
    --node_rank $NODE_ID \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $METIS_WORKER_0_HOST \
    --master_port $port \
    main.py \
    --exp_name ${subfix} \
    ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $WORKER_NUM

# bash scripts/run.sh configs/calvin_finetune/finetune_paligemma-3b_calvin.json
# bash scripts/run.sh configs/calvin_finetune/finetune_qwen25vl-3b_calvin.json
# bash scripts/run.sh configs/calvin_finetune/finetune_kosmos_calvin.json
# bash scripts/run.sh configs/calvin_finetune/finetune_qwen25vl-robobrain2_calvin.json
# bash scripts/run.sh configs/calvin_finetune/finetune_qwen3vl-30a3_calvin.json

# bash scripts/run.sh configs/oxe_training/finetune_kosmos_bridge.json # vla
# bash scripts/run.sh configs/oxe_training/finetune_paligemma-3b_bridge.json # sft
# bash scripts/run.sh configs/oxe_training/finetune_qwen25vl-3b_bridge.json

# bash scripts/run.sh configs/oxe_training/finetune_pi0_paligemma-3b_bridge.json # pi0
