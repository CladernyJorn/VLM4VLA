

export MESA_GL_VERSION_OVERRIDE=4.1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_BLOCKING_WAIT=1
# export CUDA_VISIBLE_DEVICES=1

cd $EVALUTION_ROOT
ckpt_dir=$1
config_path=$2
sudo chmod 666 -R $ckpt_dir
GPUS_PER_NODE=8

torchrun --nnodes=1 --nproc_per_node=$GPUS_PER_NODE --master_port=6067 eval/calvin/evaluate_ddp-v2.py \
--config_path $config_path \
--ckpt_path $ckpt_dir \
--ckpt_idx 0 --raw_calvin
