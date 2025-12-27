paths=(
        "/mnt/runs/checkpoints/qwen25vl/calvin_finetune/calvin_huggingface_hub_models--Qwen--Qwen2.5-VL-3B-Instruct_snapshots_c747f21f03e7d0792c30766310bd7d8de17eeeb3-bs128-lr2e-05-ws1-FCDecoder-latent1"
    )
for path in ${paths[@]}; do
    python transform_ckpt.py --ckpt_dir $path
done