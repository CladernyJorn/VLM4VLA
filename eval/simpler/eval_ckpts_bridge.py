import os

# ckpt_paths = [
#     # (
#     #     "/home/disk1//VLM4VLA/runs/torch_checkpoints_fp32/paligemma/bridge_finetune/2025-07-30/bridge_paligemma-3b-pt-224-bs512-lr2e-05-ws1-FCDecoder-latent1/epoch=0-step=2500.pt",
#     #     "/home/disk1//VLM4VLA/runs/torch_checkpoints_fp32/paligemma/bridge_finetune/2025-07-30/bridge_paligemma-3b-pt-224-bs512-lr2e-05-ws1-FCDecoder-latent1/2025-07-30_14:50:03.208413-project.json",
#     # )
#     (
#         # "/home/disk1//VLM4VLA/runs/torch_checkpoints_fp32/qwen25vl/bridge_finetune/2025-07-31/bridge_huggingface_hub_models--Qwen--Qwen2.5-VL-3B-Instruct_snapshots_c747f21f03e7d0792c30766310bd7d8de17eeeb3-bs512-lr2e-05-ws1-FCDecoder-latent1/epoch=0-step=2500.pt",
#         # "/home/disk1//VLM4VLA/runs/torch_checkpoints_fp32/qwen25vl/bridge_finetune/2025-07-31/bridge_huggingface_hub_models--Qwen--Qwen2.5-VL-3B-Instruct_snapshots_c747f21f03e7d0792c30766310bd7d8de17eeeb3-bs512-lr2e-05-ws1-FCDecoder-latent1/2025-07-31_14:38:44.994116-project.json"
#     )
# ]
base_paths = [
    # "/home/disk1//VLM4VLA/runs/torch_checkpoints_fp32/qwen3vl/bridge_finetune/2025-09-15/bridge_wqy_ckpt_qwen3_4b_qwen3-dense-4b-s3-sft32k-bridgev21actonly_0908_iter_0004000-bs512-lr2e-05-ws1-FCDecoder-latent1"
    "/home/disk1//VLM4VLA/runs/torch_checkpoints_fp32/qwen3vl/bridge_finetune/2025-09-18/bridge_wqy_ckpt_qwen3_4b_qwen3-dense-4b-s3-sft32k-bridgev21act_gSFT55_0908_iter_0004000-bs512-lr2e-05-ws1-FCDecoder-latent1"
]
ckpt_paths = []
for base_path in base_paths:
    json_path = [file for file in os.listdir(base_path) if file.endswith(".json")][0]
    ckpt_path = [(os.path.join(base_path, step_file), os.path.join(base_path, json_path))
                 for step_file in os.listdir(base_path)
                 if step_file.endswith(".pt")]
    ckpt_paths.extend(ckpt_path)

execute_step = [4, 2, 1]
device=3
# execute_step = [1]
for i, (ckpt, config) in enumerate(ckpt_paths):
    for step in execute_step:
        print(f"Running evaluation for checkpoint {ckpt} with execute step {step}")
        os.system("bash scripts/bridge.bash {} {} {} {}".format(ckpt, config, step, device))

# python eval/simpler/eval_ckpts_bridge.py