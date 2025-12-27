import os

base_paths = [
    "/mnt/workspace/jianke/jianke_z/runs/torch_checkpoints_fp32/qwen3vl/calvin_finetune/2025-11-27/calvin_codebase_qwen-vla_Qwen_Qwen3-VL-30B-A3B-Instruct-bs1024-lr4e-05-ws1-FCDecoder-latent1-strategydeepspeed_stage_2"
]
ckpt_paths = []
for base_path in base_paths:
    json_path = [file for file in os.listdir(base_path) if file.endswith(".json")][0]
    ckpt_path = [(os.path.join(base_path, step_file), os.path.join(base_path, json_path))
                 for step_file in os.listdir(base_path)
                 if step_file.endswith(".pt")]
    ckpt_paths.extend(ckpt_path)
# execute_step = [4, 2]
# execute_step = [1]
execute_step = [4]
for i, (ckpt, config) in enumerate(ckpt_paths[:1]):
    for step in execute_step:
        print(f"Running evaluation for checkpoint {ckpt} with execute step {step}")
        os.system("bash scripts/bridge_30a3.bash {} {} {}".format(ckpt, config, step))

# python eval/simpler/eval_ckpts_bridge.py