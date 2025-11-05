import os

base_paths = [
    "/home/disk1//VLM4VLA/runs/torch_checkpoints_fp32/qwen3vl/libero10_finetune/2025-09-10/libero10_VLMA-baselines_1_qwen3vl-4b-bs512-lr2e-05-ws1-FCDecoder-latent1"
]
ckpt_paths = []
for base_path in base_paths:
    json_path = [file for file in os.listdir(base_path) if file.endswith(".json")][0]
    ckpt_path = [(os.path.join(base_path, step_file), os.path.join(base_path, json_path), "libero_10")
                 for step_file in os.listdir(base_path)
                 if step_file.endswith(".pt")]
    ckpt_paths.extend(ckpt_path)
# execute_step = [1, 2, 4]
execute_step = [4, 2]
# ckpt_paths = [(
#     "/home/disk1//VLM4VLA/runs/torch_checkpoints_fp32/paligemma/libero10_finetune/2025-08-15/libero10_paligemma-3b-pt-224-bs512-lr5e-05-ws1-FCDecoder-latent1/epoch=31-step=49992.pt",
#     "/home/disk1//VLM4VLA/runs/torch_checkpoints_fp32/paligemma/libero10_finetune/2025-08-15/libero10_paligemma-3b-pt-224-bs512-lr5e-05-ws1-FCDecoder-latent1/2025-08-15_20:07:05.445433-project.json",
#     "libero_10",  # "libero_10" or "libero_spatial" or "libero_object" or "libero_goal"
# )]
for i, (ckpt, config, task_suite_name) in enumerate(ckpt_paths):
    for step in execute_step:
        print(f"Running evaluation for checkpoint {ckpt} on {task_suite_name} with execute step {step}")
        os.system("bash scripts/libero.sh {} {} {} {}".format(ckpt, config, task_suite_name, step))

# python eval/simpler/eval_ckpts_bridge.py