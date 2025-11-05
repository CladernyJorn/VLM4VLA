import os
import json
import numpy as np
# ckpt_paths = [
#     ("/home/disk1//VLM4VLA/runs/torch_checkpoints_fp32/internvl35/calvin_finetune/2025-09-05/calvin_InternVL3_5-4B-bs128-lr2e-05-ws1-FCDecoder-latent1/epoch=3-step=32932.pt","/home/disk1//VLM4VLA/runs/torch_checkpoints_fp32/internvl35/calvin_finetune/2025-09-05/calvin_InternVL3_5-4B-bs128-lr2e-05-ws1-FCDecoder-latent1/2025-09-05_23:27:28.191682-project.json"),
#     ("/home/disk1//VLM4VLA/runs/torch_checkpoints_fp32/internvl35/calvin_finetune/2025-09-05/calvin_InternVL3_5-4B-bs128-lr2e-05-ws1-FCDecoder-latent1/epoch=4-step=41165.pt","/home/disk1//VLM4VLA/runs/torch_checkpoints_fp32/internvl35/calvin_finetune/2025-09-05/calvin_InternVL3_5-4B-bs128-lr2e-05-ws1-FCDecoder-latent1/2025-09-05_23:27:28.191682-project.json")]
base_paths = [
    "/home/disk1//VLM4VLA/runs/torch_checkpoints_fp32/qwen3vl/calvin_finetune/2025-09-09/calvin_VLMA-baselines_1_qwen3vl-4b-bs128-lr2e-05-ws1-FCDecoder-latent1"
]
ckpt_paths = []
for base_path in base_paths:
    json_path = [file for file in os.listdir(base_path) if file.endswith(".json")][0]
    ckpt_path = [(os.path.join(base_path, step_file), os.path.join(base_path, json_path))
                 for step_file in os.listdir(base_path)
                 if step_file.endswith(".pt")]
    ckpt_paths.extend(ckpt_path)


def cal_acc(ckpt_path):
    accs = {}
    result_dir = os.path.join(
        os.path.dirname(ckpt_path),
        ckpt_path.split("/")[-1].split(".")[0].split("=")[-1] + "-eval",
    )
    for file in os.listdir(result_dir):
        if file.endswith(".json") and "rand" in file:
            with open(os.path.join(result_dir, file), "r") as f:
                data = json.load(f)
            for k, v in data["null"]["chain_sr"].items():
                if k not in accs:
                    accs[k] = []
                accs[k].append(v)
    accs = {k: np.mean(v) for k, v in accs.items()}
    accs["all"] = np.sum(list(accs.values()))
    print(accs)
    with open(os.path.join(result_dir, "final_results.json"), "w") as f:
        json.dump(accs, f, indent=4)
    return accs


for i, (ckpt, config) in enumerate(ckpt_paths):
    if "kosmos" in ckpt:
        import transformers

        package_dir = transformers.__path__[0]
        os.system("cp tools/modeling_kosmos2.py {}/models/kosmos2/modeling_kosmos2.py".format(package_dir))

        import importlib

        importlib.reload(transformers)

    print("evaluating checkpoint {}".format(ckpt))
    os.system("bash scripts/calvin_ddp.sh {} {}".format(ckpt, config))
    # os.system(long_cmd)
    try:
        cal_acc(ckpt)
    except Exception as e:
        print(e)
        continue
