import json
import os


def sum_values(dict1, dict2):
    for key in dict1:
        if isinstance(dict1[key], dict):
            if key in dict2:
                sum_values(dict1[key], dict2[key])
        else:
            dict1[key] = dict1[key] + dict2.get(key, 0)


def cal_file_value(root, prefix):
    summary_data = None
    for i in range(8):
        filename = f"{prefix}_rand-{i}.json"
        data = json.load(open(os.path.join(root, filename), "r"))
        if summary_data is None:
            summary_data = data
        else:
            sum_values(summary_data, data)

    summary_data["null"]["avg_seq_len"] = summary_data["null"]["avg_seq_len"] / 8
    for i in summary_data["null"]["chain_sr"]:
        summary_data["null"]["chain_sr"][i] = summary_data["null"]["chain_sr"][i] / 8

    with open(os.path.join(root, f"{prefix}_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=1)


def cal_all_res_in_dir(root):
    files = [
        _ for _ in os.listdir(root) if "results" in _ and "json" in _ and "rand" in _
    ]
    if len(files) == 0:
        return
    prefixs = set([_.split("_rand")[0] for _ in files])
    for prefix in prefixs:
        cal_file_value(root, prefix)


if __name__ == "__main__":
    log_dir = "eval/logs"
    roots = [_ for _ in os.listdir(log_dir) if "json_eval" in _]
    for root in roots:
        cal_all_res_in_dir(os.path.join(log_dir, root))
