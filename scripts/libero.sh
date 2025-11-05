# shader_dir=rt means that we turn on ray-tracing rendering; this is quite crucial for the open / close drawer task as policies often rely on shadows to infer depth
ckpt_path=$1
config_path=$2
task_suite_name=$3
execute_step=$4

CUDA_VISIBLE_DEVICES=0 python eval/libero/run_libero_eval.py --ckpt_path ${ckpt_path} --config_path ${config_path} --execute_step ${execute_step} \
  --task_suite_name ${task_suite_name} \
  --center_crop True \
  # --center_crop False \
  # --num_trials_per_task 2