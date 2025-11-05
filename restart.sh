pip install -e . &
cp /mnt/zjk/jianke_z/config /root/.ssh &

# # # # # for calvin
# cd calvin &
# sh install.sh &

# for simpler
# git clone git@github.com:lixinghang12/openvla.git
# pip install hydra-core --upgrade 
# pip install bitsandbytes pretty_errors
cd openvla &
pip install -e . &
cd dlimp_openvla &
pip install -e . 

# git clone https://hf-mirror.com/microsoft/kosmos-2-patch14-224 &&
# git clone https://hf-mirror.com/google/paligemma2-3b-pt-224
# # git clone git@hf.co:microsoft/kosmos-2-patch14-224 &&
# # git clone git@hf.co:google/paligemma2-3b-pt-224
# hf-cli google/paligemma-3b-pt-224 --token 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  --username CladernyJorn
# cp -r /root/.cache/huggingface/hub/hfmirror/paligemma2-3b-pt-224 /mnt/zjk/jianke_z/RoboVLMs/.vlms
# hf-cli BAAI/RoboBrain2.0-7B --token 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  --username CladernyJorn
# hf-cli microsoft/kosmos-2-patch14-224 --token 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  --username CladernyJorn

# hf-cli microsoft/Florence-2-large --token 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  --username CladernyJorn &&
# mv /root/.cache/huggingface/hub/hfmirror/Florence-2-large /mnt/zjk/jianke_z/RoboVLMs/.vlms &&
# hf-cli OpenGVLab/InternVL3-8B --token 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  --username CladernyJorn &&
# mv /root/.cache/huggingface/hub/hfmirror/InternVL3-8B /mnt/zjk/jianke_z/RoboVLMs/.vlms
# hf-cli  BAAI/RoboBrain2.0-7B --token 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  --username CladernyJorn &&
# mv /root/.cache/huggingface/hub/hfmirror/RoboBrain2.0-7B /mnt/zjk/jianke_z/RoboVLMs/.vlms
# hf-cli  google/paligemma-3b-pt-224 --token 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  --username CladernyJorn &&
# mv /root/.cache/huggingface/hub/hfmirror/paligemma-3b-pt-224 /mnt/zjk/jianke_z/RoboVLMs/.vlms
# bash transform_ckpt.sh &&
# scp -r -P 8576 /mnt/zjk/jianke_z/RoboVLMs/runs/torch_checkpoints_fp32/qwen25vl/calvin_finetune/2025-07-27 zhangjianke@101.6.96.163:/home/disk1/jianke_z/VLM4VLA/runs/torch_checkpoints_fp32/qwen25vl/calvin_finetune
# scp -r -P 8576 /mnt/zjk/jianke_z/RoboVLMs/runs/torch_checkpoints_fp32/qwen25vl/calvin_finetune/2025-07-28 zhangjianke@101.6.96.163:/home/disk1/jianke_z/VLM4VLA/runs/torch_checkpoints_fp32/qwen25vl/calvin_finetune
# scp -r -P 8576 /mnt/zjk/jianke_z/RoboVLMs/runs/torch_checkpoints_fp32/paligemma/calvin_finetune/2025-07-28 zhangjianke@101.6.96.163:/home/disk1/jianke_z/VLM4VLA/runs/torch_checkpoints_fp32/paligemma/calvin_finetune
