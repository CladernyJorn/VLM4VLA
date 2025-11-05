import torch
import copy
from PIL import Image
from typing import Optional, Tuple, List
from typing import Sequence
from vlm4vla.model.backbone.base_backbone import BaseRoboVLM, load_config, deep_update
from qwen_vl_utils import process_vision_info, fetch_image
import numpy as np
from einops import rearrange, repeat


class RoboQwen3VLMOE(BaseRoboVLM):

    @property
    def image_processor(self):
        # used in dataloader, just transform image to tensor before random shift
        import torchvision.transforms as T
        image_preprocess = T.Compose([
            T.Resize(
                (
                    self.configs.get("image_size", 224),
                    self.configs.get("image_size", 224),
                ),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.Lambda(lambda img: img.convert("RGB")),
            T.Lambda(lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float()),  # uint8转为float
        ])
        return image_preprocess

    @property
    def hidden_size(self):
        return self.model.config.hidden_size

    @property
    def word_embedding(self):
        return self.model.model.embed_tokens  # weight

    @property
    def text_tower(self):
        # not used
        return self.model.model

    @property
    def vision_tower(self):
        return self.model.visual

    @property
    def model(self):
        return self.backbone

    @property
    def start_image_token_id(self):
        raise NotImplementedError("start_image_token_id is not supported for qwen25vl")
        return torch.LongTensor([self.model.config.visual["image_start_id"]])

    @property
    def end_image_token_id(self):
        raise NotImplementedError("end_image_token_id is not supported for qwen25vl")
        return torch.LongTensor([self.model.config.visual["image_start_id"] + 1])

    def process_vision_info(self, images):
        image_inputs = []
        for vision_info in images:
            # image_inputs.append(fetch_image({"image": vision_info}))  # do smart resize for qwen25vl
            image_inputs.append(
                fetch_image({"image": vision_info},
                            image_patch_size=16))  # do smart resize for qwen3vl,qwen25vl use default 14,qwen3vl use 16
        return image_inputs

    def encode_images(self, images):
        raise NotImplementedError("encode_images is not supported for qwen25vl")

        return image_tensor, grid_thw

    def forward_continuous(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,  # not used (not transfered from forward_action)
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_mask: torch.Tensor = None,
        vision_gripper=None,
        raw_text=None,
        rel_state=None,  # not used (not transfered from forward_action)
        mode="train",
        **kwargs,
    ):
        # used in input:
        # lang_x(inputs), attention_mask, action_labels, action_mask
        # vision_x: 16,1,3,224,224
        # inputs: inputs after qwen25vl processor
        #       input_ids: 16,288: ["<|im_start|><|vision_start|><|image_pad|><|vision_end|><instruction_here_please>\n",...]
        #       attention_mask: 16,288
        #       pixel_values: 16*1024,1176
        #       image_grid_thw: 16,3; 16个[1,32,32]
        loss = {}
        assert vision_x is not None
        bs, seq_len = vision_x.shape[:2]  # (4, 8), seq_len 就是window size
        action_space = self.act_head_configs.get("action_space", "continuous")

        history_type = self.act_head_configs.get("history_type", "post")
        input_ids = lang_x.pop('input_ids')
        attention_mask = lang_x.pop('attention_mask')
        pixel_values = lang_x.pop('pixel_values').type(self.vision_tower.dtype)
        image_grid_thw = lang_x.pop('image_grid_thw')  # 16,3; 16个[1,32,32]
        assert input_ids.shape[0] == bs * seq_len and history_type in ["post", "pre"]

        input_embeds = self.word_embedding(input_ids)  # (4*8, 12, 1024)

        image_embeds = self.vision_tower(pixel_values, grid_thw=image_grid_thw)
        n_image_tokens = (input_ids == self.model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")

        mask = input_ids == self.model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(input_embeds)
        image_mask = mask_expanded.to(input_embeds.device)

        image_embeds = image_embeds.to(input_embeds.device, input_embeds.dtype)
        input_embeds = input_embeds.masked_scatter(image_mask, image_embeds)

        # visual_pos_masks = mask
        # deepstack_visual_embeds_multiscale = image_embeds_multiscale
        # print("deepstack_visual_embeds_multiscale", len(deepstack_visual_embeds_multiscale),
        #       deepstack_visual_embeds_multiscale[0].shape)  # len:3 3136 2560
        # print("input_embeds", input_embeds.shape)  # 16,236/237,2560

        # print("after insert image, input_embeds", input_embeds.shape)
        # print("after insert image, attention_mask", attention_mask.shape)
        # print("after insert image, attention_mask", attention_mask)

        # self.forward_text_test(input_embeds, attention_mask)

        multimodal_embeds = input_embeds
        multimodal_labels = None
        multimodal_attention_mask = attention_mask
        if vision_gripper is not None:
            # not support hand_rgb yet
            pass

        if rel_state is not None and self.use_state:
            print("Using state!!!!!!!!!!!!")
            # insert_idx = multimodal_embeds.shape[1] - int(self.tokenizer.eos_token is not None)  # insert at last
            state_token = self.encode_state(rel_state)  # bs, seq_len, d
            state_token = state_token.reshape(bs * seq_len, state_token.shape[-1]).unsqueeze(1)  # bs*seq_len, 1, d
            (
                multimodal_embeds,
                mutlimodal_labels,
                multimodal_attention_mask,
                action_token_mask,
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                state_token,
                multimodal_labels,
                multimodal_attention_mask,
                is_image=False,
                # insert_idx=insert_idx,
                insert_idx=multimodal_embeds.shape[1],  # 放在最后
                # fixed for paligemma, where |<img_token><bos><text>\n<pad>|, after insert <state>: |<img_token><bos><text>\n<pad><state>|
                fill_zero=self.act_head_configs.get("fill_zero", False),
            )

        if action_space == "continuous":

            action_tokens = repeat(
                self.action_token,
                "d -> b n d",
                b=multimodal_embeds.shape[0],
                n=self.latent_num,
            )

            (
                multimodal_embeds,
                mutlimodal_labels,
                multimodal_attention_mask,
                action_token_mask,
            ) = self.merge_multi_modal_input(
                multimodal_embeds,
                action_tokens,
                multimodal_labels,
                multimodal_attention_mask,
                is_image=False,
                # insert_idx=insert_idx,
                insert_idx=multimodal_embeds.shape[1],
                # fixed for paligemma, where |<img_token><bos><text>\n<pad>|, after insert <action>: |<img_token><bos><text>\n<pad><action>|
                fill_zero=self.act_head_configs.get("fill_zero", False),
            )
            # add one token with mask 0 in visual_pos_masks
            # visual_pos_masks = torch.cat(
            #     [visual_pos_masks,
            #      torch.zeros(visual_pos_masks.shape[0], 1).bool().to(multimodal_embeds.device)],
            #     dim=1)
            # self.forward_text_test(multimodal_embeds, multimodal_attention_mask)

        if history_type == "pre":
            multimodal_embeds = rearrange(
                multimodal_embeds, "(b l) n d -> b (l n) d",
                l=seq_len)  # original shape: (4*8, 271, 2304) -> (4, 8*271, 2304)
            if multimodal_attention_mask is not None:
                multimodal_attention_mask = rearrange(multimodal_attention_mask, "(b l) n -> b (l n)", l=seq_len)

        # copied from qwen3vlforgeneration: different from qwen25, we can not use model.forward, instead use model.model.forward
        batch_size, seq_length, _ = multimodal_embeds.shape
        delta = (0)
        position_ids = torch.arange(seq_length, device=multimodal_embeds.device, dtype=torch.float)
        position_ids = position_ids.view(1, -1).expand(batch_size, -1)
        position_ids = position_ids.add(delta)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        output = self.model.model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=multimodal_embeds,
            use_cache=False,
            # deepstack_visual_embeds_multiscale=deepstack_visual_embeds_multiscale,
            # visual_pos_masks=visual_pos_masks,
        )

        output_hs = output[0].clone()
        if history_type == "pre":
            multimodal_embeds = rearrange(multimodal_embeds, "b (l n) d -> (b l) n d", l=seq_len)
            output_hs = rearrange(output_hs, "b (l n) d -> (b l) n d", l=seq_len)

        if action_space == "continuous":
            # tmp_mask = torch.all(multimodal_embeds == self.action_token, dim=-1)
            action_hs = output_hs[action_token_mask].reshape(bs, seq_len, self.latent_num, -1)  # (4, 8, 1, 1024)

        elif action_space == "down_sample":
            action_hs = output_hs
            token_src = self.act_head_configs.get("token_source", "all")

            if token_src == "all":
                action_hs = action_hs.reshape(bs, seq_len, *action_hs.shape[1:])
            else:
                raise ValueError(f"Unsupported token source {token_src}")

        else:
            raise ValueError(f"Unsupported action space {action_space}")

        if self.use_clip_norm and mode == "train":
            clip_loss = self.clip_norm_head(action_hs, raw_text)
            self._update_loss(loss, clip_loss, "clip")

        action_logits, action_loss = self._forward_action_head(action_hs, action_labels, action_mask)

        # cur = time.time()
        # print("predict action consumes {} sec".format(cur-st))
        # st = cur

        if mode == "train":
            self._update_loss(loss, action_loss, "act")
            loss = self._format_loss(loss)
        else:
            return action_logits

        return loss

    def forward_text_test(self, input_embeds, attention_mask):
        # should not be called in training, just for test and debug
        outputs = self.model(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
        )
        last_nopadding_logits = []
        # 获取最后一个非padding的logits
        for i in range(attention_mask.shape[0]):
            last_nopadding_logits.append(outputs.logits[i, attention_mask[i].bool(), :][-1, :])
        last_nopadding_logits = torch.stack(last_nopadding_logits)
        print(last_nopadding_logits.shape)

        # 预测下一个词（取概率最大的token）
        predicted_next_token = torch.argmax(last_nopadding_logits, dim=-1)  # (batch_size,)
        print(predicted_next_token)
        response = self.processor.batch_decode(
            predicted_next_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print("response_next:", response)


if __name__ == "__main__":
    path = "/mnt/zjk/jianke_z/vlm4vla/test_image.png"

    def test_qwen25vl(model, processor):

        # img = Image.open("/mnt/zjk/jianke_z/vlm4vla/imgs/vlm4vla.png")
        messages = [{
            "role":
                "user",
            "content": [
                {
                    "type": "image",
                    "image": "/mnt/zjk/jianke_z/VLM4VLA/imgs/vlm4vla.png"
                },
                {
                    "type": "text",
                    "text": "Describe the image"
                    # "text":
                    #     "Outline the position of each object and output all the coordinates in JSON format. ([ {'bbox_2d': [x1, y1, x2, y2], 'label': 'obj_name'}, ...])\nresult:"
                }
            ]
        }]
        # text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # gg not good
        # image_inputs, video_inputs = process_vision_info(messages)
        # print("text:", text)
        # inputs = processor(
        #     text=text,
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )

        # inputs = inputs.to(model.device)
        # generated_ids = model.generate(**inputs, max_new_tokens=128)
        # # generated_ids = model(**inputs)
        # generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        # print(generated_ids_trimmed)
        # response = processor.batch_decode(generated_ids_trimmed)
        # print("response:", response)
        # <|im_start|>system
        # You are a helpful assistant.<|im_end|>
        # <|im_start|>user
        # <|vision_start|><|image_pad|><|vision_end|>Describe the image<|im_end|>
        # <|im_start|>assistant
        # print(text)
        text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image<|im_end|>\n<|im_start|>assistant\n"
        )
        # text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|image_pad|><|vision_end|>Describe the image<|im_end|>\n<|im_start|>assistant\n" 报错
        # text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image<|im_end|>\n<|im_start|>assistant\n" #正常
        # text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image<|im_end|>" # 什么都不生成（一堆\n)
        # text = "<|image_pad|>Pick up the "  # 瞎输出
        # 拟采用的两种sequence：
        # 1. 完整版
        # text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What action should the robotic arm take to pick up the pink block.<|im_end|>\n<|im_start|>assistant\n"
        # 2. 简化版：与其他模型更一致
        # text = "<|im_start|><|vision_start|><|image_pad|><|vision_end|>pick up the pink block\n"  # 输出“”
        image_inputs, video_inputs = process_vision_info(messages)
        print(image_inputs)
        print(video_inputs)
        # exit()
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(model.device)
        # print(inputs.input_ids.shape)
        # print(inputs)
        # input_ids: 1,280; 280=256+24, 其中image256, text24
        # 151644, 8948, 198,2610,
        # attention_mask: 1,280
        # pixel_values: 1024,1176
        # image_grid_thw: 1,32,32

        # Inference: Generation of the output
        outputs = model.forward(**inputs)
        # 提取最后一个输出token，并预测下一个词
        # 假设outputs为模型的输出logits
        # outputs形状通常为(batch_size, seq_len, vocab_size)
        import torch

        # 获取最后一个时间步的logits
        last_logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)

        # 预测下一个词（取概率最大的token）
        predicted_next_token = torch.argmax(last_logits, dim=-1)  # (batch_size,)
        print(predicted_next_token)

    # test
    from transformers import Qwen3_VL_MOEForConditionalGeneration, AutoProcessor

    model_name = "/mnt/zjk/jianke_z/VLMA-baselines/1/qwen3vl-30a3"
    model = Qwen3_VL_MOEForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = 'left'
    print("self.tokenizer.eos_token:", processor.tokenizer.eos_token)  # <im_end> in qwen2.5-vl
    print("self.tokenizer.bos_token:", processor.tokenizer.bos_token)  # None in qwen2.5-vl
    test_qwen25vl(model, processor)
    exit()
    configs = load_config("/mnt/zjk/jianke_z/vlm4vla/configs/calvin_finetune/finetune_qwen3vl-4b_calvin.json")
    use_hand_rgb = False
    model = RoboQwen3VL(
        configs=configs,
        train_setup_configs=configs["train_setup"],
        fwd_head_configs=None,
        window_size=configs["window_size"],
        use_hand_rgb=use_hand_rgb,
        act_head_configs=configs["act_head"],
        fwd_pred_next_n=configs["fwd_pred_next_n"],
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total trainable Qwen2.5VL-VLA Model Parameters: {total_params / (1000*1000):.2f}M")
    # 打印model第一层所有参数名字
    for name, param in model.named_children():
        """ 
        backbone:
            visual
                patch_embed
                rotary_pos_emb
                blocks
                merger
            model
                embed_tokens
                layers
                norm
                rotary_emb
            lm_head
        act_head:
            rnn, actions, gripper
        """
        total_params = sum(p.numel() for p in param.parameters() if p.requires_grad)
        print(f"-- trainable {name} Parameters: {total_params / (1000*1000):.2f}M")

        # for name, param in param.named_children():
        #     # visual, model, lm_head
        #     # rnn, actions, gripper
        #     for name, param in param.named_children():
        #         print(name)
    # test_qwen25vl(model.model.to("cuda:0"), model.processor)  # ok
    bs, seq_len = 2, 2
    device = "cuda:0"
    model = model.to(device)

    # device = 'cpu'
    fwd_next_n = configs["fwd_pred_next_n"]
    from PIL import Image
    path_dir = [path, path]
    vision_x = [Image.open(p).convert("RGB") for p in path_dir]
    # results = [model.process_image_unified(file) for file in vision_x]
    image_inputs = model.process_vision_info(vision_x)
    text = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What action should the robotic arm take to pick up the pink block<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|><|vision_start|><|image_pad|><|vision_end|>pick up the pink block\n"
    ]

    model.tokenizer.padding_side = "left"
    inputs = model.processor(
        text=text,
        images=image_inputs,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    print("inputs.pixel_values", inputs.pixel_values.shape)
    inputs = inputs.to(device)
    input_ids = inputs.pop("input_ids")
    inputs_embeds = model.word_embedding(input_ids)
    inputs.pixel_values = inputs.pixel_values.type(model.vision_tower.dtype)
    image_embeds = model.vision_tower(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
    n_image_tokens = (input_ids == model.model.config.image_token_id).sum().item()
    n_image_features = image_embeds.shape[0]
    if n_image_tokens != n_image_features:
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")

    mask = input_ids == model.model.config.image_token_id
    mask_unsqueezed = mask.unsqueeze(-1)
    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
    image_mask = mask_expanded.to(inputs_embeds.device)

    image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs["inputs_embeds"] = inputs_embeds.masked_scatter(image_mask, image_embeds)

    outputs = model.model(**inputs)

    import torch
    last_nopadding_logits = []
    # 获取最后一个非padding的logits
    for i in range(inputs["attention_mask"].shape[0]):
        last_nopadding_logits.append(outputs.logits[i, inputs["attention_mask"][i].bool(), :][-1, :])
    last_nopadding_logits = torch.stack(last_nopadding_logits)
    print(last_nopadding_logits.shape)

    # 预测下一个词（取概率最大的token）
    predicted_next_token = torch.argmax(last_nopadding_logits, dim=-1)  # (batch_size,)
    print(predicted_next_token)
    response = model.processor.batch_decode(
        predicted_next_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("response_next:", response)

    # generated_ids = model.model.generate(**inputs, max_new_tokens=128)
    # generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
    # response = model.processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # print("response:", response)
    exit()
    text = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What action should the robotic arm take to pick up the pink block.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|><|vision_start|><|image_pad|><|vision_end|>pick up the pink block\n"
    ]
    # inputs = model.processor(
    #     text=text,
    #     images=None,
    #     videos=None,
    #     truncation="only_first",
    #     return_tensors="pt",
    #     padding="longest",
    #     max_length=512,
    # )
    inputs = model.tokenizer(
        text,
        truncation="only_first",
        return_tensors="pt",
        padding="longest",
        max_length=512,
        add_special_tokens=False,
    )
    print(inputs)
    print(inputs.input_ids.shape)  # 1, 36
    # vision_gripper = torch.zeros((bs, seq_len, 3, img_size, img_size), dtype=torch.float16).to(device)
    # lang_x = torch.ones((bs, 10), dtype=torch.long).to(device) * 100
    # attention_mask = torch.ones((bs, 10)).bool().to(device)
    # action_lables = (
    #     torch.randn(bs, seq_len, fwd_next_n, 6).to(device),
    #     torch.zeros(bs, seq_len, fwd_next_n).to(device),
    # )
    # model = model.to(device).to(torch.float16)
    # test_res = model(
    #     vision_x,
    #     lang_x,
    #     attention_mask=attention_mask,
    #     position_ids=None,
    #     use_cached_vision_x=False,
    #     action_labels=action_lables,
    #     action_mask=None,
    #     caption_labels=None,
    #     caption_mask=None,
    #     past_key_values=None,
    #     use_cache=False,
    #     # vision_gripper=vision_gripper,
    #     vision_gripper=None,
    #     fwd_rgb_labels=None,
    #     fwd_hand_rgb_labels=None,
    #     fwd_mask=None,
    #     data_source="action",
    # )

    # print(test_res)
