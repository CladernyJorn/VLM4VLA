import torch
import copy
from PIL import Image
from typing import Optional, Tuple, List
from typing import Sequence
from vlm4vla.model.backbone.base_backbone import BaseRoboVLM, load_config, deep_update
import numpy as np
from einops import rearrange, repeat
from functools import partial
from transformers import AutoProcessor, AutoModelForCausalLM

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from vlm4vla.model.vlm_builder import build_vlm


class RoboEagle2(BaseRoboVLM):

    def _init_backbone(self):
        # print("using internvl35 backbone rewrite here !!!!!!!!!!!!!!!!")
        # rewrite to add tokens to model and tokenizer
        tokenizer, model = build_vlm(self.configs["vlm"], self.configs["tokenizer"])
        assert "Processor" not in self.configs["tokenizer"]["type"], "Processor is not supported for internvl35"
        self.tokenizer = tokenizer
        model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        return self.tokenizer, model

    @property
    def image_processor(self):
        # used in dataloader, just transform image to tensor before random shift
        # import torchvision.transforms as T
        # image_preprocess = T.Compose([
        #     T.Resize(
        #         (
        #             self.configs.get("image_size", 224),
        #             self.configs.get("image_size", 224),
        #         ),
        #         interpolation=T.InterpolationMode.BICUBIC,
        #     ),
        #     T.Lambda(lambda img: img.convert("RGB")),
        #     T.Lambda(lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float()),  # uint8转为float
        # ])
        image_preprocess = partial(load_image, input_size=self.configs.get("image_size", 224), max_num=1)
        return image_preprocess

    @property
    def hidden_size(self):
        return self.model.language_model.config.hidden_size

    @property
    def word_embedding(self):
        return self.model.language_model.model.embed_tokens

    @property
    def text_tower(self):
        # not used
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_model

    @property
    def model(self):
        return self.backbone

    @property
    def start_image_token_id(self):
        raise NotImplementedError("start_image_token_id is not supported for internvl35")
        return torch.LongTensor([self.model.config.visual["image_start_id"]])

    @property
    def end_image_token_id(self):
        raise NotImplementedError("end_image_token_id is not supported for internvl35")
        return torch.LongTensor([self.model.config.visual["image_start_id"] + 1])

    def encode_images(self, images):
        raise NotImplementedError("encode_images is not supported for internvl35")

        return image_tensor, grid_thw

    def forward_continuous(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,  # not used (not transfered from forward_action)
        use_cached_vision_x: bool = False,  # TODO: Do we need this? If not we can remove it from here
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_mask: torch.Tensor = None,
        caption_labels: torch.Tensor = None,
        caption_mask: torch.Tensor = None,
        past_key_values=None,  # not used (not transfered from forward_action)
        use_cache: bool = False,  # not used (not transfered from forward_action)
        vision_gripper=None,
        fwd_rgb_labels: torch.Tensor = None,
        fwd_hand_rgb_labels: torch.Tensor = None,
        fwd_mask: torch.Tensor = None,
        instr_and_action_ids=None,
        instr_and_action_labels=None,
        instr_and_action_mask=None,
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
        input_ids = lang_x
        assert input_ids.shape[0] == bs * seq_len and history_type in ["post", "pre"]
        # if history_type in ["post", "pre"]:
        #     lang_x = lang_x.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1)  # original shape: (4, 12) -> (4*8, 12)
        #     attention_mask = (attention_mask.unsqueeze(1).repeat(1, seq_len, 1).flatten(0, 1))

        # input_embeds = self.word_embedding(input_ids)  # (4*8, 12, 1024)
        # vision_x是没与处理的0-255图片
        assert seq_len == 1
        vision_x = vision_x.squeeze(1)
        vit_embeds = self.model.extract_feature(vision_x)
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.model.img_context_token_id)
        # print(selected.sum())
        assert selected.sum() != 0
        # print(selected.sum(), vit_embeds.size(0))
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
        input_embeds = input_embeds.reshape(B, N, C)

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
            # print("after insert action, insert mask", action_token_mask)
            # print("after insert action, multimodal attention mask", multimodal_attention_mask)
            # print("after insert action, forward text:")
            # self.forward_text_test(multimodal_embeds, multimodal_attention_mask)

        if history_type == "pre":
            multimodal_embeds = rearrange(
                multimodal_embeds, "(b l) n d -> b (l n) d",
                l=seq_len)  # original shape: (4*8, 271, 2304) -> (4, 8*271, 2304)
            if multimodal_attention_mask is not None:
                multimodal_attention_mask = rearrange(multimodal_attention_mask, "(b l) n -> b (l n)", l=seq_len)

        output = self.model.language_model(
            attention_mask=multimodal_attention_mask,
            inputs_embeds=multimodal_embeds,
            output_hidden_states=True,
        )

        output_hs = output.hidden_states[-1].clone()
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


# copied from internvl hf
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),  # resize to 224 first, align with other models
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j)
                        for n in range(min_num, max_num + 1)
                        for i in range(1, n + 1)
                        for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    # print(len(processed_images)) 9,3,448,448
    # print(processed_images[0].size)
    # print(processed_images[1].size)
    # print(processed_images[2].size)
    # print(processed_images[3].size)
    # print(processed_images[4].size)
    # print(processed_images[5].size)
    # print(processed_images[6].size)
    return processed_images


def load_image(image, input_size=448, max_num=1):
    # image = Image.open(image_file).convert('RGB')
    # image 是0-255图片，形状16，1，3，224，224
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values).squeeze()
    return pixel_values


if __name__ == "__main__":
    # path = "/mnt/zjk/jianke_z/vlm4vla/test_image.png"
    path = "/mnt/zjk/jianke_z/vlm4vla/imgs/vlm4vla.png"
    from transformers.image_processing_utils_fast import (
        # BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
        # BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        BaseImageProcessorFast,
        DefaultFastImageProcessorKwargs,
        divide_to_patches,
        group_images_by_shape,
        reorder_images,
    )

    def test_qwen25vl(model, processor):
        import torch
        img_path = "/mnt/zjk/jianke_z/vlm4vla/imgs/vlm4vla.png"
        img = Image.open("/mnt/zjk/jianke_z/vlm4vla/imgs/vlm4vla.png")
        messages = [{
            "role": "user",
            "content": [{
                "type": "image",
                "image": img_path
            }, {
                "type": "text",
                "text": "Describe the image"
            }]
        }]
        text_list = [processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
        image_inputs, video_inputs = processor.process_vision_info(messages)
        inputs = processor(text=text_list, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")
        model = model.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(output_text)
        # generation_config = dict(max_new_tokens=1024, do_sample=True)
        question = '<image>\nPlease describe the image shortly.'
        # response = model.chat(tokenizer, pixel_values, question, generation_config)
        # print(f'User: {question}\nAssistant: {response}')

        # text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # gg not good
        # print("text:", text)
        # <|im_start|>system
        # You are a helpful assistant.<|im_end|>
        # <|im_start|>user
        # <|vision_start|><|image_pad|><|vision_end|>Describe the image<|im_end|>
        # <|im_start|>assistant
        # print(text)
        text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<img><IMG_CONTEXT></img>Describe the image<|im_end|>\n<|im_start|>assistant\n"
        # )
        # text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|image_pad|><|vision_end|>Describe the image<|im_end|>\n<|im_start|>assistant\n" 报错
        # text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image<|im_end|>\n<|im_start|>assistant\n" #正常
        # text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image<|im_end|>" # 什么都不生成（一堆\n)
        # text = "<|image_pad|>Pick up the "  # 瞎输出
        # 拟采用的两种sequence：
        # 1. 完整版
        # text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<img><IMG_CONTEXT></img>What action should the robotic arm take to pick up the pink block.<|im_end|>\n<|im_start|>assistant\n"
        # 2. 简化版：与其他模型更一致

        # exit()
        model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        print(model.num_image_token)
        model_inputs = tokenizer(
            text.replace("<IMG_CONTEXT>", "<IMG_CONTEXT>" * model.num_image_token * 1), return_tensors="pt")
        print(model_inputs["input_ids"].shape)
        print(model_inputs["attention_mask"].shape)

        # generation_config['eos_token_id'] = tokenizer.convert_tokens_to_ids("<|im_end|>")
        # outputs = model.generate(
        #     pixel_values=pixel_values,
        #     input_ids=model_inputs["input_ids"].to(model.device),
        #     attention_mask=model_inputs["attention_mask"].to(model.device),
        #     **generation_config)
        # response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # print(response)

        input_ids = model_inputs["input_ids"].to(model.device)
        vit_embeds = model.extract_feature(pixel_values)
        input_embeds = model.language_model.get_input_embeddings()(input_ids).clone()
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == model.img_context_token_id)
        assert selected.sum() != 0
        # print(selected.sum(), vit_embeds.size(0))
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
        input_embeds = input_embeds.reshape(B, N, C)
        # outputs = model.language_model.generate(
        #     inputs_embeds=input_embeds,
        #     attention_mask=model_inputs["attention_mask"].to(model.device),
        #     use_cache=True,
        #     **generation_config,
        # )
        # response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # print(response)

        # inputs = inputs.to(model.device)
        # # Inference: Generation of the output
        # outputs = model.forward(**inputs)
        outputs = model.language_model.forward(
            inputs_embeds=input_embeds,
            attention_mask=model_inputs["attention_mask"].to(model.device),
        )

        # 获取最后一个时间步的logits
        last_logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)

        # 预测下一个词（取概率最大的token）
        predicted_next_token = torch.argmax(last_logits, dim=-1)  # (batch_size,)
        print(predicted_next_token)  # The 对了！

        # generated_ids = model.generate(**inputs, max_new_tokens=128)
        # generated_ids = model(**inputs)
        # generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        # response = processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print("response:", response)

    # test
    from transformers import AutoProcessor, AutoModel

    path = '/mnt/zjk/jianke_z/VLMA-baselines/1/nvidia/Eagle2-2B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, use_fast=True)
    test_qwen25vl(model, processor)
    exit()

    configs = load_config("/mnt/zjk/jianke_z/vlm4vla/configs/calvin_finetune/finetune_internvl35-4b_calvin.json")
    use_hand_rgb = False
    model = RoboInternVL35(
        configs=configs,
        train_setup_configs=configs["train_setup"],
        fwd_head_configs=None,
        window_size=configs["window_size"],
        use_hand_rgb=use_hand_rgb,
        act_head_configs=configs["act_head"],
        fwd_pred_next_n=configs["fwd_pred_next_n"],
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total trainable InterVL35-VLA Model Parameters: {total_params / (1000*1000):.2f}M")
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
    processor = model.image_latent_numprocessor
    image_inputs = processor(vision_x)
    text = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<img><IMG_CONTEXT></img>What action should the robotic arm take to pick up the pink block<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|><img><IMG_CONTEXT></img>pick up the pink block\n"
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
