import copy
import transformers
import torch

from vlm4vla.utils.model_utils import build_tokenizer


def build_vlm(vlm_config, tokenizer_config, precision="bf16"):
    vlm_config = copy.deepcopy(vlm_config)
    model_path = vlm_config.get("pretrained_model_name_or_path")
    model_name = vlm_config.get("name")
    model_type = vlm_config.get("type", "AutoModel")
    if model_name == "paligemma":
        from transformers.models.auto.processing_auto import AutoProcessor
        from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
        print(f"Loading model {model_type} from {model_path}")
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            attn_implementation="flash_attention_2",
        )
        tokenizer = AutoProcessor.from_pretrained(model_path)
    elif model_name == "qwen25vl":

        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
        from transformers.models.auto.processing_auto import AutoProcessor
        # from transformers import Qwen2VLImageProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cpu",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoProcessor.from_pretrained(model_path, min_pixels=28 * 28 * 256, max_pixels=1280 * 28 * 28)
    elif model_name == "qwen3vl":
        from transformers.models.auto.processing_auto import AutoProcessor
        # from transformers import Qwen2VLImageProcessor
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3_VLForConditionalGeneration
            model = Qwen3_VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map="cpu",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
        except:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map="cpu",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
        tokenizer = AutoProcessor.from_pretrained(model_path, min_pixels=28 * 28 * 256, max_pixels=1280 * 28 * 28)
    elif model_name == "qwen3vlmoe":

        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3_VL_MOEForConditionalGeneration
        from transformers.models.auto.processing_auto import AutoProcessor
        # from transformers import Qwen2VLImageProcessor

        model = Qwen3_VL_MOEForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cpu",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoProcessor.from_pretrained(model_path, min_pixels=28 * 28 * 256, max_pixels=1280 * 28 * 28)
    elif model_name == "kosmos":
        # from transformers.models.auto.modeling_auto import AutoModelForVision2Seq
        from transformers.models.auto.processing_auto import AutoProcessor
        from transformers.models.kosmos2.modeling_kosmos2 import Kosmos2ForConditionalGeneration
        model = Kosmos2ForConditionalGeneration.from_pretrained(
            model_path, device_map="cpu", torch_dtype=torch.bfloat16)
        tokenizer = AutoProcessor.from_pretrained(model_path)
        # tokenizer = AutoTokenizer.from_pretrained(
        #     model_path,
        #     model_max_length=1024,
        #     padding_side="right",
        #     use_fast=False,
        # )
    elif model_name == "florence":
        from transformers import AutoModelForCausalLM, AutoProcessor
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cpu", torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    elif model_name == "internvl35":
        # from transformers.models.internvl35.modeling_internvl35 import InternVL35ForConditionalGeneration
        from transformers import AutoTokenizer, AutoModel
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, use_flash_attn=True, trust_remote_code=True, device_map="cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    else:
        # from transformers.models.auto.modeling_auto import AutoModelForVision2Seq
        model = getattr(transformers, model_type).from_pretrained(model_path, trust_remote_code=False)
        tokenizer = build_tokenizer(tokenizer_config)

    return tokenizer, model
