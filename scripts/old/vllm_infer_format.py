# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os

# from concurrent.futures.process import ProcessPoolExecutor
import fire
# from lmformatenforcer import JsonSchemaParser
# from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data
from transformers import Seq2SeqTrainingArguments

# from data.dataset_op import DiaAnswerFormat, DiaAnswerFormat2, ImgAnswerFormat, ImgAnswerFormat2
# from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
# from llamafactory.extras.constants import IGNORE_INDEX
# from llamafactory.extras.misc import get_device_count
# from llamafactory.extras.packages import is_pillow_available, is_vllm_available
# from llamafactory.hparams import get_infer_args
# from llamafactory.model import load_tokenizer

from vllm.engine.arg_utils import AsyncEngineArgs


IMAGE_FACTOR = 28
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
MAX_RATIO = 200


# if is_pillow_available():
#     from PIL import Image
#     from PIL.Image import Image as ImageObject


# if is_vllm_available():
#     from vllm import LLM, SamplingParams
#     from vllm.lora.request import LoRARequest

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def img_op(image, min_pixels=None, max_pixels=None, size_factor: int = IMAGE_FACTOR):
    width, height = image.size
    min_pixels = min_pixels if min_pixels else MIN_PIXELS
    max_pixels = max_pixels if max_pixels else MAX_PIXELS
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))

    return image

def get_inputs(train_dataset, tokenizer, batch_size, batch_start):
    inputs, prompts, labels = [], [], []
    
    for i in range(batch_start, batch_start+batch_size):
        sample = train_dataset[i]
        if sample["images"]:
            multi_modal_data = {"image": []}
            image_num = 0
            for image in sample["images"]:
                # if image_num >= 4:
                #     break
                # image_num += 1
                if not isinstance(image, (str, ImageObject)):
                    raise ValueError(f"Expected image input is a path or PIL.Image, but got {type(image)}.")

                if isinstance(image, str):
                    image = img_op(Image.open(image).convert("RGB"))

                multi_modal_data["image"].append(image)
        else:
            multi_modal_data = None

        inputs.append({"prompt_token_ids": sample["input_ids"], "multi_modal_data": multi_modal_data})
        prompts.append(tokenizer.decode(sample["input_ids"], skip_special_tokens=False))
        # labels.append(
        #     tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, sample["labels"])), skip_special_tokens=False)
        # )
    return inputs, prompts


def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: int = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    preprocessing_num_workers=16,
    loop_batch_size=1000,
    phase="img",
    loop_start=0,
    loop_size=1.0,
):
    r"""
    Performs batch generation using vLLM engine, which supports tensor parallelism.
    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """
    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            preprocessing_num_workers=preprocessing_num_workers,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)

    # 需要控制一下输出，选择用哪个schema
    tokenizer_data = build_vllm_token_enforcer_tokenizer_data(tokenizer)
    if phase == "img":
        logits_processor = build_vllm_logits_processor(tokenizer_data, JsonSchemaParser(ImgAnswerFormat2.model_json_schema()))
    elif phase == "dia":
        logits_processor = build_vllm_logits_processor(tokenizer_data, JsonSchemaParser(DiaAnswerFormat2.model_json_schema()))
    else:
        raise ValueError("phase should be 'img' or 'dia'.")
    # inputs, prompts, labels = [], [], []

    # for sample in dataset_module["train_dataset"]:
    #     if sample["images"]:
    #         multi_modal_data = {"image": []}
    #         for image in sample["images"]:
    #             if not isinstance(image, (str, ImageObject)):
    #                 raise ValueError(f"Expected image input is a path or PIL.Image, but got {type(image)}.")

    #             if isinstance(image, str):
    #                 image = img_op(Image.open(image).convert("RGB"))

    #             multi_modal_data["image"].append(image)
    #     else:
    #         multi_modal_data = None

    #     inputs.append({"prompt_token_ids": sample["input_ids"], "multi_modal_data": multi_modal_data})
    #     prompts.append(tokenizer.decode(sample["input_ids"], skip_special_tokens=False))
    #     # labels.append(
    #     #     tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, sample["labels"])), skip_special_tokens=False)
    #     # )

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k,
        stop_token_ids=[tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids,
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=False,
        logits_processors=[logits_processor],
    )
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "tensor_parallel_size": get_device_count() or 1,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
        "max_lora_rank": 128,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 11, "video": 2}

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    llm = LLM(**engine_args)
    preds = []

    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))

    if loop_start == 0:
        with open(save_name, "w", encoding="utf-8") as f:
            f.write("\n")

    for batch in range(loop_start, dataset_module["train_dataset"].num_rows, loop_batch_size):

        inputs, prompts = get_inputs(train_dataset=dataset_module["train_dataset"], tokenizer=tokenizer, batch_size=loop_batch_size, batch_start=batch)

        results = llm.generate(inputs, sampling_params, lora_request=lora_request)
        preds = [result.outputs[0].text for result in results]

        with open(save_name, "a", encoding="utf-8") as f:
            for text, pred in zip(prompts, preds):
                f.write(json.dumps({"prompt": text, "predict": pred}, ensure_ascii=False) + "\n")
        if batch + loop_batch_size >= loop_size * dataset_module["train_dataset"].num_rows:
            break

    print("*" * 70)
    print(f"{len(prompts)} generated results have been saved at {save_name}.")
    print("*" * 70)

    return batch + loop_batch_size

def run_one(phase, loop_size=1.0):
    if phase == "img":
        adapter_name_or_path = "output/qwen2_vl-7b/dora/dora_r64_img_ep3"
        save_name="output/qwen2_vl-7b/dora/sft-infer-dora_r64_img_ep3/generated_predictions.jsonl"
        dataset="mire_format_enhance_eval_img"
    elif phase == "dia":
        adapter_name_or_path = "output/qwen2_vl-7b/dora/dora_r64_dia_ep3"
        save_name="output/qwen2_vl-7b/dora/sft-infer-dora_r64_dia_ep3/generated_predictions.jsonl"
        dataset="mire_format_enhance_eval_dia"
    else:
        raise ValueError("phase should be 'img' or 'dia'.")


    pretrained_model_name="Qwen/Qwen2-VL-7B-Instruct",
    quantization=None,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    trust_remote_code=False,
    enforce_eager=False

    engine_args = AsyncEngineArgs(model=pretrained_model_name,
                                quantization=quantization,
                                tensor_parallel_size=tensor_parallel_size,
                                gpu_memory_utilization=gpu_memory_utilization,
                                trust_remote_code=trust_remote_code,
                                disable_log_requests=True,
                                enforce_eager=enforce_eager)


    loop_times = int(1 // loop_size)
    loop_size_list = []
    for i in range(loop_times):
        loop_size_list.append(loop_size*(i+1))

    if 1 % loop_size != 0:
        loop_size_list.append(1.0)

    batch = 0
    for loop_stop_size in loop_size_list:
        batch = vllm_infer(
            model_name_or_path="Qwen/Qwen2-VL-7B-Instruct",
            # adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir="data",
            template="qwen2_vl",
            cutoff_len=4500,
            max_samples=None,
            vllm_config="{}",
            save_name=save_name,
            temperature=0.5,
            top_p=0.7,
            top_k=20,
            max_new_tokens=100,
            repetition_penalty=1.2,
            preprocessing_num_workers=16,
            loop_batch_size=100,
            phase=phase,
            loop_start=batch,
            loop_size=loop_stop_size,
        )
        print(f"batch: {batch}")

if __name__ == "__main__":
    # fire.Fire(vllm_infer)
    phase = "img"
    run_one(phase, loop_size=0.5)

    phase = "dia"
    run_one(phase, loop_size=0.5)
