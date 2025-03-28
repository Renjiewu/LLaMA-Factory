import json

from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor

from auto_gptq import BaseQuantizeConfig
from auto_gptq.modeling import Qwen2VLGPTQForConditionalGeneration
from auto_gptq import AutoGPTQForCausalLM

# from typing import Union

# import torch
# import torch.nn as nn

# import importlib
# import auto_gptq.modeling._utils as _tmp_utils

# # from accelerate.big_modeling import dispatch_model
# def get_device_hook(obj: Union[torch.Tensor, nn.Module]):
#     if isinstance(obj, torch.Tensor):
#         return obj.device
#     else:
#         try:
#             return next(obj.parameters()).device
#         except StopIteration:
#             return next(obj.buffers()).device
# _tmp_utils.get_device = get_device_hook

# importlib.reload(_tmp_utils)

# test sftp
# test sftp download
# hub.yzuu.cf
# git clone -b v0.7.1 --depth=1 https://github.com/AutoGPTQ/AutoGPTQ.git gekko pip install -vvv --no-build-isolation -e .
# libcusolver-dev-12-4 libcublas-dev-12-4 libcusparse-dev-12-4
# Specify paths and hyperparameters for quantization
model_path = "/data/models/qwen2vl_72b_all_P1P2_28122024_20eps_merged"
quant_path = "/data/models/qwen2vl_72b_all_P1P2_28122024_20eps_merged_py_script_4bit_test3"
quantize_config = BaseQuantizeConfig(
    bits=4,  # 4 or 8
    group_size=128,
    damp_percent=0.1,
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    static_groups=False,
    sym=True,
    true_sequential=True,
)

# Load your processor and model with AutoGPTQ
# device_map = {}
# device_map['language_model.model.rotary_emb'] = 0
processor = Qwen2VLProcessor.from_pretrained(model_path)
# model = Qwen2VLGPTQForConditionalGeneration.from_pretrained(model_path, quantize_config)
# max_memory = {0: "4GIB", 1: "50GIB", 2: "50GIB",'cpu': "300GIB"}
# max_memory.update({i: f"{1}GIB" for i in range(8)})
# max_memory.update({'cpu': "400GIB"})
# max_memory[0] = "1GIB"
model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config,
                                            # max_memory=max_memory
                                            )

# Then you need to prepare your data for calibaration. What you need to do is just put samples into a list,
# each of which is a typical chat message as shown below. you can specify text and image in `content` field:
# dataset = [
#     # message 0
#     [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Tell me who you are."},
#         {"role": "assistant", "content": "I am a large language model named Qwen..."},
#     ],
#     # message 1
#     [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": "file:///path/to/your/image.jpg"},
#                 {"type": "text", "text": "Output all text in the image"},
#             ],
#         },
#         {"role": "assistant", "content": "The text in the image is balabala..."},
#     ],
#     # other messages...
#     ...,
# ]
# here, we use a caption dataset **only for demonstration**. You should replace it with your own sft dataset.
def prepare_dataset(n_sample: int = 500) -> list[list[dict]]:
    dataset_path = '/data/dataset/qwen2vl_out800_entity20_pos0.9_neg0.2_with_text_9178_fixed.json'
    with open(dataset_path) as f:
        dataset = json.load(f)[:n_sample]
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample['images'][0]},
                    {"type": "text", "text": sample['messages'][0]['content'].replace('<image>', '')},
                ],
            },
            {"role": "assistant", "content": sample['messages'][1]['content']},
        ]
        for sample in dataset
    ]


dataset = prepare_dataset(n_sample=1)


def batched(iterable, n: int):
    # batched('ABCDEFG', 3) → ABC DEF G
    assert n >= 1, "batch size must be at least one"
    from itertools import islice

    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


calib_data = []
batch_size = 1
text = None
for batch in batched(dataset, batch_size):
    text = processor.apply_chat_template(batch, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(batch)
    inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    calib_data.append(inputs)
print(f'text: {text}')
print(f'dataset length: {len(calib_data)}')
print(f'\n\n------------data preparing has finished------------------\n\n')


# Then just run the calibration process by one line of code:
#model.to(device_map)
model.quantize(calib_data, cache_examples_on_gpu=False)

# Finally, save the quantized model:
model.save_quantized(quant_path, use_safetensors=True)
processor.save_pretrained(quant_path)

# Optional: save the quantized model compatible with transformers
# from optimum.gptq import GPTQQuantizer
# from transformers.utils.quantization_config import QuantizationMethod
# gptq_quantizer = GPTQQuantizer.from_dict(model.quantize_config.to_dict())
# model.model.is_quantized = True
# model.model.quantization_method = QuantizationMethod.GPTQ
# model.model.config.quantization_config = gptq_quantizer.to_dict()
# gptq_quantizer.save(model.model, quant_path, max_shard_size="4GB", safe_serialization=True)
# processor.save_pretrained(quant_path)

# Then you can obtain your own GPTQ quantized model for deployment. Enjoy!
