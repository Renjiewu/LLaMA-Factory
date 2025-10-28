from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import pandas as pd
import numpy as np
import os
# os.environ["UNSLOTH_VLLM_STANDBY"] = "1" 
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/Qwen3-4B-Base",
#     max_seq_length = max_seq_length,
#     load_in_4bit = False, # False for LoRA 16bit
#     fast_inference = True, # Enable vLLM fast inference
#     max_lora_rank = lora_rank,
#     gpu_memory_utilization = 0.9, # Reduce if out of memory
#     dtype=torch.float16, # Use bfloat16 if possible
# )

# print("model loaded")

# model = FastLanguageModel.get_peft_model(
#     model,
#     r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#     target_modules = [
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj",
#     ],
#     lora_alpha = lora_rank*2, # *2 speeds up training
#     use_gradient_checkpointing = "unsloth", # Reduces memory usage
#     random_state = 3407,
# )

# print("peft model loaded")


dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
dataset = dataset.to_pandas()[
    ["expected_answer", "problem", "generated_solution"]
]

# Try converting to number - if not, replace with NaN
is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors = "coerce").notnull()
# Select only numbers
dataset = dataset.iloc[np.where(is_number)[0]]

print(f"Dataset size: {len(dataset)}")
