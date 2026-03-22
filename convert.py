import os
from unsloth import FastLanguageModel

# Pulls the token from the Vast.ai environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")

# The repo where your adapters currently live
ADAPTER_REPO = "phinjaz/last-checkpoint" 

# The NEW repo where your finished GGUF will be saved
GGUF_REPO = "phinjaz/Qwen3-14B-Decomp-GGUF" 

print(f"Loading base model and adapters from {ADAPTER_REPO}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = ADAPTER_REPO,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True, # Keep this True! Unsloth handles the un-quantizing and merging automatically.
)

print(f"Merging, converting to Q4_K_M, and pushing to {GGUF_REPO}...")
model.push_to_hub_gguf(
    GGUF_REPO,
    tokenizer,
    quantization_method = "q4_k_m",
    token = HF_TOKEN,
)

print("Conversion and upload complete!")
