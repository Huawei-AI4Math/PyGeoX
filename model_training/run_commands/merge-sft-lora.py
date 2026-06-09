import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configuration — update these paths before running
base_model_path = "YOUR_BASE_MODEL_PATH"  # e.g. "/path/to/qwen3-8b"
adapter_path = "YOUR_ADAPTER_PATH"  # e.g. "/path/to/project/model_training/sft_training/SFT_dense/checkpoint-XXX"
save_path = "YOUR_SAVE_PATH"  # e.g. "/path/to/project/model_training/merged/qwen3-8b-SFT-Full"

print(f"📦 Task: Merging {adapter_path}")
os.makedirs(save_path, exist_ok=True)

print("🧠 Loading base model into System RAM (bf16)...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16, 
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

print("📥 Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_path, device_map="cpu")

print("🔗 Merging weights (this may take a few minutes)...")
model = model.merge_and_unload()

print("💾 Saving merged model to disk...")
model.save_pretrained(save_path, safe_serialization=True)

print("📂 Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.save_pretrained(save_path)
print(f"✨ Success! Merged model: {save_path}")