#!/usr/bin/env python
"""
Merge full-weight RL (GRPO) sparse checkpoint into a single HuggingFace model.
Checkpoint: RL_sparse_best_checkpoint/global_step30 (DeepSpeed ZeRO-3 shards).
Uses zero_to_fp32 to consolidate shards, loads into base qwen3-8b, saves as runnable HF model.

Run with an env that has deepspeed (e.g. pygeox-openrlhf):
  /path/to/pygeox-openrlhf/bin/python merge_rl_sparse_weights.py
"""
import os
import sys
import importlib.metadata

# Work around transformers dependency check when packaging version is reported as None
_orig_version = getattr(importlib.metadata, "version", None)
if _orig_version is not None:
    def _version(pkg):
        v = _orig_version(pkg)
        if pkg == "packaging" and v is None:
            return "25.0"
        return v
    importlib.metadata.version = _version

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Run on CPU to avoid OOM; vLLM can use GPU later
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Paths — update these to match your environment before running
# RL_sparse_best_checkpoint contains global_step30/ and zero_to_fp32.py
BASE_MODEL_PATH = "YOUR_BASE_MODEL_PATH"  # e.g. "/path/to/qwen3-8b"
CHECKPOINT_PARENT = "YOUR_CHECKPOINT_PATH"  # e.g. "/path/to/project/model_training/RL_sparse_best_checkpoint"
CHECKPOINT_TAG = "global_step30"  # the checkpoint step tag to merge
SAVE_PATH = "YOUR_SAVE_PATH"  # e.g. "/path/to/project/model_training/merged/qwen3-8b-RL-sparse-step30"

# Import zero_to_fp32 from the checkpoint parent (script lives next to global_step* dirs)
if CHECKPOINT_PARENT not in sys.path:
    sys.path.insert(0, CHECKPOINT_PARENT)
from zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


def main():
    print(f"📦 Merging full-weight RL sparse checkpoint: {CHECKPOINT_PARENT} / {CHECKPOINT_TAG}")
    print(f"   Base model: {BASE_MODEL_PATH}")
    print(f"   Output:     {SAVE_PATH}")
    os.makedirs(SAVE_PATH, exist_ok=True)

    print("\n🔧 Consolidating ZeRO-3 checkpoint to fp32 state dict...")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(
        CHECKPOINT_PARENT, tag=CHECKPOINT_TAG
    )

    # Strip "module." prefix if present (DeepSpeed/FSDP often wraps the model)
    keys = list(state_dict.keys())
    if any(k.startswith("module.") for k in keys):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        print("   Stripped 'module.' prefix from state dict keys.")

    print("\n🧠 Loading base model (bf16, CPU)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print("📥 Loading consolidated weights into model...")
    # Cast to bf16 to match model dtype; allow strict=False for buffer/name mismatches
    state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
    del state_dict
    load_info = model.load_state_dict(state_dict_bf16, strict=False)
    del state_dict_bf16
    if load_info.missing_keys:
        print(f"   Missing keys (expected if config differs): {len(load_info.missing_keys)}")
    if load_info.unexpected_keys:
        print(f"   Unexpected keys: {len(load_info.unexpected_keys)}")

    print("💾 Saving merged model (safetensors)...")
    model.save_pretrained(SAVE_PATH, safe_serialization=True)

    print("📂 Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.save_pretrained(SAVE_PATH)

    print(f"\n✨ Done. Merged model: {SAVE_PATH}")


if __name__ == "__main__":
    main()
