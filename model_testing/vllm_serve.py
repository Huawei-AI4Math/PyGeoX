#!/usr/bin/env python3
"""
vLLM Server for PyGeoX RL-trained Model

This script serves the RL-trained PyGeoX model using vLLM with a maximum
context length of 20,000 tokens.

Usage:
    python serve_rl_model.py [--gpu GPU_ID] [--port PORT]
"""

import os
import subprocess
import sys
import json
import argparse
from pathlib import Path

# Project root (model_testing/ -> repo root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

#159684

def start_vllm_server(
    model_path: str,
    cuda_device: int,
    port: int,
    log_file: str,
    max_num_batched_tokens: int = 40960,
    max_model_len: int = 20000,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0
):
    """
    Launch a VLLM API server for the PyGeoX RL model.

    Args:
        model_path (str): Path to the HuggingFace model directory
        cuda_device (int): CUDA device number to run on
        port (int): Port number for the API server
        log_file (str): Log file path
        max_num_batched_tokens (int): Max tokens processed in parallel. Default: 40960
        max_model_len (int): Maximum context length. Default: 20000
        temperature (float): Sampling temperature. Default: 0.7
        top_p (float): Nucleus sampling parameter. Default: 0.8
        top_k (int): Top-k sampling parameter. Default: 20
        min_p (float): Minimum probability threshold. Default: 0.0
    """
    print("=" * 80)
    print("🚀 Starting PyGeoX RL Model vLLM Server")
    print("=" * 80)
    print(f"📦 Model: {model_path}")
    print(f"🎮 GPU: {cuda_device}")
    print(f"🌐 Port: {port}")
    print(f"📊 Max Context Length: {max_model_len:,} tokens")
    print(f"📝 Log File: {log_file}")
    print("=" * 80)

    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    # Remove VLLM_USE_V1 setting - let vLLM use its default
    # env["VLLM_USE_V1"] = "0"  # Commented out - causes issues with newer vLLM
    env["VLLM_TORCH_ATTN_BACKEND"] = "flash_attn"

    # Build generation config
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    if min_p > 0:
        generation_config["min_p"] = min_p

    generation_config_str = json.dumps(generation_config)

    # Construct vLLM command
    command = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--tensor-parallel-size",
        "1",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        "0.9",
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--max-model-len",
        str(max_model_len),
        "--override-generation-config",
        generation_config_str,
        "--trust-remote-code",
    ]

    print(f"\n🔧 vLLM Configuration:")
    print(f"   - Tensor Parallel Size: 1")
    print(f"   - GPU Memory Utilization: 0.9")
    print(f"   - Max Batched Tokens: {max_num_batched_tokens:,}")
    print(f"   - Max Model Length: {max_model_len:,}")
    print(f"   - Temperature: {temperature}")
    print(f"   - Top-P: {top_p}")
    print(f"   - Top-K: {top_k}")
    if min_p > 0:
        print(f"   - Min-P: {min_p}")
    print()

    try:
        # Launch vLLM server
        print(f"🚀 Launching vLLM server...")
        print(f"💡 To test the server, run:")
        print(f"   curl http://localhost:{port}/v1/models")
        print()
        print(f"📖 API documentation available at:")
        print(f"   http://localhost:{port}/docs")
        print()
        print("⏳ Server is starting (this may take a minute)...")
        print(f"📋 Watch logs with: tail -f {log_file}")
        print()

        with open(log_file, "w") as f:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
        
        print("=" * 80)
        print(f"✅ vLLM server started successfully!")
        print(f"   - Process ID: {process.pid}")
        print(f"   - Endpoint: http://localhost:{port}")
        print(f"   - Logs: {log_file}")
        print("=" * 80)
        print()
        print("🛑 To stop the server, run:")
        print(f"   kill {process.pid}")
        print()

    except FileNotFoundError:
        print(
            "❌ Error: Python executable or vLLM module not found.",
            file=sys.stderr,
        )
        print("💡 Make sure vLLM is installed: pip install vllm", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Serve PyGeoX RL model with vLLM (max 20k tokens)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(_PROJECT_ROOT / "model_training/RL_training"),
        help="Path to the HuggingFace model directory",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=7,
        help="GPU device ID to use (default: 0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8893,
        help="Port number for the API server (default: 8890)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="vllm_rl_model.log",
        help="Log file path (default: vllm_rl_model.log)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=35000,
        help="Maximum context length in tokens (default: 20000)",
    )
    parser.add_argument(
        "--max-batched-tokens",
        type=int,
        default=40960,
        help="Maximum batched tokens (default: 40960)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Nucleus sampling top-p (default: 0.8)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling (default: 20)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Minimum probability threshold (default: 0.0)",
    )

    args = parser.parse_args()

    # Verify model path exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model path does not exist: {args.model_path}", file=sys.stderr)
        print(f"\n💡 Did you convert the checkpoint first?", file=sys.stderr)
        print(f"   Run: python {_PROJECT_ROOT / 'model_training/RL_training/convert_to_hf.py'}", file=sys.stderr)
        sys.exit(1)

    # Start the server
    start_vllm_server(
        model_path=args.model_path,
        cuda_device=args.gpu,
        port=args.port,
        log_file=args.log_file,
        max_num_batched_tokens=args.max_batched_tokens,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
    )


if __name__ == "__main__":
    main()
