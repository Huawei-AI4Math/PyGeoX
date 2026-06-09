#!/usr/bin/env python3
"""
Test one model on PyGeoX benchmark and compute rewards.
Includes retry logic with validation and feedback.
"""
import os
import sys
import json
import time
import re
import requests
import threading
import pandas as pd
import subprocess
import math
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Project root and paths (benchmark in data/PyGeoX benchmark)
# Repo root: script lives in model_testing/PygeoX-Bench/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "model_training"))
from pygeox.synthetic.llm_client import create_scene_from_json
import contextlib
from io import StringIO

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================
BENCH_DIR = Path(__file__).resolve().parent  # model_testing/PygeoX-Bench
OPENROUTER_API_KEY = os.getenv("API_KEY")
BENCHMARK_FOLDER = PROJECT_ROOT / "data" / "PyGeoX benchmark" / "json_fixed"
RESULTS_BASE_NEW = BENCH_DIR / "generations"  # output folder for generated JSONs
RESULTS_BASE_NEW.mkdir(parents=True, exist_ok=True)
EXCELLS_DIR = BENCH_DIR / "excells"  # where to save results_*.csv and metrics_*.csv
EXCELLS_DIR.mkdir(parents=True, exist_ok=True)
MAX_WORKERS = 30  # Single parameter: workers handle both generation and reward computation

# Model to test
MODEL_PATH = f"{PROJECT_ROOT}/model_training/merged/qwen3-8b-SFT-Full"
GPU_ID = 3
PORT = 8891
LOG_FILE = "vllm_gpu3.out"
MODEL_TRIES = 1

# Extract model name for file naming
MODEL_NAME = Path(MODEL_PATH).name  # e.g., "qwen3-1.7b-sparse-900"

# API base URL
BASE_API = f"http://localhost:{PORT}/v1"

# ============================================================================
# Load System Prompt
# ============================================================================
system_prompt_rl_path = PROJECT_ROOT / "model_training/system_prompt_rl.md"
with open(system_prompt_rl_path, "r") as f:
    SYSTEM_PROMPT = f.read()

print("=" * 80)
print("PyGeoX Benchmark - Single Model Testing")
print("=" * 80)
print(f"Model: {MODEL_PATH}")
print(f"Model name: {MODEL_NAME}")
print(f"GPU: {GPU_ID}")
print(f"Port: {PORT}")
print(f"Results folder: {RESULTS_BASE_NEW}")
print(f"Benchmark folder: {BENCHMARK_FOLDER}")
print("=" * 80)

print_lock = threading.Lock()

# ============================================================================
# Server Startup Function
# ============================================================================

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
    
    Returns:
        subprocess.Popen: The server process
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
    
    # Verify model path exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model path does not exist: {model_path}", file=sys.stderr)
        sys.exit(1)
    
    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
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
        print(f"⏳ Server is starting (this may take a minute)...")
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
        
        # Wait a bit for server to start
        time.sleep(5)
        
        # Check if server is responding
        max_wait = 120  # Wait up to 2 minutes
        wait_interval = 5
        waited = 0
        while waited < max_wait:
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=2)
                if response.status_code == 200:
                    print("=" * 80)
                    print(f"✅ vLLM server started successfully!")
                    print(f"   - Process ID: {process.pid}")
                    print(f"   - Endpoint: http://localhost:{port}")
                    print(f"   - Logs: {log_file}")
                    print("=" * 80)
                    return process
            except:
                pass
            time.sleep(wait_interval)
            waited += wait_interval
            if waited % 15 == 0:
                print(f"   Still waiting for server to start... ({waited}s)")
        
        print(f"⚠️  Server may not be fully ready, but continuing...")
        return process
        
    except FileNotFoundError:
        print("❌ Error: Python executable or vLLM module not found.", file=sys.stderr)
        print("💡 Make sure vLLM is installed: pip install vllm", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

# ============================================================================
# Validation Functions
# ============================================================================

def check_server_health(base_api, timeout=5):
    """Check if the vLLM server is healthy and responding."""
    try:
        response = requests.get(f"{base_api.replace('/v1', '')}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False

def make_api_call_with_retry(base_api, payload, max_retries=3, timeout=600):
    """
    Make API call with retry logic and health checks.
    Returns (response_json, success) or (None, False) if all retries fail.
    """
    for retry in range(max_retries):
        try:
            # Check server health before making request
            if not check_server_health(base_api, timeout=3):
                if retry < max_retries - 1:
                    time.sleep(2 ** retry)  # Exponential backoff
                    continue
                else:
                    return None, False
            
            # Make the API call
            response = requests.post(
                base_api + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json(), True
            
        except requests.exceptions.Timeout:
            if retry < max_retries - 1:
                time.sleep(2 ** retry)  # Exponential backoff
                continue
            else:
                return None, False
        except requests.exceptions.RequestException as e:
            if retry < max_retries - 1:
                time.sleep(2 ** retry)  # Exponential backoff
                continue
            else:
                return None, False
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(2 ** retry)
                continue
            else:
                return None, False
    
    return None, False

def extract_code_from_completion(completion):
    """Extract Python code from completion."""
    # Match code blocks
    match = re.search(r"```(?:python|py)(.*?)```", completion, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()

def truncate_conversation_for_retry(conversation_history):
    """
    Truncate conversation history to prevent token overflow.
    Keeps system prompt, initial user prompt, and only the code block from previous attempts.
    Removes long <think> blocks to save tokens.
    """
    if len(conversation_history) <= 2:
        # Only system + initial user prompt, nothing to truncate
        return conversation_history
    
    # Keep system and initial user prompt
    truncated = [conversation_history[0], conversation_history[1]]
    
    # For previous assistant responses, extract only the code block (remove long think blocks)
    for msg in conversation_history[2:]:
        if msg["role"] == "assistant":
            # Extract code block if present
            code = extract_code_from_completion(msg["content"])
            if code:
                # Create a minimal assistant response with just the code
                truncated.append({
                    "role": "assistant",
                    "content": f"```python\n{code}\n```"
                })
            # If no code block, skip this message entirely
        elif msg["role"] == "user":
            # Keep user feedback messages (they're short)
            truncated.append(msg)
    
    return truncated

def validate_code_execution(code):
    """
    Validate that code executes without errors and returns points and circles.
    Also converts NumPy types to Python native types (matching reward function behavior).
    
    Returns:
        tuple: (is_valid, points, circles, error_message)
    """
    if not code:
        return False, None, None, "No code provided"
    
    execution_context = {}
    try:
        exec(code, execution_context, execution_context)
    except Exception as e:
        return False, None, None, f"Code execution failed: {str(e)}"
    
    points_raw = execution_context.get('points', {})
    circles_raw = execution_context.get('circles', {})
    
    if not isinstance(points_raw, dict):
        return False, None, None, f"points must be a dictionary, got {type(points_raw).__name__}"
    
    if not isinstance(circles_raw, dict):
        return False, None, None, f"circles must be a dictionary, got {type(circles_raw).__name__}"
    
    # CRITICAL: Convert NumPy types to Python native types (same as reward function does)
    # This ensures that code passing our validation will also pass reward function validation
    try:
        points = {}
        for k, v in points_raw.items():
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                # Convert each coordinate to Python float (handles numpy.float64, etc.)
                try:
                    points[k] = [float(v[0]), float(v[1])]
                except (ValueError, TypeError) as e:
                    return False, None, None, f"Point '{k}' coordinate conversion failed: {str(e)}"
            else:
                return False, None, None, f"Point '{k}' must be a list/tuple of 2 coordinates, got {type(v).__name__}"
        
        circles = {}
        for k, v in circles_raw.items():
            # Convert to Python float (handles numpy.float64, etc.)
            try:
                circles[k] = float(v)
            except (ValueError, TypeError) as e:
                return False, None, None, f"Circle '{k}' radius conversion failed: {str(e)}"
        
        # Verify no NaN or inf values (scene validation might reject these)
        import math
        for k, coords in points.items():
            if any(math.isnan(c) or math.isinf(c) for c in coords):
                return False, None, None, f"Point '{k}' has NaN or inf coordinates"
        for k, radius in circles.items():
            if math.isnan(radius) or math.isinf(radius):
                return False, None, None, f"Circle '{k}' has NaN or inf radius"
        
    except Exception as e:
        return False, None, None, f"Type conversion failed: {str(e)}"
    
    return True, points, circles, None

def check_points_circles_format(points, circles, expected_keys):
    """
    Check if points and circles match expected format.
    
    Returns:
        tuple: (is_valid, missing_points, missing_circles, extra_points, extra_circles, error_message)
    """
    expected_points = set(expected_keys["points"].keys())
    expected_circles = set(expected_keys["circles"].keys())
    
    actual_points = set(points.keys())
    actual_circles = set(circles.keys())
    
    missing_points = expected_points - actual_points
    missing_circles = expected_circles - actual_circles
    extra_points = actual_points - expected_points
    extra_circles = actual_circles - expected_circles
    
    # Check point format
    for point_name, point_value in points.items():
        if not isinstance(point_value, list) or len(point_value) != 2:
            return False, missing_points, missing_circles, extra_points, extra_circles, \
                   f"Point '{point_name}' must be a list of 2 coordinates [x, y], got {point_value}"
    
    # Check circle format
    for circle_name, circle_value in circles.items():
        if not isinstance(circle_value, (int, float)):
            return False, missing_points, missing_circles, extra_points, extra_circles, \
                   f"Circle '{circle_name}' must be a number (radius), got {type(circle_value).__name__}"
    
    # Build error message first
    error_parts = []
    if missing_points:
        error_parts.append(f"Missing points: {', '.join(sorted(missing_points))}")
    if missing_circles:
        error_parts.append(f"Missing circles: {', '.join(sorted(missing_circles))}")
    if extra_points:
        error_parts.append(f"Extra points (not expected): {', '.join(sorted(extra_points))}")
    if extra_circles:
        error_parts.append(f"Extra circles (not expected): {', '.join(sorted(extra_circles))}")
    
    error_message = "; ".join(error_parts) if error_parts else None
    
    # CRITICAL: is_valid must check BOTH missing AND extra points/circles
    # The format is only valid if there are no missing AND no extra points/circles
    is_valid = len(missing_points) == 0 and len(missing_circles) == 0 and \
               len(extra_points) == 0 and len(extra_circles) == 0
    
    return is_valid, missing_points, missing_circles, extra_points, extra_circles, error_message

# ============================================================================
# Generation Functions with Retry Logic
# ============================================================================

def generate_with_retry(file_path, model_id, base_api, max_retries=MODEL_TRIES):
    """
    Generate completion with retry logic and validation.
    
    Returns:
        dict: Generation data with completion and metadata
    """
    # Load problem
    with open(file_path, "r") as f:
        problem = json.load(f)
    
    diagram_description = problem["nl_description"]
    
    expected_keys = {
        "points": {p: [None, None] for p in problem["possible_solution"]["points"].keys()},
        "circles": {c: None for c in problem["possible_solution"]["circles"].keys()}
    }
    
    extra_instructions = """
### **CRITICAL: Mandatory Output Variables**
Your Python code **must** define the following two variables at the end of the script for the solution to be valid. These allow the environment to extract and visualize your geometric construction:

1. **`points`**: A dictionary where keys are the **string labels** of points and values are **lists of exactly two coordinates** `[x, y]`.
2. **`circles`**: A dictionary where keys are the **string labels of the center point** and values are **floats representing the radius**.

**Strict Requirement:** Even if your construction involves no circles, you **must** define `circles = {}` to prevent extraction failure.

**Example Format:**
```python
 ... logic to calculate coordinates ...
points = {
    "A": [0.0, 5.0],
    "B": [-2.0, 0.0],
    "C": [2.0, 0.0]
}
circles = {
    "A": 3.5  # Radius of a circle centered at point A
}
```
"""
    
    # Initial user prompt
    user_prompt = f"""Find the coordinates and circle radiuses for:
{diagram_description}

{extra_instructions}

The correct format for `points` and `circles` is as follows (replace the None values with the correct coordinates and circle radiuses):
{json.dumps(expected_keys, indent=2)}
"""
    
    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    total_tokens = 0
    total_time = 0.0
    attempt_details = []  # Track details for each attempt
    
    for attempt in range(max_retries):
        try:
            # Make API call with retry logic
            start_time = time.time()
            payload = {
                "model": model_id,
                "messages": conversation_history,
                "max_tokens": 8196
            }
            
            result, success = make_api_call_with_retry(
                base_api, 
                payload, 
                max_retries=3, 
                timeout=600
            )
            
            if not success:
                error_msg = "API call failed after retries - server may be down or unresponsive"
                attempt_info = {
                    "attempt_number": attempt + 1,
                    "error": error_msg,
                    "elapsed_time": time.time() - start_time,
                    "tokens": 0,
                    "validation_steps": [{
                        "step": "api_call",
                        "status": "failed",
                        "error": error_msg
                    }]
                }
                attempt_details.append(attempt_info)
                
                if attempt < max_retries - 1:
                    # Truncate conversation and retry
                    conversation_history = truncate_conversation_for_retry(conversation_history)
                    conversation_history.append({
                        "role": "user",
                        "content": "The previous request failed. Please provide your solution again."
                    })
                    time.sleep(2)  # Brief pause before retry
                    continue
                else:
                    return {
                        "file": str(file_path),
                        "model": model_id,
                        "error": error_msg,
                        "user_prompt": user_prompt,
                        "system_prompt": SYSTEM_PROMPT,
                        "status": "failed",
                        "generated_tokens": total_tokens,
                        "elapsed_time": total_time,
                        "attempts": attempt + 1,
                        "conversation_history": conversation_history,
                        "attempt_details": attempt_details
                    }
            
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            # Extract completion and token usage
            completion = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            generated_tokens = usage.get("completion_tokens", 0)
            total_tokens += generated_tokens
            
            # Add assistant response to conversation
            conversation_history.append({"role": "assistant", "content": completion})
            
            # Track attempt details
            attempt_info = {
                "attempt_number": attempt + 1,
                "completion": completion,
                "tokens": generated_tokens,
                "elapsed_time": elapsed_time,
                "validation_steps": []
            }
            
            # Step 1: Add <think> tags if not present
            if "<think>" not in completion and "</think>" not in completion:
                completion = "<think> </think>" + completion
                attempt_info["validation_steps"].append({
                    "step": "think_tags",
                    "status": "added",
                    "note": "Added missing <think> tags"
                })
            else:
                attempt_info["validation_steps"].append({
                    "step": "think_tags",
                    "status": "ok",
                    "note": "Think tags already present"
                })
            
            # Step 2: Extract and validate code
            code = extract_code_from_completion(completion)
            
            if not code:
                feedback = "Your response does not contain a Python code block. Please provide your solution in a code block formatted as ```python ... ```"
                attempt_info["validation_steps"].append({
                    "step": "code_extraction",
                    "status": "failed",
                    "error": "No code block found"
                })
                attempt_details.append(attempt_info)
                
                if attempt < max_retries - 1:
                    # Truncate conversation to prevent token overflow (remove long think blocks)
                    conversation_history = truncate_conversation_for_retry(conversation_history)
                    conversation_history.append({
                        "role": "user",
                        "content": f"ERROR: {feedback}\n\nPlease fix this and provide a corrected response with a Python code block."
                    })
                    continue
                else:
                    return {
                        "file": str(file_path),
                        "model": model_id,
                        "completion": completion,
                        "user_prompt": user_prompt,
                        "system_prompt": SYSTEM_PROMPT,
                        "status": "failed",
                        "error": "No code block found after max retries",
                        "generated_tokens": total_tokens,
                        "elapsed_time": total_time,
                        "attempts": attempt + 1,
                        "conversation_history": conversation_history,
                        "attempt_details": attempt_details
                    }
            
            attempt_info["validation_steps"].append({
                "step": "code_extraction",
                "status": "ok",
                "note": "Code block extracted successfully"
            })
            
            # Step 3: Validate code execution
            is_valid, points, circles, exec_error = validate_code_execution(code)
            
            if not is_valid:
                feedback = f"Your Python code has an error: {exec_error}"
                attempt_info["validation_steps"].append({
                    "step": "code_execution",
                    "status": "failed",
                    "error": exec_error
                })
                attempt_details.append(attempt_info)
                
                if attempt < max_retries - 1:
                    # Truncate conversation to prevent token overflow (remove long think blocks)
                    conversation_history = truncate_conversation_for_retry(conversation_history)
                    conversation_history.append({
                        "role": "user",
                        "content": f"ERROR: {feedback}\n\nPlease fix the code and provide a corrected version. Make sure the code executes without errors and defines both 'points' and 'circles' dictionaries."
                    })
                    continue
                else:
                    return {
                        "file": str(file_path),
                        "model": model_id,
                        "completion": completion,
                        "user_prompt": user_prompt,
                        "system_prompt": SYSTEM_PROMPT,
                        "status": "failed",
                        "error": f"Code execution failed: {exec_error}",
                        "generated_tokens": total_tokens,
                        "elapsed_time": total_time,
                        "attempts": attempt + 1,
                        "conversation_history": conversation_history,
                        "attempt_details": attempt_details
                    }
            
            attempt_info["validation_steps"].append({
                "step": "code_execution",
                "status": "ok",
                "note": "Code executed successfully",
                "points_found": list(points.keys()),
                "circles_found": list(circles.keys())
            })
            
            # Step 4: Validate points and circles format
            is_valid_format, missing_points, missing_circles, extra_points, extra_circles, format_error = \
                check_points_circles_format(points, circles, expected_keys)
            
            if not is_valid_format:
                feedback_parts = []
                if missing_points:
                    feedback_parts.append(f"Missing required points: {', '.join(sorted(missing_points))}")
                if missing_circles:
                    feedback_parts.append(f"Missing required circles: {', '.join(sorted(missing_circles))}")
                if extra_points:
                    feedback_parts.append(f"Extra points (not needed): {', '.join(sorted(extra_points))}")
                if extra_circles:
                    feedback_parts.append(f"Extra circles (not needed): {', '.join(sorted(extra_circles))}")
                if format_error and "must be" in format_error:
                    feedback_parts.append(format_error)
                
                feedback = "; ".join(feedback_parts)
                attempt_info["validation_steps"].append({
                    "step": "format_validation",
                    "status": "failed",
                    "error": feedback,
                    "missing_points": list(missing_points),
                    "missing_circles": list(missing_circles),
                    "extra_points": list(extra_points),
                    "extra_circles": list(extra_circles)
                })
                attempt_details.append(attempt_info)
                
                if attempt < max_retries - 1:
                    # Truncate conversation to prevent token overflow (remove long think blocks)
                    conversation_history = truncate_conversation_for_retry(conversation_history)
                    conversation_history.append({
                        "role": "user",
                        "content": f"ERROR: {feedback}\n\nThe expected format is:\n{json.dumps(expected_keys, indent=2)}\n\nPlease fix your code to include all required points and circles with the correct format."
                    })
                    continue
                else:
                    return {
                        "file": str(file_path),
                        "model": model_id,
                        "completion": completion,
                        "user_prompt": user_prompt,
                        "system_prompt": SYSTEM_PROMPT,
                        "status": "failed",
                        "error": f"Format validation failed: {feedback}",
                        "generated_tokens": total_tokens,
                        "elapsed_time": total_time,
                        "attempts": attempt + 1,
                        "conversation_history": conversation_history,
                        "attempt_details": attempt_details
                    }
            
            attempt_info["validation_steps"].append({
                "step": "format_validation",
                "status": "ok",
                "note": "All points and circles validated successfully"
            })
            attempt_details.append(attempt_info)
            
            # Success!
            return {
                "file": str(file_path),
                "model": model_id,
                "completion": completion,
                "user_prompt": user_prompt,
                "system_prompt": SYSTEM_PROMPT,
                "status": "success",
                "generated_tokens": total_tokens,
                "elapsed_time": total_time,
                "attempts": attempt + 1,
                "conversation_history": conversation_history,
                "attempt_details": attempt_details
            }
            
        except Exception as e:
            attempt_info = {
                "attempt_number": attempt + 1,
                "error": str(e),
                "elapsed_time": 0.0,
                "tokens": 0,
                "validation_steps": [{
                    "step": "api_call",
                    "status": "failed",
                    "error": str(e)
                }]
            }
            attempt_details.append(attempt_info)
            
            if attempt < max_retries - 1:
                # Truncate conversation to prevent token overflow (remove long think blocks)
                conversation_history = truncate_conversation_for_retry(conversation_history)
                conversation_history.append({
                    "role": "user",
                    "content": f"An error occurred: {str(e)}\n\nPlease try again with a corrected response."
                })
                continue
            else:
                return {
                    "file": str(file_path),
                    "model": model_id,
                    "error": str(e),
                    "user_prompt": user_prompt,
                    "system_prompt": SYSTEM_PROMPT,
                    "status": "failed",
                    "generated_tokens": total_tokens,
                    "elapsed_time": total_time,
                    "attempts": attempt + 1,
                    "conversation_history": conversation_history,
                    "attempt_details": attempt_details
                }
    
    # Should not reach here, but just in case
    return {
        "file": str(file_path),
        "model": model_id,
        "error": "Max retries exceeded",
        "status": "failed",
        "generated_tokens": total_tokens,
        "elapsed_time": total_time,
        "attempts": max_retries,
        "conversation_history": conversation_history,
        "attempt_details": attempt_details
    }

def process_file_complete(file_path, model_id):
    """
    Process a single file: generate + compute reward in one go.
    Returns reward result dict or None.
    """
    try:
        # Create model-specific folder path
        safe_model_name = model_id.replace("/", "_").replace(":", "_")
        model_folder = RESULTS_BASE_NEW / safe_model_name
        model_folder.mkdir(parents=True, exist_ok=True)
        
        result_file = model_folder / f"{file_path.stem}.json"
        
        # Check if already processed - skip if result file exists with success status
        # (We'll recompute rewards if needed, but skip generation if already done)
        if result_file.exists():
            try:
                with open(result_file, "r") as f:
                    existing_data = json.load(f)
                if existing_data.get("status") == "success":
                    # Generation already done, skip to avoid duplicate work
                    # Reward will be computed separately if needed
                    with print_lock:
                        print(f"⏭️  Skipping {file_path.name} (already generated)...")
                    return None
            except:
                pass
        
        with print_lock:
            print(f"Processing {file_path.name} with {model_id}...")
        
        # Step 1: Generate with retry logic
        generation_data = generate_with_retry(file_path, model_id, BASE_API, max_retries=3)
        
        # Save generation data
        with open(result_file, "w") as f:
            json.dump(generation_data, f, indent=2)
        
        status_emoji = "✅" if generation_data.get("status") == "success" else "❌"
        attempts = generation_data.get("attempts", 1)
        with print_lock:
            print(f"{status_emoji} Generated {file_path.name} (attempts: {attempts})")
        
        # Step 2: Compute reward if generation was successful
        if generation_data.get("status") != "success":
            return None
        
        # Get the problem file path
        problem_file_path = generation_data.get("file")
        if isinstance(problem_file_path, Path):
            problem_file_path = str(problem_file_path)
        elif not isinstance(problem_file_path, str):
            return None
        
        # Get problem file name for CSV
        problem_file_name = Path(problem_file_path).name
        
        # Get completion text
        completion = generation_data.get("completion", "")
        if not completion:
            return None
        
        # Get token and time statistics
        generated_tokens = generation_data.get("generated_tokens", 0)
        elapsed_time = generation_data.get("elapsed_time", 0.0)
        
        # Handle tokens (could be int or dict)
        if isinstance(generated_tokens, dict):
            tokens = generated_tokens.get("output_tokens", 0) if "output_tokens" in generated_tokens else generated_tokens.get("total", 0)
        elif isinstance(generated_tokens, (int, float)):
            tokens = int(generated_tokens)
        else:
            tokens = 0
        
        # Handle time
        if isinstance(elapsed_time, (int, float)):
            time_sum = float(elapsed_time)
        else:
            time_sum = 0.0
        
        # Compute reward using scene.reward.reward_function directly
        try:
            # Get the problem file
            problem_file = BENCHMARK_FOLDER / problem_file_name
            if not problem_file.exists():
                return None
            
            # Load problem data
            with open(problem_file, "r") as f:
                problem_data = json.load(f)
            
            # Extract and execute code to get points and circles
            code = extract_code_from_completion(completion)
            if not code:
                # No code block found
                return {
                    "model_id": model_id,
                    "problem_file_name": problem_file_name,
                    "generated_tokens": tokens,
                    "elapsed_time": time_sum,
                    "total_reward": 0.0,
                    "reward_dict": {"error": "No code block found", "R_format": 0.0, "R_pygeox": 0.0}
                }
            
            # Execute code to get points and circles
            is_valid, points, circles, error_msg = validate_code_execution(code)
            if not is_valid:
                # Code execution failed or format invalid
                return {
                    "model_id": model_id,
                    "problem_file_name": problem_file_name,
                    "generated_tokens": tokens,
                    "elapsed_time": time_sum,
                    "total_reward": 0.0,
                    "reward_dict": {"error": error_msg, "R_format": 0.0, "R_pygeox": 0.0}
                }
            
            # Create scene and compute reward
            scene = None
            try:
                # Suppress stdout/stderr during scene creation and reward computation
                with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                    # Create scene from problem JSON
                    scene = create_scene_from_json(
                        domain=10,  # DOMAIN from reward_func_openrlhf
                        json_data=problem_data,
                        generate_objective_function=True,
                        distance_penalty=1,  # DISTANCE_PENALTY
                        min_dist=0.02  # MIN_DIST
                    )
                    
                    # Compute reward using scene.reward.reward_function
                    r_pygeox, reward_dict = scene.reward.reward_function(points, circles)
                    
                    # Check for invalid values
                    if not isinstance(r_pygeox, (int, float)) or (isinstance(r_pygeox, float) and (math.isnan(r_pygeox) or math.isinf(r_pygeox))):
                        r_pygeox = 0.0
                        reward_dict["error"] = "R_pygeox is NaN or inf"
                    
                    # Format is valid (we already validated it)
                    reward_dict["R_format"] = 1.0
                    reward_dict["R_pygeox"] = float(r_pygeox)
                    total_reward = float(r_pygeox)
                    
            except Exception as e:
                # Error during scene creation or reward computation
                total_reward = 0.0
                reward_dict = {
                    "error": f"Scene/reward computation error: {str(e)}",
                    "R_format": 1.0 if is_valid else 0.0,
                    "R_pygeox": 0.0
                }
            finally:
                # Clean up scene
                if scene is not None:
                    try:
                        if hasattr(scene, 'reward') and hasattr(scene.reward, '_scene'):
                            scene.reward._scene = None
                        del scene.reward
                    except:
                        pass
                    try:
                        if hasattr(scene, 'solver'):
                            del scene.solver
                    except:
                        pass
                    del scene
            
            # Return result dict with both reward and reward_dict
            with print_lock:
                print(f"  ✓ Reward computed for {file_path.name}: {total_reward:.2f}")
            
            return {
                "model_id": model_id,
                "problem_file_name": problem_file_name,
                "generated_tokens": tokens,
                "elapsed_time": time_sum,
                "total_reward": total_reward,
                "reward_dict": reward_dict
            }
            
        except Exception as e:
            # If reward computation fails, log the error and return with 0 reward
            with print_lock:
                print(f"⚠️  Reward computation error for {problem_file_name}: {str(e)}")
            return {
                "model_id": model_id,
                "problem_file_name": problem_file_name,
                "generated_tokens": tokens,
                "elapsed_time": time_sum,
                "total_reward": 0.0,
                "reward_dict": {"error": str(e), "R_format": 0.0, "R_pygeox": 0.0}
            }
    
    except Exception as e:
        with print_lock:
            print(f"❌ Error processing {file_path.name}: {str(e)}")
        return None

# ============================================================================
# Main Execution
# ============================================================================

# Start the vLLM server
print("\n[0/2] Starting vLLM Server")
print("-" * 80)
server_process = start_vllm_server(
    model_path=MODEL_PATH,
    cuda_device=GPU_ID,
    port=PORT,
    log_file=LOG_FILE
)

# ============================================================================
# Combined Generation + Reward Computation Phase
# ============================================================================
print(f"\n[1/2] Generation + Reward Computation Phase")
print("-" * 80)

# Collect all tasks for this model
all_tasks = []
safe_model_name = MODEL_PATH.replace("/", "_").replace(":", "_")
model_folder = RESULTS_BASE_NEW / safe_model_name
model_folder.mkdir(parents=True, exist_ok=True)

for file_path in BENCHMARK_FOLDER.glob("*.json"):
    result_file = model_folder / f"{file_path.stem}.json"
    # Only add files that haven't been generated yet
    if not result_file.exists():
        all_tasks.append(file_path)

total_files = len(list(BENCHMARK_FOLDER.glob("*.json")))
already_generated = total_files - len(all_tasks)
print(f"Total problems: {total_files}")
print(f"Already generated: {already_generated}")
print(f"To process: {len(all_tasks)}")
if len(all_tasks) > 0:
    print(f"Using {min(MAX_WORKERS, len(all_tasks))} workers")
    print("Each worker will: generate → compute reward → return result")
else:
    print("✅ All files already generated!")

# Collect all data
all_rows = []

if len(all_tasks) > 0:
    # Process files: each worker does generation + reward computation
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file_complete, file_path, MODEL_PATH): file_path 
                   for file_path in all_tasks}
        
        completed = 0
        start_time = time.time()
        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result(timeout=900)  # 15 min timeout per file
                if result is not None:
                    all_rows.append(result)
            except Exception as e:
                with print_lock:
                    print(f"❌ Task failed with exception: {str(e)}")
            
            # Progress update
            if completed % 5 == 0 or completed == len(all_tasks):
                elapsed = time.time() - start_time
                with print_lock:
                    elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}m"
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (len(all_tasks) - completed) / rate if rate > 0 and completed > 0 else 0
                    remaining_str = f"{remaining:.1f}s" if remaining < 60 else f"{remaining/60:.1f}m"
                    print(f"Progress: {completed}/{len(all_tasks)} files ({elapsed_str} elapsed, ~{remaining_str} remaining, {rate:.2f} files/s)")
    
    print(f"✅ Generation + reward computation complete!")
else:
    print("No new files to process.")

# Create DataFrame and save with model name
if all_rows:
    df = pd.DataFrame(all_rows)
    output_csv = EXCELLS_DIR / f"results_{MODEL_NAME}.csv"
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Saved {len(all_rows)} results to {output_csv}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
else:
    print("⚠️  No results to save!")
    df = pd.DataFrame()

# ============================================================================
# Metrics Computation
# ============================================================================
print(f"\n[2/2] Computing Metrics")
print("-" * 80)

data = df.copy()

# Extract difficulty from problem_file_name (first character: 1, 2, or 3)
data["difficulty"] =  data["problem_file_name"].str[0]

# Expected number of problems per difficulty (assuming 100 per difficulty)
EXPECTED_PER_DIFFICULTY = 100

# Calculate metrics per model_id per difficulty
metrics_list = []

for model_id in data["model_id"].unique():
    for difficulty in data["difficulty"].unique():
        # Filter data for this model and difficulty
        subset = data[(data["model_id"] == model_id) & (data["difficulty"] == difficulty) ] #& (data["generated_tokens"] < 3950)
        
        if len(subset) == 0:
            continue
        
        # Count problems with different reward ranges
        total_count = len(subset)
        perfectly_correct_count = (subset["total_reward"] >= 9).sum()
        reward_gt_zero_count = (subset["total_reward"] > 0).sum()
        wrong_format_count = (subset["total_reward"] <= 0).sum()
        
        # Calculate missing files: expected - actual files with reward > -10
        missing_files = EXPECTED_PER_DIFFICULTY - total_count
        
        # Calculate metrics
        perfectly_correct = perfectly_correct_count / EXPECTED_PER_DIFFICULTY
        
        if reward_gt_zero_count > 0:
            average_reward = (subset["total_reward"].sum()) / EXPECTED_PER_DIFFICULTY
        else:
            average_reward = 0.0
        
        wrong_format = (wrong_format_count + missing_files) / EXPECTED_PER_DIFFICULTY
        
        #average_time = subset["elapsed_time"].mean()
        
        metrics_list.append({
            "model_id": model_id,
            "difficulty": difficulty,
            "perfectly_correct": perfectly_correct,
            "average_reward": average_reward,
            "wrong_format": wrong_format,
            #"average_time": average_time,
            "total_count": total_count,
            "missing_files": missing_files
        })

# Create metrics DataFrame
metrics_df = pd.DataFrame(metrics_list)

# Display results
print("Metrics per Model per Difficulty:")
print("=" * 80)
print(metrics_df.to_string(index=False))
print("\n")

# Display pivot tables
print("\nPerfectly Correct Rate (total_reward >= 9):")
print(metrics_df.pivot(index="model_id", columns="difficulty", values="perfectly_correct").to_string())
print()

print("Average Reward:")
print(metrics_df.pivot(index="model_id", columns="difficulty", values="average_reward").to_string())
print()

print("Wrong Format Rate (total_reward <= 0 + missing files):")
print(metrics_df.pivot(index="model_id", columns="difficulty", values="wrong_format").to_string())
print()

# Save metrics to CSV
metrics_csv = EXCELLS_DIR / f"metrics_{MODEL_NAME}.csv"
metrics_df.to_csv(metrics_csv, index=False)
print(f"✅ Saved metrics to {metrics_csv}")

print("\n" + "=" * 80)
print("Done!")
print("=" * 80)
print(f"\n🛑 To stop the server, run:")
print(f"   kill {server_process.pid}")
