#!/usr/bin/env python3
"""
Test one model on PyGeoX benchmark and compute rewards.
"""
import os
import sys
import json
import time
import requests
import threading
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Project root and paths (benchmark in data/PyGeoX benchmark)
# Repo root: script lives in model_testing/PygeoX-Bench/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "model_training"))
from reward_func_openrlhf import llm_reward_function_new

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
MAX_WORKERS = 20
MAX_WORKERS_REWARD = 1  # Single worker for reward computation to avoid thread issues

# Model to test
BASE_API = "http://localhost:8890/v1"
MODEL = f"{PROJECT_ROOT}/model_training/merged/qwen3-8b-SFT-Full" 
# Extract model name for file naming
MODEL_NAME = Path(MODEL).name  # e.g., "qwen3-sft-dense"

# ============================================================================
# Load System Prompt
# ============================================================================
system_prompt_rl_path = PROJECT_ROOT / "model_training/system_prompt_rl.md"
with open(system_prompt_rl_path, "r") as f:
    SYSTEM_PROMPT = f.read()

print("=" * 80)
print("PyGeoX Benchmark - Single Model Testing")
print("=" * 80)
print(f"Model: {MODEL}")
print(f"Model name: {MODEL_NAME}")
print(f"Results folder: {RESULTS_BASE_NEW}")
print(f"Benchmark folder: {BENCHMARK_FOLDER}")
print("=" * 80)

print_lock = threading.Lock()

# ============================================================================
# Generation Functions
# ============================================================================

def generate_for_file(file_path, model_id):
    """Process a single file-model combination using direct API calls."""
    try:
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

**Strict Requirement:** Even if your construction involves no circles, you **must** define `circles = {}` to prevent extraction failure."""
#**Example Format:**
#```python
# ... logic to calculate coordinates ...
#points = {
#    "A": [0.0, 5.0],
#    "B": [-2.0, 0.0],
#    "C": [2.0, 0.0]
#}
#circles = {
#    "A": 3.5  # Radius of a circle centered at point A
#}
#```

        # Create user prompt
        user_prompt = f"""Find the coordinates and circle radiuses for:
{diagram_description}

{extra_instructions}

The correct format for `points` and `circles` is as follows (replace the None values with the correct coordinates and circle radiuses):
{json.dumps(expected_keys, indent=2)}
"""
        
        # Create model-specific folder path
        safe_model_name = model_id.replace("/", "_").replace(":", "_")
        model_folder = RESULTS_BASE_NEW / safe_model_name
        model_folder.mkdir(parents=True, exist_ok=True)
        
        result_file = model_folder / f"{file_path.stem}.json"
        
        # Skip if already processed
        if result_file.exists():
            with print_lock:
                print(f"Skipping {file_path.name} with {model_id} (already exists)...")
            return
        
        with print_lock:
            print(f"Processing {file_path.name} with {model_id}...")
        
        # Make API call to OpenRouter
        start_time = time.time()
        try:
            response = requests.post(
                BASE_API + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 15000
                },
                timeout=600
            )
            response.raise_for_status()
            result = response.json()
            
            elapsed_time = time.time() - start_time
            
            # Extract completion and token usage
            completion = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            generated_tokens = usage.get("completion_tokens", 0)
            
            generation_data = {
                "file": str(file_path),
                "model": model_id,
                "completion": completion,
                "user_prompt": user_prompt,
                "system_prompt": SYSTEM_PROMPT,
                "status": "success",
                "generated_tokens": generated_tokens,
                "elapsed_time": elapsed_time
            }
            
            # Save to specific folder
            with open(result_file, "w") as f:
                json.dump(generation_data, f, indent=2)
            
            with print_lock:
                print(f"✅ Completed {file_path.name} with {model_id}")
            
        except Exception as e:
            error_data = {
                "file": str(file_path),
                "model": model_id,
                "error": str(e),
                "user_prompt": user_prompt,
                "system_prompt": SYSTEM_PROMPT,
                "status": "failed"
            }
            with open(result_file, "w") as f:
                json.dump(error_data, f, indent=2)
            
            with print_lock:
                print(f"❌ Error processing {file_path.name} with {model_id}: {str(e)}")
    
    except Exception as e:
        with print_lock:
            print(f"❌ Error loading {file_path.name}: {str(e)}")

# ============================================================================
# Generation Phase
# ============================================================================
print(f"\n[1/2] Generation Phase")
print("-" * 80)

# Collect all tasks for this model
all_tasks = []
safe_model_name = MODEL.replace("/", "_").replace(":", "_")
model_folder = RESULTS_BASE_NEW / safe_model_name
model_folder.mkdir(parents=True, exist_ok=True)

for file_path in BENCHMARK_FOLDER.glob("*.json"):
    result_file = model_folder / f"{file_path.stem}.json"
    if not result_file.exists():
        all_tasks.append(file_path)

print(f"Total problems: {len(list(BENCHMARK_FOLDER.glob('*.json')))}")
print(f"Already completed: {len(list(BENCHMARK_FOLDER.glob('*.json'))) - len(all_tasks)}")
print(f"To process: {len(all_tasks)}")

if len(all_tasks) > 0:
    print(f"Using {min(MAX_WORKERS, len(all_tasks))} workers")
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(generate_for_file, file_path, MODEL): file_path 
                   for file_path in all_tasks}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 10 == 0:
                with print_lock:
                    print(f"Progress: {completed}/{len(all_tasks)} tasks completed")
    
    print(f"✅ Generation complete!")
else:
    print(f"✅ All generations already exist, skipping...")

# ============================================================================
# Reward Computation Phase
# ============================================================================
print(f"\n[2/2] Reward Computation Phase")
print("-" * 80)

def compute_reward_for_result(result_file, model_id):
    """Process a single result file and compute reward. Returns result dict or None."""
    try:
        with open(result_file, "r") as f:
            result_data = json.load(f)
        
        # Skip if status is failed
        if result_data.get("status") == "failed":
            return None
        
        # Get the problem file path
        problem_file_path = result_data.get("file")
        if isinstance(problem_file_path, Path):
            problem_file_path = str(problem_file_path)
        elif not isinstance(problem_file_path, str):
            return None
        
        # Get problem file name for CSV
        problem_file_name = Path(problem_file_path).name
        
        # Get completion text
        completion = result_data.get("completion", "")
        if not completion:
            return None
        
        # Get token and time statistics
        generated_tokens = result_data.get("generated_tokens", 0)
        elapsed_time = result_data.get("elapsed_time", 0.0)
        
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
        
        # Compute reward using llm_reward_function_new
        try:
            # Get the prompt from the problem file
            problem_file = BENCHMARK_FOLDER / problem_file_name
            if not problem_file.exists():
                return None
            
            with open(problem_file, "r") as f:
                problem = json.load(f)
            
            prompt = problem["nl_description"]
            # Add think tags if not present
            completion = "<think> </think>" + completion
            
            # Compute reward
            rewards = llm_reward_function_new(
                prompts=[prompt],
                completions=[completion],
                source_file=[str(problem_file)],
                max_workers=1,
                verbose=False,
                dense_reward=True
            )
            
            total_reward = rewards[0] if rewards else 0.0
            
            # Return result dict
            return {
                "model_id": model_id,
                "problem_file_name": problem_file_name,
                "generated_tokens": tokens,
                "elapsed_time": time_sum,
                "total_reward": total_reward
            }
            
        except Exception as e:
            # If reward computation fails, still record with 0 reward
            return {
                "model_id": model_id,
                "problem_file_name": problem_file_name,
                "generated_tokens": tokens,
                "elapsed_time": time_sum,
                "total_reward": 0.0
            }
            
    except Exception as e:
        return None

# Get all result files for this model
safe_model_name = MODEL.replace("/", "_").replace(":", "_")
model_folder = RESULTS_BASE_NEW / safe_model_name

if not model_folder.exists():
    print(f"❌ No results found in {model_folder}")
    sys.exit(1)

result_files = sorted([f for f in model_folder.glob("*.json") 
                      if "_error" not in f.name])

print(f"Found {len(result_files)} result files")
print(f"Computing rewards...")

# Collect all data
all_rows = []

# Process files with reward computation
with ThreadPoolExecutor(max_workers=MAX_WORKERS_REWARD) as executor:
    futures = {executor.submit(compute_reward_for_result, result_file, MODEL): result_file 
               for result_file in result_files}
    
    completed = 0
    for future in as_completed(futures):
        completed += 1
        if completed % 50 == 0:
            print(f"  Processed {completed}/{len(result_files)} files...")
        
        result = future.result()
        if result is not None:
            all_rows.append(result)

# Create DataFrame and save with model name
df = pd.DataFrame(all_rows)
output_csv = EXCELLS_DIR / f"results_{MODEL_NAME}.csv"
df.to_csv(output_csv, index=False)

print(f"\n✅ Saved {len(all_rows)} results to {output_csv}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# ============================================================================
# Metrics Computation
# ============================================================================
print(f"\n[3/3] Computing Metrics")
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
            average_reward = (subset["total_reward"].sum() + -5*missing_files) / EXPECTED_PER_DIFFICULTY
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