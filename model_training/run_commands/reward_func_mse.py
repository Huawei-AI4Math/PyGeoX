#OPENRLHF
#ray job submit --address="http://127.0.0.1:8265" \
#  --runtime-env-json='{"working_dir": "/openrlhf"}' \
#  -- python3 -m openrlhf.cli.train_ppo_ray \
#  --pretrain meta-llama/Meta-Llama-3-8B \
#  --use_dynamic_batch \
#  --remote_rm_url /path/to/PyGeoX/model_training/run_commands/reward_func_openrlhf.py \ 
#  --label_key source_file \
#  --prompt_data your_prompt_dataset \
#  ... # other training args

#Please change remote_rm_url and label_key

######################################################################
## CREATE NEW REWARD FUNCTION HERE
######################################################################

import sys
import os
import signal

# CRITICAL: Set environment variables BEFORE importing numba/numpy to prevent threading conflicts
# These settings help prevent resource exhaustion when running inside Ray workers
os.environ['NUMBA_NUM_THREADS'] = '1'  # Prevent numba from spawning extra threads
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP threading conflicts
os.environ['MKL_NUM_THREADS'] = '1'  # Prevent MKL threading conflicts
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Prevent OpenBLAS threading conflicts
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['RAY_IGNORE_UNHANDLED_ERRORS'] = '1'  # Prevent Ray from crashing on unhandled errors

# CRITICAL: Install signal handlers to prevent worker crashes from propagating
def _signal_handler(signum, frame):
    """Graceful handling of termination signals to prevent cascading failures."""
    pass  # Silently ignore - let Ray handle worker lifecycle

# Install handlers for common termination signals (only if not already set)
try:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
except Exception:
    pass  # May fail in some contexts, that's OK

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_script_dir, "../../"))
sys.path.append(_project_root)
import re
import json
import gc
import warnings
import contextlib
from io import StringIO
import numpy as np
from copy import deepcopy
from typing import Optional, Tuple, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, wait, ALL_COMPLETED, FIRST_COMPLETED
from pathlib import Path
from pygeox.synthetic.llm_client import create_scene_from_json
from pygeox.custom_objects import RegularPolygon
import math
import scipy
import shapely
import pandas
import matplotlib
import random
import numba
from func_timeout import func_timeout, FunctionTimedOut

# CRITICAL: Suppress all warnings to prevent Ray object store OOM
warnings.filterwarnings('ignore')


# Global reward function parameters
ALPHA = 0.7  # Weight for answer reward component
BETA = 0.3   # Weight for verification reward component
DOMAIN = 10  # Domain for scene creation [-domain, domain]
DISTANCE_PENALTY = 1  # Distance penalty for overlapping points
MIN_DIST = 0.02  # Minimum distance between points
WRONG_FORMAT_REWARD = 0.0
MAX_WORKERS = 1  # Single-threaded: Ray already parallelizes across workers, no need for internal threading
TIMEOUT_SECONDS = 90  # Balanced timeout for hard problems
BATCH_TIMEOUT_MULTIPLIER = 1.5  # Give batch extra time beyond individual task timeout
# CONFIGURE REWARD TYPE HERE
USE_DENSE_REWARD = True  # Dense reward: exp(-SSE/10), outputs in [0, 1]


def destroy_scene(scene):
    """
    Aggressively destroy a GeoScene object and all its internal references to prevent memory leaks.
    This is CRITICAL for Ray workers running many reward computations.
    """
    if scene is None:
        return
    
    try:
        # Clean up reward namespace
        if hasattr(scene, 'reward'):
            if hasattr(scene.reward, '_scene'):
                scene.reward._scene = None
            del scene.reward
        
        # Clean up solver and its internal state
        if hasattr(scene, 'solver'):
            if hasattr(scene.solver, '_objective_func'):
                scene.solver._objective_func = None
            if hasattr(scene.solver, '_constraints'):
                scene.solver._constraints = None
            if hasattr(scene.solver, '_variables'):
                scene.solver._variables = None
            del scene.solver
        
        # Clean up all geometric objects
        if hasattr(scene, '_objects'):
            for obj in list(scene._objects.values()):
                try:
                    del obj
                except:
                    pass
            scene._objects.clear()
        
        # Clean up points
        if hasattr(scene, '_points'):
            scene._points.clear()
        
        # Clean up constraints
        if hasattr(scene, '_constraints'):
            scene._constraints.clear()
        
        # Clean up any cached values
        for attr in ['_domain_consts', '_add', '_relate', '_prove']:
            if hasattr(scene, attr):
                try:
                    setattr(scene, attr, None)
                except:
                    pass
        
        # Close matplotlib figures
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass
        
        # Delete the scene object itself
        del scene
        
    except Exception:
        pass  # Silently ignore cleanup errors
    
    # Force garbage collection
    gc.collect()
        
# Default reward parameters
DEFAULT_REWARD_PARAMS = {
    "w": 6.0,
    "temperature": 0.1,
    "R_success": 4.0,
    "tol": 1e-2,
    "max_overlap_penalty": 5.0,
    "missing_penalty": 5.0
}
def llm_reward_function_new(
    prompts: List[str],
    completions: List[str],
    source_file: List[str],
    max_workers: Optional[int] = MAX_WORKERS,
    verbose: bool = False,
    dense_reward: bool = True,
) -> List[float]:
    """
    Computes reward R = R_format * R_pygeox.
    """
    
    if len(prompts) != len(completions) or len(completions) != len(source_file):
        raise ValueError("Length mismatch between prompts, completions, and source files.")
    
    if len(completions) == 0:
        return []

    # 1. Determine Batch Size
    if max_workers is None:
        import os
        cpu_count = os.cpu_count() or 1
        max_workers = min(len(completions), cpu_count, MAX_WORKERS)
    
    # Disable verbose logging to prevent Ray object store OOM
    # if verbose:
    #     print(f"DEBUG: Processing {len(completions)} items in strict batches of {max_workers}.")

    # -----------------------------------------------------------------------
    # Helper: Execute Code and Extract Variables (Nested as per your code)
    # -----------------------------------------------------------------------
    def is_invalid(val):
        if not isinstance(val, float) or math.isnan(val):
            return True
        return False

    def execute_and_extract(completion_text: str) -> Tuple[float, Any, Any, str]:
        # ... (Your existing execute_and_extract code remains identical) ...
        has_think = "<think>" in completion_text and "</think>" in completion_text
        has_code_block = "```py" in completion_text or "```python" in completion_text
        
        if not (has_think and has_code_block):
            return 0.0, None, None, "Missing <think> tags or python code block"

        match = re.search(r"```(?:python|py)(.*?)```", completion_text, re.DOTALL)
        if not match:
            return 0.0, None, None, "Could not regex match code block"
        
        code_content = match.group(1).strip()
        
        # SAFETY CHECK: Detect scipy.optimize usage to prevent crashes
        # scipy.optimize can cause threading conflicts with Ray workers and numerical instabilities
        dangerous_patterns = [
            'scipy.optimize',
            'from scipy import optimize',
            'from scipy.optimize import',
            'import scipy.optimize',
        ]
        for pattern in dangerous_patterns:
            if pattern in code_content:
                return 0.0, None, None, f"scipy.optimize detected - skipping for worker stability"
        
        execution_content = {}

        try:
            # Using multiprocessing to allow clean kills of infinite loops
            # (Using your multiprocessing version from previous steps is best here)
            # But relying on your provided 'execute_and_extract' code:
            def run_code():
                exec(code_content, execution_content, execution_content)
            
            # Use func_timeout if available as per imports, otherwise standard exec
            func_timeout(TIMEOUT_SECONDS, run_code)
            
            points_raw = execution_content.get('points', {})
            circles_raw = execution_content.get('circles', {})
            
            if not isinstance(points_raw, dict):
                execution_content.clear()
                return 0.0, None, None, "Variable 'points' is not a dictionary"
            if not isinstance(circles_raw, dict):
                execution_content.clear()
                return 0.0, None, None, "Variable 'circles' is not a dictionary"
            
            # CRITICAL: Deep copy to break references to execution context objects
            # This prevents memory leaks from exec() created objects
            points = {k: [float(v[0]), float(v[1])] if isinstance(v, (list, tuple)) and len(v) >= 2 else v 
                     for k, v in points_raw.items()}
            circles = {k: float(v) if isinstance(v, (int, float)) else v 
                      for k, v in circles_raw.items()}
            
            # Clean up execution context immediately to free memory
            points_raw.clear()
            circles_raw.clear()
            execution_content.clear()
            del points_raw, circles_raw
            gc.collect()
                
            return 1.0, points, circles, ""
            
        except BaseException as e:
            execution_content.clear()  # Clean up on error too
            return 0.0, None, None, f"Code Execution Error: {type(e).__name__}: {str(e)}"

    # -----------------------------------------------------------------------
    # Worker Function (Nested)
    # -----------------------------------------------------------------------
    def compute_single_reward(args: Tuple[str, str, str, bool]) -> Tuple[float, Dict[str, Any]]:
        """Compute reward for a single completion. Always returns valid (float, dict) tuple."""
        completion, source_path, _, dense_reward = args
        r_format = 0.0
        r_pygeox = WRONG_FORMAT_REWARD if dense_reward else 0.0
        details = {"R_format": r_format, "R_pygeox": r_pygeox, "error": None}
        
        # Force aggressive garbage collection before starting
        gc.collect()
        gc.collect()  # Run twice for cyclic references

        try:
            # Step 1: Format & Execution
            # print("Reward calculation for source file: ", source_path)
            r_format, points, circles, exec_err = execute_and_extract(completion)
            details["R_format"] = r_format
            print(f"[MSE2-DEBUG] Step1: r_format={r_format}, exec_err={exec_err}, has_points={points is not None}, has_circles={circles is not None}", flush=True)

            if r_format == 0.0:
                details["error"] = exec_err
                r_pygeox = WRONG_FORMAT_REWARD if dense_reward else 0.0
                print(f"[MSE2-DEBUG] EARLY EXIT: r_format=0, error={exec_err}", flush=True)
                return float(r_pygeox), details

            if points is None or circles is None:
                details["error"] = "Points/Circles is None"
                r_pygeox = WRONG_FORMAT_REWARD if dense_reward else 0.0
                print(f"[MSE2-DEBUG] EARLY EXIT: points/circles is None", flush=True)
                return float(r_pygeox), details

            # Step 2: Load Scene
            source_path_str = str(source_path)
            json_path = Path(source_path_str)
            if not json_path.is_absolute():
                base_paths = [Path.cwd(), Path(_project_root)]
                for base in base_paths:
                    candidate = base / source_path_str
                    if candidate.exists():
                        json_path = candidate
                        break
            
            print(f"[MSE2-DEBUG] Step2: json_path={json_path}, exists={json_path.exists()}", flush=True)
            if not json_path.exists():
                 details["error"] = f"Source file not found: {source_path}"
                 val = WRONG_FORMAT_REWARD if dense_reward else 0.0
                 print(f"[MSE2-DEBUG] EARLY EXIT: source file not found: {source_path}", flush=True)
                 return float(val), details

            with open(json_path, 'r', encoding='utf-8') as f:
                problem_data = json.load(f)
            
            scene = None  # Initialize to ensure cleanup in finally block
            try:
                # CRITICAL: Suppress all stdout/stderr during scene creation to prevent Ray object store OOM
                # PyGeoX prints warnings/info that fill up Ray's object store
                with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                    scene = create_scene_from_json(
                        domain=DOMAIN,
                        json_data=problem_data,
                        generate_objective_function=True,
                        distance_penalty=DISTANCE_PENALTY,
                        min_dist=MIN_DIST
                    )
                
                # Also suppress output during reward computation
                print(f"[MSE2-DEBUG] Step3: scene created OK, calling reward_function...", flush=True)
                with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                    r_pygeox, details = scene.reward.reward_function(points, circles)
                print(f"[MSE2-DEBUG] Step3: r_pygeox={r_pygeox}, details keys={list(details.keys()) if isinstance(details, dict) else 'NOT_DICT'}", flush=True)
                print(f"[MSE2-DEBUG] Step3: SSE={details.get('SSE') if isinstance(details, dict) else 'N/A'}, full details={details}", flush=True)
                if is_invalid(r_pygeox):
                    r_pygeox = WRONG_FORMAT_REWARD
                    details["error"] = "R_pygeox is NaN"

                if not dense_reward:
                    r_pygeox = 1.0 if r_pygeox > 8.0 else 0.0
            except Exception as e:
                print(f"[MSE2-DEBUG] Step3 EXCEPTION: {e}", flush=True)
                details["error"] = f"Validation logic error: {e}"
                r_pygeox = WRONG_FORMAT_REWARD
            finally:
                # CRITICAL: Use destroy_scene for aggressive memory cleanup
                destroy_scene(scene)
                scene = None
                
                # Clean up problem_data (JSON dict can be large)
                if 'problem_data' in dir():
                    problem_data.clear()
                    del problem_data

            details["R_pygeox"] = r_pygeox
            details["R_format"] = r_format
            
            # Clean up points and circles dicts after use
            if points is not None:
                points.clear()
                del points
            if circles is not None:
                circles.clear()
                del circles
            
            # Force aggressive GC after cleanup
            gc.collect()
            gc.collect()

            # Step 4: Total Reward using exp(-SSE/10)
            if r_format == 1.0:
                sse = details.get("SSE")
                print(f"[MSE2-DEBUG] Step4: r_format={r_format}, SSE={sse}, type(SSE)={type(sse)}", flush=True)
                if sse is not None and isinstance(sse, (int, float)) and not math.isnan(float(sse)):
                    total_reward = math.exp(-float(sse) / 10.0)
                    print(f"[MSE2-DEBUG] Step4: exp(-{sse}/10) = {total_reward}", flush=True)
                else:
                    total_reward = 0.0
                    print(f"[MSE2-DEBUG] Step4: SSE invalid, total_reward=0", flush=True)
            else:
                total_reward = 0.0
                print(f"[MSE2-DEBUG] Step4: r_format!=1.0, total_reward=0", flush=True)
            if is_invalid(total_reward) or total_reward < 0.0 or total_reward > 1.0:
                print(f"[MSE2-DEBUG] Step4: SANITY FAIL total_reward={total_reward}, setting to 0", flush=True)
                total_reward = 0.0
            details["R_total"] = total_reward
            print(f"[MSE2-DEBUG] FINAL RETURN: total_reward={total_reward}", flush=True)
            return float(total_reward), details

        except BaseException as e:
            print(f"[MSE2-DEBUG] GLOBAL EXCEPTION: {type(e).__name__}: {e}", flush=True)
            val = WRONG_FORMAT_REWARD if dense_reward else 0.0
            error_details = {"R_format": 0.0, "R_pygeox": val, "error": f"Global error: {type(e).__name__}: {str(e)}"}
            return float(val), error_details

    # -----------------------------------------------------------------------
    # Sequential Execution Loop (Simplified)
    # -----------------------------------------------------------------------
    final_results = [None] * len(completions)
    
    # Process each completion one by one
    # This is much more memory-stable than using a ThreadPoolExecutor
    for idx in range(len(completions)):
        arg = (completions[idx], source_file[idx], prompts[idx], dense_reward)
        
        try:
            # Compute reward directly without thread pool overhead
            r, d = compute_single_reward(arg)
            
            # Ensure valid types for serialization
            if not isinstance(r, (int, float)):
                r = WRONG_FORMAT_REWARD if dense_reward else 0.0
            if not isinstance(d, dict):
                d = {"error": "Invalid details object"}
            
            final_results[idx] = (float(r), d)
            print(f"[MSE2-DEBUG] Loop idx={idx}: reward={r}, error={d.get('error') if isinstance(d, dict) else 'N/A'}", flush=True)

        except BaseException as e:
            val = WRONG_FORMAT_REWARD if dense_reward else 0.0
            final_results[idx] = (float(val), {"error": f"Execution error: {type(e).__name__}: {str(e)}"})
            print(f"[MSE2-DEBUG] Loop idx={idx}: EXCEPTION={type(e).__name__}: {e}", flush=True)
        
        # Aggressive cleanup after EVERY SINGLE completion
        gc.collect()
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except:
            pass

    # -----------------------------------------------------------------------
    # Final Assembly - Ensure all returns are valid serializable objects
    # -----------------------------------------------------------------------
    rewards = []
    
    for i, res in enumerate(final_results):
        if res is None:
            # Should not happen logic-wise, but handle defensively
            r = WRONG_FORMAT_REWARD if dense_reward else 0.0
            d = {"error": "Result missing"}
        else:
            try:
                r, d = res
                # Ensure r is a valid number
                if not isinstance(r, (int, float)):
                    r = WRONG_FORMAT_REWARD if dense_reward else 0.0
            except (ValueError, TypeError) as e:
                r = WRONG_FORMAT_REWARD if dense_reward else 0.0
                d = {"error": f"Result unpacking error: {str(e)}"}

        # Final sanity checks - ensure valid float in [0, 1] range
        try:
            r = float(r)
            if is_invalid(r) or r < 0.0 or r > 1.0:
                r = 0.0
        except (ValueError, TypeError):
            r = 0.0

        rewards.append(float(r))  # Ensure it's a Python float, not numpy or other type
        
        # Only log errors to avoid Ray object store OOM
        # Keep error logging minimal
        # if verbose and d.get('error'):
        #      print(f"Sample {i} Error: {d.get('error')}")

    return rewards
    
def reward_func(queries, prompts, labels):
    """Main reward function called by OpenRLHF. Returns valid metrics dict with reward arrays."""
    try:
        answers = []
        # DO NOT print answers - causes Ray object store OOM!
        # print("PRINTING ANSWERS --------------------------------------------------------------")
        for q, p, l in zip(queries, prompts, labels):
            # The answer is everything in the query AFTER the prompt
            # We use the length of the prompt to slice the string
            answer = q[len(p):].strip()
            answers.append(answer)
            # print("answer", answer)  # DISABLED: Fills Ray object store
            # print("----------------------------------------------------------------------------")
            # print("source_file", l)


        rewards =  llm_reward_function_new(prompts = prompts, 
            completions = answers, 
            source_file = labels, 
            max_workers = MAX_WORKERS, 
            verbose = False,  # CRITICAL: Set to False to prevent Ray object store OOM!
            dense_reward = USE_DENSE_REWARD)
        
        # Ensure rewards is a valid list of floats
        if not isinstance(rewards, list):
            rewards = [0.0] * len(prompts)
        rewards = [float(r) if isinstance(r, (int, float)) and not math.isnan(r) else 0.0 for r in rewards]
        
        # Compute additional metrics for better comparison between dense/sparse training
        # These can be logged to wandb for analysis
        # For sparse rewards (0 or 1): success_rate tracks % where reward == 1
        # For dense rewards (0-10): success_rate tracks % where reward >= 8.0 (the threshold for sparse conversion)
        # This allows direct comparison: sparse "mean_reward" ≈ dense "success_rate"
        success_threshold = 0.5  # exp(-SSE/10) > 0.5 means SSE < ~6.9
        success_rate = sum(1 for r in rewards if r >= success_threshold) / len(rewards) if len(rewards) > 0 else 0.0
        mean_reward = sum(rewards) / len(rewards) if len(rewards) > 0 else 0.0
        
        # Log these metrics (if wandb is available in OpenRLHF context, they should appear)
        metrics = {
            "rewards": rewards, 
            "scores": rewards,
            "success_rate": float(success_rate),  # % of problems solved (reward >= 1.0 for sparse, or >= 8.0 for dense)
            "mean_reward": float(mean_reward)
        }
    except BaseException as e:
        # Catastrophic failure - return safe defaults
        print(f"CRITICAL ERROR in reward_func: {type(e).__name__}: {e}")
        num_samples = len(queries) if queries else 1
        default_reward = 0.0
        metrics = {
            "rewards": [float(default_reward)] * num_samples,
            "scores": [float(default_reward)] * num_samples,
            "success_rate": 0.0,
            "mean_reward": 0.0
        }

    #save_path = "reward_samples_debug.jsonl"
    #samples_to_save = []
    #for i, (q, p, l) in enumerate(zip(queries, prompts, labels)):
    #        sample = {
    #            "pid": os.getpid(),
    #            "sample_idx": i,
    #            "query": q,
    #            "prompt": p,
    #            "label": l,
    #            "reward_score": rewards[i] if 'rewards' in locals() else None 
    #        }
    #        samples_to_save.append(sample)

    #with open(save_path, "a", encoding="utf-8") as f:
    #     for entry in samples_to_save:
    #         f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return metrics
