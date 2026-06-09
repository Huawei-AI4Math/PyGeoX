"""
Reward functions for evaluating geometric constraint satisfaction.

This module provides functions for computing rewards based on how well
a geometric configuration satisfies its constraints.
"""
import re
import json
import numpy as np
from copy import deepcopy
from typing import Optional, Tuple, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pygeox.custom_objects import RegularPolygon

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Global reward function parameters
ALPHA = 0.7  # Weight for answer reward component
BETA = 0.3   # Weight for verification reward component
DOMAIN = 10  # Domain for scene creation [-domain, domain]
DISTANCE_PENALTY = 1  # Distance penalty for overlapping points
MIN_DIST = 0.025  # Minimum distance between points
GENERATE_OBJECTIVE_FUNCTION = True  # Whether to generate objective function

# Default reward parameters
DEFAULT_REWARD_PARAMS = {
    "w": 6.0,
    "temperature": 0.1,
    "R_success": 4.0,
    "tol": 5e-3,
    "max_overlap_penalty": 5.0,
    "missing_penalty": 5.0
}



class RewardNamespace:
    """
    Namespace for reward-related functions in a geometric scene.
    
    Provides methods to evaluate how well a geometric configuration
    satisfies its constraints using reward functions.
    """
    
    def __init__(self, scene):
        """
        Initialize the reward namespace.
        
        Args:
            scene: The GeoScene instance this namespace belongs to
        """
        self.scene = scene


    def _llm_reward_function(
        self,
        llm_response: str,
        alpha: float = 0.7,
        beta: float = 0.3,
        reward_params: Optional[dict] = None
    ) -> Tuple[float, Dict[str, Any]]:
        
        if not reward_params:
            reward_params = DEFAULT_REWARD_PARAMS

        # Initialize rewards
        R_format = 0.0
        R_answer_norm = 0.0
        R_verify = 0.0
        details = {}

        # 1. R_format Check
        # ---------------------------------------------------------
        required_tags = ["<think>", "</think>", "<answer>", "</answer>", "<verify>", "</verify>"]
        # Boolean check: All tags must exist
        R_format = 1.0 if all(tag in llm_response for tag in required_tags) else 0.0
        
        if R_format == 0.0:
            return 0.0, {"error": "Missing required XML tags", "R_format": 0.0}

        # 2. Extract and Parse Answer
        # ---------------------------------------------------------
        answer_json = None
        answer_match = re.search(r'<answer>(.*?)</answer>', llm_response, re.DOTALL)
        
        if answer_match:
            raw_content = answer_match.group(1).strip()
            # Robust JSON extraction: Handles ```json code blocks or plain text
            json_candidate = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_content, re.DOTALL)
            json_str = json_candidate.group(1) if json_candidate else raw_content
            
            try:
                answer_json = json.loads(json_str)
            except json.JSONDecodeError:
                details["error"] = "JSON parsing failed"
                # In paper, malformed output triggers -5 penalty 
                raw_r_answer = reward_params.get("missing_penalty", -5.0)
        else:
            raw_r_answer = reward_params.get("missing_penalty", -5.0)

        # 3. Compute R_answer (Geometric Correctness)
        # ---------------------------------------------------------
        # Only run geometry solver if we have valid JSON data
        if answer_json and "points" in answer_json and "circles" in answer_json:
            # Call your internal PyGeoX solver
            # This returns the raw score (-5 to 10 range as per paper)
            raw_r_answer, solve_details = self.reward_function(
                answer_json["points"], 
                answer_json["circles"], 
                **reward_params
            )
            details.update(solve_details)
        elif answer_json:
            # JSON valid but keys missing
            raw_r_answer = reward_params.get("missing_penalty", -5.0)
            details["error"] = "Missing 'points' or 'circles' keys"

        # --- CRITICAL FIX: LINEAR NORMALIZATION ---
        # Map [-5, 10] -> [0, 1] to preserve gradient from syntax error to partial success
        # Formula: (R + 5) / 15
        # -5 -> 0.0
        #  0 -> 0.33
        # 10 -> 1.0
        R_answer_norm = (max(-5.0, min(10.0, raw_r_answer)) + 5.0) / 15.0

        # 4. Compute R_verify (Self-Correction)
        # ---------------------------------------------------------
        # We define "Success" as achieving the bonus threshold. 
        # In your paper, max reward (10) is only achieved if constraints are satisfied.
        # So we check if the raw reward is close to max (indicating success).
        is_actually_correct = 1.0 if raw_r_answer >= 9.9 else 0.0
        
        verify_match = re.search(r'<verify>(.*?)</verify>', llm_response, re.DOTALL)
        if verify_match:
            verify_text = verify_match.group(1).strip()
            # Extract score (0 to 1)
            score_match = re.search(r'Score:\s*([0-1](?:\.\d+)?)', verify_text, re.IGNORECASE)
            
            if score_match:
                predicted_confidence = float(score_match.group(1))
                
                # --- CRITICAL FIX: BINARY TARGET ---
                # Reward the model if its confidence matches the binary reality
                # This teaches: "Know when you are right, know when you are wrong"
                R_verify = 1.0 - abs(predicted_confidence - is_actually_correct)
            else:
                # Malformed verify score
                R_verify = 0.0 
        
        # 5. Final Aggregation
        # ---------------------------------------------------------
        # R_total = R_format * (alpha * R_ans + beta * R_ver)
        R_total = R_format * (alpha * R_answer_norm + beta * R_verify)

        details["R_format"] = R_format
        details["R_answer"] = raw_r_answer
        details["R_verify"] = R_verify
        details["R_total"] = R_total
        return R_total, details
    
   
                    

    def reward_function(
        self,
        point_coords,
        circle_r,
        w=6.0,
        temperature=0.1,
        R_success=4.0,
        tol=5e-3,
        max_overlap_penalty=5.0,
        missing_penalty=5.0
    ):
        r"""
        Complex reward function for numerical solving with detailed constraint analysis.

        This version implements a **Normalized Inverse Exponential Summation** to ensure
        that each constraint contributes at most 1/n to the overall continuous reward.
        This prevents a single large error from driving the reward to zero, ensuring
        that each constraint's contribution is bounded between 0 and 1/n.

        .. math::
            R = \underbrace{w \cdot \frac{1}{n} \sum_{i=1}^{n} e^{-\frac{E_i}{T}}}_{\text{Continuous Signal}}
            + \underbrace{\mathbb{I}_{success} \cdot R_{success}}_{\text{Bonus}}
            - \underbrace{\min(P_{max}, \mathbb{C}_{overlap})}_{\text{Penalty}}

        where:
        - $n$ is the number of constraints (equality and inequality combined).
        - $E_i$ is the absolute error of the $i$-th constraint.
        - $T$ is the temperature parameter controlling the sharpness of the reward kernel.
        - $\mathbb{I}_{success}$ is 1 if $\max(\text{all errors}) < tol$, else 0.
        - $\mathbb{C}_{overlap}$ is the count of penalty errors exceeding tolerance.
        - $P_{max}$ is the maximum overlap penalty cap.

        **Key Design Feature:**
        - **Normalized Inverse Exponential Summation**: Each constraint contributes
          $\frac{1}{n} \cdot e^{-\frac{E_i}{T}}$ to the continuous reward, ensuring:
          * Minimum contribution: 0 (when error is very large)
          * Maximum contribution: $\frac{1}{n}$ (when error is 0)
          * Total maximum: 1.0 (when all constraints are perfectly satisfied)
        - This approach prevents a single large error from dominating the reward signal,
          providing more stable gradients during optimization.

        **Rationale & Defaults:**
        - **Weight ($w = 6.0$):** The continuous signal is normalized to $[0, w]$ via
          normalized exponential summation. With $w = 6.0$, the max continuous signal
          is 6.0 when all constraints are perfectly satisfied.
        - **Bonus ($R_{success} = 4.0$):** Discrete bonus for satisfying all constraints.
          A bonus of 4.0 provides a significant reward boost while maintaining the
          importance of the shaping signal.
        - **Penalties ($P_{overlap} \leq 4.0$):** Penalty is capped at max_overlap_penalty
          and counts violations. Set to match the max shaping reward to ensure invalid
          states (overlaps/missing) are always strictly worse than a valid but poor
          geometric configuration.
        - **Temperature ($T=0.1$):** Controls sensitivity. With $T=0.1$, an error of 0.1
          results in $e^{-1} \approx 0.36$ contribution per constraint. Lower $T$ requires
          higher precision to get rewards.
        - **Tolerance ($tol=5e-3$):** Maximum error threshold for considering constraints
          satisfied.

        Args:
            point_coords (dict): Mapping point names to (x, y) tuples or lists.
            circle_r (dict): Mapping circle names to radius values.
            w (float): Weight for the combined continuous signal component. Default: 6.0.
            temperature (float): Softmax temperature. Default: 0.1.
            R_success (float): Discrete bonus for satisfying all constraints. Default: 4.0.
            tol (float): Max error tolerance to trigger success. Default: 5e-3.
            max_overlap_penalty (float): Maximum penalty for degenerate overlapping points. Default: 4.0.
            missing_penalty (float): Penalty for crashing/missing values. Default: 5.0.

        Returns:
            tuple: (total_reward, reward_details) where:
                - total_reward (float): Scalar reward value.
                - reward_details (dict): Dictionary containing component breakdown:
                    - r_eq_ineq: Combined equality and inequality component reward
                    - bonus: Success bonus (0 or R_success)
                    - pen_overlap: Overlap penalty applied
                    - total_error: List of all constraint errors (equality and inequality)
                    - temperature: Temperature parameter used
                    - w: Weight parameter used
                    - tol: Tolerance parameter used
                    - max_overlap_penalty: Maximum overlap penalty parameter used
                    - missing_penalty: Missing penalty parameter used
                    - R_success: Success bonus parameter used
                    - has_overlapping_points: Boolean indicating if any points overlap
        """

        empty_return_dict = {
            "r_eq_ineq": [],
            "bonus": 0,
            "pen_overlap": 0,
            "error_list": [],
            "temperature": float(temperature),
            "w": float(w),
            "tol": float(tol),
            "max_overlap_penalty": float(max_overlap_penalty),
            "missing_penalty": float(missing_penalty),
            "R_success": None,
            "has_overlapping_points": None
        }


        solver = self.scene.solver
        
        if solver.objective_func is None:
            raise ValueError(
                "Objective function is not generated. Please generate it first "
                "with scene.solver.numerical(just_generate_objective_function=True)"
            )

        # -------------------------------------------------------------------------
        # 1. Setup & Context Resolution
        # -------------------------------------------------------------------------
        scene = solver.scene

        # -------------------------------------------------------------------------
        # 2. Variable Mapping (Dict -> Flat Vector)
        # -------------------------------------------------------------------------
        try:
            x_vec = self.get_internal_vals(point_coords, circle_r)
        except KeyError as e:
            empty_return_dict["error"] = "Missing Points or circles"
            return -missing_penalty, empty_return_dict
        except Exception as e:
            raise ValueError(f"Error in get_internal_vals. Call Rafael. {e}")

        # -------------------------------------------------------------------------
        # 3. Numerical Analysis (PyGeoX)
        # -------------------------------------------------------------------------
        try:
            # Call the scene's solver analysis to get residuals
            analysis = scene.solver.numerical_analysis(
                x=x_vec, verbose=False, points_info=point_coords
            )
        except Exception as e:
            # If the solver crashes (e.g., singular matrix), return penalty
            raise ValueError(f"Error in scene.solver.numerical_analysis. Call Rafael. {e}")
            
        # -------------------------------------------------------------------------
        # 4. Compute Reward Components
        # -------------------------------------------------------------------------

        # --- Component A: Equalities (Target: 0) ---
        # These are strict constraints like "lines must be parallel"
        # - take absolute value of errors
        eq_data = analysis.get("equalities")
        eq_errors = [
            {
                "type": "eq",
                "desc": item["desc"],
                "error": float(np.abs(item["val"]))
            }
            for item in eq_data
        ]

        # --- Component B: Inequalities (Target: >= 0 or > strict_tol) ---
        ineq_errors = []
        penalty_errors = []

        # 1. Geq constraints (val >= 0). Error is squared magnitude if val < 0.
        for item in analysis.get("geq"):
            val = item["val"]
            error = float(max(0, -val))
            if "penalty" in item["desc"]:
                penalty_errors.append({"type": "geq (penalty)", "desc": item["desc"], "error": error})
            else:
                ineq_errors.append({"type": "geq", "desc": item["desc"], "error": error})

        # 2. Strict constraints (val > tol). Error is squared magnitude if val <= tol.
        for item in analysis.get("strict", []):
            val = item["val"]
            error = float(max(0, tol - val))
            if "penalty" in item["desc"]:  # not a penalty (to prevent overlapping points, etc...)
                penalty_errors.append({"type": "gt (penalty)", "desc": item["desc"], "error": error})
            else:
                ineq_errors.append({"type": "gt", "desc": item["desc"], "error": error})


        # --- Component C: Overlap / Visual Check ---
        # points are too close to each other or area of shapes is too small
        # count how many penalty_errors are larger than tol
        penalty_errors_count = sum(1 for item in penalty_errors if item["error"] > tol)
        has_overlapping_points = bool(solver.has_any_overlapping_points(points_info=point_coords))
        pen_overlap = min(max_overlap_penalty, penalty_errors_count)

        # --- Component D: Success Check ---
        is_solved = max(eq_errors + ineq_errors, key=lambda x: x["error"])["error"] < tol
        bonus = R_success if is_solved else 0.0

        # -------------------------------------------------------------------------
        # 5. Reward Shaping (Normalized Inverse Exponential Summation)
        # -------------------------------------------------------------------------
        # This version implements a Normalized Inverse Exponential Summation to ensure
        # that each constraint contributes at most 1/n to the overall continuous reward.
        # This prevents a single large error from driving the reward to zero, ensuring
        # that each constraint's contribution is bounded between 0 and 1/n.
        # Formula: r_eq_ineq = (1/n) * sum(exp(-E_i / T)) for each constraint i
        total_error = eq_errors + ineq_errors
        if total_error:
            r_eq_ineq = 0 
            for item in total_error:
                r_eq_ineq += float((1/len(total_error)) * np.exp(-item["error"] / temperature))
        else:
            raise ValueError("No constraints found (non-penalty)?? - Check this problem, something went wrong.")
            
        # Calculate total reward
        if bonus > 0:
            total_reward = float(w + bonus - pen_overlap)
        else:
            total_reward = float((w * r_eq_ineq) - pen_overlap)

        return total_reward, {
            "r_eq_ineq": r_eq_ineq,
            "SSE": analysis.get("objective", None),
            "bonus": float(bonus),
            "pen_overlap": float(pen_overlap),
            "error_list": eq_errors + ineq_errors + penalty_errors,
            "temperature": float(temperature),
            "w": float(w),
            "tol": float(tol),
            "max_overlap_penalty": float(max_overlap_penalty),
            "missing_penalty": float(missing_penalty),
            "R_success": float(R_success),
            "has_overlapping_points": has_overlapping_points
        }

    

    def get_internal_vals(self, point_coords, circle_r):
        """
        Convert point coordinates and circle radii to internal solver format.

        Args:
            point_coords: Dictionary mapping point names to (x, y) tuples
            circle_r: Dictionary mapping circle names to radius values

        Returns:
            numpy.ndarray: Internal values array matching vars_to_solve order

        Raises:
            KeyError: If points or radii are missing
            ValueError: If variable mismatch occurs
        """
        solver = self.scene.solver
        points = list(solver.scene.objects["Point"].keys())

        # make sure they are all here, if not cancel reward
        if set(point_coords.keys()) != set(points):
            raise KeyError("Missing Points")

        # check radiusus
        radius_vars = [
            str(var.name[:-2])
            for var in solver.vars_to_solve
            if var.name.endswith("_r")
        ]
        if set(circle_r.keys()) != set(radius_vars):
            raise KeyError("Missing Radii")

        # step 2 - create internal_vals (don't modify user_coord)
        internal_vals = deepcopy(point_coords)
        # extend internal_vals with circle_r and add "_r" at the end of circle_r keys
        for key, value in circle_r.items():
            internal_vals[f"{key}_r"] = value

        # step 3 - deal with regular polygons: convert point coordinates to reduced parameters
        regular_polygons = [
            obj
            for d in solver.scene.objects.values()
            for obj in d.values()
            if isinstance(obj, RegularPolygon)
        ]

        for polygon in regular_polygons:
            polygon_points = polygon.points
            # Extract coordinates of polygon points from internal_vals
            polygon_point_coords = {
                point.name: point_coords[point.name]
                for point in polygon_points
            }

            # Convert point coordinates to reduced parameters
            reduced_params = RegularPolygon.compute_reduced_params_from_points(
                polygon_point_coords,
                polygon.num_sides
            )

            # Remove polygon point coordinates from internal_vals
            for point in polygon_points:
                if point.name in internal_vals:
                    del internal_vals[point.name]

            # Add reduced parameters to internal_vals with the polygon's parameter names
            internal_vals[f"{polygon.name}_center_x"] = reduced_params['center_x']
            internal_vals[f"{polygon.name}_center_y"] = reduced_params['center_y']
            internal_vals[f"{polygon.name}_circumradius"] = reduced_params['circumradius']
            internal_vals[f"{polygon.name}_orientation"] = reduced_params['orientation']

        # step 4 - convert each (x,y) tuple to separate x and y entries
        expanded_vals = {}
        for key, value in internal_vals.items():
            # Split tuple into x and y
            if not (key.endswith("_x") or key.endswith("_y") or
                    key.endswith("_r") or key.endswith("_circumradius") or
                    key.endswith("_orientation")):
                expanded_vals[f"{key}_x"] = value[0]
                expanded_vals[f"{key}_y"] = value[1]
            else:
                expanded_vals[key] = value

        # step 5 - match order with scene.solver.vars_to_solve and convert to numpy array
        if solver.vars_to_solve is not None:
            # Convert sympy symbols to string names for matching
            var_names = [str(var) for var in solver.vars_to_solve]
            if set(var_names).issubset(set(list(expanded_vals.keys()))):
                # Create numpy array matching the order of vars_to_solve
                internal_vals_array = np.array([
                    expanded_vals.get(var_name) for var_name in var_names
                ])
                return internal_vals_array
            else:
                raise ValueError("Variable Mismatch")
        else:
            raise ValueError("No Variables to Solve")


# ---------------------------------------------------------------------------
# Module-level wrapper for ms-swift: call directly via
#   from pygeox.reward import llm_reward_function_for_swift
# This avoids circular imports by loading GeoScene only inside the function.
# ---------------------------------------------------------------------------
def llm_reward_function_for_swift(
    prompts: List[str],
    completions: List[str],
    source_file: List[str],
    max_workers: Optional[int] = None,
    use_wandb: bool = False,
    verbose: bool = False,
    wandb_log_prefix: str = "reward") -> List[float]:
    """
    Compute rewards for LLM responses in a format compatible with ms-swift RL training.
    
    This function processes multiple prompts, completions, and source files in parallel.
    For each source file, it loads the problem JSON, creates a scene, and computes rewards
    using the scene's reward namespace. It supports wandb logging for tracking reward statistics.
    
    Uses global constants: ALPHA, BETA, DOMAIN, DISTANCE_PENALTY, MIN_DIST, GENERATE_OBJECTIVE_FUNCTION.
    
    Args:
        prompts (List[str]): List of input prompts (for compatibility, not directly used).
        completions (List[str]): List of LLM completions/responses to evaluate.
        source_file (List[str]): List of source file paths (relative or absolute) to JSON problem files.
        max_workers (Optional[int]): Maximum number of parallel workers. If None, uses
            the number of CPU cores. Default: None.
        use_wandb (bool): Whether to log metrics to wandb. Default: False.
        wandb_log_prefix (str): Prefix for wandb log keys. Default: "reward".
    
    Returns:
        List[float]: List of reward values for each completion, in the same order as inputs.
    
    Example:
        >>> prompts = ["Solve this geometry problem...", "Another problem..."]
        >>> completions = [
        ...     "<think>...</think><answer>...</answer><verify>Score: 1</verify>",
        ...     "<think>...</think><answer>...</answer><verify>Score: 0</verify>"
        ... ]
        >>> source_file = [
        ...     "generated_data_penalty/json_processed_2_fixed/1obj_2rel_2extra_gen0016.json",
        ...     "generated_data_penalty/json_processed_2_fixed/2obj_3rel_1extra_gen2963.json"
        ... ]
        >>> rewards = reward_namespace.llm_reward_function_for_swift(
        ...     prompts=prompts,
        ...     completions=completions,
        ...     source_file=source_file,
        ...     use_wandb=True
        ... )
        >>> print(f"Rewards: {rewards}")
    """
    if len(prompts) != len(completions) or len(completions) != len(source_file):
        raise ValueError(
            f"Length mismatch: prompts ({len(prompts)}) != completions ({len(completions)}) "
            f"!= source_file ({len(source_file)})"
        )
    
    if len(completions) == 0:
        return []
    
    if max_workers is None:
        import os
        max_workers = min(len(completions), os.cpu_count() or 1)
    
    # Prepare arguments for parallel processing
    def compute_reward(args: Tuple[str, str, str]) -> Tuple[float, Dict[str, Any]]:
        """Wrapper function for parallel execution."""
        completion, source_path, _ = args
        
        try:
            # Resolve source file path (handle both relative and absolute paths)
            source_path_str = str(source_path)
            
            if Path(source_path_str).is_absolute():
                json_path = Path(source_path_str)
            else:
                # Try relative to common data directories
                # Repo root (pygeox/reward.py -> pygeox -> repo root)
                _repo_root = Path(__file__).resolve().parent.parent
                base_paths = [
                    Path.cwd(),
                    Path.cwd().parent,
                    _repo_root,
                ]
                json_path = None
                for base in base_paths:
                    candidate = base / source_path_str
                    if candidate.exists():
                        json_path = candidate
                        break
                
                if json_path is None:
                    # Try as absolute path (in case it's already correct)
                    json_path = Path(source_path_str)
                    if not json_path.exists():
                        raise FileNotFoundError(f"Could not find source file: {source_path_str}")
            
            # Load problem JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                problem = json.load(f)
                        
            # Create scene from JSON with default parameters
            scene = create_scene_from_json(
                domain=DOMAIN,
                json_data=problem,
                generate_objective_function=GENERATE_OBJECTIVE_FUNCTION,
                distance_penalty=DISTANCE_PENALTY,
                min_dist=MIN_DIST
            )
            
            # Create reward namespace for this scene
            reward_ns = RewardNamespace(scene)
            
            # Compute reward using global constants
            reward, detail = reward_ns._llm_reward_function(
                llm_response=completion,
                alpha=ALPHA,
                beta=BETA,
                reward_params=DEFAULT_REWARD_PARAMS
            )
            
            return reward, detail
            
        except Exception as e:
            # Handle errors gracefully
            return 0.0, {"error": str(e), "source": source_path}
    
    # Parallelize reward computation
    rewards = []
    details_list = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(compute_reward, (completion, src, prompt)): idx
            for idx, (completion, src, prompt) in enumerate(zip(completions, source_file, prompts))
        }
        
        # Collect results in order
        results = [None] * len(completions)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                reward, detail = future.result()
                if verbose:
                    print("--------------------------------")
                    print("Total Reward: ", detail["R_total"])
                    print("Reward format: ", detail["R_format"])
                    print("Reward answer: ", detail["R_answer"])
                    print("Reward verify: ", detail["R_verify"])
                results[idx] = (reward, detail)
            except Exception as e:
                # Handle errors gracefully
                results[idx] = (0.0, {"error": str(e)})
        
        # Extract rewards and details
        for reward, detail in results:
            rewards.append(reward)
            details_list.append(detail)
    
    # Compute statistics for wandb logging
    rewards_array = np.array(rewards)
    statistics = {
        "mean": float(np.mean(rewards_array)),
        "std": float(np.std(rewards_array)),
        "min": float(np.min(rewards_array)),
        "max": float(np.max(rewards_array)),
        "median": float(np.median(rewards_array)),
        "count": len(rewards),
        "positive_count": int(np.sum(rewards_array > 0)),
        "zero_count": int(np.sum(rewards_array == 0)),
    }
    
    # Extract component statistics if available
    if details_list and "R_format" in details_list[0]:
        r_format_values = [d.get("R_format", 0) for d in details_list]
        r_answer_values = [d.get("R_answer", 0) for d in details_list]
        r_verify_values = [d.get("R_verify", 0) for d in details_list]
        
        statistics["R_format_mean"] = float(np.mean(r_format_values))
        statistics["R_answer_mean"] = float(np.mean(r_answer_values))
        statistics["R_verify_mean"] = float(np.mean(r_verify_values))
    
    # Log to wandb if enabled
    if WANDB_AVAILABLE and use_wandb:
        log_dict = {
            f"{wandb_log_prefix}/mean": statistics["mean"],
            f"{wandb_log_prefix}/std": statistics["std"],
            f"{wandb_log_prefix}/min": statistics["min"],
            f"{wandb_log_prefix}/max": statistics["max"],
            f"{wandb_log_prefix}/median": statistics["median"],
            f"{wandb_log_prefix}/count": statistics["count"],
            f"{wandb_log_prefix}/positive_ratio": statistics["positive_count"] / statistics["count"] if statistics["count"] > 0 else 0.0,
        }
        
        # Add component statistics
        if "R_format_mean" in statistics:
            log_dict[f"{wandb_log_prefix}/R_format_mean"] = statistics["R_format_mean"]
            log_dict[f"{wandb_log_prefix}/R_answer_mean"] = statistics["R_answer_mean"]
            log_dict[f"{wandb_log_prefix}/R_verify_mean"] = statistics["R_verify_mean"]
        
        # Log weights and hyperparameters if available in details
        if details_list and "w" in details_list[0]:
            log_dict[f"{wandb_log_prefix}/weight"] = details_list[0].get("w")
        if details_list and "temperature" in details_list[0]:
            log_dict[f"{wandb_log_prefix}/temperature"] = details_list[0].get("temperature")
        
        # Log global constants
        log_dict[f"{wandb_log_prefix}/alpha"] = ALPHA
        log_dict[f"{wandb_log_prefix}/beta"] = BETA
        log_dict[f"{wandb_log_prefix}/domain"] = DOMAIN
        log_dict[f"{wandb_log_prefix}/distance_penalty"] = DISTANCE_PENALTY
        log_dict[f"{wandb_log_prefix}/min_dist"] = MIN_DIST
        
        wandb.log(log_dict)
    
    return rewards

