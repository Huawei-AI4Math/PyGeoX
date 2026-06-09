"""
Data generation pipeline for creating synthetic geometric diagrams.

This module provides functionality to generate multiple geometric diagrams
with different configurations and save all outputs (JSON, images).
"""

import sys
import json
import re
import yaml
import os
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from concurrent.futures import (
    ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError,
    ThreadPoolExecutor
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for multiprocessing
import matplotlib.pyplot as plt  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Add parent directory to path so we can import pygeox
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pygeox.synthetic import generate_diagram_with_llm  # noqa: E402


def create_output_directories(base_dir: Path) -> Dict[str, Path]:
    """
    Create output directories for JSON and images.

    Args:
        base_dir: Base directory where all outputs will be saved

    Returns:
        Dictionary mapping folder names to Path objects
    """
    dirs = {
        "json": base_dir / "json",
        "image": base_dir / "image",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def load_config(config_path: Optional[Path] = None) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file. If None, looks for
                    data_generation_config.yaml in project root.

    Returns:
        Dictionary containing all configuration values
    """
    if config_path is None:
        # Look for config in project root
        config_path = project_root / "data_generation_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Please create a config file or specify the path."
        )

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Load LLM environment variables from .env if not specified in config
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    # Override LLM config with .env values if config values are null
    if config.get("llm", {}).get("api_key") is None:
        config.setdefault("llm", {})["api_key"] = os.getenv("API_KEY")
    if config.get("llm", {}).get("open_ai_base_url") is None:
        config.setdefault("llm", {})["open_ai_base_url"] = (
            os.getenv("OPEN_AI_BASE_URL")
        )

    # Convert configurations list of lists to list of tuples
    if "configurations" in config:
        config["configurations"] = [
            tuple(c) for c in config["configurations"]
        ]

    return config


def _parse_function_call(func_str: str):
    """Parse a string like 'LineSegment(A,B)' into (name, args)."""
    match = re.match(r'(\w+)\((.*?)\)', func_str.strip())
    if not match:
        raise ValueError(f"Invalid function call format: {func_str}")
    func_name = match.group(1)
    args_str = match.group(2)
    # Parse arguments (split by comma, handling nested parentheses)
    args = []
    current_arg = ""
    paren_depth = 0
    for char in args_str:
        if char == '(':
            paren_depth += 1
            current_arg += char
        elif char == ')':
            paren_depth -= 1
            current_arg += char
        elif char == ',' and paren_depth == 0:
            args.append(current_arg.strip())
            current_arg = ""
        else:
            current_arg += char
    if current_arg.strip():
        args.append(current_arg.strip())
    return func_name, args


def _get_add_method_name(obj_type_str: str) -> str:
    """Convert object type string to scene.add method name."""
    # Convert CamelCase to snake_case
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', obj_type_str).lower()
    return name


def _get_relate_method_name(rel_name: str) -> str:
    """Convert relationship name to scene.relate method name."""
    # Convert CamelCase to snake_case
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', rel_name).lower()
    return name


def generate_pygeox_code_from_json(json_data: dict) -> str:
    """
    Generate PyGeoX code string from JSON data.

    Args:
        json_data: JSON data containing Points, Objs, and Rels

    Returns:
        String containing PyGeoX code
    """
    code_lines = []
    code_lines.append("from pygeox import GeoScene")
    code_lines.append("")
    code_lines.append("scene = GeoScene()")
    code_lines.append("")
    code_lines.append("### objects")
    code_lines.append("")

    # Create points with variable assignment
    points = json_data.get("Points", [])
    if points:
        point_vars_str = ", ".join(points)
        point_names_str = ", ".join([f'"{p}"' for p in points])
        code_lines.append(
            f"{point_vars_str} = scene.add.points([{point_names_str}])"
        )
        code_lines.append("")

    # Create objects with variable assignment (no name argument)
    objs = json_data.get("Objs", {})
    for obj_name, obj_str in objs.items():
        obj_type, args = _parse_function_call(obj_str)
        add_method_name = _get_add_method_name(obj_type)

        # Format arguments
        formatted_args = []
        for arg in args:
            # Check if it's a number
            if re.match(r'^-?\d+\.?\d*$', arg):
                formatted_args.append(arg)
            # Check if it's a boolean
            elif arg.lower() in ('true', 'false'):
                formatted_args.append(arg.capitalize())
            # Otherwise, it's a point or object name (use as-is)
            else:
                formatted_args.append(arg)

        args_str = ", ".join(formatted_args)
        code_lines.append(
            f"{obj_name} = scene.add.{add_method_name}({args_str})"
        )

    code_lines.append("")
    code_lines.append("### relationships")
    code_lines.append("")

    # Add relationships
    rels = json_data.get("Rels", [])
    for rel_str in rels:
        rel_name, args = _parse_function_call(rel_str)
        relate_method_name = _get_relate_method_name(rel_name)

        # Format arguments
        formatted_args = []
        for arg in args:
            # Check if it's a number
            if re.match(r'^-?\d+\.?\d*$', arg):
                formatted_args.append(arg)
            # Check if it's a boolean
            elif arg.lower() in ('true', 'false'):
                formatted_args.append(arg.capitalize())
            # Otherwise, it's a point or object name (use as-is)
            else:
                formatted_args.append(arg)

        args_str = ", ".join(formatted_args)
        code_lines.append(f"scene.relate.{relate_method_name}({args_str})")

    # Add extra relationships section if they exist
    extra_rel = json_data.get("extra_rel", [])
    if extra_rel:
        code_lines.append("")
        code_lines.append("### Extra relationships")
        code_lines.append("")
        for rel_code in extra_rel:
            # Replace scene.get_object(x).property with x.property
            # Pattern: scene.get_object(identifier).property
            rel_code = re.sub(
                r'scene\.get_object\(([^)]+)\)\.(\w+)',
                r'\1.\2',
                rel_code
            )
            # Remove quotes around object names before property access
            # Pattern: 'object_name'.property or "object_name".property
            rel_code = re.sub(
                r"['\"](\w+)['\"]\.(\w+)",
                r'\1.\2',
                rel_code
            )
            # Remove quotes around object names in function arguments
            # Pattern: scene.constraint.eq('object_name'.property, ...)
            # This handles cases where quotes are around the whole
            # object.property
            rel_code = re.sub(
                r"['\"](\w+\.\w+)['\"]",
                r'\1',
                rel_code
            )
            code_lines.append(rel_code)

    code_lines.append("")
    code_lines.append("scene.solver.numerical()")
    code_lines.append("scene.plot()")

    return "\n".join(code_lines)


def generate_unique_id(
    num_objects: int, num_relationships: int, extra_rel_num: int,
    generation_id: int
) -> str:
    """
    Generate a unique ID for a generation.

    Args:
        num_objects: Number of objects in configuration
        num_relationships: Number of relationships in configuration
        extra_rel_num: Number of extra relationships in configuration
        generation_id: Generation ID within the configuration (0-indexed)

    Returns:
        Unique string ID in format
        "{num_objects}obj_{num_relationships}rel_{extra_rel_num}extra_gen{generation_id+1:04d}"
        (1-indexed for display)
    """
    # Use 1-indexed for display (gen0001, gen0002, etc.)
    return (
        f"{num_objects}obj_{num_relationships}rel_"
        f"{extra_rel_num}extra_gen{generation_id+1:04d}"
    )


def save_generation_outputs(
    unique_id: str,
    scene,
    json_data: dict,
    output_dirs: Dict[str, Path]
) -> Dict[str, str]:
    """
    Save all outputs for a single generation.

    Args:
        unique_id: Unique identifier for this generation
        scene: Generated GeoScene object
        json_data: JSON data returned from generation
        output_dirs: Dictionary of output directory paths

    Returns:
        Dictionary with file paths for all saved outputs
    """
    file_paths = {}

    # Save scene plot to image folder
    try:
        image_path = output_dirs["image"] / f"{unique_id}.png"
        fig = scene.plot(return_fig=True)
        fig.savefig(image_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        file_paths["image"] = str(image_path)
    except Exception as e:
        file_paths["image"] = f"ERROR: {str(e)}"

    return file_paths


def _generate_single_diagram(args):
    """
    Worker function for parallel generation of a single diagram.

    Args:
        args: Tuple containing (gen_id, num_objects, num_relationships,
              unique_id, output_base_dir_str, model, max_retries,
              extra_rel_num, strict_tol, distance_penalty, generation_timeout)

    Returns:
        Tuple of (unique_id, json_data, success, error_message)
    """
    (gen_id, num_objects, num_relationships, unique_id,
     output_base_dir_str, model, max_retries,
     extra_rel_num, strict_tol, distance_penalty,
     generation_timeout) = args

    # Reconstruct paths (needed for multiprocessing)
    output_base_dir = Path(output_base_dir_str)
    output_dirs = {
        "json": output_base_dir / "json",
        "image": output_base_dir / "image",
    }

    # Initialize JSON data with metadata
    json_data = {
        "unique_id": unique_id,
        "num_objects": num_objects,
        "num_relationships": num_relationships,
        "extra_rel_num": extra_rel_num,
        "generation_id": gen_id,
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "error_message": None,
        "has_overlapping_points": None
    }

    # Store generated_json_data, scene, and messages outside try block
    # to save even on failure
    generated_json_data = None
    scene = None
    messages = None

    try:
        # Generate diagram with timeout
        def run_generation():
            return generate_diagram_with_llm(
                num_objects=num_objects,
                num_relationships=num_relationships,
                extra_rel_num=extra_rel_num,
                model=model,
                max_retries=max_retries,
                strict_tol=strict_tol,
                distance_penalty=distance_penalty
            )

        # Use ThreadPoolExecutor to add timeout to generation
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_generation)
            try:
                scene, generated_json_data, possible_solution, messages = (
                    future.result(timeout=generation_timeout)
                )
            except FutureTimeoutError:
                raise TimeoutError(
                    f"Diagram generation exceeded {generation_timeout}s "
                    f"timeout. The diagram may be too complex or have "
                    f"contradictions."
                )

        # Merge generated JSON data into our JSON structure
        if generated_json_data:
            json_data.update(generated_json_data)

        # Add LLM conversation messages to JSON if available
        if messages is not None:
            json_data["llm_messages"] = messages

        # Generate PyGeoX code from JSON
        try:
            if generated_json_data:
                pygeox_code = generate_pygeox_code_from_json(
                    generated_json_data
                )
                json_data["pygeox_code"] = pygeox_code
        except Exception as e:
            # If code generation fails, leave it empty
            json_data["pygeox_code"] = f"Error generating code: {str(e)}"

        # Check for overlapping points (even if solver succeeded)
        if scene is not None:
            # Convert numpy bool_ to Python bool for JSON serialization
            json_data["has_overlapping_points"] = bool(
                scene.solver.has_any_overlapping_points()
            )

        # Save all outputs
        file_paths = save_generation_outputs(
            unique_id=unique_id,
            scene=scene,
            json_data=json_data,
            output_dirs=output_dirs
        )

        # Mark as successful
        json_data["success"] = True
        json_data["image_path"] = file_paths.get("image", "")
        json_data["possible_solution"] = possible_solution
        return (unique_id, json_data, True, None)

    except Exception as e:
        # Capture error information
        error_message = str(e)
        json_data["success"] = False
        json_data["error_message"] = error_message

        # Try to extract json_data and messages from exception
        # if it's a GenerationError
        if hasattr(e, 'json_data') and e.json_data:
            generated_json_data = e.json_data
        if hasattr(e, 'messages') and e.messages:
            messages = e.messages

        # Even on failure, try to save the JSON data if we have it
        # This ensures Points, Objs, Rels are saved even when generation fails
        if generated_json_data:
            json_data.update(generated_json_data)

            # Try to generate PyGeoX code even on failure
            try:
                pygeox_code = generate_pygeox_code_from_json(
                    generated_json_data
                )
                json_data["pygeox_code"] = pygeox_code
            except Exception:
                json_data["pygeox_code"] = None

        # Add LLM conversation messages to JSON if available
        # (even on failure)
        if messages is not None:
            json_data["llm_messages"] = messages

        # Try to check overlapping points if scene was created
        if scene is not None:
            try:
                # Convert numpy bool_ to Python bool for JSON serialization
                json_data["has_overlapping_points"] = bool(
                    scene.solver.has_any_overlapping_points()
                )
            except Exception:
                pass  # If we can't check, leave it as None

        return (unique_id, json_data, False, error_message)


def generate_data(
    config: Optional[Dict] = None,
    config_path: Optional[Path] = None
) -> None:
    """
    Generate data for all configurations.

    Args:
        config: Configuration dictionary. If None, loads from config_path.
        config_path: Path to config YAML file. Used if config is None.
    """
    # Load config if not provided
    if config is None:
        config = load_config(config_path)

    # Extract configuration values
    gen_config = config.get("generation", {})
    solver_config = config.get("solver", {})
    timeout_config = config.get("timeout", {})
    configurations = config.get("configurations", [])

    n_per_config = gen_config.get("n_per_config", 3)
    output_base_dir = Path(
        gen_config.get("output_base_dir", "generated_data")
    )
    model = gen_config.get("model", "")
    max_retries = int(gen_config.get("max_retries", 3))
    start_config = int(gen_config.get("start_config", 0))
    start_gen = int(gen_config.get("start_gen", 0))
    n_cores = int(gen_config.get("n_cores", 1))
    strict_tol = float(solver_config.get("strict_tol", 3e-5))
    distance_penalty = float(solver_config.get("distance_penalty", 0))
    generation_timeout = float(timeout_config.get("generation_timeout", 180))
    parallel_task_timeout = float(timeout_config.get("parallel_task_timeout", 360))

    # Set up output directory
    output_base_dir.mkdir(parents=True, exist_ok=True)
    output_dirs = create_output_directories(output_base_dir)

    # Determine number of cores
    if n_cores is None:
        n_cores = 1
    elif n_cores < 1:
        n_cores = 1

    # Process each configuration
    for config_id, (
        num_objects, num_relationships, extra_rel_num
    ) in enumerate(configurations):
        if config_id < start_config:
            continue

        print(f"\n{'='*60}")
        msg = (
            f"Processing Config {config_id}: {num_objects} objects, "
            f"{num_relationships} relationships, "
            f"{extra_rel_num} extra relationships"
        )
        print(msg)
        if n_cores > 1:
            print(f"Using {n_cores} cores for parallel generation")
        print(f"{'='*60}")

        # Prepare generation tasks
        tasks = []
        skipped_count = 0
        for gen_id in range(n_per_config):
            if config_id == start_config and gen_id < start_gen:
                continue

            unique_id = generate_unique_id(
                num_objects, num_relationships, extra_rel_num, gen_id
            )

            # Check if JSON file already exists
            json_path = output_dirs["json"] / f"{unique_id}.json"
            if json_path.exists():
                print(f"  Skipping {unique_id} (JSON file already exists)")
                skipped_count += 1
                continue

            print(f"  Preparing task: gen_id={gen_id}, unique_id={unique_id}")
            tasks.append((
                gen_id, num_objects, num_relationships, unique_id,
                str(output_base_dir), model, max_retries, extra_rel_num,
                strict_tol, distance_penalty, generation_timeout
            ))

        print(f"  Total tasks prepared: {len(tasks)}")
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} existing files")

        # Generate samples (sequentially or in parallel)
        if n_cores == 1 or len(tasks) == 0:
            # Sequential processing
            for task in tasks:
                unique_id, json_data, success, error_message = (
                    _generate_single_diagram(task)
                )

                if success:
                    print(f"✓ Successfully generated {unique_id}")
                else:
                    print(f"✗ Failed to generate {unique_id}: "
                          f"{error_message}")

                # Save JSON file
                json_path = output_dirs["json"] / f"{unique_id}.json"
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2,
                                  ensure_ascii=False)
                    print(f"  JSON saved to {json_path}")
                except Exception as e:
                    print(f"  ✗ Failed to save JSON: {str(e)}")
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(_generate_single_diagram, task): task
                    for task in tasks
                }

                # Process completed tasks as they finish
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    unique_id = task[3]  # unique_id is at index 3
                    try:
                        # Wait for result with timeout
                        unique_id, json_data, success, error_message = (
                            future.result(timeout=parallel_task_timeout)
                        )

                        if success:
                            print(f"✓ Successfully generated {unique_id}")
                        else:
                            print(f"✗ Failed to generate {unique_id}: "
                                  f"{error_message}")

                        # Save JSON file
                        json_path = output_dirs["json"] / f"{unique_id}.json"
                        try:
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(json_data, f, indent=2,
                                          ensure_ascii=False)
                            print(f"  JSON saved to {json_path}")
                        except Exception as e:
                            print(f"  ✗ Failed to save JSON: {str(e)}")
                    except FutureTimeoutError:
                        # Task exceeded timeout
                        print(f"✗ Timeout: {unique_id} exceeded "
                              f"{parallel_task_timeout}s, cancelling...")
                        future.cancel()
                        # Try to save a failure record
                        json_path = output_dirs["json"] / f"{unique_id}.json"
                        timeout_json_data = {
                            "unique_id": unique_id,
                            "num_objects": num_objects,
                            "num_relationships": num_relationships,
                            "extra_rel_num": extra_rel_num,
                            "generation_id": task[0],
                            "success": False,
                            "error_message": (
                                f"Task exceeded {parallel_task_timeout}s "
                                f"timeout"
                            ),
                            "timestamp": datetime.now().isoformat()
                        }
                        try:
                            with open(json_path, 'w', encoding='utf-8') as f:
                                json.dump(timeout_json_data, f, indent=2,
                                          ensure_ascii=False)
                            print(f"  Timeout record saved to {json_path}")
                        except Exception as e:
                            print(f"  ✗ Failed to save timeout record: "
                                  f"{str(e)}")
                    except Exception as e:
                        print(f"✗ Exception generating {unique_id}: {str(e)}")

    print(f"\n{'='*60}")
    print("Data generation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic geometric diagrams using config file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data_generation_config.yaml",
        help=(
            "Path to config YAML file. "
            "If not specified, looks for data_generation_config.yaml "
            "in project root."
        )
    )

    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    generate_data(config_path=config_path)
