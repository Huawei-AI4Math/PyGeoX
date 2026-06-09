"""
Python code generation from natural language descriptions.

This module provides functionality to generate Python code that calculates
geometric coordinates from natural language descriptions using LLM.
"""

import sys
import json
import re
import io
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add parent directory to path so we can import pygeox
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pygeox.synthetic.llm_client import (  # noqa: E402
    send_prompt_to_openrouter,
    create_scene_from_json
)


def generate_python_code_from_description(
    nl_description: str,
    model: str = "google/gemini-2.0-flash-exp",
    points_info: Optional[Dict[str, Tuple[float, float]]] = None,
    circles_info: Optional[Dict[str, float]] = None,
    max_rounds: int = 5
) -> Tuple[list, Dict[str, Tuple[float, float]], Dict[str, float]]:
    """
    Generate Python code from natural language description and execute it
    to extract points and circles. Supports multi-round execution.

    Args:
        nl_description: Natural language description of the diagram
        model: LLM model to use (default: google/gemini-2.0-flash-exp)
        points_info: Optional dict mapping point names to (x, y) coordinates
        circles_info: Optional dict mapping circle center names to radii
        max_rounds: Maximum number of execution rounds (default: 5)

    Returns:
        Tuple of (messages, extracted_points, extracted_circles)
        where messages is the entire conversation history
    """

    # Build expected output format based on provided point and circle names (not values)
    point_names = sorted(points_info.keys()) if points_info else []
    circle_names = sorted(circles_info.keys()) if circles_info else []

    example_points = (
        "points = {" + ", ".join([f'"{p}": [x, y]' for p in point_names]) + "}"
        if point_names else "points = {}"
    )
    example_circles = (
        "circles = {" + ", ".join([f'"{c}": r' for c in circle_names]) + "}"
        if circle_names else "circles = {} #EMPTY! Do not fill."
    )
    ref_info = (
        "\nExpected output format:\n"
        "```python\n"
        f"{example_points}\n"
        f"{example_circles}\n"
        "```"
    )

    initial_prompt = f"""Please generate Python code that calculates the coordinates of points and radii of circles for a geometry diagram described as:

{nl_description}

{ref_info}

This is a multi-round conversation with up to {max_rounds} rounds. You can use print() statements to show your intermediate calculation steps, which will be visible in subsequent rounds.

REQUIREMENTS:
1. You can use: math, numpy (as np), sympy, scipy
2. You MUST create two variables:
   - `points` = a dictionary mapping point names to [x, y] coordinates
   - `circles` = a dictionary mapping circle center names to radius values
3. Format: Output the Python code wrapped in ```python ``` code blocks.

EXAMPLE OUTPUT:
```python
import math
import numpy as np

# Calculate coordinates
points = {"A": [100, 150], "B": [200, 150], "O": [150, 200]}
circles = {"O": 100}
```

IMPORTANT:
- Please note that the domain of the problem is [-10, 10] for both x and y coordinates. Do not consider points outside this domain.
- Please note that there should not be overlapping points.
- The `points` dictionary must map point names (strings) to lists of [x, y] coordinates
- The `circles` dictionary must map circle center names (strings) to radius values (floats)
- Use the EXACT point names and circle center names listed above
- You can use any mathematical calculations needed to determine the coordinates
- You can use print() statements to show intermediate steps if needed"""

    messages = [{"role": "user", "content": initial_prompt}]

    for round_num in range(max_rounds):
        # Get response from LLM
        response = send_prompt_to_openrouter(
            messages=messages, model=model
        )
        messages.append({"role": "assistant", "content": response})

        # Extract Python code from response
        python_code = None
        python_match = re.search(
            r'```python\s*\n(.*?)\n```', response, re.DOTALL
        )
        if python_match:
            python_code = python_match.group(1)
        else:
            # Try without the python tag
            python_match = re.search(
                r'```\s*\n(.*?)\n```', response, re.DOTALL
            )
            if python_match:
                python_code = python_match.group(1)
            else:
                # If no code block, try to use entire response
                python_code = response.strip()

        # Try to execute the code and extract points and circles
        try:
            # Create a local namespace for execution
            local_namespace = {
                'math': __import__('math'),
                'numpy': __import__('numpy'),
                'np': __import__('numpy'),
                'sympy': __import__('sympy'),
                'scipy': __import__('scipy')
            }

            # Capture stdout to get print outputs
            stdout_capture = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout_capture

            try:
                # Execute the code
                exec(
                    python_code,
                    {"__builtins__": __builtins__},
                    local_namespace
                )
            finally:
                # Restore stdout
                sys.stdout = old_stdout

            # Get the captured print output
            print_output = stdout_capture.getvalue()

            # Check if points and circles exist
            if ('points' in local_namespace and
                    'circles' in local_namespace):

                # Success - return the messages and extracted data
                return (
                    messages,
                    local_namespace['points'],
                    local_namespace['circles']
                )
            else:
                # Variables not found, continue to next round
                print(
                    f"  Round {round_num + 1}: Final objects not found, "
                    f"continue thinking..."
                )
                if print_output:
                    print(f"  Print output from code:\n{print_output}")

                if round_num < max_rounds - 1:
                    # Build continue message with print output if available
                    if print_output:
                        continue_message = (
                            f"""Round {round_num + 2} of {max_rounds}: The """
                            f"""Python code was executed, but the required """
                            f"""variables `points` and `circles` were not """
                            f"""found in the code.

Print output from the code execution:
```
{print_output}
```

Please continue thinking and when you finish, add the required """
                            f"""variables `points` and `circles` to your """
                            f"""Python code."""
                        )
                    else:
                        continue_message = (
                            f"""Round {round_num + 2} of {max_rounds}: The """
                            f"""Python code was executed, but the required """
                            f"""variables `points` and `circles` were not """
                            f"""found in the code.

Please continue thinking and when you finish, add the required """
                            f"""variables `points` and `circles` to your """
                            f"""Python code."""
                        )

                    messages.append(
                        {"role": "user", "content": continue_message}
                    )
                    continue
                else:
                    # Last round failed
                    print(f"  Failed after {max_rounds} rounds")
                    return messages, {}, {}

        except Exception as e:
            # Restore stdout in case of exception
            if 'old_stdout' in locals():
                sys.stdout = old_stdout

            # Try to get any print output that was captured before error
            print_output = ""
            if 'stdout_capture' in locals():
                print_output = stdout_capture.getvalue()

            # Execution error, ask LLM to fix it
            print(
                f"  Round {round_num + 1}: Execution error: {str(e)}"
            )
            if print_output:
                print(
                    f"  Print output from code before error:\n{print_output}"
                )

            if round_num < max_rounds - 1:
                if print_output:
                    error_message = (
                        f"""Round {round_num + 2} of {max_rounds}: The """
                        f"""Python code had an execution error:

{str(e)}

Print output from the code execution (before error):
```
{print_output}
```

Please fix the code and ensure:
1. All imports are correct
2. The code executes without errors
3. Both `points` and `circles` variables are created at the end"""
                    )
                else:
                    error_message = (
                        f"""Round {round_num + 2} of {max_rounds}: The """
                        f"""Python code had an execution error:

{str(e)}

Please fix the code and ensure:
1. All imports are correct
2. The code executes without errors
3. Both `points` and `circles` variables are created at the end"""
                    )

                messages.append(
                    {"role": "user", "content": error_message}
                )
                continue
            else:
                # Last round failed
                print(f"  Failed after {max_rounds} rounds")
                return messages, {}, {}

    # Should not reach here, but just in case
    if 'messages' in locals():
        return messages, {}, {}
    else:
        return [], {}, {}


def process_json_files_for_python(
    json_dir: Path,
    output_code_dir: Optional[Path] = None,
    model: str = "google/gemini-2.5-flash"
) -> None:
    """
    Process all JSON files in a directory and generate Python code.

    Args:
        json_dir: Directory containing JSON files
        output_code_dir: Directory to save Python code files
                      (default: json_dir/../agent_code)
        model: LLM model to use
    """
    json_dir = Path(json_dir)
    if output_code_dir is None:
        output_code_dir = json_dir.parent / "agent_code"
    else:
        output_code_dir = Path(output_code_dir)

    output_code_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = list(json_dir.glob("*.json"))

    print(f"Found {len(json_files)} JSON files to process")
    print(f"Output directory: {output_code_dir}")

    for json_file in json_files:
        print(f"\nProcessing {json_file.name}...")

        try:
            # Load JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract nl_description
            nl_description = data.get("nl_description", "")
            if not nl_description:
                print(f"  ✗ No nl_description found in {json_file.name}")
                continue

            # if solver did not initially work during data generation, then skip
            if not data.get("success"):
                print(f"  ✗ Solver did not initially work during data generation. Skipping {json_file.name}")
                continue

            # Extract possible_solution if available
            possible_solution = data.get("possible_solution", {})
            points_info = possible_solution.get("points", {})
            circles_info = possible_solution.get("circles", {})

            # STEP 1. Generate Python code and extract points/circles
            messages, extracted_points, extracted_circles = (
                generate_python_code_from_description(
                    nl_description=nl_description,
                    model=model,
                    points_info=points_info if points_info else None,
                    circles_info=circles_info if circles_info else None
                )
            )

            # Initialize reward
            reward = -5

            if not extracted_points and not extracted_circles:
                print("  ✗ Error: Failed to extract points and circles")
            elif (not isinstance(extracted_points, dict) or
                  not isinstance(extracted_circles, dict)):
                print("  ✗ Error: Failed to extract points and circles")
            else:
                print(f"  Extracted points: {extracted_points}")
                print(f"  Extracted circles: {extracted_circles}")

                # STEP 2. Extract correct Points, Objs, Rels from JSON
                scene = create_scene_from_json(
                    10, data, generate_objective_function=True,
                    distance_penalty=1, min_dist=0.025
                )

                # STEP 3. Calculate reward
                reward, reward_dict = scene.reward.reward_function(
                    extracted_points, extracted_circles
                )

                print(f"  Reward: {reward}")

            # Convert points from tuples to lists for JSON serialization
            if isinstance(extracted_points, dict):
                points_for_json = {
                    name: [float(coord[0]), float(coord[1])]
                    for name, coord in extracted_points.items()
                }
            else:
                points_for_json = {}

            # Prepare JSON output with conversation, results, and reward
            output_data = {
                "conversation": messages,
                "results": {
                    "points": points_for_json,
                    "circles": {
                        name: float(radius)
                        for name, radius in extracted_circles.items()
                    } if isinstance(extracted_circles, dict) else {}
                },
                "reward": reward,
                "reward_dict": reward_dict
            }

            # Save conversation and results to JSON file
            code_dir = output_code_dir
            code_dir.mkdir(parents=True, exist_ok=True)
            code_file = code_dir / f"{json_file.stem}.json"
            with open(code_file, "w", encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)


        except Exception as e:
            print(f"  ✗ Error processing {json_file.name}: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Python code generation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Python code from JSON descriptions"
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        default="generated_data/json",
        help="Directory containing JSON files (default: generated_data/json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_data/agent_code",
        help=(
            "Directory to save Python code files "
            "(default: json_dir/../agent_code)"
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-2.5-flash",
        help="LLM model to use (default: google/gemini-2.5-flash)"
    )

    args = parser.parse_args()

    process_json_files_for_python(
        json_dir=Path(args.json_dir),
        output_code_dir=Path(args.output_dir) if args.output_dir else None,
        model=args.model
    )
