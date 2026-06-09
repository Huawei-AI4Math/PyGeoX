"""
Test synthetic data generation results.

This module processes JSON files and calculates metrics,
saving results to an Excel file.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd

# Add parent directory to path so we can import pygeox
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pygeox.synthetic.llm_client import create_scene_from_json  # noqa: E402


def process_json_files(
    json_dir: Path,
    output_dir: Optional[Path] = None
) -> None:
    """
    Process all JSON files in a directory and calculate metrics,
    saving results to a single Excel file.

    Args:
        json_dir: Directory containing JSON files
        output_dir: Directory to save Excel file
                    (default: json_dir/../generated_data)
    """
    json_dir = Path(json_dir)
    # Find all JSON files
    json_files = list(json_dir.glob("*.json"))

    print(f"Found {len(json_files)} JSON files to process")

    # List to collect all results
    all_results: List[Dict] = []

    for json_file in json_files:
        print(f"\nProcessing {json_file.name}...")

        # Initialize result row with filename
        result_row = {
            "filename": json_file.name,
            "unique_id": json_file.stem,
            "reward": None,
            "pygeox_code_success": False,
            "has_overlapping_points": None,
            "original_generation_success": None,
            "error_message": None
        }

        try:
            # Load JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Add metadata from JSON
            # Get success value - JSON already has it as boolean (True/False)
            # or None if key doesn't exist
            success_value = data.get("success")
            # Ensure it's a proper boolean (not string "true"/"false")
            # JSON.load() should already give us a boolean, but be safe
            if (success_value is not None and
                    not isinstance(success_value, bool)):
                # If somehow it's a string, convert properly
                if isinstance(success_value, str):
                    success_value = success_value.lower() == "true"
                else:
                    success_value = bool(success_value)

            result_row.update({
                "num_objects": data.get("num_objects", None),
                "num_relationships": data.get("num_relationships", None),
                "extra_rel_num": data.get("extra_rel_num", None),
                "has_overlapping_points": data.get(
                    "has_overlapping_points", None
                ),
                "original_generation_success": success_value,
            })

            # STEP 1. Extract Points, Objs, Rels from JSON and create scene
            try:
                scene = create_scene_from_json(
                    10, data, generate_objective_function=True, distance_penalty=1, min_dist = 0.025
                )
                reward, reward_dict = scene.reward.reward_function(
                    data["possible_solution"]["points"],
                    data["possible_solution"]["circles"]
                )
            except Exception as e:
                result_row["error_message"] = (
                    f"Error calculating reward: {str(e)}"
                )
                print(f"  ✗ Error calculating reward: {str(e)}")
                result_row["reward"] = None
                result_row["reward_dict"] = None
                continue

            result_row["reward"] = float(reward)
            result_row["reward_dict"] = json.dumps(reward_dict)

            # Check if data.get("pygeox_code") runs without errors
            try:
                if data.get("pygeox_code") is not None:
                    pygeox_code = data.get("pygeox_code")
                    # Remove last two lines of pygeox_code
                    pygeox_code = pygeox_code.split("\n")[:-2]
                    pygeox_code = "\n".join(pygeox_code)
                    exec(pygeox_code)
                    result_row["pygeox_code_success"] = True
                else:
                    result_row["pygeox_code_success"] = False
            except Exception as e:
                print(f"  ✗ Error executing {json_file.name}: {str(e)}")
                result_row["pygeox_code_success"] = False
                if result_row["error_message"]:
                    result_row["error_message"] += (
                        f"; PyGeoX code error: {str(e)}"
                    )
                else:
                    result_row["error_message"] = (
                        f"PyGeoX code error: {str(e)}"
                    )

        except Exception as e:
            error_msg = f"Error processing {json_file.name}: {str(e)}"
            print(f"  ✗ {error_msg}")
            result_row["error_message"] = error_msg

        # Add result to collection
        all_results.append(result_row)

    # Save all results to Excel file
    if all_results:
        try:
            # Create DataFrame
            df = pd.DataFrame(all_results)

            # Determine output path
            if output_dir is None:
                output_dir = json_dir.parent / "generated_data"
            else:
                output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save to Excel
            excel_path = output_dir / "test_results.xlsx"
            df.to_excel(excel_path, index=False, engine='openpyxl')
            print(f"\n{'='*60}")
            print(f"All results saved to {excel_path}")
            print(f"Total files processed: {len(all_results)}")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"Error saving Excel file: {str(e)}")
            print(f"{'='*60}\n")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n{'='*60}")
        print("No results to save.")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    process_json_files(Path("generated_data/json"))
