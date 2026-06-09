"""
Generate thinking data for SFT training with multiple approaches.

This script processes JSON files from generated_data/json and generates:
- "constructive_approach_data" using constructive prompt
- "code_approach_data" using generate_code_prompt
- "R1_approach_data" using general_thinking_prompt (skipped for now)

Each approach data contains: "think", "answer", "verify" fields.

Only processes files where "success" is True.
Outputs are saved to generated_data/json_processed_approaches/.
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .llm_client import send_prompt_to_openrouter


def load_prompt_file(filename: str) -> str:
    """Load a prompt file from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / filename
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def hide_numerical_values(obj):
    """Recursively replace numerical values with None while preserving structure."""
    if isinstance(obj, dict):
        return {key: hide_numerical_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # For lists, replace all numeric values with None
        return [None if isinstance(item, (int, float)) else hide_numerical_values(item) for item in obj]
    elif isinstance(obj, (int, float)):
        return None
    else:
        return obj


def get_ground_truth_json(data: Dict) -> str:
    """Extract or construct ground truth JSON from data, hiding numerical values."""
    # Try to get ground truth from data if it exists
    if "ground_truth" in data:
        hidden_truth = hide_numerical_values(data["ground_truth"])
        return json.dumps(hidden_truth, indent=2)
    
    # Otherwise, construct from points and circles if available
    ground_truth = {}
    if "points" in data:
        ground_truth["points"] = data["points"]
    if "circles" in data:
        ground_truth["circles"] = data["circles"]
    
    if ground_truth:
        hidden_truth = hide_numerical_values(ground_truth)
        return json.dumps(hidden_truth, indent=2)
    
    # Fallback: return empty structure
    return json.dumps({"points": {}, "circles": {}}, indent=2)


def build_constructive_prompt(
    nl_description: str,
    ground_truth_json: str,
    base_prompt: str
) -> str:
    """Build the constructive approach prompt."""
    prompt = f"""{base_prompt}

## INPUT
{nl_description}

##  ANSWER FORMAT:
```json
{ground_truth_json}
```

Please solve this problem using constructive methods and provide your response in the required JSON format.
"""
    return prompt


def build_code_prompt(
    nl_description: str,
    ground_truth_json: str,
    base_prompt: str
) -> str:
    """Build the code/optimization approach prompt."""
    prompt = f"""{base_prompt}

## INPUT
[PROBLEM DESCRIPTION]:
{nl_description}

[ANSWER FORMAT]:
```json
{ground_truth_json}
```

Please solve this problem using numerical optimization and provide your response in the required JSON format.
"""
    return prompt


def build_r1_prompt(
    nl_description: str,
    ground_truth_json: str,
    base_prompt: str
) -> str:
    """Build the R1/general thinking prompt."""
    prompt = f"""{base_prompt}

## INPUT
{nl_description}

## ANSWER FORMAT:
```json
{ground_truth_json}
```

Please solve this problem with extensive reasoning and provide your response in the required JSON format.
"""
    return prompt


def parse_llm_response(response: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Parse LLM response to extract JSON with think, answer, verify fields.
    
    Returns:
        Tuple of (parsed_dict, error_message)
    """
    try:
        # Try to extract JSON from code blocks first
        json_match = re.search(
            r'```json\s*\n(.*?)\n```', response, re.DOTALL
        )
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object in the response
            json_match = re.search(
                r'\{[\s\S]*"think"[\s\S]*"answer"[\s\S]*"verify"[\s\S]*\}', response, re.DOTALL
            )
            if json_match:
                json_str = json_match.group(0)
            else:
                # If no code block, try to parse entire response as JSON
                json_str = response.strip()
        
        # Clean up JSON string - remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        parsed_response = json.loads(json_str)
        
        # Validate required fields
        if "think" not in parsed_response:
            return None, "Missing 'think' field in LLM response"
        if "answer" not in parsed_response:
            return None, "Missing 'answer' field in LLM response"
        if "verify" not in parsed_response:
            return None, "Missing 'verify' field in LLM response"
        
        # Convert answer to string if it's a JSON object
        if isinstance(parsed_response["answer"], (dict, list)):
            parsed_response["answer"] = json.dumps(parsed_response["answer"], indent=2)
        
        # Check if fields are non-empty
        if not str(parsed_response.get("think", "")).strip():
            return None, "Empty 'think' field in LLM response"
        if not str(parsed_response.get("answer", "")).strip():
            return None, "Empty 'answer' field in LLM response"
        if not str(parsed_response.get("verify", "")).strip():
            return None, "Empty 'verify' field in LLM response"
        
        return parsed_response, None
        
    except json.JSONDecodeError as e:
        return None, f"Failed to parse LLM response as JSON: {str(e)}"
    except Exception as e:
        return None, f"Error parsing response: {str(e)}"


def generate_approach_data(
    prompt: str,
    model: str,
    max_retries: int = 3
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Generate approach data by calling LLM with the given prompt.
    
    Returns:
        Tuple of (parsed_response_dict, error_message)
    """
    response = None
    for attempt in range(max_retries):
        try:
            response = send_prompt_to_openrouter(
                prompt=prompt,
                model=model
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                return None, f"LLM call failed after {max_retries} attempts: {str(e)}"
            time.sleep(1)  # Brief delay before retry
    
    if response is None:
        return None, "Failed to get LLM response"
    
    return parse_llm_response(response)


def process_single_json_file(
    json_path: Path,
    output_dir: Path,
    constructive_prompt_base: str,
    code_prompt_base: str,
    r1_prompt_base: Optional[str],
    model: str = "openai/gpt-4o",
    max_retries: int = 3,
    skip_r1: bool = True
) -> tuple[bool, str, Optional[str]]:
    """
    Process a single JSON file and generate approach data.
    
    Args:
        json_path: Path to input JSON file
        output_dir: Directory to save processed JSON file
        constructive_prompt_base: Base prompt for constructive approach
        code_prompt_base: Base prompt for code/optimization approach
        r1_prompt_base: Base prompt for R1 approach (optional)
        model: LLM model to use
        max_retries: Maximum number of retry attempts
        skip_r1: Whether to skip R1 approach generation
        
    Returns:
        Tuple of (success, filename, error_message)
    """
    try:
        # Check if output file already exists and has all required fields
        output_path = output_dir / json_path.name
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                has_constructive = "constructive_approach_data" in existing_data and existing_data.get("constructive_approach_data")
                has_code = "code_approach_data" in existing_data and existing_data.get("code_approach_data")
                if has_constructive and has_code:
                    return (False, json_path.name, "already processed (has approach data fields)")
            except (json.JSONDecodeError, Exception):
                # If we can't read the existing file, proceed to regenerate
                pass
        
        # Load JSON file from input
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Skip if success is not True
        if not data.get("success", False):
            return (False, json_path.name, "success field is not True")
        
        # Check if required fields exist
        if "nl_description" not in data or not data.get("nl_description"):
            return (False, json_path.name, "missing or empty nl_description field")
        
        # Get ground truth JSON
        ground_truth_json = get_ground_truth_json(data)
        
        # Generate constructive approach data
        constructive_prompt = build_constructive_prompt(
            nl_description=data["nl_description"],
            ground_truth_json=ground_truth_json,
            base_prompt=constructive_prompt_base
        )
        
        constructive_data, error = generate_approach_data(
            prompt=constructive_prompt,
            model=model,
            max_retries=max_retries
        )
        if constructive_data is None:
            return (False, json_path.name, f"constructive approach failed: {error}")
        
        # Generate code approach data
        code_prompt = build_code_prompt(
            nl_description=data["nl_description"],
            ground_truth_json=ground_truth_json,
            base_prompt=code_prompt_base
        )
        
        code_data, error = generate_approach_data(
            prompt=code_prompt,
            model=model,
            max_retries=max_retries
        )
        if code_data is None:
            return (False, json_path.name, f"code approach failed: {error}")
        
        # Generate R1 approach data (if not skipped)
        r1_data = None
        if not skip_r1 and r1_prompt_base:
            r1_prompt = build_r1_prompt(
                nl_description=data["nl_description"],
                ground_truth_json=ground_truth_json,
                base_prompt=r1_prompt_base
            )
            
            r1_data, error = generate_approach_data(
                prompt=r1_prompt,
                model=model,
                max_retries=max_retries
            )
            if r1_data is None:
                # R1 failure is not fatal, just log it
                print(f"Warning: R1 approach failed for {json_path.name}: {error}")
        
        # Add approach data fields to data
        data["constructive_approach_data"] = constructive_data
        data["code_approach_data"] = code_data
        if r1_data is not None:
            data["R1_approach_data"] = r1_data
        
        # Save to output directory
        output_path = output_dir / json_path.name
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return (True, json_path.name, None)
    
    except json.JSONDecodeError as e:
        return (False, json_path.name, f"JSON decode error: {str(e)}")
    except Exception as e:
        return (False, json_path.name, f"Unexpected error: {str(e)}")


def process_json_files(
    input_folder: Path,
    output_folder: Path,
    model: str = "openai/gpt-4o",
    max_workers: int = 10,
    max_retries: int = 3,
    skip_r1: bool = True
) -> Dict[str, int]:
    """
    Process all JSON files in the input folder and generate approach data.
    
    Args:
        input_folder: Path to folder containing input JSON files
        output_folder: Path to folder for output JSON files
        model: LLM model to use
        max_workers: Number of parallel workers
        max_retries: Maximum retry attempts per file
        skip_r1: Whether to skip R1 approach generation
        
    Returns:
        Dictionary with statistics: {"total": int, "success": int, "failed": int, "skipped": int}
    """
    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load prompt files once
    constructive_prompt_base = load_prompt_file("constructive prompt.md")
    code_prompt_base = load_prompt_file("generate_code_prompt.md")
    r1_prompt_base = None
    if not skip_r1:
        try:
            r1_prompt_base = load_prompt_file("general_thinking_prompt.md")
        except FileNotFoundError:
            print("Warning: general_thinking_prompt.md not found, skipping R1 approach")
            skip_r1 = True
    
    # Find all JSON files
    json_files = list(input_folder.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_folder}")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    
    print(f"Found {len(json_files)} JSON files to process")
    print(f"Using model: {model}")
    print(f"Max workers: {max_workers}")
    print(f"Output directory: {output_folder}")
    print(f"Skip R1: {skip_r1}")
    print("-" * 80)
    
    stats = {"total": len(json_files), "success": 0, "failed": 0, "skipped": 0, "already_processed": 0}
    completed = 0
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                process_single_json_file,
                json_path,
                output_folder,
                constructive_prompt_base,
                code_prompt_base,
                r1_prompt_base,
                model,
                max_retries,
                skip_r1
            ): json_path
            for json_path in json_files
        }
        
        # Process results as they complete
        for future in as_completed(future_to_file):
            json_path = future_to_file[future]
            try:
                success, filename, error = future.result()
                completed += 1
                
                if success:
                    stats["success"] += 1
                    print(f"[{completed}/{stats['total']}] ✓ {filename}")
                else:
                    if error and "success field is not True" in error:
                        stats["skipped"] += 1
                        print(f"[{completed}/{stats['total']}] ⊘ {filename} (skipped: {error})")
                    elif error and "already processed" in error:
                        stats["already_processed"] += 1
                        print(f"[{completed}/{stats['total']}] ⊘ {filename} (already processed)")
                    else:
                        stats["failed"] += 1
                        print(f"[{completed}/{stats['total']}] ✗ {filename} (error: {error})")
            except Exception as e:
                stats["failed"] += 1
                completed += 1
                print(f"[{completed}/{stats['total']}] ✗ {json_path.name} (exception: {str(e)})")
    
    print("-" * 80)
    print(f"Processing complete!")
    print(f"Total: {stats['total']}")
    print(f"Success: {stats['success']}")
    print(f"Already processed: {stats['already_processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    
    return stats


if __name__ == "__main__":

    #I think best to use a big model - R1 distill....  qwen3-120b-a22b-instruct-fp8

    project_root = Path(__file__).parent.parent.parent
    input_folder = project_root / "generated_data_penalty/json"
    output_folder = project_root / "generated_data_penalty/json_processed_2"
    model = "/mnt/disk0/models/Qwen3-Coder-30B-A3B-Instruct-FP8"  
    max_workers = 10
    max_retries = 2
    skip_r1 = True  # Skip R1 as requested

    # Resolve relative to project root if needed
    if not input_folder.is_absolute():
        # Get project root (parent of pygeox folder)
        project_root = Path(__file__).parent.parent.parent
        input_folder = (project_root / input_folder).resolve()
        output_folder = (project_root / output_folder).resolve()
    
    if not input_folder.exists():
        print(f"Error: Input folder does not exist: {input_folder}")
        exit(1)
    

    # Process files in parallel
    stats = process_json_files(
        input_folder=input_folder,
        output_folder=output_folder,
        model=model,
        max_workers=max_workers,
        max_retries=max_retries,
        skip_r1=skip_r1
    )
