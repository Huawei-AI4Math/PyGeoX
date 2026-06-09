"""
Generate thinking data for SFT training.

This script processes JSON files from generated_data/json and generates:
- "thinking" field (pygeox_sft_thinking) using an LLM
- "verification" field (pygeox_sft_verify) using an LLM

The LLM receives:
- nl_description: Natural language description of the geometric diagram
- pygeox_code: The PyGeoX code that implements the diagram
- pygeox_documentation: Documentation about PyGeoX API

Only processes files where "success" is True.
Outputs are saved to generated_data/json_processed/.
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .llm_client import send_prompt_to_openrouter


def load_pygeox_documentation() -> str:
    """Load PyGeoX documentation from markdown file."""
    doc_path = Path(__file__).parent / "prompts" / "pygeox_documentation.md"
    with open(doc_path, "r", encoding="utf-8") as f:
        return f.read()


def generate_thinking_prompt(
    nl_description: str,
    pygeox_code: str,
    pygeox_documentation: str
) -> str:
    """Generate the prompt for the LLM to create thinking and verification data."""
    prompt = f"""You are generating training data for a thinking model. Given a natural language description of a geometric diagram and the corresponding PyGeoX code that implements it, generate both a detailed "thinking" process and a "verification" phase.

NATURAL LANGUAGE DESCRIPTION:
{nl_description}

PYGEOX CODE:
```python
{pygeox_code}
```

PYGEOX DOCUMENTATION:
{pygeox_documentation}

TASK:
Generate two things:

### 1. FIELD "think": The Engineer's Monologue

Write a first-person, stream-of-consciousness internal monologue of a developer writing the Target Code from scratch based on the Request.

**Style & Requirements:**

  * **Mental Sandbox:** You ARE allowed to write short pseudo-code or Python snippets inside the thought process to test ideas.
  * **Explicit Doc Retrieval (CRITICAL):** Before you "decide" to use a specific PyGeoX function, you must explicitly "recall" it from the provided documentation.
      * *Bad:* "I will create a circle."
      * *Good:* "I need a circle. Checking the docs... I see `scene.add_circle(center, radius)`. It requires a Point object for the center. I should create that first."
  * **Self-Correction:** You must include 1-2 moments of self-doubt or correction.
      * *Example:* "Wait, does `add_line` take two points or a segment? Let me check the doc... Ah, it takes two points."
  * **Structure:**
    1.  **Deconstruct:** Break down the natural language description into geometric objects, relationships, and constraints (e.g., "I need a triangle ABC where AB is fixed...").
    2.  **Map to API:** Select the correct functions from the documentation.
    2.1 Show understanding of the PyGeoX API including details of the API (which methods to use, what arguments they take, implementation details)
    3.  **Order of Operations:** Explain why points come before lines, or why constraints come last.
    4.  **Final Check:** Confirm the code matches the user request.


### 2. FIELD "verify": The Auditor (Unit Test)
A verification phase that checks if the constructed diagram is correct by writing Python code to probe object properties. The verification should:
   - CRITICAL RESTRICTIONS: The Python code in verification is FORBIDDEN from using:
     * scene.relate.* (any relationship methods)
     * scene.constraint.* (any constraint methods)
     * scene.add.* (any object creation methods)
   - ONLY allowed Python constructions are:
     1. print(object.property) - e.g., print(circle1.radius), print(line_AB.length), print(square1.area)
     2. print(scene.angle(A, B, C)) - to print angle measure at vertex B between points A and C
     3. print(A.x) - to print the x-coordinate of point A
     4. print(A.y) - to print the y-coordinate of point A
   - State what the expected values should be based on the natural language description
   - Compare the printed values with expected values
   - Give a Score at the end: 0 (wrong) or 1 (correct)
   - Format: Start with "Let me check if my generated diagram is correct. If it is correct then [expected properties]"
   - Then show Python code in a code block using ONLY the allowed constructions above
   - Then show the Answer with the printed values
   - Then conclude with "So I conclude it is [correct/incorrect/partially correct]."
   - End with "Score: [0/1]"

OUTPUT FORMAT:
You must output a JSON object with exactly two fields:
{{
  "think": "<thinking text as a continuous narrative>",
  "verify": "<verification phase text with code blocks and score>"
}}

NOTE: Only generate correct thinking and verification data (Score: 1)
"""
    return prompt


def process_single_json_file(
    json_path: Path,
    output_dir: Path,
    pygeox_documentation: str,
    model: str = "openai/gpt-4o",
    max_retries: int = 3
) -> tuple[bool, str, Optional[str]]:
    """
    Process a single JSON file and generate thinking data.
    
    Args:
        json_path: Path to input JSON file
        output_dir: Directory to save processed JSON file
        pygeox_documentation: PyGeoX documentation text
        model: LLM model to use
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (success, filename, error_message)
    """
    try:
        # Check if output file already exists and has verification field
        #output_path = output_dir / json_path.name
        #if output_path.exists():
        #    try:
        #        with open(output_path, "r", encoding="utf-8") as f:
        #            existing_data = json.load(f)
        #        if "pygeox_sft_verify" in existing_data and existing_data.get("pygeox_sft_verify"):
        #            return (False, json_path.name, "already processed (has pygeox_sft_verify field)")
        #    except (json.JSONDecodeError, Exception):
        #        # If we can't read the existing file, proceed to regenerate
        #        pass
        
        #add here skip logic
        
        # Load JSON file from input
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Skip if success is not True
        if not data.get("success", False):
            return (False, json_path.name, "success field is not True")
        
        # Check if required fields exist
        if "nl_description" not in data or not data.get("nl_description"):
            return (False, json_path.name, "missing or empty nl_description field")
        
        if "pygeox_code" not in data or not data.get("pygeox_code"):
            return (False, json_path.name, "missing or empty pygeox_code field")
        
        # Generate thinking and verification using LLM
        pygeox_code = data["pygeox_code"]
        # Replace all instances of scene.get_object('A') -> A
        # Example: scene.constraint.eq(scene.angle(scene.get_object('D'), scene.get_object('A'), scene.get_object('E')), 45)
        # -> scene.constraint.eq(scene.angle(D, A, E), 45)
        pygeox_code = re.sub(r'scene\.get_object\((\w+)\)', r'\1', pygeox_code)
        prompt = generate_thinking_prompt(
            nl_description=data["nl_description"],
            pygeox_code=pygeox_code,
            pygeox_documentation=pygeox_documentation
        )
        
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
                    return (False, json_path.name, f"LLM call failed after {max_retries} attempts: {str(e)}")
                time.sleep(1)  # Brief delay before retry
        
        if response is None:
            return (False, json_path.name, "Failed to get LLM response")
        
        # Parse JSON response
        try:
            # Try to extract JSON from code blocks first
            json_match = re.search(
                r'```json\s*\n(.*?)\n```', response, re.DOTALL
            )
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no code block, try to parse entire response as JSON
                json_str = response.strip()
            
            # Clean up JSON string - remove trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            parsed_response = json.loads(json_str)
            
            if "think" not in parsed_response:
                return (False, json_path.name, "Missing 'think' field in LLM response")
            if "verify" not in parsed_response:
                return (False, json_path.name, "Missing 'verify' field in LLM response")
            
            thinking = parsed_response["think"]
            verification = parsed_response["verify"]
            
        except json.JSONDecodeError as e:
            return (False, json_path.name, f"Failed to parse LLM response as JSON: {str(e)}")
        except KeyError as e:
            return (False, json_path.name, f"Missing required field in LLM response: {str(e)}")
        
        # Add thinking and verification fields to data
        data["pygeox_sft_thinking"] = thinking
        data["pygeox_sft_verify"] = verification
        data["pygeox_code_fixed"] = pygeox_code   
        data["fixed_thinking"] = True
        
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
    max_retries: int = 3
) -> Dict[str, int]:
    """
    Process all JSON files in the input folder and generate thinking data.
    
    Args:
        input_folder: Path to folder containing input JSON files
        output_folder: Path to folder for output JSON files
        model: LLM model to use
        max_workers: Number of parallel workers
        max_retries: Maximum retry attempts per file
        
    Returns:
        Dictionary with statistics: {"total": int, "success": int, "failed": int, "skipped": int}
    """
    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load documentation once
    pygeox_documentation = load_pygeox_documentation()
    
    # Find all JSON files
    json_files = list(input_folder.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_folder}")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    
    print(f"Found {len(json_files)} JSON files to process")
    print(f"Using model: {model}")
    print(f"Max workers: {max_workers}")
    print(f"Output directory: {output_folder}")
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
                pygeox_documentation,
                model,
                max_retries
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
    #process id: 1468488
    project_root = Path(__file__).parent.parent.parent
    input_folder = project_root / "generated_data_no_penalty/json_processed_fixed"
    output_folder = project_root / "generated_data_no_penalty/json_processed_fixed_2"
    model = "/mnt/disk0/models/gpt-oss-20b"
    max_workers = 150
    max_retries = 2

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
        max_retries=max_retries
    )
