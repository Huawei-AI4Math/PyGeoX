"""
Regenerate nl_description fields in JSON files.

This script processes JSON files from generated_data_no_penalty/json_processed
and regenerates the nl_description field by converting the computer language
(Objs and Rels) into natural language using an LLM.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from .llm_client import send_prompt_to_openrouter
from .definitions import OBJ_DICT, REL_DICT


def _extract_object_types(objs: Dict[str, str]) -> Set[str]:
    """Extract object type names from Objs dictionary."""
    obj_types = set()
    for obj_value in objs.values():
        # Extract the type name before the opening parenthesis
        # e.g., "circle(A)" -> "circle", "line_segment(B, D)" -> "line_segment"
        match = re.match(r'^(\w+)\(', obj_value)
        if match:
            obj_type = match.group(1)
            obj_types.add(obj_type)
    return obj_types


def _extract_relationship_types(rels: list) -> Set[str]:
    """Extract relationship type names from Rels list."""
    rel_types = set()
    for rel in rels:
        # Extract the relationship name before the opening parenthesis
        # e.g., "is_diameter(line_BD, circle1)" -> "is_diameter"
        match = re.match(r'^(\w+)\(', rel)
        if match:
            rel_type = match.group(1)
            rel_types.add(rel_type)
    return rel_types


def build_nl_description_prompt(objs: Dict[str, str], rels: list, points: list, extra_rel: list, old_nl_description: str = "") -> str:
    """Build a prompt to convert Objs and Rels into natural language."""
    objs_str = json.dumps(objs, indent=2)
    rels_str = json.dumps(rels, indent=2)
    points_str = json.dumps(points, indent=2)
    extra_rel_str = json.dumps(extra_rel, indent=2)
    
    old_desc_section = ""
    if old_nl_description:
        old_desc_section = f"""
CURRENT DESCRIPTION (to evaluate):
{old_nl_description}
"""
    
    extra_rel_section = ""
    if extra_rel:
        extra_rel_section = f"""
EXTRA RELATIONSHIPS (extra_rel):
{extra_rel_str}
"""
    
    points_list_str = ", ".join(points) if points else "none"
    
    # Extract only the object types that are actually used
    used_obj_types = _extract_object_types(objs)
    object_definitions_lines = []
    for obj_type in sorted(used_obj_types):
        if obj_type in OBJ_DICT:
            docstring = OBJ_DICT[obj_type]["docstring"]
            object_definitions_lines.append(f"- {obj_type}: {docstring}")
    
    object_definitions = ""
    if object_definitions_lines:
        object_definitions = """
AVAILABLE GEOMETRIC OBJECTS (use these definitions only):
""" + "\n".join(object_definitions_lines)
    
    # Extract only the relationship types that are actually used
    used_rel_types = _extract_relationship_types(rels)
    relationship_definitions_lines = []
    for rel_type in sorted(used_rel_types):
        if rel_type in REL_DICT:
            docstring = REL_DICT[rel_type]["docstring"]
            # Clean up the docstring - normalize whitespace but keep the full content
            # Replace multiple newlines with single newline, strip leading/trailing whitespace
            docstring_clean = re.sub(r'\n\s*\n', '\n', docstring).strip()
            relationship_definitions_lines.append(f"- {rel_type}: {docstring_clean}")
    
    relationship_definitions = ""
    if relationship_definitions_lines:
        relationship_definitions = """
AVAILABLE RELATIONSHIPS (use these definitions only):
""" + "\n".join(relationship_definitions_lines)

    # Constraint definitions from pygeox_documentation.md (only if extra_rel exists)
    constraint_definitions = ""
    if extra_rel:
        constraint_definitions = """
ADDITIONAL CONSTRAINT OPERATIONS (for extra_rel):
- scene.constraint.eq(object1.property, object2.property): Constrains two expressions to be equal
- scene.constraint.eq(object1.property, 5.0): Constrains an expression to equal a specific value
- scene.constraint.geq(object1.property, value): Constrains expression to be >= value
- scene.constraint.leq(object1.property, value): Constrains expression to be <= value
- scene.constraint.eq(scene.angle(A, B, C), 45): Constrains an angle to equal a specific value in degrees
"""

    # Build the definitions section
    definitions_section = ""
    if object_definitions:
        definitions_section += object_definitions + "\n"
    if relationship_definitions:
        definitions_section += relationship_definitions + "\n"
    if constraint_definitions:
        definitions_section += constraint_definitions

    prompt = f"""You are given a geometric diagram described in computer language. 
Your task if to generate a natural language description of the diagram. The natural language must describe the points, objects, relationships, and extra relationships.

POINTS:
{points_str}

GEOMETRIC OBJECTS (Objs):
{objs_str}

RELATIONSHIPS (Rels):
{rels_str}

{extra_rel_section}
{definitions_section}

EXAMPLES OF CORRECT FORMAT - FOLLOW THESE EXACTLY:

Example 1:
Input:
  Points: ["A", "B", "C", "D"]
  Objs: {{"circle1": "circle(A)", "line_BD": "line_segment(B, D)", "line_AC": "line_segment(A, C)"}}
  Rels: ["congruent(line_BD, line_AC)", "perpendicular(line_BD, line_AC)"]
  extra_rel: ["scene.constraint.eq(scene.get_object('line_AC').length, scene.get_object('line_BD').length)", "scene.constraint.eq(scene.get_object('circle1').diameter, 2 * scene.get_object('line_AC').length)"]

Correct Output:
  "Diagram description: The diagram contains points A, B, C, D. There is a circle with center A. There are line segments BD and AC. Line BD is congruent to line AC. Line BD is perpendicular to line AC. Further, length of line AC is equal to length of line BD and the diameter of the circle is 2 times the length of line AC."

Example 2:
Input:
  Points: ["A", "B", "C", "D", "E", "F"]
  Objs: {{"circle1": "circle(A)", "line1": "line_segment(B, C)", "line2": "line_segment(A, D)", "line3": "line_segment(E, F)"}}
  Rels: ["perpendicular(line1, line2)", "line_intersects_circle_at(line1, circle1, B, C)", "point_lies_on(D, circle1)"]
  extra_rel: []

Correct Output:
  "Diagram description: The diagram contains points A, B, C, D, E, F. There is a circle with center A. There are line segments BC, AD, and EF. Line BC is perpendicular to line AD. Line BC intersects the circle at points B and C. Point D lies on the circle."

CRITICAL FORMATTING REQUIREMENTS - YOU MUST FOLLOW THIS EXACT STRUCTURE:

The natural language description MUST follow this strict format in this exact order. Follow the examples above precisely:

1. START with: "Diagram description: The diagram contains points {points_list_str}."

2. Then list ALL geometric objects from the Objs section, one by one:
   - For each object, state: "There is a [object_type] with [parameters]." or "There are [object_type]s [descriptive_names]."
   - When describing line segments, use the point names: "line_segment(B, C)" should be described as "line segment BC" or "line BC"
   - Use the object definitions above to understand what each object type means, but describe them directly without adding extra information.

3. Then list ALL relationships from the Rels section, one by one:
   - IMPORTANT: When referring to objects in relationships, use their DESCRIPTIVE NAMES based on their point parameters, NOT their object names.
   - For example: "line_segment(B, C)" should be referred to as "line BC" or "line segment BC", NOT as "line1"
   - For example: "circle(A)" should be referred to as "the circle" or "circle with center A", NOT as "circle1"
   - Use the relationship definitions above to understand what each relationship type means, but describe them directly.

4. Finally, if there are any extra relationships in the extra_rel section, describe them:
   - State the constraint directly: "Further, [constraint description]."
   - When referring to objects, use their descriptive names (e.g., "line AC" not "line_AC")
   - Use the constraint definitions above to understand the constraints.

IMPORTANT RULES - BE PRECISE:
- DO NOT hallucinate objects, relationships, or constraints that are not in the provided data
- DO NOT add extra geometric facts or properties not explicitly stated
- DO NOT ask questions or provide analysis
- DO NOT add interpretations, implications, or derived properties
- ONLY use the object types, relationship types, and constraint types defined above
- Follow the exact order: points → objects → relationships → extra relationships
- Use direct language - just state what is there, nothing more
- Make MINIMAL adaptations - only convert the computer language to natural language, do not add information
- When describing relationships, ALWAYS use descriptive names based on point parameters (e.g., "line BC", "circle with center A"), NEVER use object names from Objs (e.g., "line1", "circle1")
- Follow the examples above exactly - use the same style and structure
- Do not deviate from this structure

Output your response in the following JSON format:
```json
{{
    "new_nl_description": "Diagram description: The diagram contains points {points_list_str}. ..."
}}
```
"""
    return prompt


def regenerate_nl_description(
    json_data: Dict,
    model: str,
    max_retries: int = 2
) -> Tuple[bool, Optional[str]]:
    """
    Regenerate nl_description from Objs and Rels using LLM.
    
    Args:
        json_data: The JSON data containing Objs and Rels
        model: The model to use
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (success, result_dict) where result_dict contains
        nl_description_correct and new_nl_description
    """
    objs = json_data.get("Objs", {})
    rels = json_data.get("Rels", [])
    points = json_data.get("Points", [])
    extra_rel = json_data.get("extra_rel", [])
    old_nl_description = json_data.get("nl_description", "")
    
    if not objs or not rels:
        return False, None
    
    prompt = build_nl_description_prompt(objs, rels, points, extra_rel, old_nl_description)
    messages = [{"role": "user", "content": prompt}]
    
    for attempt in range(max_retries):
        try:
            response = send_prompt_to_openrouter(messages=messages, model=model)
            messages.append({"role": "assistant", "content": response})
            
            # Extract JSON from response
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
            
            try:
                result = json.loads(json_str)
                nl_description_correct = result.get("nl_description_correct", False)
                new_nl_description = result.get("new_nl_description", None)
                
                return True, {
                    "nl_description_correct": nl_description_correct,
                    "new_nl_description": new_nl_description
                }
            except json.JSONDecodeError as e:
                if attempt == max_retries - 1:
                    print(f"JSON parsing error after {max_retries} attempts: {str(e)}")
                    print(f"Response: {response[:500]}...")
                    return False, None
                
                # Retry with error message
                retry_message = f"""PREVIOUS RESPONSE:
{response}

ERROR:
Invalid JSON format. Please provide a valid JSON response with the exact format:
```json
{{
    "nl_description_correct": true/false,
    "new_nl_description": null or "Diagram description: ..."
}}
```
"""
                messages.append({"role": "user", "content": retry_message})
                continue
                
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error in regenerate_nl_description: {str(e)}")
                return False, None
            
            # Retry
            retry_message = f"""An error occurred: {str(e)}

Please try again with a valid JSON response."""
            messages.append({"role": "user", "content": retry_message})
            continue
    
    return False, None


def process_single_file(
    file_path: Path,
    output_folder: Path,
    model: str,
    max_retries: int = 2
) -> Dict:
    """
    Process a single JSON file to regenerate nl_description.
    
    Args:
        file_path: Path to input JSON file
        output_folder: Path to output folder for modified files
        model: The model to use
        max_retries: Maximum retry attempts
        
    Returns:
        Dictionary with processing results
    """
    try:
        # Read JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # Write to output folder
        output_folder.mkdir(parents=True, exist_ok=True)
        output_file = output_folder / file_path.name
        
        #if old description not correct, skip (already processed)
        if not json_data["old_nl_description_correct"]:
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            return {
                "file": str(file_path),
                "status": "skipped",
                "message": "Already processed (old_nl_description_correct key exists)",
                "updated": False
            }
        
        # Regenerate description
        success, result = regenerate_nl_description(json_data, model, max_retries)
        
        if not success:
            return {
                "file": str(file_path),
                "status": "error",
                "message": "Failed to regenerate description"
            }
        
        nl_description_correct = result["nl_description_correct"]
        new_nl_description = result["new_nl_description"]
        
        # Update JSON data
        if new_nl_description is not None:
            json_data["nl_description"] = new_nl_description
        # else: keep the old description
        
        # Add old_nl_description_correct field
        json_data["old_nl_description_correct"] = False #nl_description_correct
    
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return {
            "file": str(file_path),
            "status": "success",
            "nl_description_correct": nl_description_correct,
            "updated": new_nl_description is not None
        }
        
    except Exception as e:
        return {
            "file": str(file_path),
            "status": "error",
            "message": str(e)
        }


def process_json_files(
    input_folder: Path,
    output_folder: Path,
    model: str,
    max_workers: int = 15,
    max_retries: int = 2
) -> Dict:
    """
    Process all JSON files in the input folder to regenerate nl_description.
    
    Args:
        input_folder: Path to folder containing JSON files
        output_folder: Path to folder for output files
        model: The model to use
        max_workers: Maximum number of parallel workers
        max_retries: Maximum retry attempts per file
        
    Returns:
        Dictionary with processing statistics
    """
    # Find all JSON files
    json_files = list(input_folder.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_folder}")
        return {"total": 0, "success": 0, "error": 0}
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total": len(json_files),
        "success": 0,
        "error": 0,
        "skipped": 0,
        "updated": 0,
        "kept_old": 0
    }
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_single_file, file_path, output_folder, model, max_retries): file_path
            for file_path in json_files
        }
        
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                
                if result["status"] == "success":
                    stats["success"] += 1
                    if result.get("updated", False):
                        stats["updated"] += 1
                    else:
                        stats["kept_old"] += 1
                    print(f"✓ {file_path.name}: {'Updated' if result.get('updated') else 'Kept old'}")
                elif result["status"] == "skipped":
                    stats["skipped"] += 1
                    print(f"⊘ {file_path.name}: Skipped (already marked as correct)")
                else:
                    stats["error"] += 1
                    print(f"✗ {file_path.name}: {result.get('message', 'Unknown error')}")
                    
            except Exception as e:
                stats["error"] += 1
                print(f"✗ {file_path.name}: Exception - {str(e)}")
    
    return stats


if __name__ == "__main__":
    # Configuration
    project_root = Path(__file__).parent.parent.parent
    input_folder = project_root / "data/PyGeoX benchmark/json_fixed"
    output_folder = project_root / "data/PyGeoX benchmark/json_fixed_2"
    model = "openai/gpt-oss-20b"#/mnt/disk0/models/gpt-oss-20b"
    max_workers = 25
    max_retries = 2
    
    # Resolve relative to project root if needed
    if not input_folder.is_absolute():
        # Get project root (parent of pygeox folder)
        project_root = Path(__file__).parent.parent.parent
        input_folder = (project_root / input_folder).resolve()
    
    if not output_folder.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        output_folder = (project_root / output_folder).resolve()
    
    if not input_folder.exists():
        print(f"Error: Input folder does not exist: {input_folder}")
        exit(1)
    
    print(f"Processing JSON files in: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Using model: {model}")
    print(f"Max workers: {max_workers}")
    print()
    
    # Process files
    stats = process_json_files(
        input_folder=input_folder,
        output_folder=output_folder,
        model=model,
        max_workers=max_workers,
        max_retries=max_retries
    )
    
    # Print summary
    print()
    print("=" * 60)
    print("Processing Summary:")
    print(f"  Total files: {stats['total']}")
    print(f"  Success: {stats['success']}")
    print(f"  Skipped (already correct): {stats.get('skipped', 0)}")
    print(f"  Errors: {stats['error']}")
    print(f"  Updated descriptions: {stats['updated']}")
    print(f"  Kept old descriptions: {stats['kept_old']}")
    print("=" * 60)

