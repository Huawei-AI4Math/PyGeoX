"""
LLM client for sending prompts via OpenRouter API.

This module provides functionality to send prompts to LLMs through OpenRouter,
using API keys stored in a .env file.
"""

import os
import json
import re
import traceback
import copy
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv
from openai import OpenAI

from .definitions import generate_llm_prompt
from pygeox.geoscene import GeoScene


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


def send_prompt_to_openrouter(
    prompt: str = None,
    messages: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
    env_file: Optional[str] = None
) -> str:
    """
    Send a prompt or conversation to OpenRouter API and return the response.

    Reads the OpenRouter API key from a .env file in the main project folder.

    Args:
        prompt: The user prompt to send to the LLM (if messages not provided)
        messages: List of message dicts with 'role' and 'content' keys
                  (if provided, prompt is ignored)
        model: The model to use 
        env_file: Optional path to .env file. If None, looks for .env in
                  the project root (parent of pygeox folder)

    Returns:
        The response text from the LLM

    Raises:
        ValueError: If API key is not found in .env file or neither prompt
                    nor messages provided
        Exception: If API request fails
    """

    #print(messages[-1]["content"])
    #print("=" * 100)

    # Determine project root (parent of pygeox folder)
    if env_file is None:
        # Get the directory where this file is located
        current_file = Path(__file__)
        # Go up: synthetic -> pygeox -> project root
        project_root = current_file.parent.parent.parent
        env_file = project_root / ".env"
    else:
        env_file = Path(env_file)

    # Load environment variables from .env file
    load_dotenv(env_file)

    # Get API key from environment
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            f"API_KEY not found in {env_file}. "
            "Please add API_KEY=your_key_here to the .env file."
        )

    # Prepare messages
    if messages is None:
        if prompt is None:
            raise ValueError("Either prompt or messages must be provided")
        messages = [{"role": "user", "content": prompt}]

    # Initialize OpenAI client with OpenRouter base URL
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPEN_AI_BASE_URL")
    )

    # Send request to OpenRouter
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )


        #print(response.choices[0].message.content)
        # Extract and return the response content
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Failed to get response from OpenRouter: {str(e)}")


def create_objects_from_json(
    scene,
    json_data: dict,
    points_dict: dict
) -> dict:
    """
    Create all objects from JSON data.

    Args:
        scene: The GeoScene instance
        json_data: JSON data containing "Objs" dictionary
        points_dict: Dictionary mapping point names to Point objects

    Returns:
        Dictionary mapping object names to created objects

    Raises:
        ValueError: If object creation fails
    """
    objs_dict = json_data.get("Objs", {})
    created_objects = {}

    for obj_name, obj_str in objs_dict.items():
        obj_type, args = _parse_function_call(obj_str)
        add_method_name = _get_add_method_name(obj_type)

        # Resolve arguments to actual objects
        resolved_args = []
        for arg in args:
            # Check if it's a point
            if arg in points_dict:
                resolved_args.append(points_dict[arg])
            # Check if it's a number (for radius, etc.)
            elif re.match(r'^-?\d+\.?\d*$', arg):
                resolved_args.append(
                    float(arg) if '.' in arg else int(arg)
                )
            # Check if it's a boolean
            elif arg.lower() in ('true', 'false'):
                resolved_args.append(arg.lower() == 'true')
            # Otherwise, assume it's an object name
            else:
                if arg not in created_objects:
                    raise ValueError(
                        f"Object '{arg}' referenced but not yet created"
                    )
                resolved_args.append(created_objects[arg])

        # Get the add method and call it
        add_method = getattr(scene.add, add_method_name, None)
        if add_method is None:
            raise ValueError(
                f"Unknown object type: {obj_type} "
                f"(method: {add_method_name})"
            )

        # Call with resolved arguments and name
        obj = add_method(*resolved_args, name=obj_name)
        created_objects[obj_name] = obj

    return created_objects


def add_relationships_from_json(
    scene,
    json_data: dict,
    points_dict: dict,
    created_objects: dict
):
    """
    Add all relationships from JSON data.

    Args:
        scene: The GeoScene instance
        json_data: JSON data containing "Rels" list
        points_dict: Dictionary mapping point names to Point objects
        created_objects: Dictionary mapping object names to objects

    Raises:
        ValueError: If relationship addition fails
    """
    rels_list = json_data.get("Rels", [])

    for rel_str in rels_list:
        rel_name, args = _parse_function_call(rel_str)
        relate_method_name = _get_relate_method_name(rel_name)

        # Resolve arguments to actual objects
        resolved_args = []
        for arg in args:
            # Check if it's a point
            if arg in points_dict:
                resolved_args.append(points_dict[arg])
            # Check if it's a number
            elif re.match(r'^-?\d+\.?\d*$', arg):
                resolved_args.append(
                    float(arg) if '.' in arg else int(arg)
                )
            # Check if it's a boolean
            elif arg.lower() in ('true', 'false'):
                resolved_args.append(arg.lower() == 'true')
            # Otherwise, assume it's an object name
            else:
                if arg not in created_objects:
                    raise ValueError(
                        f"Object '{arg}' referenced in relationship "
                        f"but not found"
                    )
                resolved_args.append(created_objects[arg])

        # Get the relate method and call it
        relate_method = getattr(scene.relate, relate_method_name, None)
        if relate_method is None:
            raise ValueError(
                f"Unknown relationship: {rel_name} "
                f"(method: {relate_method_name})"
            )

        # Call with resolved arguments
        relate_method(*resolved_args)


def create_scene_from_json(domain, json_data: dict, generate_objective_function=False, distance_penalty=0, min_dist=0.1):
    """
    Create a GeoScene from JSON data with a specified domain.

    Args:
        domain: Domain value for GeoScene initialization
        json_data: JSON data containing "Points", "Objs", and "Rels"

    Returns:
        GeoScene: A fully constructed PyGeoX scene with all objects
        and relationships (not yet solved)

    Raises:
        ValueError: If JSON data is invalid or scene creation fails
    """
    # Check if all required keys are present
    if not all(key in json_data for key in ["Objs", "Rels", "Points"]):
        raise ValueError(
            "Invalid JSON data: missing required keys "
            "(must have 'Points', 'Objs', 'Rels')"
        )

    # Create PyGeoX scene
    scene = GeoScene(domain)

    # Step 1: Create all points
    point_names = json_data.get("Points", [])
    points_dict = {}
    for point_name in point_names:
        point = scene.add.point(name=point_name)
        points_dict[point_name] = point

    # Step 2: Create all objects
    created_objects = create_objects_from_json(
        scene, json_data, points_dict
    )

    # Step 3: Add all relationships
    add_relationships_from_json(
        scene, json_data, points_dict, created_objects
    )

    # Step 4: Relationships
    for entry in json_data.get("extra_rel", []):
        exec(entry)

    # Step 5: Generate objective function
    if generate_objective_function: 
        scene.solver.numerical(
            just_generate_objective_function=True,
            distance_penalty=distance_penalty,
            min_dist=min_dist,
            verbose = False
        )

    return scene


def extract_float_properties(scene, object_names):
    object_list = sum([list(obj.values()) for obj in scene.objects.values()], [])
    
    property_fields = {}

    for obj in object_list:
        props = []
        for attr_name in dir(obj):
            if attr_name.startswith("_"):
                continue  # Skip private/protected
            try:
                attr = getattr(type(obj), attr_name, None)
                # Check if attribute is a property and has return annotation 'Coordinate'
                if isinstance(attr, property):
                    # Check type hint of the property
                    if hasattr(attr.fget, "__annotations__"):
                        return_type = attr.fget.__annotations__.get("return", None)
                        # Accept both typenames and types due to how annotations might be stored
                        if str(return_type) == "typing.Union[int, float, sympy.core.expr.Expr]":
                            props.append(attr_name)
            except Exception:
                continue
        if props:
            property_fields[str(obj.name)] = props

    # Only return the properties of the objects in object_names
    property_fields = {
        k: v for k, v in property_fields.items() if k in object_names
    }
    property_fields_str = []

    for k, v in property_fields.items():
        for prop in v:
            property_fields_str.append(f"{k}.{prop}")

    return property_fields_str  # Show mapping from object to its property fields


def add_extra_relationships(
    scene,
    object_names,
    n,
    messages: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None
):
    """
    Add extra relationships to a scene by asking LLM to generate constraints.

    Args:
        scene: The GeoScene instance
        object_names: List of object names to consider
        n: Number of extra relationships to add
        messages: Previous conversation history to append to (optional)
        model: The model to use 

    Returns:
        tuple: (updated_scene, success_flag) - Returns original scene if fails
    """
    import random

    # Create deepcopy at the beginning
    original_scene = copy.deepcopy(scene)

    # Initialize messages if not provided
    if messages is None:
        messages = []

    # STEP 1 - make list of properties
    property_fields_str = extract_float_properties(scene, object_names)

    if len(property_fields_str) < 2:
        return original_scene, False, ""  # Need at least 2 properties

    # Randomly sample up to 10 properties (or all if less than 10)
    sample_size = min(10, len(property_fields_str))
    sampled_properties = random.sample(property_fields_str, sample_size)

    # STEP 2 - add prompt to LLM to generate extra relationships
    prompt = f"""Based on the previous conversation and the geometric diagram that has been built, I need you to add {n} extra constraint relationships.

AVAILABLE PROPERTIES (sampled from the scene) with format "object.property":
{chr(10).join(sampled_properties)}

IMPORTANT CONSTRAINTS:
- The coordinate domain is [-10, 10] x [-10, 10], so all constraints must be compatible with this domain
- The constraints should be mathematically valid and solvable, and lead to no degenerate solutions (overlapping points, etc.)
- They should be compatible with the existing geometric construction

Please generate {n} constraint relationships using the following format:

```json
{{
    "extra_rel": [
        "scene.constraint.eq(scene.get_object('object1').property1  , scene.get_object('object2').property2)",
        "scene.constraint.geq(scene.get_object('object1').property1, 5.0)",
        "scene.constraint.leq(scene.get_object('object1').property1, (1/4) * scene.get_object('object2').property2**2)"
        "scene.constraint.leq(scene.get_object('object1').property1, (1/4) * scene.get_object('object2').property2**2)"
    ],
    "description": "Description of all constraints"
}}
```

CONSTRAINT TYPES AVAILABLE:
1. scene.constraint.eq(a, b) - Equality: a == b
2. scene.constraint.geq(a, b) - Greater or equal: a >= b  
3. scene.constraint.leq(a, b) - Less or equal: a <= b

Additionally feel free to add angle constraints, even tough these are not included in the AVAILABLE PROPERTIES.
- When setting angle constraints, use the following format:
  - scene.constraint.eq(scene.angle(scene.get_object('A'), scene.get_object('B'), scene.get_object('C')), 45) 

RULES:
- First argument must always be an object property (e.g., "circle1.area", "line1.length")
- Second argument can be either:
  - Another object property (e.g., "square1.perimeter")
  - A float value (e.g., 5.0, 3.14)
  - An algebraic expression involving properties (e.g., "(1/4) * property2**2", "property1 + property2")
- You can use algebraic manipulations like multiplication, division, addition, subtraction, powers
- Make sure the constraints are meaningful and compatible with the domain [-10, 10] x [-10, 10]
- The description field should be a single string describing all constraints

Generate exactly {n} constraints that are compatible with the existing construction. Make it interesting!

EXAMPLE:
AVAILABLE PROPERTIES (sampled from the scene) with format "object.property":
Triangle_ABC.radius
line_EF.length
circle1.area

OUTPUT FORMAT:
```json
{{
    "extra_rel": [
        "scene.constraint.eq(scene.get_object('Triangle_ABC').radius, 5)"
    ],
    "description": "Triangle ABC has radius 5"
}}
```
"""
    # Append prompt to conversation history
    messages.append({"role": "user", "content": prompt})

    # Get response from LLM
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

        # Clean up JSON string - remove trailing commas in arrays/objects
        json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing comma before }
        json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing comma before ]

        try:
            extra_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"JSON string: {json_str[:500]}...")
            raise ValueError(f"Invalid JSON response: {str(e)}")

        extra_rel = extra_data.get("extra_rel", [])
        description = extra_data.get("description", "")

        if len(extra_rel) != n:
            print(f"Warning: Expected {n} relationships, got {len(extra_rel)}")

        # STEP 3 - Execute constraints directly using exec()
        for rel_str in extra_rel:
            try:
                exec(rel_str)
            except Exception as e:
                print(f"Error executing constraint '{rel_str}': {str(e)}")
                return original_scene, False, "", extra_rel

        # STEP 4 - Try to solve with numerical solver
        try:
            scene.solver.numerical(
                strict_tol=5e-4,
                distance_penalty=0,
                verbose = False
            )

            if scene.solver.status != "success":
                print("Solver failed after adding extra relationships")
                return original_scene, False, "", extra_rel

            return scene, True, description, extra_rel

        except Exception as e:
            print(f"Error solving scene: {str(e)}")
            return original_scene, False, "", extra_rel

    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"Error in add_extra_relationships: {str(e)}")
        print(f"Traceback: {error_traceback}")
        # extra_rel might not be defined if JSON parsing failed
        extra_rel = []
        return original_scene, False, "", extra_rel

def generate_diagram_with_llm(
    num_objects: int,
    num_relationships: int,
    extra_rel_num: int,
    model: Optional[str] = None,
    max_retries: int = 3,
    strict_tol: float = 3e-5,
    distance_penalty: float = 0
):
    """
    Generate a geometric diagram by sending the LLM prompt to OpenRouter
    and create a PyGeoX scene from the response.

    This function combines generate_llm_prompt with send_prompt_to_openrouter,
    parses the JSON response, and creates a complete GeoScene. If there are
    errors, it will retry by appending the error message to the conversation
    history and sending it back to the LLM.

    Args:
        num_objects: Number of objects to include in the diagram
        num_relationships: Target number of relationships to include
        extra_rel_num: Number of extra relationships to add
        model: The model to use
        max_retries: Maximum number of retry attempts (default: 3)
        strict_tol: Strict tolerance for solver (default: 3e-5)
        distance_penalty: Distance penalty for solver (default: 0)

    Returns:
        GeoScene: A fully constructed PyGeoX scene with all objects
        and relationships
    """
    prompt = generate_llm_prompt(num_objects, num_relationships)

    # Initialize conversation history with the initial prompt
    messages = [{"role": "user", "content": prompt}]

    last_json_data = None
    for attempt in range(max_retries):
        try:
            # Get response from LLM using conversation history
            response = send_prompt_to_openrouter(
                messages=messages, model=model
            )

            # Add assistant response to conversation history
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

            try:
                json_data = json.loads(json_str)
                # Store for potential return on failure
                last_json_data = json_data
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON response: {json_str[:200]}... "
                    f"Error: {str(e)}"
                )

            # Create PyGeoX scene from JSON data
            scene = create_scene_from_json(10, json_data, generate_objective_function=False)
            # Solve once after all relationships are added
            scene.solver.numerical(
                strict_tol=strict_tol,
                distance_penalty=distance_penalty,
                verbose = False
            )


            if scene.solver.status != "success":
                raise ValueError(
                    "Solver failed to find a solution. Maybe there are "
                    "contradictions or the diagram is not solvable. Please "
                    "change objects/relationships and try again with a "
                    "simpler diagram."
                )
            #if scene.solver.has_any_overlapping_points():
            #    raise ValueError(
            #        "Solver found overlapping points (not allowed!). "
            #        "Please change objects/relationships and try again "
            #        "with a simpler diagram."
            #    )

            # Add extra relationships if requested
            if extra_rel_num > 0:
                print("Scene solving was a success. Adding extra relationships now.")
                scene, success, description, extra_rel = add_extra_relationships(
                    scene, list(json_data.get("Objs", {}).keys()), extra_rel_num, messages, model = model
                )

                if success:
                    if description:
                        json_data["nl_description"] += " The additional constraints are: " + description

                    # Append the additional constraints to the JSON data
                    if extra_rel:
                        json_data["extra_rel"] = extra_rel
                else:
                    json_data["extra_rel"] = []
            else:   
                json_data["extra_rel"] = []

            # Extract points and circle radii for possible solution
            points_dict = scene.get_points()
            circles_dict = {}
            if "Circle" in scene.objects_solved:
                for circle_name, circle_obj in scene.objects_solved[
                    "Circle"
                ].items():
                    if (hasattr(circle_obj, 'center') and
                            hasattr(circle_obj, 'radius')):
                        center_name = circle_obj.center.name
                        circles_dict[center_name] = float(circle_obj.radius)

            possible_solution = {
                "points": points_dict,
                "circles": circles_dict
            }

            # Success - return the scene, json_data, possible_solution, and messages
            return scene, json_data, possible_solution, messages

        except Exception as e:
            # Get full traceback for more informative error messages
            error_traceback = traceback.format_exc()
            error_msg = str(e)

            # If this is the last attempt, raise error with json_data
            # and full traceback
            if attempt == max_retries - 1:
                # Create a custom exception that includes json_data and messages
                class GenerationError(ValueError):
                    def __init__(self, message, json_data=None, messages=None):
                        super().__init__(message)
                        self.json_data = json_data
                        self.messages = messages

                error_text = (
                    f"Failed after {max_retries} attempts.\n\n"
                    f"Last error: {error_msg}\n\n"
                    f"Full traceback:\n{error_traceback}"
                )
                raise GenerationError(
                    error_text,
                    json_data=last_json_data,
                    messages=messages
                )

            # Retry by appending error message to conversation history
            retry_message = f"""PREVIOUS ATTEMPT RESULT:
{response}

ERROR ENCOUNTERED:
{error_msg}

FULL TRACEBACK:
{error_traceback}

Please fix the JSON response to address the error above. Make sure:
1. All object names in relationships exist in the 'Objs' section
2. All point names used in objects exist in the 'Points' section
3. No dot notation is used (e.g., use "line1" not "polygon1.side_AB")
4. The JSON is valid and properly formatted
5. The diagram is solvable and doesn't have contradictions
6. No overlapping points are created
"""

            # Append the retry message to conversation history
            messages.append({"role": "user", "content": retry_message})

    # Should not reach here, but just in case
    raise ValueError("Failed to generate valid diagram after retries")
