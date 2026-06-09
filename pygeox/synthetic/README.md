# PyGeoX Synthetic Data Generation

This namespace provides tools and definitions for generating synthetic geometric scenes for training machine learning models.

## Overview

The `pygeox.synthetic` module contains comprehensive dictionaries of all valid geometric objects and relationships available in PyGeoX, structured for use in synthetic scene generation algorithms.

## Structure

### `definitions.py`

Contains two main dictionaries:

1. **`OBJ_DICT`**: Dictionary of all geometric objects that can be created
   - Keys: Object method names (e.g., "line", "triangle", "square")
   - Values: Dictionaries with:
     - `input_types`: List of required input types
     - `docstring`: Documentation string
     - `object_type`: Category ("linelike", "circle", "arc", "polygon", "angle")

2. **`REL_DICT`**: Dictionary of all geometric relationships/constraints
   - Keys: Relationship method names (e.g., "parallel", "perpendicular", "congruent")
   - Values: Dictionaries with:
     - `types`: List of argument types (including Optional types)
     - `docstring`: Documentation string

### Helper Functions

- `get_object_definitions()`: Get a copy of all object definitions
- `get_relationship_definitions()`: Get a copy of all relationship definitions
- `get_objects_by_type(object_type)`: Get all objects of a specific type
- `get_relationships_by_arity(arity)`: Get relationships with a specific number of required arguments

## Usage

```python
from pygeox.synthetic import OBJ_DICT, REL_DICT, get_objects_by_type

# Access object definitions
print(OBJ_DICT["triangle"])
# {'input_types': [Point, Point, Point], 'docstring': '...', 'object_type': 'polygon'}

# Access relationship definitions
print(REL_DICT["parallel"])
# {'types': [LineLike, LineLike], 'docstring': '...'}

# Get all polygon objects
polygons = get_objects_by_type("polygon")
# ['regular_pentagon', 'regular_hexagon', 'triangle', 'square', ...]
```

## Object Categories

- **linelike**: Lines, line segments, rays, chords
- **circle**: Circles
- **arc**: Minor arcs, major arcs, semicircles
- **polygon**: Triangles, quadrilaterals, regular polygons
- **angle**: Angle objects

## Relationship Categories

Relationships are categorized by the number of required arguments:
- **2 arguments**: parallel, perpendicular, congruent, similar, etc.
- **3 arguments**: collinear, acute_angle, right_angle, lines_intersect_at, etc.
- **4 arguments**: line_intersects_circle_at, translation, etc.

## Notes

- All type definitions use the actual PyGeoX classes (Point, LineLike, Circle, etc.)
- Optional parameters are indicated using `Optional[Type]` in the type lists
- The definitions match the actual method signatures in `pygeox.add_objects` and `pygeox.relationship`
- These definitions are designed for use with constructive grammar-based scene generation algorithms

## Data Generation Pipeline

The synthetic data generation pipeline consists of multiple stages that process geometric problems and generate training data.

### Folder Structure

The data generation pipeline creates the following folder structure:

```
generated_data/                    # Base directory (or generated_data_penalty/, generated_data_no_penalty/)
├── json/                         # Raw generated JSON files from data_generation_pipeline.py
│   ├── 1obj_2rel_2extra_gen1234.json
│   └── ...
├── image/                        # Generated geometric diagram images
│   ├── 1obj_2rel_2extra_gen1234.png
│   └── ...
├── json_processed/               # Processed JSON files with thinking/verification (from generate_thinking_data1.py)
│   ├── 1obj_2rel_2extra_gen1234.json  # Contains: pygeox_sft_thinking, pygeox_sft_verify
│   └── ...
└── json_processed_approaches/    # Processed JSON files with approach data (from generate_thinking_data2.py)
    ├── 1obj_2rel_2extra_gen1234.json  # Contains: constructive_approach_data, code_approach_data, R1_approach_data
    └── ...
```

### Processing Scripts

1. **`data_generation_pipeline.py`**
   - Generates initial geometric problems
   - Creates JSON files with problem descriptions, PyGeoX code, and ground truth
   - Generates corresponding diagram images
   - Output: `json/` and `image/` folders

2. **`generate_thinking_data1.py`**
   - Processes JSON files from `json/` folder
   - Generates thinking and verification data using LLM
   - Adds fields: `pygeox_sft_thinking`, `pygeox_sft_verify`
   - Only processes files where `success` is `True`
   - Output: `json_processed/` folder

3. **`generate_thinking_data2.py`**
   - Processes JSON files from `json/` folder
   - Generates multiple approach data using different prompts:
     - **Constructive approach**: Uses constructive geometric methods
     - **Code approach**: Uses numerical optimization methods
     - **R1 approach**: Uses general reasoning (optional)
   - Each approach contains: `think`, `answer`, `verify` fields
   - Only processes files where `success` is `True`
   - Output: `json_processed_approaches/` folder

### Prompt Files

The `prompts/` directory contains system prompts used for generating training data:

- `constructive prompt.md`: Prompt for constructive geometric approach
- `generate_code_prompt.md`: Prompt for numerical optimization approach
- `general_thinking_prompt.md`: Prompt for general reasoning approach (R1)
- `pygeox_documentation.md`: PyGeoX API documentation

### JSON File Structure

**Input JSON** (from `json/` folder):
```json
{
  "success": true,
  "nl_description": "Natural language description...",
  "pygeox_code": "Python code...",
  "points": {"A": [x, y], ...},
  "circles": {"O": radius, ...}
}
```

**Processed JSON** (from `json_processed/` folder):
```json
{
  ... (all input fields) ...,
  "pygeox_sft_thinking": "Thinking process...",
  "pygeox_sft_verify": "Verification code... Score: 1"
}
```

**Approach Data JSON** (from `json_processed_approaches/` folder):
```json
{
  ... (all input fields) ...,
  "constructive_approach_data": {
    "think": "...",
    "answer": "...",
    "verify": "..."
  },
  "code_approach_data": {
    "think": "...",
    "answer": "...",
    "verify": "..."
  },
  "R1_approach_data": {  # Optional
    "think": "...",
    "answer": "...",
    "verify": "..."
  }
}
```

### Usage

```bash
# Generate initial data
python -m pygeox.synthetic.data_generation_pipeline

# Generate thinking/verification data
python -m pygeox.synthetic.generate_thinking_data1

# Generate approach data
python -m pygeox.synthetic.generate_thinking_data2
```

