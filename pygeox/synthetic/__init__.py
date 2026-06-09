"""
Synthetic Data Generation namespace for PyGeoX.

This module provides tools for generating synthetic geometric scenes for training
machine learning models. It includes definitions of all valid objects and relationships
that can be used in PyGeoX scenes, along with generators for creating diverse,
solvable geometric configurations.
"""

from .definitions import (
    OBJ_DICT,
    REL_DICT,
    Coordinate,
    valid_objects,
    get_object_definitions,
    get_relationship_definitions,
    get_objects_by_type,
    sample_objects,
    get_valid_relationships_from_object_names,
    generate_llm_prompt,
    OBJECT_WEIGHTS,
    RELATIONSHIP_WEIGHTS,
)
from .llm_client import (
    send_prompt_to_openrouter,
    generate_diagram_with_llm,
)

__all__ = [
    "OBJ_DICT",
    "REL_DICT",
    "Coordinate",
    "valid_objects",
    "get_object_definitions",
    "get_relationship_definitions",
    "get_objects_by_type",
    "sample_objects",
    "get_valid_relationships_from_object_names",
    "generate_llm_prompt",
    "OBJECT_WEIGHTS",
    "RELATIONSHIP_WEIGHTS",
    "send_prompt_to_openrouter",
    "generate_diagram_with_llm",
]

