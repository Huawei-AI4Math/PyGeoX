"""
Definitions of valid objects and relationships for synthetic data generation.

This module contains comprehensive dictionaries of all geometric objects and
relationships available in PyGeoX, structured for use in synthetic scene generation.
"""

from typing import List, Optional, Union
import random
import sympy
from pygeox.basic_objects import (
    Point, LineLike, LineSegment, Circle, MajorArc, MinorArc, Line, Ray
)
from pygeox.custom_objects import (
    Polygon, Triangle, Quadrilateral, RegularPolygon,
    RegularPentagon, RegularHexagon, RegularHeptagon, RegularOctagon,
    EquilateralTriangle, IsoscelesTriangle, RightTriangle,
    RightIsoscelesTriangle, ObtuseTriangle, AcuteTriangle, ScaleneTriangle,
    Parallelogram, Rectangle, Square, Rhombus, Trapezoid,
    IsoscelesTrapezoid, RightTrapezoid, Kite, Rhomboid
)

# Type aliases
valid_objects = Union[Point, LineLike, Circle, MajorArc, MinorArc, Polygon]
Coordinate = Union[int, float, sympy.Expr]

# ============================================================================
# OBJECT DEFINITIONS
# ============================================================================

OBJ_DICT = {
    "circle": {
        "input_types": [Point],
        "docstring": "Add a circle. Arg is center of the circle.",
        "object_type": Circle
    },
    "minor_arc": {
        "input_types": [Point, Point, Point],
        "docstring": "Create a minor arc. Args are center of the circle, start point of the arc, and end point of the arc.",
        "object_type": MinorArc
    },
    "major_arc": {
        "input_types": [Point, Point, Point],
        "docstring": "Create a major arc. Args are center of the circle, start point of the arc, and end point of the arc.",
        "object_type": MajorArc
    },
    "regular_pentagon": {
        "input_types": [Point, Point, Point, Point, Point],
        "docstring": "Add a regular pentagon. Args are five vertices in order.",
        "object_type": RegularPentagon
    },
    "regular_hexagon": {
        "input_types": [Point, Point, Point, Point, Point, Point],
        "docstring": "Add a regular hexagon. Args are six vertices in order.",
        "object_type": RegularHexagon
    },
    "regular_heptagon": {
        "input_types": [Point, Point, Point, Point, Point, Point, Point],
        "docstring": "Add a regular heptagon. Args are seven vertices in order.",
        "object_type": RegularHeptagon
    },
    "regular_octagon": {
        "input_types": [Point, Point, Point, Point, Point, Point, Point, Point],
        "docstring": "Add a regular octagon. Args are eight vertices in order.",
        "object_type": RegularOctagon
    },
    "triangle": {
        "input_types": [Point, Point, Point],
        "docstring": "Add a general triangle. Args are three vertices.",
        "object_type": Triangle
    },
    "equilateral_triangle": {
        "input_types": [Point, Point, Point],
        "docstring": "Add an equilateral triangle. Args are three vertices.",
        "object_type": EquilateralTriangle
    },
    "isosceles_triangle": {
        "input_types": [Point, Point, Point],
        "docstring": "Add an isosceles triangle. Args are three vertices (A,B,C) where AB = BC.",
        "object_type": IsoscelesTriangle
    },
    "right_triangle": {
        "input_types": [Point, Point, Point],
        "docstring": "Add a right triangle. Args are three vertices (A,B,C) where angle B is right.",
        "object_type": RightTriangle
    },
    "right_isosceles_triangle": {
        "input_types": [Point, Point, Point],
        "docstring": "Add a right isosceles triangle. Args are three vertices (A,B,C) where angle B is right and AB = BC.",
        "object_type": RightIsoscelesTriangle
    },
    "obtuse_triangle": {
        "input_types": [Point, Point, Point],
        "docstring": "Add an obtuse triangle. Args are three vertices (A,B,C) where at least one internal angle is greater than 90 degrees.",
        "object_type": ObtuseTriangle
    },
    "acute_triangle": {
        "input_types": [Point, Point, Point],
        "docstring": "Add an acute triangle. Args are three vertices (A,B,C) where all internal angles are less than 90 degrees.",
        "object_type": AcuteTriangle
    },
    "scalene_triangle": {
        "input_types": [Point, Point, Point],
        "docstring": "Add a scalene triangle. Args are three vertices (A,B,C) where all side lengths are different.",
        "object_type": ScaleneTriangle
    },
    "quadrilateral": {
        "input_types": [Point, Point, Point, Point],
        "docstring": "Add a quadrilateral. Args are four vertices.",
        "object_type": Quadrilateral
    },
    "parallelogram": {
        "input_types": [Point, Point, Point, Point],
        "docstring": "Add a parallelogram. Opposite sides are constrained parallel.",
        "object_type": Parallelogram
    },
    "rectangle": {
        "input_types": [Point, Point, Point, Point],
        "docstring": "Add a rectangle. All angles are constrained 90°; opposite sides parallel.",
        "object_type": Rectangle
    },
    "square": {
        "input_types": [Point, Point, Point, Point],
        "docstring": "Add a square. All sides equal, all angles 90°, opposite sides parallel.",
        "object_type": Square
    },
    "rhombus": {
        "input_types": [Point, Point, Point, Point],
        "docstring": "Add a rhombus. All sides equal; opposite sides parallel; angles not necessarily 90°.",
        "object_type": Rhombus
    },
    "trapezoid": {
        "input_types": [Point, Point, Point, Point],
        "docstring": "Add a trapezoid ABCD. Constrains AB ∥ CD.",
        "object_type": Trapezoid
    },
    "isosceles_trapezoid": {
        "input_types": [Point, Point, Point, Point],
        "docstring": "Add an isosceles trapezoid ABCD. Constrains AB ∥ CD and BC = DA.",
        "object_type": IsoscelesTrapezoid
    },
    "right_trapezoid": {
        "input_types": [Point, Point, Point, Point],
        "docstring": "Add a right trapezoid ABCD. Constrains AB ∥ CD and angles A, D = 90°.",
        "object_type": RightTrapezoid
    },
    "kite": {
        "input_types": [Point, Point, Point, Point],
        "docstring": "Add a kite ABCD. Constrains AB = BC and CD = DA.",
        "object_type": Kite
    },
    "rhomboid": {
        "input_types": [Point, Point, Point, Point],    
        "docstring": "Add a rhomboid ABCD. Opposite sides parallel; adjacent sides unequal; no right-angle constraint.",
        "object_type": Rhomboid
    },
    "semicircle": {
        "input_types": [Point, Point, Point],
        "docstring": "Add a semicircle (180° major arc). Args (O,A,B) where O is center and A and B are endpoints on the circle. Enforces collinearity of center, A, and B (direction not forced).",
        "object_type": MajorArc
    },
}

# ============================================================================
# RELATIONSHIP DEFINITIONS
# ============================================================================

REL_DICT = {
    "parallel": {
        "types": [LineLike, LineLike],
        "docstring": "Add a constraint that two lines are parallel.\n\n        The cross product of their direction vectors must be zero.\n\n        Args:\n            line1: The first line-like object (Line, Ray, LineSegment).\n            line2: The second line-like object."
    },
    "perpendicular": {
        "types": [LineLike, LineLike, Optional[Point], Optional[bool]],
        "docstring": "Add a constraint that two lines are perpendicular.\n\n        The dot product of their direction vectors must be zero.\n\n        Optionally, if a foot point is provided, constrain that it lies on both lines.\n\n        Args:\n            line1: The first line-like object (Line, Ray, LineSegment).\n            line2: The second line-like object (Line, Ray, LineSegment).\n            foot: Optional point representing the foot of the perpendicular.\n            plot: Optional flag for plotting (default True).\n\n        Note:\n            If foot is provided and is not an endpoint of either line, it will be\n            automatically constrained to lie on both lines."
    },
    "collinear": {
        "types": [Point, Point, Point, Optional[bool]],
        "docstring": "Add constraints to ensure three points are collinear.\n\n        The signed area of the triangle formed by the points is zero.\n\n        If force_direction is True, also enforce that p3 lies in the direction from p1 to p2.\n\n        Args:\n            p1: First point.\n            p2: Second point.\n            p3: Third point.\n            force_direction: Whether to enforce directional order of points along the line (default True).\n\n        Note:\n            If any two points are equal, the constraint is skipped with a warning."
    },
    "point_lies_on": {
        "types": [Point, Union[LineLike, Circle, MajorArc, MinorArc]],
        "docstring": "Constrain a point to lie on a geometric object.\n\n        For LineSegment: point must be between the endpoints.\n        For Ray: point must be on the ray side of the start point.\n        For Line: point must be collinear (no position constraint).\n        For Circle: point lies on the circumference.\n        For Arc: point lies on the circumference and within angular bounds.\n\n        Args:\n            point: The point to constrain.\n            obj: The geometric object (Line, Ray, LineSegment, Circle, MajorArc, MinorArc).\n\n        Note:\n            If point is already an endpoint/defining point of the object, constraint is skipped."
    },
    "perpendicular_bisector_at": {
        "types": [LineSegment, LineLike, Optional[Point]],
        "docstring": "Constrain a line to be the perpendicular bisector of a line segment.\n\n        Optionally constrain the intersection point of the bisector and segment.\n\n        Args:\n            line1: The line segment to bisect.\n            perp_line: The line to be constrained as the perpendicular bisector (Line, Ray, LineSegment).\n            P: Optional intersection point of perp_line and line1."
    },
    "angle_bisector": {
        "types": [Point, Point, Point, LineLike],
        "docstring": "Constrain a line to be the angle bisector of the angle formed by three points.\n\n        The bisector line must start at the vertex point.\n\n        Args:\n            point1: One endpoint of the angle.\n            vertex: The vertex point of the angle.\n            point2: The other endpoint of the angle.\n            bisector_line: The line constrained to bisect the angle (Line, Ray, LineSegment).\n            IMPORTANT: bisector_line.point1 must be the vertex point.\n\n        Raises:\n            ValueError: If bisector_line.point1 is not the vertex."
    },
    "tangent_to_circle": {
        "types": [Union[LineLike, Circle], Circle, Point],
        "docstring": "Constrain a line or circle to be tangent to a circle at a given point.\n\n        Args:\n            line_or_circle: The tangent line or circle (Line, Ray, LineSegment, Circle).\n            circle: The circle to which tangency applies.\n            A: The point of tangency."
    },
    "is_chord": {
        "types": [LineSegment, Circle],
        "docstring": "Constrain a line segment to be a chord of a circle.\n\n        Args:\n            line: The line segment to constrain as a chord (LineSegment).\n            circle: The circle on which the chord lies (Circle)."
    },
    "line_intersects_circle_at": {
        "types": [LineLike, Circle, Point, Point],
        "docstring": "Constrain a line to intersect a circle at one or two given points.\n        The line will be a chord of the circle, and the given points will lie on both the line and the circle.\n\n        Args:\n            line: The line that intersects the circle (Line, Ray, LineSegment).\n            circle: The circle being intersected (Circle).\n            *points: One or two intersection points.\n\n        Raises:\n            ValueError: If not one or two points are provided."
    },
    "line_extension_intersects_circle_at": {
        "types": [LineSegment, Circle, Point, Point],
        "docstring": "Constrain the extension of a line segment to intersect a circle at one or two points.\n        Points must be collinear with the line and lie on the circle.\n\n        Args:\n            line: The line segment (LineSegment).\n            circle: The circle being intersected (Circle).\n            *points: One or two intersection points.\n\n        Raises:\n            ValueError: If not one or two points are provided."
    },
    "acute_angle": {
        "types": [Point, Point, Point],
        "docstring": "Constrain an angle to be acute (less than 90 degrees).\n\n        Uses the dot product of the two vectors forming the angle.\n\n        Args:\n            point1: A point on one arm of the angle.\n            vertex: The vertex of the angle.\n            point2: A point on the other arm of the angle."
    },
    "right_angle": {
        "types": [Point, Point, Point],
        "docstring": "Constrain an angle to be a right angle (exactly 90 degrees).\n\n        This is equivalent to the dot product of the two vectors being zero.\n\n        Args:\n            point1: A point on one arm of the angle.\n            vertex: The vertex of the angle.\n            point2: A point on the other arm of the angle."
    },
    "obtuse_angle": {
        "types": [Point, Point, Point],
        "docstring": "Constrain an angle to be obtuse (greater than 90 degrees).\n\n        Args:\n            point1: A point on one arm of the angle.\n            vertex: The vertex of the angle.\n            point2: A point on the other arm of the angle."
    },
    "lines_intersect_at": {
        "types": [LineLike, LineLike, Point],
        "docstring": "Constrain two lines to intersect at a specific point.\n\n        Args:\n            line1: The first line-like object (Line, Ray, LineSegment).\n            line2: The second line-like object (Line, Ray, LineSegment).\n            intersection_point: The point where the lines intersect."
    },
    "line_extensions_intersect_at": {
        "types": [LineLike, LineLike, Point],
        "docstring": "Constrain the extensions of two line-like objects to intersect at a point.\n\n        Args:\n            line1: The first line-like object (Line, Ray, LineSegment).\n            line2: The second line-like object (Line, Ray, LineSegment).\n            intersection_point: The point of intersection."
    },
    "congruent": {
        "types": [valid_objects, valid_objects],
        "docstring": "Constrain two geometric objects to be congruent (have the same shape and size).\n\n        Args:\n            obj1: The first geometric object (LineSegment, Polygon, Circle, MajorArc, MinorArc).\n            obj2: The second geometric object (LineSegment, Polygon, Circle, MajorArc, MinorArc).\n\n        Raises:\n            ValueError: If polygons have different numbers of vertices.\n            NotImplementedError: If object types are not supported for congruence."
    },
    "similar": {
        "types": [Union[Circle, Triangle, Quadrilateral, RegularPolygon], Union[Circle, Triangle, Quadrilateral, RegularPolygon]],
        "docstring": "Constrain two polygons or circles to be similar (have the same shape).\n\n        Args:\n            obj1: The first object (Circle, Polygon, MajorArc, MinorArc).\n            obj2: The second object (Circle, Polygon, MajorArc, MinorArc).\n\n        Raises:\n            ValueError: If polygons have different numbers of vertices.\n            NotImplementedError: If object types are not supported for similarity.\n\n        Note:\n            All circles are similar. Regular polygons of the same type are similar."
    },
    "is_radius": {
        "types": [LineSegment, Circle],
        "docstring": "Constrain a line segment to be a radius of a circle.\n\n        One endpoint must be the center, and the other must lie on the circle.\n\n        Args:\n            line: The line segment to constrain (LineSegment).\n            circle: The circle (Circle).\n\n        Raises:\n            ValueError: If neither endpoint of the line is the center of the circle."
    },
    "is_midpoint": {
        "types": [Point, Union[LineSegment, MajorArc, MinorArc]],
        "docstring": "Constrain a point to be the midpoint of a line segment or arc.\n\n        Args:\n            point: The midpoint point.\n            obj: The line segment or arc (LineSegment, MajorArc, MinorArc).\n\n        Note:\n            For arcs, the midpoint is computed along the arc's angular path."
    },
    "is_diameter": {
        "types": [LineSegment, Circle],
        "docstring": "Constrain a line segment to be a diameter of a circle.\n\n        Args:\n            line: The line segment to constrain (LineSegment).\n            circle: The circle (Circle)."
    },
    "is_circumcircle": {
        "types": [Circle, Polygon],
        "docstring": "Constrain a circle to be the circumcircle of a polygon.\n\n        The circle's center must equal the polygon's circumcenter, and the circle's\n        radius must equal the polygon's circumradius.\n\n        Args:\n            circle: The circle to constrain as the circumcircle (Circle).\n            polygon: The polygon (must have circumcenter and circumradius properties).\n                Supported types: Triangle, RegularPolygon, Square, Rectangle, CyclicPolygon.\n\n        Raises:\n            AttributeError: If the polygon does not have circumcenter or circumradius properties."
    },
    "is_incircle": {
        "types": [Circle, Union[Triangle, RegularPolygon, Square]],
        "docstring": "Constrain a circle to be the incircle of a polygon.\n\n        The circle's center must equal the polygon's incenter, and the circle's\n        radius must equal the polygon's inradius.\n\n        Args:\n            circle: The circle to constrain as the incircle (Circle).\n            polygon: The polygon (must have incenter and inradius properties).\n                Supported types: Triangle, RegularPolygon, Square.\n\n        Raises:\n            AttributeError: If the polygon does not have incenter or inradius properties."
    },
    "is_orthocenter": {
        "types": [Point, Triangle],
        "docstring": "Constrain a point to be the orthocenter of a triangle.\n\n        The orthocenter is the intersection point of the three altitudes of a triangle.\n\n        Args:\n            point: The point to constrain as the orthocenter (Point).\n            triangle: The triangle (Triangle).\n\n        Raises:\n            TypeError: If triangle is not a Triangle instance."
    },
    "is_centroid": {
        "types": [Point, Polygon],
        "docstring": "Constrain a point to be the centroid of a polygon.\n\n        The centroid is the average of all vertices (vertex centroid).\n\n        Args:\n            point: The point to constrain as the centroid (Point).\n            polygon: The polygon (Polygon or any subclass)."
    },
    "is_median": {
        "types": [LineSegment, Triangle, Point],
        "docstring": "Constrain a line segment to be a median of a triangle.\n\n        A median connects a vertex to the midpoint of the opposite side.\n\n        Args:\n            line: The line segment to constrain as a median (LineSegment).\n            triangle: The triangle (Triangle).\n            vertex: The vertex Point from which the median originates (must be one of triangle.points) (Point).\n\n        Raises:\n            TypeError: If triangle is not a Triangle instance.\n            ValueError: If vertex is not one of the triangle's vertices."
    },
    "is_altitude": {
        "types": [LineSegment, Triangle, Point],
        "docstring": "Constrain a line segment to be an altitude of a triangle.\n\n        An altitude is perpendicular from a vertex to the opposite side.\n\n        Args:\n            line: The line segment to constrain as an altitude (LineSegment).\n            triangle: The triangle (Triangle).\n            vertex: The vertex Point from which the altitude originates (must be one of triangle.points) (Point).\n\n        Raises:\n            TypeError: If triangle is not a Triangle instance.\n            ValueError: If vertex is not one of the triangle's vertices."
    },
    "translation": {
        "types": [valid_objects, valid_objects, Optional[Coordinate], Optional[Coordinate]],
        "docstring": "Constrain two objects to be a translation of each other.\n\n        Args:\n            obj1: The translated object (Point, LineLike, Circle, BaseArc, Polygon).\n            obj2: The reference object (must be same type as obj1).\n            dx: Optional x-component of the translation vector. If None, x-translation is free.\n            dy: Optional y-component of the translation vector. If None, y-translation is free.\n\n        Raises:\n            TypeError: If obj1 and obj2 are not of the same type, or if object types are unsupported."
    },
    "scale": {
        "types": [valid_objects, valid_objects, Coordinate],
        "docstring": "Constrain two objects to be a scaled version of each other by a scale factor.\n\n        Angles are preserved and lengths are scaled proportionally.\n\n        Args:\n            obj1: The scaled object (Circle, BaseArc, LineLike, RegularPolygon, Polygon).\n            obj2: The reference object (must be same type as obj1).\n            scale: The scaling factor (should not be zero).\n\n        Raises:\n            TypeError: If obj1 and obj2 are not of the same type.\n            ValueError: If scale factor is zero or invalid."
    },
    "rotation_around_point": {
        "types": [Union[LineSegment, Polygon], Union[LineLike, Polygon], Point, Coordinate],
        "docstring": "Constrain two objects to be a rotation of each other around a point by a given angle.\n\n        Args:\n            obj1: The rotated object (LineSegment or Polygon).\n            obj2: The reference object (LineLike or Polygon).\n            point: The rotation center (must be a common vertex of both objects).\n            angle: The rotation angle in degrees (CCW positive).\n\n        Raises:\n            TypeError: If obj1 and obj2 are not of the same type.\n            ValueError: If point is not a common vertex of both objects."
    },
    "mirror_across_line": {
        "types": [valid_objects, valid_objects, LineLike],
        "docstring": "Constrain two objects to be mirror images of each other across a line.\n\n        Args:\n            obj1: The mirrored object (Point, LineSegment, Circle, MajorArc, MinorArc, Polygon).\n            obj2: The reference object (must be same type as obj1).\n            axis_line: The line of reflection (Line, Ray, LineSegment).\n\n        Raises:\n            TypeError: If obj1 and obj2 are not of the same type.\n            NotImplementedError: If object types are not supported for mirroring."
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_object_definitions():
    """
    Get all object definitions.
    
    Returns:
        dict: Dictionary mapping object names to their definitions.
    """
    return OBJ_DICT.copy()


def get_relationship_definitions():
    """
    Get all relationship definitions.
    
    Returns:
        dict: Dictionary mapping relationship names to their definitions.
    """
    return REL_DICT.copy()


def get_objects_by_type(object_type: str) -> List[str]:
    """
    Get all object names of a specific type.
    
    Args:
        object_type: One of "linelike", "circle", "arc", "polygon", "angle"
    
    Returns:
        List of object names matching the type.
    """
    return [
        name for name, defn in OBJ_DICT.items()
        if defn.get("object_type") == object_type
    ]

# ============================================================================
# OBJECT WEIGHTS (based on frequency in typical geometry problems)
# ============================================================================

OBJECT_WEIGHTS = {
    # --- TRIANGLES (Target: ~30%) ---
    # High weights here because triangles are fundamental for logic (similarity/congruence)
    "triangle": 3,              # Generic/Random triangle
    "right_triangle": 2,        # Crucial for Pythagoras
    "equilateral_triangle": 2,  # Crucial for 60-60-60 properties
    "isosceles_triangle": 2,    # Crucial for base angle theorems
    "scalene_triangle": 1,     # distinct from generic "triangle"
    "obtuse_triangle": 1,
    "acute_triangle": 1,
    "right_isosceles_triangle": 1.5, # The 45-45-90 case

    # --- QUADRILATERALS (Target: ~30%) ---
    # Lowered Square/Rect to allow other shapes to breathe
    "square": 3,
    "rectangle": 3,
    "parallelogram": 2,
    "rhombus": 2,
    "trapezoid": 2,
    "isosceles_trapezoid": 2,   # Very common in circle geometry
    "right_trapezoid": 1.5,
    "kite": 1.5,
    "quadrilateral": 1.5,       # Generic irregular quad

    # --- CIRCLES & ARCS (Target: ~20%) ---
    "circle": 7,                # The base case needs high frequency
    "semicircle": 2,            # Boosted: Thales theorem relies on this
    "minor_arc": 1.5,
    "major_arc": 1.5,

    # --- POLYGONS & RARE (Target: ~20%) ---
    # These test the model's ability to generalize n-gon logic
    "regular_pentagon": 2,
    "regular_hexagon": 2,       # Boosted: Hexagons appear often (6 equilaterals)
    "regular_heptagon": 1,
    "regular_octagon": 1,
    "rhomboid": 1,
}

# ============================================================================
# RELATIONSHIP WEIGHTS (based on frequency in typical geometry problems)
# ============================================================================

RELATIONSHIP_WEIGHTS = {
    # --- STRUCTURAL "GLUE" (Target: High Priority) ---
    # Boosted per request. These define the physical connectivity of the diagram.
    "point_lies_on": 8,             # Restored: Essential for defining segments
    "collinear": 8,                 # Restored: Essential for straight lines
    "lines_intersect_at": 8,        # Restored: The most common way points are created
    "line_extensions_intersect_at": 8, # Crucial for exterior angles/cyclic quads
    "line_intersects_circle_at": 7,    # High freq for secants/chords
    "line_extension_intersects_circle_at": 7,

    # --- CORE THEOREM TRIGGERS (Target: Medium-High) ---
    # Kept robust so the model learns *why* things are connected.
    "parallel": 6,
    "perpendicular": 6,
    "tangent_to_circle": 6,         # High value: Tangency is a strict constraint
    "angle_bisector": 5,
    "congruent": 5,
    "similar": 5,

    # --- TRIANGLE & CIRCLE CENTERS (Target: Medium) ---
    "is_midpoint": 5,               # Slightly higher as it's a very common construction
    "perpendicular_bisector_at": 4,
    "is_altitude": 4,
    "is_circumcircle": 3,
    "is_incircle": 3,
    "is_orthocenter": 3,
    "is_centroid": 3,
    "is_median": 3,

    # --- GEOMETRIC FEATURES (Target: Medium-Low) ---
    "right_angle": 4,               # Boosted slightly: often goes with perpendicular
    "is_radius": 3,
    "is_diameter": 3,
    "is_chord": 3,
    "obtuse_angle": 2,
    "acute_angle": 2,

    # --- TRANSFORMATIONS (Target: Low) ---
    "rotation_around_point": 2,
    "mirror_across_line": 2,
    "translation": 1,
    "scale": 1,
}


# ============================================================================
# SAMPLING FUNCTIONS
# ============================================================================

def sample_objects(num_objects: int, object_list: Optional[List[str]] = None) -> List[str]:
    """
    Randomly sample objects with weighted probabilities.

    Objects are weighted based on their frequency in typical geometry problems.
    Repetitions are allowed.
    
    NOTE: Only samples from circles and polygons.
    Lines/rays/line_segments should be added by the LLM as needed for relationships.

    Args:
        num_objects: Number of objects to sample
        object_list: Optional list of object names to sample from.
                    If None, uses all objects from OBJ_DICT.

    Returns:
        List of sampled object names (may contain duplicates)

    Example:
        >>> objects = sample_objects(5)
        >>> # Returns 5 object names, weighted by frequency
        >>> # ['triangle', 'square', 'circle', 'regular_hexagon', 'triangle']
    """
    if object_list is None:
        object_list = list(OBJ_DICT.keys())

    # Validate that all objects in the list exist
    valid_objects = [obj for obj in object_list if obj in OBJ_DICT]
    if not valid_objects:
        raise ValueError("No valid objects found in object_list")

    # Get weights for each object (default weight 1 if not specified)
    weights = [OBJECT_WEIGHTS.get(obj, 1) for obj in valid_objects]

    # Sample with replacement using weighted random choice
    sampled = random.choices(valid_objects, weights=weights, k=num_objects)

    return sampled


def get_valid_relationships_from_object_names(
    object_names: List[str]
) -> List[str]:
    """
    Get valid relationships based on object names (not instances).

    Uses class type checks with actual class types from object_type field.

    Args:
        object_names: List of object names from OBJ_DICT

    Returns:
        List of valid relationship names
    """
    # Get actual object types that will be created
    object_types = []
    for obj_name in object_names:
        if obj_name not in OBJ_DICT:
            continue
        obj_type = OBJ_DICT[obj_name].get("object_type")
        if obj_type:
            object_types.append(obj_type)
    
    if not object_types:
        return []
    
    valid_rels = []
    
    # Types to ignore (LineLike types are always available - LLM will add them as needed)
    ignored_types = {Coordinate, bool, Point, int, float, sympy.Expr, LineLike, Line, Ray, LineSegment}
    
    # Helper to check if type should be ignored
    def should_ignore_type(t):
        """Check if type should be ignored."""
        if t in ignored_types:
            return True
        # Check Optional
        if (hasattr(t, '__origin__') and t.__origin__ is Union and
                type(None) in t.__args__):
            return True
        # Check if Union contains only ignored types
        if (hasattr(t, '__origin__') and t.__origin__ is Union):
            non_none = [a for a in t.__args__ if a is not type(None)]
            if all(a in ignored_types for a in non_none):
                return True
        # Check List[Point]
        if (hasattr(t, '__origin__') and t.__origin__ is list):
            inner = t.__args__[0] if t.__args__ else Point
            if inner == Point:
                return True
        return False
    
    # Check each relationship
    for rel_name, rel_def in REL_DICT.items():
        required_types = rel_def["types"]
        is_valid = True
        
        # Helper to check if a type can be satisfied by available objects
        def can_satisfy_type(req_type, available_objs_list):
            """Check if req_type can be satisfied by any object in available_objs_list."""
            # Check if it's a Union type
            if (hasattr(req_type, '__origin__') and req_type.__origin__ is Union):
                # Check if any type in the Union is ignored (can be created)
                for ut in req_type.__args__:
                    if ut is not type(None) and ut in ignored_types:
                        return True
                
                # Check if any non-ignored type in the Union matches available objects
                for ut in req_type.__args__:
                    if ut is not type(None) and ut not in ignored_types:
                        # Check if this type matches any available object
                        for obj_class in available_objs_list:
                            if obj_class == ut:
                                return True
                            elif ut == valid_objects:
                                # Check if object matches any type in valid_objects
                                for vt in [Point, LineLike, Circle, MajorArc, MinorArc, Polygon]:
                                    try:
                                        if (obj_class == vt or
                                                issubclass(obj_class, vt)):
                                            return True
                                    except (TypeError, Exception):
                                        pass
                            else:
                                try:
                                    if issubclass(obj_class, ut):
                                        return True
                                except (TypeError, Exception):
                                    pass
                return False
            
            # Non-Union type checks
            for obj_class in available_objs_list:
                if obj_class == req_type:
                    return True
                elif req_type == valid_objects:
                    # Check if object matches any type in valid_objects
                    for vt in [Point, LineLike, Circle, MajorArc, MinorArc, Polygon]:
                        try:
                            if (obj_class == vt or
                                    issubclass(obj_class, vt)):
                                return True
                        except (TypeError, Exception):
                            pass
                else:
                    try:
                        if issubclass(obj_class, req_type):
                            return True
                    except (TypeError, Exception):
                        pass
            return False
        
        # Check each required type
        for req_type in required_types:
            if should_ignore_type(req_type):
                continue
            
            # Check if this type can be satisfied
            if not can_satisfy_type(req_type, object_types):
                is_valid = False
                break
        
        if is_valid:
            valid_rels.append(rel_name)
    
    return valid_rels


def generate_llm_prompt(
    num_objects: int,
    num_relationships: int
) -> str:
    """
    Generate an LLM prompt for creating a geometric diagram.

    Samples objects, determines valid relationships, and creates a comprehensive
    prompt instructing the LLM to generate a valid geometric configuration.

    Args:
        num_objects: Number of objects to include
        num_relationships: Target number of relationships to include

    Returns:
        String prompt for the LLM
    """
    # Step 1: Sample objects
    sampled_objects = sample_objects(num_objects)
    
    # Step 2: Get valid relationships
    valid_relationships = get_valid_relationships_from_object_names(sampled_objects)
    
    if not valid_relationships:
        return "Error: No valid relationships found for the sampled objects."
    
    # Step 3: Sample relationships (weighted)
    weights = [RELATIONSHIP_WEIGHTS.get(rel, 1) for rel in valid_relationships]
    selected_relationships =  random.choices(
        valid_relationships,
        weights=weights,
        k=min(num_relationships//2, len(valid_relationships))
    )
    
    # Build objects section
    objects_section = ""
    for i, obj_name in enumerate(sampled_objects, 1):
        obj_def = OBJ_DICT.get(obj_name, {})
        input_types = obj_def.get("input_types", [])
        obj_type = obj_def.get("object_type", "unknown")
        docstring = obj_def.get("docstring", "").split("\n")[0]
        input_types_str = [t.__name__ if hasattr(t, '__name__') else str(t) for t in input_types]
        objects_section += f"{i}. {obj_name} ({obj_type})\n"
        objects_section += f"   Input types: {input_types_str}\n"
        objects_section += f"   Description: {docstring}\n\n"
    
    # Build relationships section
    relationships_section = ""
    unique_rels = list(set(valid_relationships))

    relationships_to_avoid_unless_preferred = ["similar", "congruent", "scale", "translation"]

    for i, rel_name in enumerate(unique_rels, 1):
        
        #if similar not in prefered relationships, then it should be avoided unless it is a preferred relationship
        if rel_name in relationships_to_avoid_unless_preferred and rel_name not in selected_relationships:
            continue

        rel_def = REL_DICT.get(rel_name)
        rel_types = rel_def.get("types")
        docstring = rel_def.get("docstring")
        
        # Format types (mark Optional, Coordinate, bool)
        type_strs = []
        for rt in rel_types:
            is_optional = False
            if hasattr(rt, '__origin__') and rt.__origin__ is Union:
                if type(None) in rt.__args__:
                    is_optional = True
                    args = [
                        a.__name__ if hasattr(a, '__name__') else str(a)
                        for a in rt.__args__ if a is not type(None)
                    ]
                    type_str = f"Optional[{', '.join(args)}]"
                else:
                    args = [
                        a.__name__ if hasattr(a, '__name__') else str(a)
                        for a in rt.__args__
                    ]
                    type_str = f"Union[{', '.join(args)}]"
            elif hasattr(rt, '__origin__') and rt.__origin__ is list:
                inner = rt.__args__[0] if rt.__args__ else "Point"
                inner_name = inner.__name__ if hasattr(inner, '__name__') else inner
                type_str = f"List[{inner_name}]"
            elif hasattr(rt, '__name__'):
                type_str = rt.__name__
            else:
                type_str = str(rt)
            
            # Mark optional
            if is_optional:
                type_str += " (optional)"
            # Mark Coordinate and bool
            if rt == Coordinate:
                type_str += " (scalar value, e.g., 5, 3.14)"
            elif rt == bool:
                type_str += " (True or False)"
            
            type_strs.append(type_str)
        
        relationships_section += f"{i}. {rel_name}\n"
        relationships_section += f"   Required types: {', '.join(type_strs)}\n"
        relationships_section += f"   Description: {docstring}\n\n"
    
    prompt = f"""{"=" * 80}
GEOMETRIC DIAGRAM GENERATION
{"=" * 80}

Create an interesting geometric diagram with EXACTLY {num_objects} PRIMARY objects (circles/polygons) and exactly {num_relationships} relationships.

REQUIRED PRIMARY OBJECTS:
{"-" * 80}
{objects_section}

AVAILABLE RELATIONSHIPS:
{"-" * 80}
{relationships_section}

PREFERRED RELATIONSHIPS (MUST use  {num_relationships//2} from these):
{"-" * 80}
{selected_relationships}

You MUST use at least {num_relationships//2} relationships from the PREFERRED RELATIONSHIPS list above.
For the remaining relationships, you can choose from any relationship in the AVAILABLE RELATIONSHIPS section.

CRITICAL RULES:
{"-" * 80}
1. PRIMARY OBJECTS: STRICT LIMIT - EXACTLY {num_objects} PRIMARY OBJECTS
   - You MUST create EXACTLY {num_objects} primary objects (circles/polygons) from the REQUIRED PRIMARY OBJECTS list
   - DO NOT create more than {num_objects} primary objects
   - DO NOT create fewer than {num_objects} primary objects
   - Count only circles and polygons as primary objects (NOT line segments, lines, or rays)
   - Auxiliary line-like objects (LineSegment, Line, Ray) are allowed in addition to the {num_objects} primary objects

2. OBJECTS: Create required primary objects + auxiliary line-like objects as needed
   - Required: EXACTLY {num_objects} primary objects (circles/polygons from REQUIRED PRIMARY OBJECTS list above)
   - Auxiliary: Add Line/LineSegment/Ray objects in 'Objs' when relationships require them
   - Format: "line_name": "line_segment(Point1, Point2)" or "line(Point1, Point2)" or "ray(Point1, Point2)"
   - Example: For parallel(line1, line2), create both line objects in 'Objs' first

3. RELATIONSHIPS: Select exactly {num_relationships} relationships
   - MUST use at least {num_relationships//2} relationships from PREFERRED RELATIONSHIPS
   - Remaining relationships can be chosen from AVAILABLE RELATIONSHIPS. Make the diagram interesting!
   - CRITICAL: Check the "Required types" for each relationship in AVAILABLE RELATIONSHIPS
   - You MUST ensure all arguments match the required types exactly:
     * For "similar(obj1, obj2)": Both objects MUST be the same type (both circles, both triangles, both quadrilaterals, etc.)
       - INVALID: "similar(triangle_EFG, square_ABCD)" - triangle and square are different types
       - VALID: "similar(triangle_ABC, triangle_DEF)" - both are triangles
     * For "point_lies_on(point, obj)": Second argument MUST be LineLike, Circle, MajorArc, or MinorArc
       - INVALID: "point_lies_on(A, square1)" - square1 is a polygon, not allowed
       - VALID: "point_lies_on(A, circle1)" or "point_lies_on(A, line_AB)"
     * For "is_midpoint(point, line)": The point MUST NOT be an endpoint of the line
       - INVALID: "is_midpoint(D, line_AD)" - D is an endpoint of line_AD
       - VALID: "is_midpoint(E, line_AD)" where E is a different point
   - DO NOT add relationships that are already implicit in object definitions:
     * Squares/rectangles: already have parallel opposite sides, right angles, equal sides
     * Regular polygons: already have equal sides and angles
     * Circles: already have all radii equal
     * Isosceles triangles: already have two equal sides
   - Focus on relationships that CONNECT different objects together
   - Create a connected diagram where all objects relate to each other (no isolated objects)
   - Use object names from 'Objs' as plain strings: "parallel(line1, line2)" NOT "parallel(\"line1\", \"line2\")"
   - DO NOT use dot notation: "square1.side_AB" is INVALID

4. POINTS: Only list points that are actually used
   - Use single uppercase letters (A, B, C, D, ...)
   - All points must be within domain [-10, 10] for both x and y coordinates
   - ONLY include points that are used in 'Objs' or 'Rels'
   - DO NOT add extra unused points
   - Points are defined by their usage in objects and relationships

5. CONNECTIVITY & INTEREST:
   - Ensure all objects are connected through relationships (no isolated objects)
   - Create interesting, non-trivial configurations suitable for challenging geometry problems
   - Use relationships that create geometric dependencies and interesting properties

6. VALIDATION:
   - No contradictions (e.g., don't make same lines both parallel and perpendicular)
   - All relationships must be geometrically possible
   - Coordinate values: use reasonable scalars (1-5) for lengths/radii
   - Boolean arguments: use True/False as specified
   - ALWAYS verify argument types match the "Required types" in AVAILABLE RELATIONSHIPS

OUTPUT FORMAT (IMPORTANT: Order is Objs, Rels, Points):
{"-" * 80}
```json
{{
  "Objs": {{
    "obj_name": "ObjectType(Point1, Point2, ...)",
    "line_name": "line_segment(Point1, Point2)",
    ...
  }},
  "Rels": [
    "RelationshipName(obj1, obj2, ...)",
    ...
  ],
  "Points": ["A", "B", "C", "D", ...],
  "nl_description": "A natural language description of the diagram. Only describe the geometric configuration, key relationships. Write in a clear, mathematical style suitable for a geometry exam. Please just restrict to providing a description of the diagram, not a question or analysis."
}}
```

7. NATURAL LANGUAGE DESCRIPTION:
   - Write a clear, concise description of the diagram in "nl_description" field
   - Start with "Diagram description: " followed by the description
   - Style: Write like a geometry exam problem description (descriptive, not a question)
   - Include: What objects are present, key relationships, and interesting geometric properties
   - DO NOT ask questions or request analysis - just describe the diagram
   - Please include all the required details: if you add a right triangle, mention which vertex is the right angle, for instance.
   - Example: "Diagram description: In the diagram, square ABCD is inscribed in circle O. Point E lies on the circle, and line segment AE is a radius. Line BC is tangent to the circle at point B."

EXAMPLE:
{"-" * 80}
```json
{{
  "Objs": {{
    "square1": "square(A, B, C, D)",
    "circle1": "circle(E)",
    "line_AE": "line_segment(A, E)",
    "line_BC": "line_segment(B, C)"
  }},
  "Rels": [
    "point_lies_on(E, circle1)",
    "is_radius(line_AE, circle1)",
    "tangent_to_circle(line_BC, circle1)"
  ],
  "Points": ["A", "B", "C", "D", "E"],
  "nl_description": "Diagram description: In the diagram, square ABCD is inscribed in circle O with center E. Point E lies on the circle, and line segment AE is a radius of the circle. Line BC is tangent to the circle at point B."
}}
```
Notes: 
- If you need to reference a polygon side in a relationship, create a separate
  line_segment object for that side in 'Objs' (e.g., "line_CE": "line_segment(C, E)")
  
{"=" * 80}
"""
    
    return prompt



