# PyGeoX Documentation

## Overview

PyGeoX is a Python library for creating and solving geometric constraint problems. It allows you to define geometric objects (points, lines, circles, polygons) and relationships between them, then solve for valid configurations.

## Core Concepts

### GeoScene

The main class for creating geometric scenes. Initialize with a domain size:

```python
from pygeox import GeoScene

scene = GeoScene(domain=10)  # Domain is [-10, 10] x [-10, 10]
```

### Object Hierarchies

Understanding the object type hierarchies is important for using relationships correctly:

**LineLike** (base class for line-like objects):
- `Line`: Infinite line through two points
- `Ray`: Ray from a start point through another point
- `LineSegment`: Finite line segment between two points

**Polygon** (base class for polygons):
- `RegularPolygon`: Regular polygons with equal sides and angles
  - `RegularPentagon`, `RegularHexagon`, `RegularHeptagon`, `RegularOctagon`
- `Triangle`: Three-sided polygons
  - `EquilateralTriangle`, `IsoscelesTriangle`, `RightTriangle`, `ObtuseTriangle`, `AcuteTriangle`, `ScaleneTriangle`, `RightIsoscelesTriangle`
- `Quadrilateral`: Four-sided polygons
  - `Square`, `Rectangle`, `Parallelogram`, `Rhombus`, `Trapezoid`, `IsoscelesTrapezoid`, `RightTrapezoid`, `Kite`, `Rhomboid`
- `CyclicPolygon`: Polygons that can be inscribed in a circle

**Important Notes**:
- When a relationship accepts `LineLike`, it accepts `Line`, `Ray`, or `LineSegment`
- When a relationship accepts `Polygon`, it accepts any polygon type (Triangle, Quadrilateral, RegularPolygon, etc.)
- When a relationship requires a specific type (e.g., `Triangle`), only that type or its subclasses are valid
- For transformations and similarity/congruence, objects must be the **exact same type** (e.g., both `Triangle`, both `Square`, not `Triangle` and `Quadrilateral`)

### Creating Objects

All objects are created through `scene.add` methods. Objects must be created before relationships can be established.

#### Points

```python
# Create a single point
A = scene.add.point(name="A")

# Create multiple points at once
A, B, C = scene.add.points(["A", "B", "C"])
```

#### Line-like Objects

```python
# Line segment between two points
line_AB = scene.add.line_segment(A, B)

# Infinite line through two points
line = scene.add.line(A, B)

# Ray from point A through point B
ray = scene.add.ray(A, B)
```

#### Circles

```python
# Circle with center point
circle1 = scene.add.circle(O)  # O is the center point
```

#### Arcs

```python
# Minor arc (less than 180 degrees)
arc = scene.add.minor_arc(center, start_point, end_point)

# Major arc (greater than 180 degrees)
arc = scene.add.major_arc(center, start_point, end_point)

# Semicircle (180 degrees)
semicircle = scene.add.semicircle(center, start_point, end_point)
```

#### Triangles

```python
# General triangle
triangle = scene.add.triangle(A, B, C)

# Right triangle (angle at B is 90 degrees)
right_tri = scene.add.right_triangle(A, B, C)

# Equilateral triangle
equilateral = scene.add.equilateral_triangle(A, B, C)

# Isosceles triangle (AB = BC)
isosceles = scene.add.isosceles_triangle(A, B, C)

# Right isosceles triangle
right_isosceles = scene.add.right_isosceles_triangle(A, B, C)

# Scalene triangle (all sides different)
scalene = scene.add.scalene_triangle(A, B, C)

# Obtuse triangle
obtuse = scene.add.obtuse_triangle(A, B, C)

# Acute triangle
acute = scene.add.acute_triangle(A, B, C)
```

#### Quadrilaterals

```python
# General quadrilateral
quad = scene.add.quadrilateral(A, B, C, D)

# Square (all sides equal, all angles 90°)
square = scene.add.square(A, B, C, D)

# Rectangle (all angles 90°, opposite sides parallel)
rectangle = scene.add.rectangle(A, B, C, D)

# Parallelogram (opposite sides parallel)
parallelogram = scene.add.parallelogram(A, B, C, D)

# Rhombus (all sides equal, opposite sides parallel)
rhombus = scene.add.rhombus(A, B, C, D)

# Trapezoid (AB ∥ CD)
trapezoid = scene.add.trapezoid(A, B, C, D)

# Isosceles trapezoid (AB ∥ CD and BC = DA)
isosceles_trapezoid = scene.add.isosceles_trapezoid(A, B, C, D)

# Right trapezoid (AB ∥ CD and angles A, D = 90°)
right_trapezoid = scene.add.right_trapezoid(A, B, C, D)

# Kite (AB = BC and CD = DA)
kite = scene.add.kite(A, B, C, D)

# Rhomboid (opposite sides parallel, adjacent sides unequal)
rhomboid = scene.add.rhomboid(A, B, C, D)
```

#### Regular Polygons

```python
# Regular pentagon
pentagon = scene.add.regular_pentagon(A, B, C, D, E)

# Regular hexagon
hexagon = scene.add.regular_hexagon(A, B, C, D, E, F)

# Regular heptagon
heptagon = scene.add.regular_heptagon(A, B, C, D, E, F, G)

# Regular octagon
octagon = scene.add.regular_octagon(A, B, C, D, E, F, G, H)
```

#### Angles

```python
# Create an angle object
angle_ABC = scene.add.angle(A, B, C)
# Creates an angle at vertex B between points A and C
# Returns the angle measure (symbolic expression or numeric value)
# Optional parameters:
#   plot_sign: Whether to plot the angle sign (default: True)
#   plot_text: Whether to plot angle label (default: False)
#   name: Optional name for the angle object
```

**Note**: Angle objects can be used in constraints:
```python
angle1 = scene.add.angle(A, B, C)
scene.constraint.eq(angle1, 45)  # Constrain angle to 45 degrees
```

### Adding Relationships

Relationships are added through `scene.relate` methods. They constrain the geometric configuration.

#### Point Relationships

```python
# Point lies on a line, circle, or arc
scene.relate.point_lies_on(P, line_AB)
scene.relate.point_lies_on(P, circle1)
scene.relate.point_lies_on(P, arc)

# Valid object types for point_lies_on:
# - Line: point must be collinear (no position constraint)
# - Ray: point must be on the ray side of the start point
# - LineSegment: point must be between the endpoints
# - Circle: point lies on the circumference
# - MajorArc, MinorArc: point lies on the circumference and within angular bounds
# 
# Invalid: Polygons (use point_lies_on with individual line segments instead)

# Point is midpoint of a line segment or arc
scene.relate.is_midpoint(M, line_AB)

# Three points are collinear
scene.relate.collinear(A, B, C, force_direction=True)
# force_direction: If True (default), enforces that p3 lies in the direction from p1 to p2.
#                  If False, only ensures the signed area of the triangle is zero (points are collinear but order is not enforced).
# Note: If any two points are equal, the constraint is skipped with a warning.
```

#### Line Relationships

```python
# Two lines are parallel
scene.relate.parallel(line1, line2)

# Two lines are perpendicular
scene.relate.perpendicular(line1, line2)
# With optional foot point
scene.relate.perpendicular(line1, line2, foot_point, plot=True)
# foot_point: Optional point representing the "foot" of the perpendicular - the point where
#             the perpendicular from one line to the other intersects. If provided and not
#             an endpoint of either line, it will be automatically constrained to lie on both lines.

# Two lines intersect at a point
scene.relate.lines_intersect_at(line1, line2, intersection_point)
# Valid input types: line1 and line2 can be Line, Ray, or LineSegment
# intersection_point: The point where the lines intersect (must lie on both lines)

# Extensions of two lines intersect at a point
scene.relate.line_extensions_intersect_at(line1, line2, intersection_point)
# Valid input types: line1 and line2 can be Line, Ray, or LineSegment
# intersection_point: The point where the extended lines intersect (may be outside the original line segments)

# Line is perpendicular bisector of a line segment
scene.relate.perpendicular_bisector_at(line_segment, bisector_line, intersection_point=None)
# Constrains a line to be the perpendicular bisector of a line segment.
# Args:
#   line_segment: The line segment to bisect (LineSegment)
#   bisector_line: The line to be constrained as the perpendicular bisector (Line, Ray, or LineSegment)
#   intersection_point: Optional intersection point of bisector_line and line_segment.
#                      If provided, constrains this point to be the midpoint of the segment.

# Line is angle bisector
scene.relate.angle_bisector(point1, vertex, point2, bisector_line)
# Constrains a line to be the angle bisector of the angle formed by three points.
# Args:
#   point1: One endpoint of the angle
#   vertex: The vertex point of the angle (where the two rays meet)
#   point2: The other endpoint of the angle
#   bisector_line: The line constrained to bisect the angle (Line, Ray, or LineSegment)
#                  IMPORTANT: bisector_line.point1 must be the vertex point.
# Raises:
#   ValueError: If bisector_line.point1 is not the vertex.
```

#### Circle Relationships

```python
# Line or circle is tangent to a circle at a point
scene.relate.tangent_to_circle(line_AB, circle1, point_of_tangency)

# Line segment is a chord of a circle
scene.relate.is_chord(line_AB, circle1)

# Line intersects circle at one or two points
scene.relate.line_intersects_circle_at(line, circle, point1, point2=None)

# Extension of line segment intersects circle
scene.relate.line_extension_intersects_circle_at(line_segment, circle, point1, point2=None)

# Line segment is a radius
scene.relate.is_radius(line_OA, circle1)

# Line segment is a diameter
scene.relate.is_diameter(line_AB, circle1)

# Circle is circumcircle of a polygon
scene.relate.is_circumcircle(circle1, polygon)
# Constrains a circle to be the circumcircle (circle passing through all vertices) of a polygon.
# Supported polygon types: Triangle, RegularPolygon, Square, Rectangle, CyclicPolygon
# The circle's center must equal the polygon's circumcenter, and radius must equal circumradius.
# Raises: AttributeError if polygon does not have circumcenter or circumradius properties.

# Circle is incircle of a polygon
scene.relate.is_incircle(circle1, polygon)
# Constrains a circle to be the incircle (circle tangent to all sides) of a polygon.
# Supported polygon types: Triangle, RegularPolygon, Square
# The circle's center must equal the polygon's incenter, and radius must equal inradius.
# Raises: AttributeError if polygon does not have incenter or inradius properties.
```

#### Angle Relationships

```python
# Right angle (90 degrees)
scene.relate.right_angle(point1, vertex, point2)

# Acute angle (less than 90 degrees)
scene.relate.acute_angle(point1, vertex, point2)

# Obtuse angle (greater than 90 degrees)
scene.relate.obtuse_angle(point1, vertex, point2)
```

#### Triangle Relationships

```python
# Point is orthocenter of triangle
scene.relate.is_orthocenter(H, triangle)

# Point is centroid of polygon
scene.relate.is_centroid(G, polygon)

# Line segment is median of triangle
scene.relate.is_median(median_line, triangle, vertex)

# Line segment is altitude of triangle
scene.relate.is_altitude(altitude_line, triangle, vertex)
# Constrains a line segment to be an altitude of a triangle.
# An altitude is the perpendicular line from a vertex to the opposite side.
# Args:
#   altitude_line: The line segment to constrain as an altitude (LineSegment)
#   triangle: The triangle (Triangle)
#   vertex: The vertex Point from which the altitude originates (must be one of triangle.points)
# Raises:
#   TypeError: If triangle is not a Triangle instance
#   ValueError: If vertex is not one of the triangle's vertices
```

#### Congruence and Similarity

```python
# Two objects are congruent (same shape and size)
scene.relate.congruent(obj1, obj2)
# Valid object types: LineSegment, Polygon, Circle, MajorArc, MinorArc
# Both objects must be the same type (e.g., both triangles, both squares, both circles).
# Raises:
#   ValueError: If polygons have different numbers of vertices
#   NotImplementedError: If object types are not supported for congruence

# Two objects are similar (same shape)
scene.relate.similar(obj1, obj2)
# Valid object types: Circle, Triangle, Quadrilateral, RegularPolygon
# Both objects must be the same type (e.g., both triangles, both quadrilaterals, both circles).
# Note: All circles are similar. Regular polygons of the same type are similar.
# Raises:
#   ValueError: If polygons have different numbers of vertices
#   NotImplementedError: If object types are not supported for similarity
```

#### Transformations

```python
# Translation
scene.relate.translation(obj1, obj2, dx=None, dy=None)
# Constrains two objects to be a translation of each other.
# Valid object types: Point, LineLike, Circle, BaseArc, Polygon
# Both objects must be the same type (e.g., both triangles, both circles).
# Args:
#   obj1: The translated object
#   obj2: The reference object (must be same type as obj1)
#   dx: Optional x-component of translation vector (if None, x-translation is free)
#   dy: Optional y-component of translation vector (if None, y-translation is free)
# Raises: TypeError if obj1 and obj2 are not of the same type

# Scale
scene.relate.scale(obj1, obj2, scale_factor)
# Constrains two objects to be a scaled version of each other.
# Valid object types: Circle, BaseArc, LineLike, RegularPolygon, Polygon
# Both objects must be the same type (e.g., both triangles, both circles).
# Angles are preserved and lengths are scaled proportionally.
# Args:
#   obj1: The scaled object
#   obj2: The reference object (must be same type as obj1)
#   scale_factor: The scaling factor (should not be zero)
# Raises:
#   TypeError: If obj1 and obj2 are not of the same type
#   ValueError: If scale factor is zero or invalid

# Rotation around a point
scene.relate.rotation_around_point(obj1, obj2, center_point, angle_degrees)
# Constrains two objects to be a rotation of each other around a point.
# Valid object types: LineSegment, Polygon
# Both objects must be the same type (e.g., both triangles, both line segments).
# Args:
#   obj1: The rotated object (LineSegment or Polygon)
#   obj2: The reference object (LineLike or Polygon)
#   center_point: The rotation center (must be a common vertex of both objects)
#   angle_degrees: The rotation angle in degrees (CCW positive)
# Raises:
#   TypeError: If obj1 and obj2 are not of the same type
#   ValueError: If center_point is not a common vertex of both objects

# Mirror across a line
scene.relate.mirror_across_line(obj1, obj2, axis_line)
# Constrains two objects to be mirror images of each other across a line.
# Valid object types: Point, LineSegment, Circle, MajorArc, MinorArc, Polygon
# Both objects must be the same type (e.g., both triangles, both circles).
# Args:
#   obj1: The mirrored object
#   obj2: The reference object (must be same type as obj1)
#   axis_line: The line of reflection (Line, Ray, or LineSegment)
# Raises:
#   TypeError: If obj1 and obj2 are not of the same type
#   NotImplementedError: If object types are not supported for mirroring
```

### Constraints

For additional numerical constraints, use `scene.constraint`:

```python
# Equality constraint
scene.constraint.eq(object1.property, object2.property)
scene.constraint.eq(object1.property, 5.0)
# Constrains two expressions to be equal

# Greater than or equal
scene.constraint.geq(object1.property, value)
# Constrains expression to be >= value

# Less than or equal
scene.constraint.leq(object1.property, value)
# Constrains expression to be <= value

# Angle constraint
scene.constraint.eq(scene.angle(A, B, C), 45)
```

**Available Properties for Constraints**:
- Object properties: `area`, `length`, `radius`, `perimeter`, `side_length`, `height`, etc.
- Point coordinates: `point.x`, `point.y`
- Algebraic expressions: `circle1.area + square1.area`, `line1.length * 2`, etc.
- Angles: Use `scene.add.angle(point1, vertex, point2)` to create an angle object, then access its properties

**Constraint Examples**:
```python
# Constrain areas to be equal
scene.constraint.eq(circle1.area, square1.area)

# Constrain length to be at least 5
scene.constraint.geq(line_AB.length, 5.0)

# Constrain sum of areas
scene.constraint.eq(circle1.area + triangle1.area, 100.0)

# Constrain ratio
scene.constraint.eq(circle1.radius, square1.side_length * 2)
```

### Solving

After defining objects and relationships, solve the scene:

```python
# Solve numerically
scene.solver.numerical(
)

# Check solver status
if scene.solver.status == "success":
    # Access solved coordinates
    points = scene.get_points()
else:
    # Possible status values: "not solved", "success", "failure"
    print(f"Solver status: {scene.solver.status}")
```

**Solver Parameters**:
- `strict_tol`: Maximum allowed constraint violation. Lower values mean stricter constraints (default: 3e-5)
- `distance_penalty`: Penalty weight to prevent points from overlapping. Use > 0 to avoid degenerate cases (default: 0)
- `verbose`: If True, prints solver iteration progress and constraint violations

**Solver Status**:
- `"not solved"`: Scene has not been solved yet
- `"success"`: All constraints satisfied within tolerance
- `"failure"`: Solver could not find a solution satisfying all constraints

**Tips**:
- If solver fails, try reducing `strict_tol` or increasing `distance_penalty`
- For complex scenes, you may need to reduce the domain size: `GeoScene(domain=5)`
- Ensure all constraints are geometrically consistent (no contradictions)

### Accessing Objects

```python
# Get all points
points = scene.get_points()  # Returns dict: {"A": [x, y], ...}
# Returns dictionary mapping point names to (x, y) coordinate tuples

# Get a specific object by name
obj = scene.get_object("object_name")
# Returns the geometric object with the given name, or None if not found
# Searches across all object types (points, lines, circles, polygons, etc.)

# Get solved object (after solving)
solved_obj = scene.get_object("object_name", solved=True)
# Prefers solved objects, falls back to unsolved if solved version is missing

# Access object properties
area = circle1.area
length = line_AB.length
radius = circle1.radius
perimeter = polygon.perimeter
```

**Note**: Object names are optional when creating objects. If not provided, objects are still accessible through the variables you assign them to.

### Object Properties

Common properties available on geometric objects:

- **Points**: 
  - `x`: x-coordinate
  - `y`: y-coordinate
  - `coordinate`: [x, y] coordinate pair

- **LineSegments**: 
  - `length`: Length of the segment
  - `slope`: Slope of the line
  - `intercept`: y-intercept
  - `point1`: First endpoint
  - `point2`: Second endpoint

- **Lines and Rays**: 
  - `slope`: Slope of the line
  - `intercept`: y-intercept
  - `point1`, `point2`: Defining points

- **Circles**: 
  - `center`: Center point
  - `radius`: Radius of the circle
  - `area`: Area (π × radius²)
  - `circumference`: Circumference (2 × π × radius)

- **Polygons**: 
  - `area`: Area of the polygon
  - `perimeter`: Sum of all side lengths
  - `side_length`: Length of a side (for regular polygons or when applicable)
  - `internal_angle`: Internal angle measure (for regular polygons)
  - `height`: Height of the polygon (when applicable, e.g., triangles)
  - `circumradius`: Radius of circumscribed circle (for triangles, regular polygons, squares, rectangles)
  - `inradius`: Radius of inscribed circle (for triangles, regular polygons, squares)

### Important Notes

1. **Domain**: All coordinates must be within the domain specified when creating the scene (default: [-10, 10] x [-10, 10])
   - Points are constrained to lie within this domain
   - If solver fails, try reducing domain size: `GeoScene(domain=5)`
   - Domain affects the scale of your diagram

2. **Order Matters**: 
   - Create points first
   - Then create objects (which reference the points)
   - Then add relationships (which reference the objects)
   - Finally solve
   - You cannot add relationships before objects exist

3. **Object Names**: When creating objects, you can optionally provide a `name` parameter:
   ```python
   square1 = scene.add.square(A, B, C, D, name="square1")
   ```
   - Names are useful for retrieving objects later: `scene.get_object("square1")`
   - If not provided, objects are still accessible through the variables you assign

4. **Point Ordering in Polygons**: The order of points matters when creating polygons:
   ```python
   # These are different polygons (different vertex order)
   scene.add.square(A, B, C, D)  # Vertices in order A -> B -> C -> D
   scene.add.square(B, C, D, A)  # Same square (cyclic permutation)
   scene.add.square(A, C, B, D)  # Different polygon (different order)
   ```
   - Points should be provided in order (clockwise or counterclockwise)
   - The polygon is defined by connecting points in the given order

5. **Type Requirements**: Relationships have strict type requirements. Understanding object hierarchies helps:
   - `point_lies_on` requires the second argument to be `LineLike`, `Circle`, or `Arc` (not `Polygon`)
   - `similar` requires both objects to be the **exact same type** (e.g., both `Triangle`, both `Square`)
   - When a function accepts `LineLike`, you can use `Line`, `Ray`, or `LineSegment`
   - When a function accepts `Polygon`, you can use any polygon type, but for transformations they must match exactly

6. **Implicit Constraints**: Some objects have built-in constraints that are automatically enforced:
   - Squares/rectangles: already have parallel opposite sides, right angles, equal sides (for squares)
   - Regular polygons: already have equal sides and angles
   - Circles: already have all radii equal
   - Isosceles triangles: already have two equal sides
   - Right triangles: already have a 90-degree angle
   - Equilateral triangles: already have all sides and angles equal
   - Etc. When you add an object, you can assume all the object implicit constraints are already satisfied.
   - Don't add redundant relationships that are already implicit

7. **Extra Relationships**: You can add custom constraints using `scene.constraint` methods for properties not covered by standard relationships:
   - Use for numerical constraints (areas, lengths, ratios)
   - Use for angle constraints (though `scene.relate.right_angle` etc. are preferred)
   - Use for complex algebraic relationships between object properties

### Example Complete Workflow

```python
from pygeox import GeoScene

# Create scene
scene = GeoScene(domain=10)

# Create points
A, B, C, D = scene.add.points(["A", "B", "C", "D"])

# Create objects
square1 = scene.add.square(A, B, C, D)
circle1 = scene.add.circle(O)
line_AC = scene.add.line_segment(A, C)

# Add relationships
scene.relate.point_lies_on(O, circle1)
scene.relate.is_radius(line_AC, circle1)

# Add extra constraints if needed
scene.constraint.eq(circle1.area, 25.0)

# Solve
scene.solver.numerical()

#Plot (if needed)
scene.plot()

# Check result
if scene.solver.status == "success":
    points = scene.get_points()
    print(f"Solved! Points: {points}")
    
    # Access solved object properties
    solved_square = scene.get_object("square1", solved=True)
    print(f"Square area: {solved_square.area}")
```

## Common API Patterns

### Error Handling

Most relationship functions will raise errors if:
- Object types don't match requirements (e.g., `TypeError` if wrong type)
- Required properties are missing (e.g., `AttributeError` if polygon lacks circumcenter)
- Arguments are invalid (e.g., `ValueError` if vertex is not part of triangle)

```python
try:
    scene.relate.similar(triangle1, square1)  # Will raise TypeError
except TypeError as e:
    print(f"Type error: {e}")
```

### Checking Object Types

When working with objects, you can check their types:
```python
from pygeox.basic_objects import LineLike, LineSegment, Line, Ray
from pygeox.custom_objects import Polygon, Triangle, Quadrilateral

# Check if object is a line-like object
if isinstance(line_obj, LineLike):
    # Works for Line, Ray, or LineSegment
    
# Check specific types
if isinstance(poly_obj, Triangle):
    # Works for Triangle and all its subclasses
```

### Working with Solved vs Unsolved Objects

After solving, objects have two versions:
- **Unsolved objects**: Original symbolic objects with symbolic expressions
- **Solved objects**: Objects with numeric values substituted

```python
# Before solving: objects contain symbolic expressions
line_length = line_AB.length  # May be a SymPy expression

# After solving:
scene.solver.numerical()
if scene.solver.status == "success":
    # Get solved version
    solved_line = scene.get_object("line_AB", solved=True)
    solved_length = solved_line.length  # Now a numeric value
```


