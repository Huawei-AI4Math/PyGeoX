**PyGeoX** is an interactive Python library designed to create, manipulate, visualize, and solve geometric problems symbolically and numerically. Ideal for educational purposes, research, and interactive geometry explorations, PyGeoX simplifies geometric constructions and constraint solving with an intuitive, readable Python syntax.

---

## Philosophy and Design

PyGeoX follows a object-oriented approach, where geometry becomes intuitive Python code. It supports creating geometric scenes, adding and relating geometric objects, defining precise constraints, and seamlessly solving complex geometry problems.

The philosophy behind PyGeoX revolves around simplicity, clarity, and flexibility:

* **Simple Syntax:** Clearly defined commands mimic geometric language.
* **Automatic Constraint Management:** Constraints are implicitly added when relationships are defined.
* **Rich Property Access:** Directly access properties like areas, lengths, midpoints, and more.

---

## Getting Started with PyGeoX

The main functionalities are described below. Please click `Examples` in the left panel to check several use cases.

### Create a Scene

Start your geometry exploration by defining a scene:

```python
from pygeox import GeoScene

scene = GeoScene(10)  # creates a scene with domain from -10 to 10
```

### Add Objects

* PyGeoX contains **37 fundamental geometric objects:** Points, Lines, Segments, Rays, Circles, Arcs, Triangles, Quadrilaterals, Regular Polygons, and more.

* Please click `Add geometric objects` on the left panel to check all objects that can be added.

* PyGeoX makes creating geometric figures simple:

```python
A, B, C = scene.add.points(['A', 'B', 'C'])
triangle = scene.add.triangle(A, B, C)
circle = scene.add.circle(center=A, radius=5)
```

* Each object has **2-15 detailed properties** such as area, perimeter, midpoint, centroid, circumcircle, incircle, altitudes, and medians.

* Example property access:

```python
area = triangle.area
length = line_AB.length
mid_segment = triangle.midsegment
```

* Please click `Basic Objects` and `Polygons` on the left panel to see all object properties.


### Define Relationships

* PyGeoX contains **27 geometric relationships:** parallel, tangent, similarity, inside/outside containment, etc.

* Please click `Add relationships` on the left panel to check all relationships that can be added.

* Relationships automatically introduce constraints to `scene`:

```python
# Make line AB parallel to line CD
scene.relate.parallel(line_AB, line_CD)

# Point P lies on circle
scene.relate.point_lies_on(P, circle)
```

* This implicitly adds constraints like:

```python
# Example internal constraint:
# (line_AB.direction_vector × line_CD.direction_vector) = 0
```



### Add Custom Constraints

* Please click `Add constraints` on the left panel to check all constraints that can be added.

* Define explicit mathematical constraints:

```python
import sympy
scene.constraint.eq(triangle.area, sympy.cos(circle.perimeter) / circle.area)
scene.constraint.gt(line_AB.length, circle.radius)
```

---

## Solving Geometric Problems

* PyGeoX provides both analytical (symbolic) and numerical solvers.

```python
scene.solver.analytical()
scene.solver.numerical(method="basinhopping")
```

* The numerical solver avoid degenerate solutions (e.g., points too close, radius too small) by adding extra constraints.

* Please click `Solvers` on the left panel to see all available options.

### Visualize Results

Easily visualize solutions with built-in plotting:

```python
scene.plot()
```

---

## Under Construction

### Animation and Interactive Explorations

Future PyGeoX versions will support animations and dynamic geometry, such as moving points:

```python
anim_lambda = scene.animate.add_parameter("lambda")
scene.animate.run(frames=50)
scene.animate.gif("animation.gif")
```

### Automated Geometry Proofs

PyGeoX is developing an integrated proving system, enabling automated geometric proofs:

```python
with scene.proving() as prover:
    scene.relate.collinear(A, B, C)

scene.prove()
```

