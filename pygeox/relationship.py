"""
RelationshipNamespace class provides geometric relationship constraints 
between various geometric objects such as points, lines, circles, arcs, and polygons.
These constraints can be used within a scene to enforce geometric properties like 
parallelism, perpendicularity, collinearity, congruence, similarity, and containment.

Each method adds symbolic constraints to the scene's constraint solver to represent
the desired geometric relationships.
"""

from typing import List, Optional, Union

import sympy
from typeguard import typechecked

from .basic_objects import (
    BaseArc,
    Circle,
    Line,
    LineLike,
    LineSegment,
    MajorArc,
    MinorArc,
    Point,
    Ray,
)
from .custom_objects import *

valid_objects = Union[Point, LineLike, Circle, MajorArc, MinorArc, Polygon]
Coordinate = Union[int, float, sympy.Expr]


def _track_inside(method):
    def wrapper(self, *args, **kwargs):
        self._inside = True
        try:
            return method(self, *args, **kwargs)
        finally:
            self._inside = False
    return wrapper

def _decorate_all_methods(cls):
    for attr_name, attr in cls.__dict__.items():
        if callable(attr) and not attr_name.startswith("__"):
            setattr(cls, attr_name, _track_inside(attr))
    return cls

@_decorate_all_methods
@typechecked
class RelationshipNamespace:
    """
    Namespace class for defining geometric relationship constraints
    in a scene.

    Attributes:
        scene: The geometric scene containing objects and constraint solvers.
    """

    def __init__(self, scene):
        """
        Initialize with a reference to the scene that contains geometric objects and constraints.

        Args:
            scene: The scene instance where constraints will be applied.
        """
        self.scene = scene

        #check if we are "inside" this namespace
        #useful to know if we should relate condition to a specific relationship_id 
        self._inside = False  

    def parallel(self, line1: LineLike, line2: LineLike):
        """
        Add a constraint that two lines are parallel.

        The cross product of their direction vectors must be zero.

        Args:
            line1: The first line-like object (Line, Ray, LineSegment).
            line2: The second line-like object.
        """
        self.scene._add_relationship("parallel", {"line1": line1, "line2": line2})
        
        u_x, u_y = line1.direction_vector
        v_x, v_y = line2.direction_vector
        cross = u_x * v_y - u_y * v_x
        

        self.scene.constraint.eq(
            cross, 0, f"{line1.name} || {line2.name} (cross product zero)"
        )

    def perpendicular(
        self, line1: LineLike, line2: LineLike, foot: Optional[Point] = None, plot: Optional[bool] = True
    ):
        """
        Add a constraint that two lines are perpendicular.

        The dot product of their direction vectors must be zero.

        Optionally, if a foot point is provided, constrain that it lies on both lines.

        Args:
            line1: The first line-like object.
            line2: The second line-like object.
            foot: Optional point representing the foot of the perpendicular.
        """
        self.scene._add_relationship("perpendicular", {"line1": line1, "line2": line2, "foot": foot, "plot": plot})

        # Foot not present
        u_x, u_y = line1.direction_vector
        v_x, v_y = line2.direction_vector
        dot = u_x * v_x + u_y * v_y
        self.scene.constraint.eq(
            dot, 0, f"{line1.name} ⟂ {line2.name} (dot product zero at intersection)"
        )

        if foot:
            if foot not in [line1.point1, line1.point2]:
                self.point_lies_on(foot, line1)
            if foot not in [line2.point1, line2.point2]:
                self.point_lies_on(foot, line2)

    def collinear(self, p1: Point, p2: Point, p3: Point, force_direction: Optional[bool] = True):
        """
        Add constraints to ensure three points are collinear.

        The signed area of the triangle formed by the points is zero.

        If force_direction is True, also enforce that p3 lies in the direction from p1 to p2.

        Args:
            p1: First point.
            p2: Second point.
            p3: Third point.
            force_direction: Whether to enforce directional order of points along the line.
        """
        self.scene._add_relationship("collinear", {"p1": p1, "p2": p2, "p3": p3, "force_direction": force_direction})

        if p1.name == p2.name or p1.name == p3.name or p2.name == p3.name:
            print(f"[WARNING] Collinearity skipped: two of the points are equal")
            return

        # Collinearity (area = 0)
        expr = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
        self.scene.constraint.eq(expr, 0, f"{p1.name}, {p2.name}, {p3.name} collinear")

        if force_direction:
            # Direction constraint: dot product > 0
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            vx = p3.x - p2.x
            vy = p3.y - p2.y
            dot = dx * vx + dy * vy
            self.scene.constraint.gt(
                dot, 0, f"{p3.name} after {p2.name} in direction {p1.name}->{p2.name}"
            )

    def point_lies_on(
        self, point: Point, obj: Union[LineLike, Circle, MajorArc, MinorArc]
    ):
        """
        Constrain a point to lie on a geometric object (line, ray, segment, circle, or arc).

        For lines and rays, this includes collinearity and positional constraints
        (e.g., point must lie between segment endpoints or on ray direction).

        For circles and arcs, the point lies on the circumference,
        and for arcs, also within the angular bounds.

        Args:
            point: The point to constrain.
            obj: The geometric object (line, circle, or arc). Semicircles are not supported.
        
        Raises:
            ValueError: If obj is a semicircle (MajorArc with name starting with "semicircle_").
        """
        # Exclude semicircles - they are MajorArc instances but should not be used here
        if isinstance(obj, MajorArc) and obj.name.startswith("semicircle_"):
            raise ValueError(
                f"Semicircles are not supported in point_lies_on. "
                f"Received semicircle: {obj.name}. "
                f"Use a regular MajorArc or MinorArc instead."
            )
        
        self.scene._add_relationship("point_lies_on", {"point": point, "obj": obj})

        # Handle LineLike objects (LineSegment, Line, Ray)
        if isinstance(obj, LineLike):
            if point == obj.point1 or point == obj.point2:
                print(
                    f"[Warning] Point {point.name} already defines {obj.__class__.__name__} {obj.name}. "
                    "Constraint might be redundant."
                )
                return

            # All LineLike objects require collinearity with their their defining points.
            # The collinearity check: (y - y1)(x2 - x1) - (y2 - y1)(x - x1) = 0
            x0, y0 = point.x, point.y
            x1, y1 = obj.point1.x, obj.point1.y
            x2, y2 = obj.point2.x, obj.point2.y
            collinear_expr = (y0 - y1) * (x2 - x1) - (y2 - y1) * (x0 - x1)
            self.scene.constraint.eq(
                collinear_expr, 0, f"{point.name} collinear with {obj.name}"
            )

            if isinstance(obj, LineSegment):
                # For a LineSegment, the point must be *between* the two endpoints.
                # Vector AB (from point1 to point2)
                ab_x = x2 - x1
                ab_y = y2 - y1
                # Vector AP (from point1 to point)
                ap_x = x0 - x1
                ap_y = y0 - y1

                dot_product = ab_x * ap_x + ab_y * ap_y
                ab_squared_length = ab_x**2 + ab_y**2

                # Point must be after point1 along the segment's direction
                self.scene.constraint.geq(
                    dot_product,
                    0,
                    f"{point.name} is after {obj.point1.name} along {obj.name}",
                )
                # Point must be before point2 along the segment's direction
                self.scene.constraint.leq(
                    dot_product,
                    ab_squared_length,
                    f"{point.name} is before {obj.point2.name} along {obj.name}",
                )

            elif isinstance(obj, Ray):
                # For a Ray, the point must be on the ray's side of the start_point.
                ray_dx, ray_dy = obj.direction_vector

                # Vector from ray's start_point to the point being constrained
                vec_start_to_point_x = x0 - obj.start_point.x
                vec_start_to_point_y = y0 - obj.start_point.y

                # Dot product must be non-negative for the point to be on the ray
                dot_product_ray = (
                    ray_dx * vec_start_to_point_x + ray_dy * vec_start_to_point_y
                )
                self.scene.constraint.geq(
                    dot_product_ray,
                    0,
                    f"{point.name} is on the ray side of {obj.start_point.name} along {obj.name}",
                )

            elif isinstance(obj, Line):
                # For an infinite Line, collinearity is the only constraint needed.
                pass

        elif isinstance(obj, Circle):
            self.scene.constraint.eq(
                point.distance_squared(obj.center),
                obj.radius**2,
                f"{point.name} lies on {obj.name}",
            )
        elif isinstance(obj, BaseArc):
            if point.name == obj.start_point.name or point.name == obj.end_point.name:
                print(f"[WARNING] Point {point.name} already belongs to Arc {obj.name}")
                return
            cx, cy = obj.center.x, obj.center.y
            r = obj.radius
            sx, sy = obj.start_point.x, obj.start_point.y
            ex, ey = obj.end_point.x, obj.end_point.y
            px, py = point.x, point.y

            # Constraint 1: Point lies on the circle
            self.scene.constraint.eq(
                (px - cx) ** 2 + (py - cy) ** 2,
                r**2,
                f"{point.name} lies on circle of {obj.name}",
            )

            # Vectors
            us_x, us_y = sx - cx, sy - cy
            ue_x, ue_y = ex - cx, ey - cy
            up_x, up_y = px - cx, py - cy

            # Cross products
            cross_se = us_x * ue_y - us_y * ue_x
            cross_sp = us_x * up_y - us_y * up_x
            cross_pe = up_x * ue_y - up_y * ue_x

            if isinstance(obj, MinorArc):
                self.scene.constraint.geq(
                    cross_se, 0, f"{obj.name} is a minor arc (CCW)"
                )
                self.scene.constraint.geq(
                    cross_sp, 0, f"{point.name} is after start on minor arc {obj.name}"
                )
                self.scene.constraint.geq(
                    cross_pe, 0, f"{point.name} is before end on minor arc {obj.name}"
                )
            else:
                self.scene.constraint.leq(
                    cross_sp,
                    0,
                    f"{point.name} is not after start on major arc {obj.name}",
                )
                self.scene.constraint.leq(
                    cross_pe,
                    0,
                    f"{point.name} is not before end on major arc {obj.name}",
                )

    def points_lie_on(
        self, points: List[Point], obj: Union[LineLike, Circle, MajorArc, MinorArc]
    ):
        """
        Constrain multiple points to lie on the given geometric object.

        Args:
            points: List of points to constrain.
            obj: The geometric object.
        """
        for point in points:
            self.point_lies_on(point, obj)

    def perpendicular_bisector_at(
        self, perp_line: LineLike, line1: LineSegment, P: Optional[Point] = None
    ):
        """
        Constrain a line to be the perpendicular bisector of a line segment.

        Optionally constrain the intersection point of the bisector and segment.

        Args:
            line1: The line segment to bisect.
            perp_line: The line to be constrained as the perpendicular bisector.
            P: Optional intersection point of perp_line and line1.
        """
        self.scene._add_relationship("perpendicular_bisector_at", {"perp_line": perp_line, "line1": line1, "P": P})

        if P:
            if not (P.name == perp_line.point1.name or P.name == perp_line.point2.name):
                self.scene.relate.point_lies_on(P, perp_line)

            self.scene.constraint.eq(P.x, (line1.point1.x + line1.point2.x) / 2)
            self.scene.constraint.eq(P.y, (line1.point1.y + line1.point2.y) / 2)
            self.scene.relate.perpendicular(line1, perp_line)
        else:
            self.scene.relate.point_lies_on(line1.midpoint, perp_line)
            self.scene.relate.perpendicular(line1, perp_line)

    def angle_bisector(
        self, point1: Point, vertex: Point, point2: Point, bisector_line: LineLike
    ):
        """
        Constrain a line to be the angle bisector of the angle formed by three points.

        The bisector line must start at the vertex point.

        Args:
            point1: One endpoint of the angle.
            vertex: The vertex point of the angle.
            point2: The other endpoint of the angle.
            bisector_line: The line constrained to bisect the angle.
        """
        self.scene._add_relationship("angle_bisector", {"point1": point1, "vertex": vertex, "point2": point2, "bisector_line": bisector_line})

        if bisector_line.point1.name != vertex.name:
            raise ValueError("Point1 of bisector_line must be vertex!")

        # Angle at B between A and D
        angle_ABD = vertex.angle(point1, bisector_line.point2)
        # Angle at B between D and C
        angle_DBC = vertex.angle(bisector_line.point2, point2)

        # Enforce equality of the two angles
        self.scene.constraint.eq(
            angle_ABD,
            angle_DBC,
            f"Angle bisector: angle ABD = angle DBC at {vertex.name}",
        )

    def tangent_to_circle(
        self, line_or_circle: Union[LineLike, Circle], circle: Circle, A: Point
    ):
        """
        Constrain a line or circle to be tangent to a circle at a given point.

        Args:
            line_or_circle: The tangent line or circle.
            circle: The circle to which tangency applies.
            A: The point of tangency.
        """
        self.scene._add_relationship("tangent_to_circle", {"line_or_circle": line_or_circle, "circle": circle, "A": A})

        # 1. A is on the line
        self.point_lies_on(A, line_or_circle)

        # 2. A is on the circle
        self.point_lies_on(A, circle)

        # 3. Tangency: the radius at A is perpendicular to the tangent line
        # Construct the radius line from the center to A
        if isinstance(line_or_circle, LineLike):
            radius_line = Line(
                circle.center, A, name=f"radius_{circle.center.name}_{A.name}"
            )
            self.perpendicular(line_or_circle, radius_line, plot = False)
        else:
            self.collinear(
                A, line_or_circle.center, circle.center, force_direction=False
            )

    def is_chord(self, line: LineSegment, circle: Circle):
        """
        Constrain a line segment to be a chord of a circle (both endpoints lie on the circle).

        Args:
            line: The line segment to constrain.
            circle: The circle.
        """
        self.scene._add_relationship("chord", {"line": line, "circle": circle})

        self.point_lies_on(line.point1, circle)
        self.point_lies_on(line.point2, circle)

    def line_intersects_circle_at(self, line: LineLike, circle: Circle, *points):
        """
        Constrain a line to intersect a circle at one or two specified points.

        Args:
            line: The line.
            circle: The circle.
            points: One or two points of intersection.

        Raises:
            ValueError: If not one or two points are provided.
        """
        self.scene._add_relationship("line_intersects_circle_at", {"line": line, "circle": circle, "points": points})

        if not (1 <= len(points) <= 2):
            raise ValueError("You must provide one or two intersection points.")

        for pt in points:
            # Point on line
            self.point_lies_on(pt, line)
            # Point on circle
            self.point_lies_on(pt, circle)

    def line_extension_intersects_circle_at(
        self, line: LineSegment, circle: Circle, *points
    ):
        """
        Constrain the extension of a line segment to intersect a circle at given points.

        Points must be collinear with the line and lie on the circle.

        Args:
            line: The line segment.
            circle: The circle.
            points: One or two points of intersection.

        Raises:
            ValueError: If not one or two points are provided.
        """
        self.scene._add_relationship("line_extension_intersects_circle_at", {"line": line, "circle": circle, "points": points})

        if not (1 <= len(points) <= 2):
            raise ValueError("You must provide one or two intersection points.")

        for pt in points:
            # Use the existing collinear function
            self.collinear(line.point1, line.point2, pt)
            # Point on circle
            self.point_lies_on(pt, circle)

    # In geoscene.relate:
    def acute_angle(self, A: Point, B:Point, C:Point):
        """
        Constrain that angle ABC is acute (less than 90 degrees).

        Args:
            A, B, C: Points defining the angle at B.
        """
        self.scene._add_relationship("acute_angle", {"A": A, "B": B, "C": C})

        # Interior angle at B: use dot product of BA and BC
        dot = (A.x - B.x) * (C.x - B.x) + (A.y - B.y) * (C.y - B.y)
        self.scene.constraint.gt(dot, 0, f"Acute angle at {B.name}")

    def right_angle(self, A:Point, B:Point, C:Point):
        """
        Constrain that angle ABC is right (exactly 90 degrees).

        Args:
            A, B, C: Points defining the angle at B.
        """
        self.scene._add_relationship("right_angle", {"A": A, "B": B, "C": C})

        dot = (A.x - B.x) * (C.x - B.x) + (A.y - B.y) * (C.y - B.y)
        self.scene.constraint.eq(dot, 0, f"Right angle at {B.name}")

    def obtuse_angle(self, A:Point, B:Point, C:Point):
        """
        Constrain that angle ABC is obtuse (greater than 90 degrees).

        Args:
            A, B, C: Points defining the angle at B.
        """
        self.scene._add_relationship("obtuse_angle", {"A": A, "B": B, "C": C})

        dot = (A.x - B.x) * (C.x - B.x) + (A.y - B.y) * (C.y - B.y)
        self.scene.constraint.lt(dot, 0, f"Obtuse angle at {B.name}")

    def lines_intersect_at(self, line1: LineLike, line2: LineLike, P: Point):
        """
        Constrain that two lines intersect at a given point.

        Args:
            line1: First line.
            line2: Second line.
            P: Intersection point.
        """
        self.scene._add_relationship("lines_intersect_at", {"line1": line1, "line2": line2, "P": P})

        self.point_lies_on(P, line1)
        self.point_lies_on(P, line2)

    def line_extensions_intersect_at(self, line1: LineLike, line2: LineLike, P: Point):
        """
        Constrain that the extensions of two line segments intersect at a given point.
        line1 and line2 can be of different types, for instance line1 is ray and line2 is line segment.
        In this case ray intersects extension of line2 at P.

        Args:
            line1: First line segment, ray, infinite line.
            line2: Second line segment, ray, infinite line.
            P: Intersection point.
        """
        self.scene._add_relationship("line_extensions_intersect_at", {"line1": line1, "line2": line2, "P": P})

        self.collinear(line1.point1, line1.point2, P, force_direction=False)
        self.collinear(line2.point1, line2.point2, P, force_direction=False)

    def inside(
        self,
        obj: Optional[Union[Point, LineSegment, Polygon, Circle]],
        region: Optional[Union[Circle, Triangle, Quadrilateral, RegularPolygon]],
        inclusive: Optional[bool] = False,
    ):
        """
        Constrain that an object lies inside a given region.

        Supports points, line segments, polygons, and circles inside
        circular or polygonal regions.

        Args:
            obj: Object to constrain inside the region.
            region: The containing region.
            inclusive: Whether boundary is included.
        """
        #raise NotImplementedError("This function is really bad implemented. Do not use.")
        self.scene._add_relationship("inside", {"obj": obj, "region": region, "inclusive": inclusive})

        # Point inside region
        if isinstance(obj, Point):
            if isinstance(region, Circle):
                if inclusive:
                    self.scene.constraint.leq(
                        obj.distance_squared(region.center) - region.radius**2,
                        0,
                        f"{obj.name} inside (inclusive) {region.name}",
                    )
                else:
                    self.scene.constraint.lt(
                        obj.distance_squared(region.center) - region.radius**2,
                        0,
                        f"{obj.name} inside {region.name}",
                    )
            elif isinstance(region, (Triangle, Quadrilateral, RegularPolygon)):
                # Use cross product method for convex polygons
                points = region.points
                n = len(points)
                for i in range(n):
                    P1 = points[i]
                    P2 = points[(i + 1) % n]
                    cross = (P2.x - P1.x) * (obj.y - P1.y) - (P2.y - P1.y) * (
                        obj.x - P1.x
                    )
                    if inclusive:
                        self.scene.constraint.geq(
                            cross,
                            0,
                            f"{obj.name} inside (inclusive) {region.name} (edge {i})",
                        )
                    else:
                        self.scene.constraint.gt(
                            cross, 0, f"{obj.name} inside {region.name} (edge {i})"
                        )

        # Line, Polygon inside region: all points inside
        elif isinstance(obj, LineSegment):
            self.inside(obj.point1, region, inclusive=inclusive)
            self.inside(obj.point2, region, inclusive=inclusive)
        elif isinstance(obj, Polygon):
            for pt in obj.points:
                self.inside(pt, region, inclusive=inclusive)

        elif isinstance(obj, Circle):
            if isinstance(region, Circle):
                if inclusive:
                    self.scene.constraint.leq(
                        obj.center.distance_squared(region.center),
                        (region.radius - obj.radius) ** 2,
                        f"{obj.name} inside (inclusive) {region.name}",
                    )
                else:
                    self.scene.constraint.lt(
                        obj.center.distance_squared(region.center),
                        (region.radius - obj.radius) ** 2,
                        f"{obj.name} inside {region.name}",
                    )
            else:
                raise NotImplementedError(
                    "Circle inside/outside polygons not implemented."
                )

    def outside(
        self,
        obj: Optional[Union[Point, LineSegment, Polygon, Circle]],
        region: Optional[Union[Circle, Triangle, Quadrilateral, RegularPolygon]],
        inclusive: bool = False,
    ):
        """
        Constrain that an object lies outside a given region.

        Supports points, line segments, polygons, and circles outside
        circular or polygonal regions.

        Args:
            obj: Object to constrain outside the region.
            region: The excluding region.
            inclusive: Whether boundary is included.
        """
        #raise NotImplementedError("This function is really bad implemented. Do not use.")
        self.scene._add_relationship("outside", {"obj": obj, "region": region, "inclusive": inclusive})

        # Point outside region
        if isinstance(obj, Point):
            if isinstance(region, Circle):
                if inclusive:
                    self.scene.constraint.geq(
                        obj.distance_squared(region.center) - region.radius**2,
                        0,
                        f"{obj.name} outside (inclusive) {region.name}",
                    )
                else:
                    self.scene.constraint.gt(
                        obj.distance_squared(region.center) - region.radius**2,
                        0,
                        f"{obj.name} outside {region.name}",
                    )
            elif isinstance(region, (Triangle, Quadrilateral, RegularPolygon)):
                # Use cross product method for convex polygons

                #TODO: not enough that points lie outside the region, lines cannot intersect
                #TODO: maybe one way is to use the circumcircles, say that circumcircles
                points = region.points
                n = len(points)
                for i in range(n):
                    P1 = points[i]
                    P2 = points[(i + 1) % n]
                    cross = (P2.x - P1.x) * (obj.y - P1.y) - (P2.y - P1.y) * (
                        obj.x - P1.x
                    )
                    if inclusive:
                        self.scene.constraint.leq(
                            cross,
                            0,
                            f"{obj.name} outside (inclusive) {region.name} (edge {i})",
                        )
                    else:
                        self.scene.constraint.lt(
                            cross, 0, f"{obj.name} outside {region.name} (edge {i})"
                        )

        # Line, Polygon outside region: all points outside
        elif isinstance(obj, LineSegment):
            self.outside(obj.point1, region, inclusive=inclusive)
            self.outside(obj.point2, region, inclusive=inclusive)
        elif isinstance(obj, Polygon):
            for pt in obj.points:
                self.outside(pt, region, inclusive=inclusive)

        elif isinstance(obj, Circle):
            if isinstance(region, Circle):
                if inclusive:
                    self.scene.constraint.geq(
                        obj.center.distance_squared(region.center),
                        (obj.radius + region.radius) ** 2,
                        f"{obj.name} outside (inclusive) {region.name}",
                    )
                else:
                    self.scene.constraint.gt(
                        obj.center.distance_squared(region.center),
                        (obj.radius + region.radius) ** 2,
                        f"{obj.name} outside {region.name}",
                    )
            else:
                raise NotImplementedError(
                    "Circle inside/outside polygons not implemented."
                )

    def congruent(
        self,
        obj1: Union[LineSegment, Polygon, Circle, MajorArc, MinorArc],
        obj2: Union[LineSegment, Polygon, Circle, MajorArc, MinorArc],
    ):
        """
        Constrain that two objects are congruent (same shape and size).

        Supports line segments, polygons, circles, and arcs.

        Args:
            obj1: First object.
            obj2: Second object.
        """

        if not isinstance(obj2, type(obj1)):
            raise TypeError("Error: Object 1 must be of the same type as object 2 for congruence constraint.")

        self.scene._add_relationship("congruent", {"obj1": obj1, "obj2": obj2})
        
        if isinstance(obj1, LineSegment) and isinstance(obj2, LineSegment):
            self.scene.constraint.eq(
                obj1.length,
                obj2.length,
                f"Congruent: {obj1.name} and {obj2.name} (same lenght)",
            )    
        elif isinstance(obj1, Circle) and isinstance(obj2, Circle):
            self.scene.constraint.eq(
                obj1.radius,
                obj2.radius,
                f"Congruent: {obj1.name} and {obj2.name} (radii equal)",
            )

        elif (isinstance(obj1, MajorArc) and isinstance(obj2, MajorArc)) or (
            isinstance(obj1, MinorArc) and isinstance(obj2, MinorArc)
        ):
            self.scene.constraint.eq(
                obj1.radius,
                obj2.radius,
                f"Congruent: {obj1.name} and {obj2.name} (radii equal)",
            )
            self.scene.constraint.eq(
                obj1.central_angle,
                obj2.central_angle,
                f"Congruent: {obj1.name} and {obj2.name} (central angles equal)",
            )

        elif isinstance(obj1, RegularPolygon) and isinstance(obj2, RegularPolygon):
            self.scene.constraint.eq(
                obj1.circumradius,
                obj2.circumradius,
                f"Congruence: {obj1.name} and {obj2.name} have same circumradius",
            )
        elif isinstance(obj1, Polygon) and isinstance(obj2, Polygon):
            n = len(obj1.points)
            if n != len(obj2.points):
                raise ValueError(
                    "Polygons must have the same number of vertices for congruence."
                )

            # Corresponding side lengths equal
            for i in range(n):
                side1 = obj1.points[i].distance_squared(obj1.points[(i + 1) % n])
                side2 = obj2.points[i].distance_squared(obj2.points[(i + 1) % n])
                angle1 = obj1.points[i].angle(
                    obj1.points[i - 1], obj1.points[(i + 1) % n]
                )
                angle2 = obj2.points[i].angle(
                    obj2.points[i - 1], obj2.points[(i + 1) % n]
                )
                self.scene.constraint.eq(
                    side1, side2, f"Congruent: side {i} of {obj1.name} and {obj2.name}"
                )
                self.scene.constraint.eq(
                    angle1,
                    angle2,
                    f"Similarity: angle {i} of {obj1.name} and {obj2.name}",
                )
        else:
            raise NotImplementedError(
                "Congruence between the give object types not yet implemented."
            )

    def similar(
        self,
        obj1: Union[Polygon, Circle, MajorArc, MinorArc],
        obj2: Union[Polygon, Circle, MajorArc, MinorArc],
    ):
        """
        Constrain that two objects are similar (same shape, proportional size).

        Supports polygons, circles, and arcs.

        Args:
            obj1: First object.
            obj2: Second object.
        """
        self.scene._add_relationship("similar", {"obj1": obj1, "obj2": obj2})

        # Circles
        if isinstance(obj1, Circle) and isinstance(obj2, Circle):
            print(f"[WARNING] All circles are similar: {obj1.name}, {obj2.name}")

        elif (isinstance(obj1, MajorArc) and isinstance(obj2, MajorArc)) or (
            isinstance(obj1, MinorArc) and isinstance(obj2, MinorArc)
        ):
            self.scene.constraint.eq(
                obj1.central_angle,
                obj2.central_angle,
                f"Similar: {obj1.name} and {obj2.name} (central angles equal)",
            )
        elif isinstance(obj1, RegularPolygon) and isinstance(obj2, RegularPolygon):
            print("[WARNING] Regular polygons are already similar. No condition added.")
            
        elif isinstance(obj1, Polygon) and isinstance(obj2, Polygon):
            n = len(obj1.points)
            if n != len(obj2.points):
                raise ValueError(
                    "Polygons must have the same number of vertices for similarity."
                )

            # All corresponding side ratios are equal: s0/t0 = s1/t1 = ... = sn/tn
            for i in range(1, n):
                side1_0 = obj1.points[0].distance_squared(obj1.points[1])
                side2_0 = obj2.points[0].distance_squared(obj2.points[1])
                side1_i = obj1.points[i].distance_squared(obj1.points[(i + 1) % n])
                side2_i = obj2.points[i].distance_squared(obj2.points[(i + 1) % n])
                # Cross-multiplied to avoid division:
                # side1_0 * side2_i == side2_0 * side1_i
                self.scene.constraint.eq(
                    side1_0 * side2_i, side2_0 * side1_i, f"Similarity: side {i} ratio"
                )

            # All corresponding angles must be equal
            for i in range(n):
                angle1 = obj1.points[i].angle(
                    obj1.points[i - 1], obj1.points[(i + 1) % n]
                )
                angle2 = obj2.points[i].angle(
                    obj2.points[i - 1], obj2.points[(i + 1) % n]
                )
                self.scene.constraint.eq(
                    angle1,
                    angle2,
                    f"Similarity: angle {i} of {obj1.name} and {obj2.name}",
                )
        else:
            raise NotImplementedError(
                "Congruence between the give object types not yet implemented."
            )

    def is_radius(self, line: LineSegment, circle: Circle):
        """
        Constrain a line segment to be a radius of a circle.

        One endpoint must be the center, and the other must lie on the circle.

        Args:
            line: Line segment to constrain.
            circle: Circle to constrain radius of.
        """
        self.scene._add_relationship("is_radius", {"line": line, "circle": circle})

        # Check if either endpoint is the center
        p1_is_center = (line.point1.x == circle.center.x) and (
            line.point1.y == circle.center.y
        )
        p2_is_center = (line.point2.x == circle.center.x) and (
            line.point2.y == circle.center.y
        )

        if not (p1_is_center or p2_is_center):
            raise ValueError(
                f"Neither endpoint of {line.name} is the center of {circle.name}."
            )

        if p1_is_center:
            self.scene.constraint.eq(
                line.point2.distance_squared(circle.center),
                circle.radius**2,
                f"{line.name} endpoint on circle (2)",
            )
        else:  # p2_is_center
            self.scene.constraint.eq(
                line.point1.distance_squared(circle.center),
                circle.radius**2,
                f"{line.name} endpoint on circle (1)",
            )



    def is_midpoint(self, point: Point, obj: Union[LineSegment, MinorArc, MajorArc]):
        """
        Constrain a point to be the midpoint of a line segment or an arc.

        Args:
            point: The midpoint.
            obj: The line segment or arc.
        """
        # This part remains the same
        self.scene._add_relationship("is_midpoint", {"point": point, "obj": obj})

        if isinstance(obj, LineSegment):
            # The logic for LineSegment is correct
            x_mid = (obj.point1.x + obj.point2.x) / 2
            y_mid = (obj.point1.y + obj.point2.y) / 2

        else:  # Handles MinorArc and MajorArc
            # --- NEW, ROBUST LOGIC FOR ARCS ---

            cx, cy = obj.center.x, obj.center.y
            r = obj.radius
            sx, sy = obj.start_point.x, obj.start_point.y
            ex, ey = obj.end_point.x, obj.end_point.y

            # Vectors from the center to the start and end points
            us_x, us_y = sx - cx, sy - cy
            ue_x, ue_y = ex - cx, ey - cy

            # The sum of the two vectors gives a vector that bisects the angle
            # between them, pointing toward the minor arc's midpoint.
            mid_dir_x = us_x + ue_x
            mid_dir_y = us_y + ue_y
            
            # For a MajorArc, we need the opposite direction.
            if isinstance(obj, MajorArc):
                mid_dir_x = -mid_dir_x
                mid_dir_y = -mid_dir_y

            # We normalize the direction vector and scale it by the radius.
            # Note: This could cause a division by zero if start and end points are
            # diametrically opposite, as mid_dir would be the zero vector.
            # A full implementation might use sympy.Piecewise to handle this edge case.
            len_mid_dir = sympy.sqrt(mid_dir_x**2 + mid_dir_y**2)

            # Final midpoint coordinates are the center plus the scaled direction vector.
            x_mid = cx + r * mid_dir_x / len_mid_dir
            y_mid = cy + r * mid_dir_y / len_mid_dir


        desc = f"{point.name} = mid_{obj.name}"
        self.scene.constraint.eq(point.x, x_mid, desc)
        self.scene.constraint.eq(point.y, y_mid, desc)

    def is_diameter(self, line: LineSegment, circle: Circle):
        """
        Constrain a line segment to be a diameter of a circle.

        Both endpoints lie on the circle and the line passes through the center.

        Args:
            line: The line segment.
            circle: The circle.
        """
        self.scene._add_relationship("is_diameter", {"line": line, "circle": circle})

        # Both endpoints on the circle
        self.scene.constraint.eq(
            line.point1.distance_squared(circle.center),
            circle.radius**2,
            f"{line.point1.name} on {circle.name}",
        )
        self.scene.constraint.eq(
            line.point2.distance_squared(circle.center),
            circle.radius**2,
            f"{line.point2.name} on {circle.name}",
        )
        # Center is collinear with endpoints
        self.collinear(line.point1, circle.center, line.point2)

    def is_circumcircle(self, circle: Circle, polygon: Union[Triangle, RegularPolygon, Square, Rectangle, CyclicPolygon]):
        """
        Constrain a circle to be the circumcircle of a polygon.

        The circle's center must equal the polygon's circumcenter, and the circle's
        radius must equal the polygon's circumradius.

        Args:
            circle: The circle to constrain as the circumcircle.
            polygon: The polygon (must have circumcenter and circumradius properties).

        Raises:
            AttributeError: If the polygon does not have circumcenter or circumradius properties.
        """
        self.scene._add_relationship("is_circumcircle", {"circle": circle, "polygon": polygon})

        # Check if polygon has required properties
        if not hasattr(polygon, "circumcenter"):
            raise AttributeError(
                f"{polygon.__class__.__name__} does not have a circumcenter property. "
                "is_circumcircle is only supported for Triangle, RegularPolygon, Square, Rectangle, and CyclicPolygon."
            )
        if not hasattr(polygon, "circumradius"):
            raise AttributeError(
                f"{polygon.__class__.__name__} does not have a circumradius property. "
                "is_circumcircle is only supported for Triangle, RegularPolygon, Square, Rectangle, and CyclicPolygon."
            )

        # Constrain center to match circumcenter
        self.scene.constraint.eq(
            circle.center.x,
            polygon.circumcenter.x,
            f"{circle.name}.center.x = {polygon.name}.circumcenter.x",
        )
        self.scene.constraint.eq(
            circle.center.y,
            polygon.circumcenter.y,
            f"{circle.name}.center.y = {polygon.name}.circumcenter.y",
        )

        # Constrain radius to match circumradius
        self.scene.constraint.eq(
            circle.radius,
            polygon.circumradius,
            f"{circle.name}.radius = {polygon.name}.circumradius",
        )

    def is_incircle(self, circle: Circle, polygon: Union[Triangle, RegularPolygon, Square]):
        """
        Constrain a circle to be the incircle of a polygon.

        The circle's center must equal the polygon's incenter, and the circle's
        radius must equal the polygon's inradius.

        Args:
            circle: The circle to constrain as the incircle.
            polygon: The polygon (must have incenter and inradius properties).

        Raises:
            AttributeError: If the polygon does not have incenter or inradius properties.
        """
        self.scene._add_relationship("is_incircle", {"circle": circle, "polygon": polygon})

        # Check if polygon has required properties
        if not hasattr(polygon, "incenter"):
            raise AttributeError(
                f"{polygon.__class__.__name__} does not have an incenter property. "
                "is_incircle is only supported for Triangle, RegularPolygon, and Square."
            )
        if not hasattr(polygon, "inradius"):
            raise AttributeError(
                f"{polygon.__class__.__name__} does not have an inradius property. "
                "is_incircle is only supported for Triangle, RegularPolygon, and Square."
            )

        # Constrain center to match incenter
        self.scene.constraint.eq(
            circle.center.x,
            polygon.incenter.x,
            f"{circle.name}.center.x = {polygon.name}.incenter.x",
        )
        self.scene.constraint.eq(
            circle.center.y,
            polygon.incenter.y,
            f"{circle.name}.center.y = {polygon.name}.incenter.y",
        )

        # Constrain radius to match inradius
        self.scene.constraint.eq(
            circle.radius,
            polygon.inradius,
            f"{circle.name}.radius = {polygon.name}.inradius",
        )

    def is_orthocenter(self, point: Point, triangle: Triangle):
        """
        Constrain a point to be the orthocenter of a triangle.

        The orthocenter is the intersection point of the three altitudes of a triangle.

        Args:
            point: The point to constrain as the orthocenter.
            triangle: The triangle (Triangle).

        Raises:
            TypeError: If triangle is not a Triangle instance.
        """
        self.scene._add_relationship("is_orthocenter", {"point": point, "triangle": triangle})

        if not isinstance(triangle, Triangle):
            raise TypeError(
                f"is_orthocenter is only supported for Triangle. "
                f"Got {triangle.__class__.__name__}."
            )

        # Constrain point to match orthocenter coordinates
        self.scene.constraint.eq(
            point.x,
            triangle.orthocenter.x,
            f"{point.name}.x = {triangle.name}.orthocenter.x",
        )
        self.scene.constraint.eq(
            point.y,
            triangle.orthocenter.y,
            f"{point.name}.y = {triangle.name}.orthocenter.y",
        )

    def is_centroid(self, point: Point, polygon: Polygon):
        """
        Constrain a point to be the centroid of a polygon.

        The centroid is the average of all vertices (vertex centroid).

        Args:
            point: The point to constrain as the centroid.
            polygon: The polygon (Polygon or any subclass).
        """
        self.scene._add_relationship("is_centroid", {"point": point, "polygon": polygon})

        # Constrain point to match centroid coordinates
        self.scene.constraint.eq(
            point.x,
            polygon.centroid.x,
            f"{point.name}.x = {polygon.name}.centroid.x",
        )
        self.scene.constraint.eq(
            point.y,
            polygon.centroid.y,
            f"{point.name}.y = {polygon.name}.centroid.y",
        )

    def is_median(self, line: LineSegment, triangle: Triangle, vertex: Point):
        """
        Constrain a line segment to be a median of a triangle.

        A median connects a vertex to the midpoint of the opposite side.

        Args:
            line: The line segment to constrain as a median.
            triangle: The triangle (Triangle).
            vertex: The vertex Point from which the median originates (must be one of triangle.points).

        Raises:
            TypeError: If triangle is not a Triangle instance.
            ValueError: If vertex is not one of the triangle's vertices.
        """
        self.scene._add_relationship("is_median", {"line": line, "triangle": triangle, "vertex": vertex})

        if not isinstance(triangle, Triangle):
            raise TypeError(
                f"is_median is only supported for Triangle. "
                f"Got {triangle.__class__.__name__}."
            )
        
        if vertex not in triangle.points:
            raise ValueError(
                f"vertex must be one of the triangle's vertices. "
                f"Got {vertex.name}, triangle vertices: {[p.name for p in triangle.points]}"
            )

        # Find the index of the vertex
        vertex_index = triangle.points.index(vertex)
        medians = triangle.medians
        median = medians[vertex_index]
        
        # Constrain line to match the median (either direction)
        self.scene.constraint.eq(
            line.point1.x, median.point1.x,
            f"{line.name}.point1.x = {median.name}.point1.x"
        )
        self.scene.constraint.eq(
            line.point1.y, median.point1.y,
            f"{line.name}.point1.y = {median.name}.point1.y"
        )
        self.scene.constraint.eq(
            line.point2.x, median.point2.x,
            f"{line.name}.point2.x = {median.name}.point2.x"
        )
        self.scene.constraint.eq(
            line.point2.y, median.point2.y,
            f"{line.name}.point2.y = {median.name}.point2.y"
        )

    def is_altitude(self, line: LineSegment, triangle: Triangle, vertex: Point):
        """
        Constrain a line segment to be an altitude of a triangle.

        An altitude is perpendicular from a vertex to the opposite side.

        Args:
            line: The line segment to constrain as an altitude.
            triangle: The triangle (Triangle).
            vertex: The vertex Point from which the altitude originates (must be one of triangle.points).

        Raises:
            TypeError: If triangle is not a Triangle instance.
            ValueError: If vertex is not one of the triangle's vertices.
        """
        self.scene._add_relationship("is_altitude", {"line": line, "triangle": triangle, "vertex": vertex})

        if not isinstance(triangle, Triangle):
            raise TypeError(
                f"is_altitude is only supported for Triangle. "
                f"Got {triangle.__class__.__name__}."
            )
        
        if vertex not in triangle.points:
            raise ValueError(
                f"vertex must be one of the triangle's vertices. "
                f"Got {vertex.name}, triangle vertices: {[p.name for p in triangle.points]}"
            )

        # Find the index of the vertex
        vertex_index = triangle.points.index(vertex)
        altitudes = triangle.altitudes
        altitude = altitudes[vertex_index]
        
        # Constrain line to match the altitude (either direction)
        self.scene.constraint.eq(
            line.point1.x, altitude.point1.x,
            f"{line.name}.point1.x = {altitude.name}.point1.x"
        )
        self.scene.constraint.eq(
            line.point1.y, altitude.point1.y,
            f"{line.name}.point1.y = {altitude.name}.point1.y"
        )
        self.scene.constraint.eq(
            line.point2.x, altitude.point2.x,
            f"{line.name}.point2.x = {altitude.name}.point2.x"
        )
        self.scene.constraint.eq(
            line.point2.y, altitude.point2.y,
            f"{line.name}.point2.y = {altitude.name}.point2.y"
        )

    def translation(self, obj1: 'valid_objects', obj2: 'valid_objects',
                                dx: Optional['Coordinate'] = None, dy: Optional['Coordinate'] = None) -> None:
        """
        Constrains obj1 to be a translation of obj2 by (dx, dy) units.
        If dx or dy are not provided, it implies they are free to be any value
        that satisfies the translation for the other coordinate.
        
        Args:
            obj1: The object to constrain (will be at obj2's position + translation).
            obj2: The reference object.
            dx: The translation amount along the x-axis. If None, the x-component
                of the translation is not explicitly constrained here, but implied
                by the overall translation of the object's defining points.
            dy: The translation amount along the y-axis. If None, the y-component
                of the translation is not explicitly constrained here.

        Raises:
            TypeError: If obj1 and obj2 are not of compatible types or if an
                    unsupported object type is provided.
            ValueError: If a translation is attempted on a non-point object
                        without corresponding structural points (e.g., trying to
                        translate a LineLike object where point1/point2 are not present).
        """
        self.scene._add_relationship("translation", {"obj1": obj1, "obj2": obj2, "dx": dx, "dy": dy})

        if not isinstance(obj2, type(obj1)):
            raise TypeError("Different object types for translation constraint.")

        # --- Apply Constraints based on Object Type ---
        if isinstance(obj1, Point):
            if dx is not None:
                self.scene.constraint.eq(obj1.x, obj2.x + dx, f"{obj1.name} is translation if {obj2.name} (x)")
            if dy is not None:
                self.scene.constraint.eq(obj1.y, obj2.y + dy, f"{obj1.name} is translation if {obj2.name} (y)")
        
        elif isinstance(obj1, Circle):
            # For a circle, only its center needs to be translated. Its radius should be invariant.
            self.translation(obj1.center, obj2.center, dx, dy)
            self.scene.constraint.eq(obj1.radius, obj2.radius, f"{obj1.name} is translation if {obj2.name} (equal radius)") # Implicit: radius must be same

        elif isinstance(obj1, BaseArc):
            self.translation(obj1.center, obj2.center, dx, dy)
            self.translation(obj1.start_point, obj2.start_point, dx, dy)
            self.translation(obj1.end_point, obj2.end_point, dx, dy)
            
        elif isinstance(obj1, LineLike): # Assuming LineLike has point1 and point2
            self.translation(obj1.point1, obj2.point1, dx, dy)
            self.translation(obj1.point2, obj2.point2, dx, dy)

        elif isinstance(obj1, RegularPolygon):
            self.translation(obj1.center, obj2.center, dx, dy)
            self.scene.constraint.eq(obj1.inradius, obj2.inradius, f"{obj1.name} is translation if {obj2.name} (equal inradius)") # Or apothem, or circumradius
            self.scene.constraint.eq(obj1.circumradius, obj2.orientation, f"{obj1.name} is translation if {obj2.name} (equal orientation)") # Or apothem, or circumradius

        # Handle a general polygon or a collection of points
        elif hasattr(obj1, 'points') and hasattr(obj2, 'points') and len(obj1.points) == len(obj2.points):
            for p1, p2 in zip(obj1.points, obj2.points):
                # Ensure p1 and p2 are indeed points
                if isinstance(p1, Point) and isinstance(p2, Point):
                    self.translation(p1, p2, dx, dy)
                else:
                    raise TypeError(f"Points in object collection are not of type 'Point' for translation constraint.")
        else:
            raise TypeError(f"Cannot constrain translation for objects of type {type(obj1).__name__}.")

    def scale(self, obj1: 'valid_objects', obj2: 'valid_objects',
                    scale: 'Coordinate') -> None:
        """
        Constrain obj1 to be a scaled version of obj2 by a scale factor.

        Angles are preserved and lengths are scaled proportionally.

        Args:
            obj1: The scaled object.
            obj2: The reference object.
            scale: Scaling factor.

        Raises:
            TypeError: If obj1 and obj2 are incompatible types.
            ValueError: If scale factor is zero or invalid.
        """
        self.scene._add_relationship("scale", {"obj1": obj1, "obj2": obj2, "scale": scale})

        if not isinstance(obj2, type(obj1)):
            raise TypeError("Incompatible object types for scale constraint.")

        elif isinstance(obj1, Circle):
            # For a circle, only its center needs to be translated. Its radius should be invariant.
            self.scene.constraint.eq(obj1.radius, scale*obj2.radius, f"{obj1.name} is scaling of {obj2.name}") # Implicit: radius must be same

        elif isinstance(obj1, BaseArc):
            self.scene.constraint.eq(obj1.radius, scale*obj2.radius, f"{obj1.name} is scaling of {obj2.name}")
            
        elif isinstance(obj1, LineLike): 
            self.scene.constraint.eq(obj1.length, scale*obj2.length, f"{obj1.name} is scaling of {obj2.name}")

        elif isinstance(obj1, RegularPolygon):
            self.scene.constraint.eq(obj1.circumradius, scale*obj2.circumradius, f"{obj1.name} is scaling of {obj2.name}") # Or apothem, or circumradius

        # Handle a general polygon or a collection of points
        # All internal angles are the same
        # Side lenghts proportional by scale
        elif isinstance(obj1, Polygon):
            p1 = obj1.points
            p2 = obj2.points
            n = len(p1)
            for i in range(n):
                angle1 = self.scene.angle(p1[i-1],p1[i],p1[i % n-1])
                angle2 = self.scene.angle(p2[i-1],p2[i],p2[i % n-1])
                self.scene.constraint.eq(angle1, angle2, f"{obj1.name} is scaling of {obj2.name} (same internal angles)")
            for a1, a2 in zip(obj1.sides, obj2.sides):
                self.scene.constraint.eq(a1.length,scale*a2.length, f"{obj1.name} is scaling of {obj2.name} (sides related by scale factor)")   
        else:
            raise TypeError(f"Cannot constrain translation for objects of type {type(obj1).__name__}.")
        
    def rotation_around_point(self, obj1: Union[LineSegment, Polygon], obj2: Union[LineLike, Polygon],
                              point: Point, angle: 'Coordinate') -> None:
        """
        Constrain obj1 to be a rotation of obj2 around a point by a given angle (in degrees).

        Both polygons share the rotation center vertex.

        Args:
            obj1: Rotated polygon.
            obj2: Reference polygon.
            point: Rotation center (common vertex).
            angle: Rotation angle in degrees (CCW positive).

        Raises:
            TypeError: If obj1 or obj2 is not a Polygon.
            ValueError: If point is not a common vertex of both polygons.
        """
        self.scene._add_relationship("rotation_around_point", {"obj1": obj1, "obj2": obj2, "point": point, "angle": angle})

        if not isinstance(obj1, type(obj2)):
            raise TypeError("obj1 and obj2 must be of the same type.")

        #get all points of obj1 and obj2
        if isinstance(obj1, Polygon):
            obj1_points = obj1.points
            obj2_points = obj2.points
        else:
            obj1_points = [obj1.point1, obj1.point2]
            obj2_points = [obj2.point1, obj2.point2]

        # Check if 'point' is a common vertex in both polygons' point lists
        point_in_obj1 = any(p.name == point.name for p in obj1_points)
        point_in_obj2 = any(p.name == point.name for p in obj2_points)

        if not (point_in_obj1 and point_in_obj2):
            raise ValueError(f"Point {point.name} is not a common vertex of {obj1.name} and {obj2.name}.")
        
        # First, ensure congruence between the two polygons.
        #self.congruent(obj1, obj2) -> not needed anymore

        # Get the actual Point object instances for the rotation center from both lists
        center_obj1_index = obj1_points.index(point)
        center_obj2_index = obj2_points.index(point)

        def restart_vector_from_index(vector: list, index: int) -> list:
            return vector[index % len(vector):] + vector[:index % len(vector)] if vector else []

        obj1_non_center_points = restart_vector_from_index(obj1_points, center_obj1_index)
        obj2_non_center_points = restart_vector_from_index(obj2_points, center_obj2_index)

        del obj1_non_center_points[0]
        del obj2_non_center_points[0]

        angle_rad = sympy.pi * angle / 180 # Convert degrees to radians directly using sympy.pi
        cos_theta = sympy.cos(angle_rad)
        sin_theta = sympy.sin(angle_rad)

        # Iterate through each point in obj2 (excluding the center)
        for i in range(len(obj1_non_center_points)):
            p1_dest = obj1_non_center_points[i] # Destination point from obj1
            p2_src = obj2_non_center_points[i]   # Source point from obj2

            # Calculate the expected coordinates of p1_dest based on rotating p2_src around 'point'
            # 1. Translate p2_src relative to the rotation center 'point'
            p2_rel_x = p2_src.x - point.x
            p2_rel_y = p2_src.y - point.y

            # 2. Rotate the relative coordinates
            rotated_rel_x = p2_rel_x * cos_theta - p2_rel_y * sin_theta
            rotated_rel_y = p2_rel_x * sin_theta + p2_rel_y * cos_theta

            # 3. Translate back to the original coordinate system
            expected_p1_x = rotated_rel_x + point.x
            expected_p1_y = rotated_rel_y + point.y

            # Constrain p1_dest's coordinates to be equal to the expected rotated coordinates
            self.scene.constraint.eq(
                p1_dest.x,
                expected_p1_x,
                f"{p1_dest.name}.x is rotated {angle}deg from {p2_src.name}.x around {point.name}"
            )
            self.scene.constraint.eq(
                p1_dest.y,
                expected_p1_y,
                f"{p1_dest.name}.y is rotated {angle}deg from {p2_src.name}.y around {point.name}"
            )



    def mirror_across_line(
        self,
        obj1: valid_objects,
        obj2: valid_objects,
        axis_line: LineLike
    ):
        """
        Constrain obj1 to be the mirror image of obj2 across the given line.

        Supports points, line segments, circles, arcs, and polygons.

        Args:
            obj1: The mirrored object.
            obj2: The reference object.
            axis_line: The line of reflection.

        Raises:
            TypeError: If obj1 and obj2 are incompatible types.
            NotImplementedError: If types are unsupported for mirroring.
        """
        self.scene._add_relationship("mirror_across_line", {"obj1": obj1, "obj2": obj2, "axis_line": axis_line})

        if not isinstance(obj2, type(obj1)):
            raise TypeError("Different object types for mirror constraint.")

        if isinstance(obj1, Point):
            A = axis_line.point1
            B = axis_line.point2
            dx = B.x - A.x
            dy = B.y - A.y
            denom = dx**2 + dy**2
            d = ((obj2.x - A.x) * dx + (obj2.y - A.y) * dy) / denom
            Fx = A.x + d * dx
            Fy = A.y + d * dy
            Rx = 2 * Fx - obj2.x
            Ry = 2 * Fy - obj2.y
            self.scene.constraint.eq(obj1.x, Rx, f"{obj1.name} is mirror across line {axis_line.name} of {obj2.name}: {obj1.name} (x)")
            self.scene.constraint.eq(obj1.y, Ry, f"{obj1.name} is mirror across line {axis_line.name} of {obj2.name}: {obj1.name} (y)")

        elif isinstance(obj1, LineSegment) and isinstance(obj2, LineSegment):
            self.mirror_across_line(obj1.point1, obj2.point1, axis_line)
            self.mirror_across_line(obj1.point2, obj2.point2, axis_line)

        elif isinstance(obj1, Circle) and isinstance(obj2, Circle):
            self.mirror_across_line(obj1.center, obj2.center, axis_line)
            self.scene.constraint.eq(obj1.radius, obj2.radius, f"{obj1.name} is mirror across line {axis_line.name} of {obj2.name} (equal radius)" )

        elif (isinstance(obj1, MajorArc) and isinstance(obj2, MajorArc)) or (
            isinstance(obj1, MinorArc) and isinstance(obj2, MinorArc)
        ):
            self.mirror_across_line(obj1.center, obj2.center, axis_line)
            self.mirror_across_line(obj1.start_point, obj2.start_point, axis_line)
            self.mirror_across_line(obj1.end_point, obj2.end_point, axis_line)

        elif isinstance(obj1, Polygon) and isinstance(obj2, Polygon):
            for p1, p2 in zip(obj1.points, obj2.points):
                self.mirror_across_line(p1, p2, axis_line)

        else:
            raise NotImplementedError(
                "Mirror between the given object types not yet implemented."
            )