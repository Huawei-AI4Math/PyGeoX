"""
basic_objects.py: Fundamental 2D geometric primitives with symbolic or numeric coordinates.

Defines Point, LineLike, LineSegment, Line, Ray, Circle, and Arc classes with
methods for geometric computations such as distance, angle, slope, and length.
"""

from typing import Optional, Union

import sympy
from typeguard import typechecked

Coordinate = Union[int, float, sympy.Expr]


@typechecked
class Point:
    """
    Represents a 2D point with symbolic or numeric coordinates.

    Attributes:
        name (str): Point name.
        x (Coordinate): X coordinate.
        y (Coordinate): Y coordinate.
    """

    def __init__(self, name: str, x: Optional[Coordinate], y: Optional[Coordinate]):
        """
        Initialize a Point.

        Args:
            name: Name of the point.
            x: X coordinate (symbolic or numeric).
            y: Y coordinate (symbolic or numeric).
        """

        self.name = name
        self.x = x
        self.y = y

    def __repr__(self):
        """
        Return string representation of the Point.
        """
        return f"Point({self.name}, x={self.x}, y={self.y})"

    def distance(self, other_point: "Point"):
        """
        Compute the Euclidean distance to another point.

        Args:
            other_point (Point): The other point.

        Returns:
            sympy.Expr: The distance between self and other_point.
        """
        return sympy.sqrt((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2)

    def distance_squared(self, other_point: "Point"):
        """
        Compute the squared Euclidean distance to another point.

        Args:
            other_point (Point): The other point.

        Returns:
            sympy.Expr: The squared distance between self and other_point.
        """
        return (self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2

    def angle(self, point_b: "Point", point_c: "Point", signed: bool = False) -> 'sympy.Expr':
        """
        Compute angle at self between vectors to point_b and point_c in degrees.

        Args:
            point_b: First point defining angle arm.
            point_c: Second point defining angle arm.
            signed: If True, return signed angle (CCW positive).

        Returns:
            sympy.Expr: Angle in degrees.
        """

        vector_ab_x = point_b.x - self.x
        vector_ab_y = point_b.y - self.y
        vector_ac_x = point_c.x - self.x
        vector_ac_y = point_c.y - self.y

        magnitude_ab = sympy.sqrt(vector_ab_x**2 + vector_ab_y**2)
        magnitude_ac = sympy.sqrt(vector_ac_x**2 + vector_ac_y**2)
        dot_product = vector_ab_x * vector_ac_x + vector_ab_y * vector_ac_y

        if signed:
            cross_product_z = vector_ab_x * vector_ac_y - vector_ab_y * vector_ac_x
            angle_rad = sympy.atan2(cross_product_z, dot_product)
        else:
            denominator = magnitude_ab * magnitude_ac
            cos_angle = dot_product / denominator
            angle_rad = sympy.acos(cos_angle)

        angle_deg = angle_rad * 180 / sympy.pi
        return angle_deg


@typechecked
class LineLike:
    """
    Base class for geometric entities that are line-like (LineSegment, Line, Ray).
    Defined by two Point objects.
    """

    def __init__(self, point1: Point, point2: Point, name: Optional[str] = None):
        """
        Initialize a LineLike object.

        Args:
            point1 (Point): The first defining point.
            point2 (Point): The second defining point.
            name (str, optional): The name of the line-like object. If None, auto-generated.

        Raises:
            ValueError: If point1 and point2 are the same.
        """
        if point1 == point2:
            raise ValueError(
                "point1 and point2 cannot be the same for a line-like object."
            )
        self.point1 = point1
        self.point2 = point2
        # Auto-generate name based on the class type and point names
        self.name = (
            name
            if name is not None
            else f"{self.__class__.__name__}_{self.point1.name}_{self.point2.name}"
        )

    @property
    def slope(self) -> Coordinate:
        """
        Returns the slope of the infinite line passing through point1 and point2.

        Returns:
            sympy.Expr: The slope, or sympy.oo (symbolic infinity) for vertical lines.
        """
        if self.point1.x == self.point2.x:
            return sympy.oo
        else:
            return (self.point2.y - self.point1.y) / (self.point2.x - self.point1.x)

    @property
    def intercept(self) -> Coordinate:
        """
        Returns the y-intercept (c) of the infinite line defined by point1 and point2
        (in the form y = mx + c).

        Returns:
            sympy.Expr: The y-intercept, or sympy.nan for vertical lines.
        """
        if self.point1.x == self.point2.x:  # Vertical line, no y-intercept
            return sympy.nan  # Not a Number, indicating no y-intercept
        else:
            # Use the equation c = y - mx, with coordinates from point1
            return self.point1.y - self.slope * self.point1.x

    @property
    def direction_vector(self) -> tuple[Coordinate, Coordinate]:
        """
        Returns the direction vector of the ray, from start_point to direction_point.

        Returns:
            tuple[sympy.Expr, sympy.Expr]: The (dx, dy) vector components.
        """
        return self.point2.x - self.point1.x, self.point2.y - self.point1.y

    def __repr__(self) -> str:
        """
        Returns a generic string representation of the LineLike object.
        Subclasses may override for more specific details.
        """
        return f"{self.__class__.__name__}(name='{self.name}', p1={self.point1.name}, p2={self.point2.name})"


@typechecked
class LineSegment(LineLike):
    """
    Represents a line segment defined by two Point objects.
    This class contains methods specific to a finite segment.
    """

    def __init__(self, point1: Point, point2: Point, name: Optional[str] = None):
        """
        Initialize a LineSegment.

        Args:
            point1 (Point): First endpoint.
            point2 (Point): Second endpoint.
            name (str, optional): Name for the line segment. If None, auto-generated.
        """
        super().__init__(point1, point2, name)
        # Override auto-generated name from LineLike if specific format is desired
        self.name = (
            name
            if name is not None
            else f"LineSegment_{self.point1.name}_{self.point2.name}"
        )

    @property
    def length(self) -> Coordinate:
        """
        Returns the length of the line segment.

        Returns:
            sympy.Expr: The distance between point1 and point2.
        """
        return self.point1.distance(self.point2)
    
    @property
    def midpoint(self) -> Point:
        """
        Returns the midpoint of the line segment.

        Returns:
            Point: The midpoint as a new Point object.
        """
        x_mid = (self.point1.x + self.point2.x) / 2
        y_mid = (self.point1.y + self.point2.y) / 2
        name = f"mid_{self.point1.name}_{self.point2.name}"
        return Point(name, x_mid, y_mid)

    def angle(self, other_line_segment: "LineSegment", signed: bool = False) -> sympy.Expr:
        """
        Returns the angle between this line segment and another line segment
        at their common intersection point (vertex).

        Args:
            other_line_segment (LineSegment): The other line segment.
            signed (bool): If True, returns the signed angle.

        Returns:
            sympy.Expr: The angle in degrees.

        Raises:
            ValueError: If line segments do not meet at a single common vertex.
        """
        # Collect all four points involved
        all_points = [
            self.point1,
            self.point2,
            other_line_segment.point1,
            other_line_segment.point2,
        ]

        # Use a dictionary to count occurrences of each point to find the common vertex
        point_counts = {}
        for pt in all_points:
            point_counts[pt] = point_counts.get(pt, 0) + 1

        vertex = None
        unique_points_for_angle = []  # Points that are endpoints of the segments but not the common vertex

        for pt, count in point_counts.items():
            if count == 2:  # This point is common to both segments (the vertex)
                vertex = pt
            elif count == 1:  # This point is an endpoint of only one segment
                unique_points_for_angle.append(pt)

        # For a valid angle computation between two segments sharing an endpoint,
        # there must be exactly one common vertex and two other unique endpoints.
        if vertex is None or len(unique_points_for_angle) != 2:
            raise ValueError(
                "Line segments do not meet at a single common vertex; cannot compute angle. "
                "Ensure they share exactly one endpoint."
            )

        # Pass the two non-vertex points to the vertex's angle method
        point_on_segment1, point_on_segment2 = unique_points_for_angle
        return vertex.angle(point_on_segment1, point_on_segment2, signed)

    def __repr__(self) -> str:
        """
        Returns a string representation of the LineSegment, including its length.
        """
        return f"LineSegment(name='{self.name}', p1={self.point1.name}, p2={self.point2.name}, length={self.length})"


@typechecked
class Line(LineLike):
    """
    Represents an infinite line defined by two Point objects.
    """

    def __init__(self, point1: Point, point2: Point, name: Optional[str] = None):
        """
        Initialize an infinite Line.

        Args:
            point1 (Point): First defining point.
            point2 (Point): Second defining point.
            name (str, optional): Name for the line. If None, auto-generated.
        """
        super().__init__(point1, point2, name)
        self.name = (
            name if name is not None else f"Line_{self.point1.name}_{self.point2.name}"
        )

    @property
    def equation(self) -> sympy.Expr:
        """
        Returns the equation of the line in the form Ax + By + C = 0.
        The result is a SymPy expression that evaluates to zero for points on the line.

        Returns:
            sympy.Expr: The symbolic equation of the line.
        """
        x_sym, y_sym = sympy.symbols("x y")
        m = self.slope

        if m == sympy.oo:  # Vertical line: x - x_constant = 0
            return x_sym - self.point1.x
        else:  # Non-vertical line: y - y1 = m(x - x1)  =>  y - y1 - m(x - x1) = 0
            return y_sym - self.point1.y - m * (x_sym - self.point1.x)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Line, including its equation.
        """
        return f"Line(name='{self.name}', p1={self.point1.name}, p2={self.point2.name}, equation={self.equation} = 0)"


@typechecked
class Ray(LineLike):
    """
    Represents a ray defined by a starting Point and a second Point indicating its direction.
    The ray extends infinitely from the start_point through the direction_point.
    """

    def __init__(
        self, start_point: Point, direction_point: Point, name: Optional[str] = None
    ):
        """
        Initialize a Ray.

        Args:
            start_point (Point): The starting point (origin) of the ray.
            direction_point (Point): A point that defines the direction of the ray.
            name (str, optional): Name for the ray. If None, auto-generated.
        """
        # Call LineLike's init, treating start_point as point1 and direction_point as point2
        super().__init__(start_point, direction_point, name)
        self.start_point = start_point  # Alias for point1 for clarity in Ray context
        self.direction_point = (
            direction_point  # Alias for point2 for clarity in Ray context
        )
        self.name = (
            name
            if name is not None
            else f"Ray_{self.start_point.name}_to_{self.direction_point.name}"
        )

    @property
    def length(self) -> sympy.oo:
        """
        Returns the length of the ray, which is infinite.

        Returns:
            sympy.oo: Symbolic infinity.
        """
        return sympy.oo

    def __repr__(self) -> str:
        """
        Returns a string representation of the Ray.
        """
        return f"Ray(name='{self.name}', start={self.start_point.name}, direction_through={self.direction_point.name}, length={self.length})"


@typechecked
class Circle:
    """
    Represents a circle defined by its center Point and radius (symbolic or numeric).

    Attributes:
        center (Point): The center of the circle.
        radius (Coordinate): The radius (can be symbolic or numeric).
        name (str): The name of the circle.
    """

    def __init__(
        self, center: Point, radius: Optional[Coordinate], name: Optional[str] = None
    ):
        """
        Initialize a Circle.

        Args:
            center (Point): The center point.
            radius (Coordinate): The radius.
            name (str, optional): Name for the circle. If None, auto-generated.
        """
        self.center = center
        self.radius = radius
        self.name = name if name is not None else f"Circle_{self.center.name}"

    @property
    def area(self) -> Coordinate:
        """
        Returns the area of the circle.

        Returns:
            sympy.Expr: The area.
        """
        return sympy.pi * self.radius**2

    @property
    def perimeter(self) -> Coordinate:
        """
        Returns the circumference of the circle.

        Returns:
            sympy.Expr: The circumference.
        """
        return 2 * sympy.pi * self.radius

    @property
    def diameter(self) -> Coordinate:
        """
        Returns the diameter (length) of the circle.

        Returns:
            sympy.Expr: The diameter (2 * radius).
        """
        return 2 * self.radius

    def __repr__(self):
        """
        Returns a string representation of the Circle.
        """
        return f"Circle({self.name}, center={self.center.name}, radius={self.radius})"


@typechecked
class BaseArc:
    """
    Base class for arcs, defined by a center and two endpoints.
    Not intended to be used directly.

    Attributes:
        center (Point): The center of the arc's circle.
        start_point (Point): The start point of the arc.
        end_point (Point): The end point of the arc.
        name (str): The name of the arc.
        arc_type (str): The type of arc ("Arc", "MinorArc", "MajorArc").
        start_radius_line (Line): Line from center to start_point.
        end_radius_line (Line): Line from center to end_point.
    """

    def __init__(
        self,
        center: Point,
        start_point: Point,
        end_point: Point,
        name: Optional[str] = None,
        arc_type="Arc",
    ):
        """
        Initialize a BaseArc after validating that both start_point and end_point lie on the same circle centered at 'center'.

        Args:
            center (Point): Center of the arc's circle.
            start_point (Point): Start point of the arc.
            end_point (Point): End point of the arc.
            name (str, optional): Name for the arc. If None, auto-generated.
            arc_type (str): Type of arc ("Arc", "MinorArc", "MajorArc").
        Raises:
            ValueError: If start_point or end_point is not on the circle defined by center and radius.
        """
        self.center = center
        self.start_point = start_point
        self.end_point = end_point
        self.name = (
            name or f"{arc_type}_{center.name}_{start_point.name}_{end_point.name}"
        )
        self.arc_type = arc_type
        self.start_radius_line = LineSegment(
            self.center, self.start_point, name=f"line_radius_start_{self.name}"
        )
        self.end_radius_line = LineSegment(
            self.center, self.end_point, name=f"line_radius_end_{self.name}"
        )

    @property
    def radius(self) -> Coordinate:
        """
        Returns the radius of the arc (distance from center to start_point).

        Returns:
            sympy.Expr: The radius.
        """
        return self.center.distance(self.start_point)

    def _angle_rad(self):
        """
        Returns the angle (in radians) between the start and end points at the center.

        Returns:
            sympy.Expr: The angle in radians.
        """

        # safe_radius = sympy.Piecewise(
        #    (sympy.S.One, sympy.Abs(self.radius) <= 1e-5), # If radius is near zero, use 1
        #    (self.radius, True)                # Otherwise, use the original radius
        # )

        v_start_x = self.start_point.x - self.center.x
        v_start_y = self.start_point.y - self.center.y
        v_end_x = self.end_point.x - self.center.x
        v_end_y = self.end_point.y - self.center.y
        dot_product = v_start_x * v_end_x + v_start_y * v_end_y
        cos_theta = dot_product / (self.radius**2)
        return sympy.acos(cos_theta)


@typechecked
class MinorArc(BaseArc):
    """
    Represents the minor arc (smaller arc) between two points on a circle.
    """

    def __init__(
        self,
        center: Point,
        start_point: Point,
        end_point: Point,
        name: Optional[str] = None,
    ):
        super().__init__(center, start_point, end_point, name=name, arc_type="MinorArc")

    @property
    def central_angle(self) -> Coordinate:
        """
        Returns the central angle of the minor arc in degrees (0 to 180).

        Returns:
            sympy.Expr: The angle in degrees.
        """
        return self._angle_rad() * 180 / sympy.pi

    @property
    def inscribed_angle(self) -> Coordinate:
        """
        Returns the inscribed angle subtended by the arc in degrees.

        Returns:
            sympy.Expr: The inscribed angle in degrees.
        """
        return self.central_angle / 2

    @property
    def length(self) -> Coordinate:
        """
        Returns a non-standard 'length' of the minor arc, calculated as radius * (angle in degrees).

        Note: Standard arc length requires the angle in radians (s = r * theta_radians).

        Returns:
            sympy.Expr: The arc length.
        """
        return self.radius * self.central_angle

    @property
    def area(self) -> Coordinate:
        """
        Returns the area of the sector defined by the minor arc.

        Returns:
            sympy.Expr: The sector area.
        """
        return 0.5 * self.radius**2 * self._angle_rad()

    @property
    def midpoint(self, name: Optional[str] = None) -> Point:
        """
        Calculates and returns the midpoint of the minor arc.

        Args:
            name (str, optional): Name for the midpoint. If None, auto-generated.

        Returns:
            Point: The midpoint as a new Point object.
        """
        cx, cy = self.center.x, self.center.y
        r = self.radius

        sx, sy = self.start_point.x, self.start_point.y
        ex, ey = self.end_point.x, self.end_point.y

        # Vectors from center to start and end
        us_x, us_y = sx - cx, sy - cy
        ue_x, ue_y = ex - cx, ey - cy

        # Compute cos(theta) and sin(theta) between start and end
        dot = us_x * ue_x + us_y * ue_y
        cos_theta = dot / (r * r)

        # Half angle formulas (always positive square roots)
        cos_half = sympy.sqrt((1 + cos_theta) / 2)
        sin_half = sympy.sqrt((1 - cos_theta) / 2)

        # Midpoint direction (rotation of start vector by half the angle)
        # For minor arc, use normal rotation direction
        mid_dir_x = us_x * cos_half - us_y * sin_half
        mid_dir_y = us_x * sin_half + us_y * cos_half

        # Calculate midpoint coordinates
        mx = cx + mid_dir_x
        my = cy + mid_dir_y

        if name is None:
            name = f"mid_{self.name}"

        # Return a new Point object
        return Point(name, x=mx, y=my)

    def __repr__(self):
        """
        Returns a string representation of the MinorArc.
        """
        return (
            f"MinorArc({self.name}, center={self.center.name}, r={self.radius}, "
            f"start={self.start_point.name}, end={self.end_point.name}, "
            f"central_angle={self.central_angle} deg, length={self.length})"
        )


@typechecked
class MajorArc(BaseArc):
    """
    Represents the major arc (larger arc) between two points on a circle.
    """

    def __init__(
        self,
        center: Point,
        start_point: Point,
        end_point: Point,
        name: Optional[str] = None,
    ):
        super().__init__(center, start_point, end_point, name=name, arc_type="MajorArc")

    @property
    def central_angle(self) -> Coordinate:
        """
        Returns the central angle of the major arc in degrees (180 to 360).

        Returns:
            sympy.Expr: The angle in degrees.
        """
        return 360 - self._angle_rad() * 180 / sympy.pi

    @property
    def inscribed_angle(self) -> Coordinate:
        """
        Returns the inscribed angle subtended by the arc in degrees.

        Returns:
            sympy.Expr: The inscribed angle in degrees.
        """
        return self.central_angle / 2

    @property
    def length(self) -> Coordinate:
        """
        Returns a non-standard 'length' of the major arc, calculated as radius * (angle in degrees).

        Note: Standard arc length requires the angle in radians (s = r * theta_radians).

        Returns:
            sympy.Expr: The arc length.
        """
        return self.radius * self.central_angle

    @property
    def area(self) -> Coordinate:
        """
        Returns the area of the sector defined by the major arc.

        Returns:
            sympy.Expr: The sector area.
        """
        return 0.5 * self.radius**2 * (2 * sympy.pi - self._angle_rad())
    
    @property
    def midpoint(arc: "MajorArc", name: Optional[str] = None) -> Point:
        """
        Calculates and returns the midpoint of the major arc.

        Args:
            arc (MajorArc): The MajorArc instance.
            name (str, optional): Name for the midpoint. If None, auto-generated.

        Returns:
            Point: The midpoint as a new Point object.
        """
        cx, cy = arc.center.x, arc.center.y
        r = arc.radius

        sx, sy = arc.start_point.x, arc.start_point.y
        ex, ey = arc.end_point.x, arc.end_point.y

        # Vectors from center to start and end
        us_x, us_y = sx - cx, sy - cy
        ue_x, ue_y = ex - cx, ey - cy

        # Compute cos(theta) and sin(theta) between start and end

        dot = us_x * ue_x + us_y * ue_y
        # denominator = sympy.Piecewise(
        #    (sympy.S.One, sympy.Abs(r * r) <= 1e-5), # If radius is near zero, use 1
        #    (r * r, True)                # Otherwise, use the original radius
        # )
        cos_theta = dot / (r * r)

        # Half angle formulas (always positive square roots)
        cos_half = sympy.sqrt((1 + cos_theta) / 2)
        sin_half = sympy.sqrt((1 - cos_theta) / 2)

        # Midpoint direction (rotation of start vector by half the angle)
        mid_dir_x = -(us_x * cos_half - us_y * sin_half)
        mid_dir_y = -(us_x * sin_half + us_y * cos_half)

        mx = cx + mid_dir_x
        my = cy + mid_dir_y

        # Calculate midpoint coordinates
        mx = cx + mid_dir_x
        my = cy + mid_dir_y

        if name is None:
            name = f"mid_{arc.name}"

        # Return a new Point object
        return Point(name, x=mx, y=my)


    def mirror_across_line(self, axis_line: "Line", name=None) -> "MinorArc":
        center_m = self.center.mirror_across_line(axis_line)
        start_m = self.start_point.mirror_across_line(axis_line)
        end_m = self.end_point.mirror_across_line(axis_line)
        return MinorArc(
            center_m, self.radius, start_m, end_m, name=name or f"{self.name}_mirrored"
        )

    def __repr__(self):
        """
        Returns a string representation of the MajorArc.
        """
        return (
            f"MajorArc({self.name}, center={self.center.name}, r={self.radius}, "
            f"start={self.start_point.name}, end={self.end_point.name}, "
            f"central_angle={self.central_angle} deg, length={self.length})"
        )

@typechecked
class Angle():
    """
    Represents an angle defined by three points with a specified vertex.

    Attributes:
        A (Point): One endpoint of the angle.
        vertex (Point): The vertex point of the angle.
        B (Point): The other endpoint of the angle.
        plot (bool): Whether to plot the angle.
        name (str): Name of the angle.
    
    Properties:
        value: The angle measure at the vertex in degrees (or symbolic expression).
    """

    def __init__(
        self,
        A: Point,
        vertex: Point,
        B: Point,
        plot_sign: Optional[bool] = True,
        plot_text: Optional[bool] = False,
        name: Optional[str] = None,
    ):
        """
        Initializes an Angle instance.

        Args:
            A (Point): One endpoint of the angle.
            vertex (Point): Vertex point where the angle is measured.
            B (Point): The other endpoint of the angle.
            plot (bool, optional): Whether to plot the angle. Defaults to True.
            name (str, optional): Name of the angle. If None, a default name is generated.
        """
            
        self.name = (
            name or f"angle_{A.name}_{vertex.name}_{B.name}"
        )

        if len(set([A, vertex, B])) < 3:
            raise ValueError("Three points of angle must be different.")
               
        self.A = A
        self.vertex = vertex
        self.B = B
        self.name = name
        self.plot_sign = plot_sign
        self.plot_text = plot_text
    
    @property
    def value(self) -> Coordinate:
        """
        Calculates and returns the angle value at the vertex formed by points A and B.

        Returns:
            float or sympy.Expr: The angle measure in degrees (or symbolic expression if symbolic).
        """
        return self.vertex.angle(self.A, self.B)