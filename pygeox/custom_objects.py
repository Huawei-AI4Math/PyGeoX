"""
custom_objects.py: Classes for polygons and specialized geometric shapes within a geometric scene.

Defines general Polygon and its subclasses including CyclicPolygon, RegularPolygon,
and various specific polygons like Triangles, Quadrilaterals, and their special cases
(e.g., EquilateralTriangle, Rectangle, Square).

These classes encapsulate geometric properties, constraints, and relationships,
enabling symbolic geometry modeling, constraint solving, and scene management.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import sympy
from typeguard import typechecked

from .basic_objects import Circle, Line, LineLike, LineSegment, Point, Ray

Coordinate = Union[int, float, sympy.Expr]

############## Polygons


@typechecked
class Polygon:
    """
    Represents a general polygon defined by an ordered list of Point objects.

    Args:
        geoscene: The geometric scene/context to which this polygon belongs.
        *points (Point): The vertices of the polygon, in order.
        name (str, optional): Name for the polygon. If None, an automatic name is generated.
        add_to_scene_as_type (str, optional): Type string for scene registration. Default is 'Polygon'.
        force_convex (bool, optional): If True, enforces convexity and simplicity constraint (forcing also counterclockwise direction).

    Returns:
        Polygon: A new Polygon object registered in the scene.
    """

    def __init__(
        self,
        geoscene,
        *points: Point,
        name: Optional[str] = None,
        add_to_scene_as_type: Optional[str] = "Polygon",
        force_convex: bool = False,
    ):
        self.geoscene = geoscene
        self.points = list(points)  # List of Point objects in order

        # Auto-generate name if not provided
        if name is None:
            point_names = [p.name for p in self.points]
            self.name = f"Polygon_{'_'.join(point_names)}"
        else:
            self.name = name

        # Create lines connecting consecutive points to form the polygon sides
        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[
                (i + 1) % len(self.points)
            ]  # Connect last point back to first
            geoscene.add.line_segment(p1, p2)

        # Centralized add_object call
        # Use the provided type string, or default to 'Polygon'
        self.geoscene.add_object(add_to_scene_as_type, self.name, self)

        # convex/simple constraints
        if force_convex:
            for i, angle in enumerate(self.internal_angles):
                # Convexity: all cross products of consecutive edges have the same sign
                n = len(self.points)
                for i in range(n):
                    p0 = self.points[i]
                    p1 = self.points[(i + 1) % n]
                    p2 = self.points[(i + 2) % n]
                    # Vector from p0 to p1 and p1 to p2
                    dx1 = p1.x - p0.x
                    dy1 = p1.y - p0.y
                    dx2 = p2.x - p1.x
                    dy2 = p2.y - p1.y
                    cross = dx1 * dy2 - dy1 * dx2
                    self.geoscene.constraint.gt(
                        cross, 0, f"Convex: {self.name} cross {i} > 0"
                    )

                # Simplicity: sum of internal angles = (n-2)*180
                # angle_sum = sum(self.internal_angles)
                # self.geoscene.constraint.eq(angle_sum, (n - 2) * 180, f"Simple: {self.name} angle sum")

    @property
    def area(self) -> Coordinate:
        """
        Calculates the area of the polygon using the shoelace formula.

        Returns:
            sympy.Expr: The area. Units: square of input coordinates.
        """
        if len(self.points) >= 3:
            return 0.5 * abs(
                sum(
                    self.points[i].x * self.points[(i + 1) % len(self.points)].y
                    - self.points[(i + 1) % len(self.points)].x * self.points[i].y
                    for i in range(len(self.points))
                )
            )
        else:
            return 0

    @property
    def perimeter(self) -> Coordinate:
        """
        Calculates the perimeter of the polygon by summing side lengths.

        Returns:
            sympy.Expr: The perimeter. Units: same as input coordinates.
        """
        return sum(side.length for side in self.sides)

    @property
    def centroid(self):
        """
        Returns the vertex centroid (average of all vertices).

        Returns:
            Point: The centroid as a Point object.
        """
        n = len(self.points)
        x = sum(p.x for p in self.points) / n
        y = sum(p.y for p in self.points) / n
        return Point(f"{self.name}_centroid", x, y)

    @property
    def circumcircle(self):
        """
        Returns the circumcircle of the polygon if circumcenter and circumradius exist.

        Returns:
            Circle: The circumcircle.
        Raises:
            NotImplementedError: If circumcenter or circumradius is not defined for this polygon type.
        """
        center = getattr(self, "circumcenter", None)
        radius = getattr(self, "circumradius", None)
        if center is not None and radius is not None:
            return Circle(center, radius, name=f"{self.name}_circumcircle")
        raise NotImplementedError(
            f"circumcenter or circumradius is not defined for {self.__class__.__name__}"
        )

    @property
    def incircle(self):
        """
        Returns the incircle of the polygon if incenter and inradius exist.

        Returns:
            Circle: The incircle.
        Raises:
            NotImplementedError: If incenter or inradius is not defined for this polygon type.
        """
        center = getattr(self, "incenter", None)
        radius = getattr(self, "inradius", None)
        if center is not None and radius is not None:
            return Circle(center, radius, name=f"{self.name}_incircle")
        raise NotImplementedError(
            f"incenter or inradius is not defined for {self.__class__.__name__}"
        )

    @property
    def sides(self) -> List[LineSegment]:
        """
        Returns the sides of the polygon as LijneSegment objects, in order. Sides [AB, BC, CD, ...] for Polygon(A,B,C,D,...).

        Returns:
            list[LijneSegment]: The sides of the polygon.
        """
        n = len(self.points)
        return [
            LineSegment(
                self.points[i],
                self.points[(i + 1) % n]
            )
            for i in range(n)
        ]

    def internal_angle_at_point(self, vertex_point: Point):
        """
        Returns the internal angle (in degrees) at a given vertex of the polygon,
        correctly handling convex and concave cases.

        Parameters:
            vertex_point (Point): The vertex at which to compute the internal angle.

        Returns:
            sympy expression: Internal angle in degrees, in range (0, 360)
        """
        points = self.points
        index = points.index(vertex_point)

        prev_point = points[index - 1]
        next_point = points[(index + 1) % len(points)]

        # Vectors from vertex to prev and next
        vec1 = (prev_point.x - vertex_point.x, prev_point.y - vertex_point.y)
        vec2 = (next_point.x - vertex_point.x, next_point.y - vertex_point.y)

        # Reverse order to get internal turning angle
        angle_radians = sympy.Mod(sympy.atan2(vec1[1], vec1[0]) - sympy.atan2(vec2[1], vec2[0]), 2 * sympy.pi)
        angle_degrees = sympy.deg(angle_radians)

        return angle_degrees


    @property
    def internal_angles(self):
        """
        Returns the internal angles at each vertex. Angles [DAB,ABC,BCD,..] for Polygon(A,B,C,D,...).

        Returns:
            list[float]: The internal angles at each vertex in degrees.
        """

        return [self.internal_angle_at_point(p) for p in self.points]

    @property
    def num_sides(self):
        """
        Returns a string representation of the Polygon object.

        Returns:
            str: String representation.
        """
        return len(self.points)

    def __repr__(self):
        """Returns a string representation of the Polygon object."""
        point_names = ", ".join([p.name for p in self.points])
        return f"Polygon({self.name}, points=[{point_names}])"


#@typechecked
class CyclicPolygon(Polygon):
    """
    Polygon whose vertices all lie on a single circle (circumcircle).
    Circumcenter and circumradius are always defined.
    """

    def __init__(
        self,
        geoscene,
        *points: Point,
        name: Optional[str] = None,
        add_to_scene_as_type: Optional[str] = "CyclicPolygon",
    ):
        super().__init__(
            geoscene, *points, name=name, add_to_scene_as_type=add_to_scene_as_type
        )
        # Symbolic circumcenter and circumradius
        self._circumcenter_x = sympy.Symbol(f"{self.name}_circumcenter_x")
        self._circumcenter_y = sympy.Symbol(f"{self.name}_circumcenter_y")
        self._circumradius = sympy.Symbol(f"{self.name}_circumradius")
        # Add constraints: all points equidistant from circumcenter
        for p in self.points:
            geoscene.constraint.eq(
                (p.x - self._circumcenter_x) ** 2 + (p.y - self._circumcenter_y) ** 2,
                self._circumradius**2,
                f"Cyclic: {self.name} {p.name} on circumcircle",
            )

    @property
    def circumcenter(self):
        return Point(
            f"{self.name}_circumcenter", self._circumcenter_x, self._circumcenter_y
        )

    @property
    def circumradius(self) -> Coordinate:
        return self._circumradius


@typechecked
class RegularPolygon(Polygon):
    """
    Represents a regular polygon (all sides and angles equal).
    It imposes regularity constraints on the provided points.
    Defines its own reduced parameter set (center, radius, orientation).
    """

    def __init__(
        self,
        geoscene,
        *points: Point,
        name: Optional[str] = None,
        add_to_scene_as_type: Optional[str] = "RegularPolygon",
    ):
        if len(points) < 3:
            raise ValueError(
                f"RegularPolygon requires at least 3 points. Got {len(points)}."
            )

        if name is None:
            point_names = [p.name for p in points]
            name = f"RegularPolygon_{'_'.join(point_names)}"

        # Pass the specific type string here for scene registration, defaulting if not provided
        super().__init__(
            geoscene, *points, name=name, add_to_scene_as_type=add_to_scene_as_type
        )

        # defining the reduced parameters - center point
        x_domain = self.geoscene._domain_consts["x"]
        y_domain = self.geoscene._domain_consts["y"]
        center_x = sympy.Symbol(
            f"{self.name}_center_x",
            domain=sympy.Interval(*x_domain) if x_domain else None,
        )
        center_y = sympy.Symbol(
            f"{self.name}_center_y",
            domain=sympy.Interval(*y_domain) if y_domain else None,
        )
        self._center = Point(f"{self.name}_center", center_x, center_y)

        # defining the reduced parameters - radius
        r_domain = self.geoscene._domain_consts["r"]
        self._circumradius = sympy.Symbol(
            f"{self.name}_circumradius", domain=sympy.Interval(*r_domain)
        )

        # defining the reduced parameters - orientation
        self._orientation = sympy.Symbol(
            f"{self.name}_orientation", domain=sympy.Interval(0, sympy.pi)
        )

        self.geoscene.add_parameter(
            symbol=self._center.x, type="x", full=False, origin=name, bound=x_domain
        )
        self.geoscene.add_parameter(
            symbol=self._center.y, type="y", full=False, origin=name, bound=y_domain
        )
        self.geoscene.add_parameter(
            symbol=self._circumradius, type="r", full=False, origin=name, bound=r_domain
        )
        self.geoscene.add_parameter(
            symbol=self._orientation,
            type="theta",
            full=False,
            origin=name,
            bound=[0, sympy.pi],
        )

        # Add mapping constraints from reduced parameters to full parameters (vertex coordinates)
        for i in range(self.num_sides):
            vertex = self.points[i]
            angle = (2 * sympy.pi * i / self.num_sides) + self._orientation
            self.geoscene.add_mapping_constraint(
                sympy.Eq(
                    vertex.x, self._center.x + self._circumradius * sympy.cos(angle)
                ),
                f"Mapping {vertex.name}_x to reduced params of {self.name}",
            )
            self.geoscene.add_mapping_constraint(
                sympy.Eq(
                    vertex.y, self._center.y + self._circumradius * sympy.sin(angle)
                ),
                f"Mapping {vertex.name}_y to reduced params of {self.name}",
            )

    @property
    def circumradius(self) -> Coordinate:
        """Returns the circumradius of the regular polygon."""
        return self._circumradius

    @circumradius.setter
    def circumradius(self, value):
        """Sets the circumradius of the regular polygon."""
        self._circumradius = value

    @property
    def orientation(self) -> Coordinate:
        """Returns the orientation (rotation) of the regular polygon in radians."""
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        """Sets the orientation (rotation) of the regular polygon in radians."""
        self._orientation = value

    @property
    def center(self):
        """Returns the center point of the regular polygon."""
        return self._center

    @center.setter
    def center(self, value):
        """Sets the center point of the regular polygon."""
        self._center = value

    @property
    def incenter(self):
        """
        Returns:
            Point: The incenter of the regular polygon, which coincides with the center for regular polygons.
        """
        return self.center

    @property
    def circumcenter(self):
        """
        Returns:
            Point: The circumcenter of the regular polygon, which coincides with the center for regular polygons.
        """
        return self.center

    @property
    def area(self) -> Coordinate:
        """Calculates the area of the regular polygon using its properties."""
        return (
            (self.num_sides / 2)
            * self.circumradius**2
            * sympy.sin(2 * sympy.pi / self.num_sides)
        )

    @property
    def perimeter(self) -> Coordinate:
        """Calculates the perimeter of the regular polygon using its properties."""
        return (
            self.num_sides
            * 2
            * self.circumradius
            * sympy.sin(sympy.pi / self.num_sides)
        )

    @property
    def side_length(self) -> Coordinate:
        """Side length of the regular polygon."""
        return 2 * self.circumradius * sympy.sin(sympy.pi / self.num_sides)

    @property
    def inradius(self) -> Coordinate:
        """Inradius (apothem) of the regular polygon."""
        return self.circumradius * sympy.cos(sympy.pi / self.num_sides)

    @property
    # todo
    def height(self) -> Coordinate:
        """Height (distance from one side to the opposite vertex) of the regular polygon."""
        if self.num_sides % 2 == 0:
            # Even-sided polygons: height = 2 * circumradius
            return 2 * self.circumradius
        else:
            # Odd-sided polygons: height = circumradius + inradius
            return self.circumradius + self.inradius

    @property
    # todo
    def width(self) -> Coordinate:
        """Width (distance between two farthest separated points) of the regular polygon."""
        if self.num_sides % 2 == 0:
            # Even-sided polygons: width = 2 * circumradius
            return 2 * self.circumradius
        else:
            # Odd-sided polygons: width = 2 * circumradius * sympy.cos(π/(2n))
            return 2 * self.circumradius * sympy.cos(sympy.pi / (2 * self.num_sides))

    @property
    def diagonal(self) -> Coordinate:
        """Longest diagonal of the regular polygon."""
        return self.width

    @property
    def internal_angle(self) -> Coordinate:
        """
        Returns:
            float: The internal angle (in degrees) of the regular polygon.
        """
        return (self.num_sides - 2) * 180 / self.num_sides

    @staticmethod
    def compute_reduced_params_from_points(
        point_coords: Dict[str, Union[Tuple[float, float], List[float]]],
        num_sides: int
    ) -> Dict[str, float]:
        """
        Compute reduced parameters (center_x, center_y, circumradius, 
        orientation) from point coordinates for a regular polygon.
        
        This is the reverse of the mapping: given vertex coordinates, 
        compute the reduced parameter representation.
        
        Args:
            point_coords: Dictionary mapping point names to (x, y) tuples.
                         Should contain coordinates for all vertices in order.
                         In Python 3.7+, dict preserves insertion order.
            num_sides: Number of sides of the regular polygon.
        
        Returns:
            Dictionary with keys: 'center_x', 'center_y', 'circumradius', 
            'orientation'
        """
        import math
        
        # Extract coordinates in order (dict preserves order in Python 3.7+)
        coords_list = [point_coords[name] for name in point_coords.keys()]
        
        # Compute center as average of all vertices (centroid)
        center_x = sum(x for x, y in coords_list) / num_sides
        center_y = sum(y for x, y in coords_list) / num_sides
        
        # Compute circumradius as distance from center to first vertex
        first_x, first_y = coords_list[0]
        dx = first_x - center_x
        dy = first_y - center_y
        circumradius = math.sqrt(dx * dx + dy * dy)
        
        # Compute orientation as angle of first vertex relative to center
        # Using atan2 to get angle in [-pi, pi]
        angle = math.atan2(dy, dx)
        
        # Normalize to [0, 2*pi)
        if angle < 0:
            angle = angle + 2 * math.pi
        
        # Now angle is in [0, 2*pi)
        # Since the forward mapping for i=0 is: angle_0 = orientation,
        # and orientation domain is [0, pi], the first vertex angle should be in [0, pi]
        # The key issue: when angle is exactly pi, angle % pi = 0 (wrong!)
        # So we handle the pi case specially
        tol = 1e-10
        if abs(angle - math.pi) < tol:
            # Angle is exactly pi (or very close)
            orientation = math.pi
        elif angle < math.pi:
            # Angle is in [0, pi), use directly
            orientation = angle
        else:
            # angle is in (pi, 2*pi)
            # This shouldn't normally happen for first vertex, but handle it
            # Map to [0, pi] by subtracting pi, but check if result makes sense
            # Actually, if angle is close to 2*pi, it's equivalent to 0
            if abs(angle - 2*math.pi) < tol:
                orientation = 0.0
            else:
                # For angles in (pi, 2*pi), we can't directly represent them
                # in [0, pi] domain. Use modulo but be careful.
                remainder = angle % math.pi
                # If remainder is 0, it means angle was a multiple of pi
                # Since angle is in (pi, 2*pi), if remainder is 0, angle was 2*pi
                # which we already handled. Otherwise, remainder is in (0, pi)
                orientation = remainder if remainder > tol else math.pi
        
        return {
            'center_x': center_x,
            'center_y': center_y,
            'circumradius': circumradius,
            'orientation': orientation
        }

    def __repr__(self):
        """Returns a string representation of the RegularPolygon object."""
        point_names = ", ".join([p.name for p in self.points])
        return (
            f"RegularPolygon({self.name}, sides={self.num_sides}, points=[{point_names}], "
            f"reduced_params=(center_x={self.center.x}, center_y={self.center.y}, "
            f"circumradius={self.circumradius}, orientation={self.orientation} rad))"
        )


@typechecked
class RegularPentagon(RegularPolygon):
    """
    A specific type of RegularPolygon with 5 sides.
    """

    def __init__(
        self,
        geoscene,
        *points: Point,
        name: Optional[str] = None,
        add_to_scene_as_type: Optional[str] = "RegularPentagon",
    ):
        if len(points) != 5:
            raise ValueError(
                f"RegularPentagon requires exactly 5 points. Got {len(points)}."
            )
        if name is None:
            point_names = [p.name for p in points]
            name = f"RegularPentagon_{'_'.join(point_names)}"
        # Pass the specific type string here for scene registration
        super().__init__(
            geoscene, *points, name=name, add_to_scene_as_type=add_to_scene_as_type
        )


@typechecked
class RegularHexagon(RegularPolygon):
    """
    A specific type of RegularPolygon with 6 sides.
    """

    def __init__(
        self,
        geoscene,
        *points: Point,
        name: Optional[str] = None,
        add_to_scene_as_type: Optional[str] = "RegularHexagon",
    ):
        if len(points) != 6:
            raise ValueError(
                f"RegularHexagon requires exactly 6 points. Got {len(points)}."
            )
        if name is None:
            point_names = [p.name for p in points]
            name = f"RegularHexagon_{'_'.join(point_names)}"
        super().__init__(
            geoscene, *points, name=name, add_to_scene_as_type=add_to_scene_as_type
        )


@typechecked
class RegularHeptagon(RegularPolygon):
    """
    A specific type of RegularPolygon with 7 sides.
    """

    def __init__(
        self,
        geoscene,
        *points: Point,
        name: Optional[str] = None,
        add_to_scene_as_type: Optional[str] = "RegularHeptagon",
    ):
        if len(points) != 7:
            raise ValueError(
                f"RegularHeptagon requires exactly 7 points. Got {len(points)}."
            )
        if name is None:
            point_names = [p.name for p in points]
            name = f"RegularHeptagon_{'_'.join(point_names)}"
        super().__init__(
            geoscene, *points, name=name, add_to_scene_as_type=add_to_scene_as_type
        )


@typechecked
class RegularOctagon(RegularPolygon):
    """
    A specific type of RegularPolygon with 8 sides.
    """

    def __init__(
        self,
        geoscene,
        *points: Point,
        name: Optional[str] = None,
        add_to_scene_as_type: Optional[str] = "RegularOctagon",
    ):
        if len(points) != 8:
            raise ValueError(
                f"RegularOctagon requires exactly 8 points. Got {len(points)}."
            )
        if name is None:
            point_names = [p.name for p in points]
            name = f"RegularOctagon_{'_'.join(point_names)}"
        super().__init__(
            geoscene, *points, name=name, add_to_scene_as_type=add_to_scene_as_type
        )


############ Triangles


@typechecked
class Triangle(Polygon):
    """
    Represents a general triangle defined by three Point objects.
    """
    def __init__(
        self,
        geoscene,
        p1: Point,
        p2: Point,
        p3: Point,
        name: Optional[str] = None,
        add_to_scene_as_type: Optional[str] = "Triangle",
    ):
        if name is None:
            name = f"Triangle_{p1.name}_{p2.name}_{p3.name}"
        # Pass the specific type string for scene registration
        super().__init__(
            geoscene, p1, p2, p3, name=name, add_to_scene_as_type=add_to_scene_as_type
        )

    @property
    def area(self) -> Coordinate:
        """Calculates the area of the triangle using Heron's formula."""
        a, b, c = [side.length for side in self.sides]
        s = (a + b + c) / 2
        return sympy.sqrt(s * (s - a) * (s - b) * (s - c))

    @property
    def heights(self):
        """Returns the heights from each vertex to the opposite side."""
        a, b, c = [side.length for side in self.sides]
        area = self.area
        return [2 * area / a, 2 * area / b, 2 * area / c]

    @property
    def incenter(self):
        A, B, C = self.points
        a = B.distance(C)
        b = C.distance(A)
        c = A.distance(B)
        return Point(
            f"{self.name}_incenter",
            x=(a * A.x + b * B.x + c * C.x) / (a + b + c),
            y=(a * A.y + b * B.y + c * C.y) / (a + b + c),
        )

    @property
    def circumcenter(self):
        A, B, C = self.points
        D = 2 * (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))
        Ux = (
            (A.x**2 + A.y**2) * (B.y - C.y)
            + (B.x**2 + B.y**2) * (C.y - A.y)
            + (C.x**2 + C.y**2) * (A.y - B.y)
        ) / D
        Uy = (
            (A.x**2 + A.y**2) * (C.x - B.x)
            + (B.x**2 + B.y**2) * (A.x - C.x)
            + (C.x**2 + C.y**2) * (B.x - A.x)
        ) / D
        return Point(f"{self.name}_circumcenter", x=Ux, y=Uy)

    @property
    def orthocenter(self):
        A, B, C = self.points
        x, y = sympy.symbols(f"{self.name}_orthocenter_x {self.name}_orthocenter_y")
        eq1 = sympy.Eq((y - A.y), -((B.x - C.x) / (B.y - C.y)) * (x - A.x))
        eq2 = sympy.Eq((y - B.y), -((A.x - C.x) / (A.y - C.y)) * (x - B.x))
        sol = sympy.solve([eq1, eq2], (x, y), dict=True)
        if sol:
            ox, oy = sol[0][x], sol[0][y]
        else:
            ox, oy = 0, 0  # fallback
        return Point(f"{self.name}_orthocenter", x=ox, y=oy)

    @property
    def inradius(self) -> Coordinate:
        a, b, c = [side.length for side in self.sides]
        s = (a + b + c) / 2
        return self.area / s

    @property
    def circumradius(self) -> Coordinate:
        a, b, c = [side.length for side in self.sides]
        return (a * b * c) / (4 * self.area)

    @property
    def medians(self) -> list:
        """
        Returns:
            list[LineSegment]: The three medians of the triangle, each from a vertex to the midpoint of the opposite side.
        """
        A, B, C = self.points
        mid_AB = Point(f"{self.name}_mid_AB", x=(A.x + B.x) / 2, y=(A.y + B.y) / 2)
        mid_BC = Point(f"{self.name}_mid_BC", x=(B.x + C.x) / 2, y=(B.y + C.y) / 2)
        mid_CA = Point(f"{self.name}_mid_CA", x=(C.x + A.x) / 2, y=(C.y + A.y) / 2)
        return [
            LineSegment(A, mid_BC, name=f"{self.name}_median_A"),
            LineSegment(B, mid_CA, name=f"{self.name}_median_B"),
            LineSegment(C, mid_AB, name=f"{self.name}_median_C"),
        ]

    @property
    def altitudes(self) -> list:
        """
        Returns:
            list[LineSegment]: The three altitudes of the triangle, each from a vertex perpendicular to the opposite side.
        """
        A, B, C = self.points

        # Helper to find foot of perpendicular from P to line QR
        def foot(P, Q, R, name):
            x0, y0 = P.x, P.y
            x1, y1 = Q.x, Q.y
            x2, y2 = R.x, R.y
            dx, dy = x2 - x1, y2 - y1
            t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx**2 + dy**2)
            xf = x1 + t * dx
            yf = y1 + t * dy
            return Point(name, x=xf, y=yf)

        foot_A = foot(A, B, C, f"{self.name}_altitude_foot_A")
        foot_B = foot(B, C, A, f"{self.name}_altitude_foot_B")
        foot_C = foot(C, A, B, f"{self.name}_altitude_foot_C")
        return [
            LineSegment(A, foot_A, name=f"{self.name}_altitude_A"),
            LineSegment(B, foot_B, name=f"{self.name}_altitude_B"),
            LineSegment(C, foot_C, name=f"{self.name}_altitude_C"),
        ]

    @property
    def midsegments(self) -> list:
        """
        Returns:
            list[LineSegment]: The three midsegments of the triangle, each connecting the midpoints of two sides.
        """
        A, B, C = self.points
        mid_AB = Point(f"{self.name}_mid_AB", x=(A.x + B.x) / 2, y=(A.y + B.y) / 2)
        mid_BC = Point(f"{self.name}_mid_BC", x=(B.x + C.x) / 2, y=(B.y + C.y) / 2)
        mid_CA = Point(f"{self.name}_mid_CA", x=(C.x + A.x) / 2, y=(C.y + A.y) / 2)
        return [
            LineSegment(mid_AB, mid_BC, name=f"{self.name}_midsegment_AB_BC"),
            LineSegment(mid_BC, mid_CA, name=f"{self.name}_midsegment_BC_CA"),
            LineSegment(mid_CA, mid_AB, name=f"{self.name}_midsegment_CA_AB"),
        ]

    @property
    def angle_bisectors(self) -> list:
        """
        Returns:
            list[LineSegment]: The three angle bisectors of the triangle, each bisecting one angle and ending at the opposite side.
        """
        A, B, C = self.points

        # Calculate the intersection point of the angle bisector on the opposite side (using the Angle Bisector Theorem)
        def get_bisector_point(p1, p2, p3):
            # p1是顶点，p2和p3是角的两边端点
            a = p2.distance(p3)  # 对边长度
            b = p1.distance(p3)  # 边p1-p3长度
            c = p1.distance(p2)  # 边p1-p2长度
            ratio = b / c  # 根据角平分线定理
            # 计算分割点坐标
            x = (b * p2.x + c * p3.x) / (b + c)
            y = (b * p2.y + c * p3.y) / (b + c)
            return Point(f"{self.name}_bisect_{p1.name}_{p2.name}{p3.name}", x=x, y=y)

        # Calculate the intersection points of the three angle bisectors on their opposite sides
        bisect_A = get_bisector_point(A, B, C)
        bisect_B = get_bisector_point(B, A, C)
        bisect_C = get_bisector_point(C, A, B)

        return [
            LineSegment(A, bisect_A, name=f"{self.name}_bisector_A"),
            LineSegment(B, bisect_B, name=f"{self.name}_bisector_B"),
            LineSegment(C, bisect_C, name=f"{self.name}_bisector_C"),
        ]

    def __repr__(self):
        """Returns a string representation of the Triangle object."""
        point_names = ", ".join([p.name for p in self.points])
        return f"Triangle({self.name}, points=[{point_names}])"

@typechecked
class EquilateralTriangle(Triangle):
    """
    Triangle with all sides equal and all angles 60 degrees.
    """

    def __init__(
        self,
        geoscene,
        p1: Point,
        p2: Point,
        p3: Point,
        name: Optional[str] = None,
        add_to_scene_as_type: Optional[str] = "EquilateralTriangle",
    ):
        if name is None:
            name = f"EquilateralTriangle_{p1.name}_{p2.name}_{p3.name}"
        # Pass the specific type string for scene registration
        super().__init__(
            geoscene, p1, p2, p3, name=name, add_to_scene_as_type=add_to_scene_as_type
        )
        a, b, c = [side.length for side in self.sides]
        geoscene.constraint.eq(a, b, f"Equilateral Triangle: {self.name} a=b")
        geoscene.constraint.eq(b, c, f"Equilateral Triangle: {self.name} b=c")

    @property
    def inradius(self) -> Coordinate:
        # Inradius = (side length) / (2 * sqrt(3))
        a = self.sides[0].length
        return a / (2 * sympy.sqrt(3))

    @property
    def circumradius(self) -> Coordinate:
        # Circumradius = (side length) / sqrt(3)
        a = self.sides[0].length
        return a / sympy.sqrt(3)

@typechecked
class IsoscelesTriangle(Triangle):
    """
    Triangle with at least two equal sides. IsoscelesTriangle(A,B,C) means that AB=BC.
    """

    def __init__(
        self,
        geoscene,
        p1: Point,
        p2: Point,
        p3: Point,
        name: Optional[str] = None,
        add_to_scene_as_type: Optional[str] = "IsoscelesTriangle",
    ):
        if name is None:
            name = f"IsoscelesTriangle_{p1.name}_{p2.name}_{p3.name}"
        super().__init__(
            geoscene, p1, p2, p3, name=name, add_to_scene_as_type=add_to_scene_as_type
        )
        a, b, _ = [side.length for side in self.sides]
        geoscene.constraint.eq(a, b, f"Isosceles Triangle : {self.name} a=b")


@typechecked
class RightTriangle(Triangle):
    """
    Triangle with one right angle. RightTriangle(A,B,C) means that B is the right angle.
    """

    def __init__(
        self,
        geoscene,
        p1: Point,
        p2: Point,
        p3: Point,
        name: Optional[str] = None,
        add_to_scene_as_type: Optional[str] = "RightTriangle",
    ):
        if name is None:
            name = f"RightTriangle_{p1.name}_{p2.name}_{p3.name}"
        super().__init__(
            geoscene, p1, p2, p3, name=name, add_to_scene_as_type=add_to_scene_as_type
        )
        # Assuming sides[0] and sides[1] form the right angle based on point order (p1-p2, p2-p3)
        geoscene.relate.perpendicular(self.sides[0], self.sides[1])


@typechecked
class RightIsoscelesTriangle(RightTriangle, IsoscelesTriangle):
    """
    Triangle with a right angle and two equal sides. RightIsoscelesTriangle(A,B,C) means that AB=BC and B is the right angle.
    """

    def __init__(
        self,
        geoscene,
        p1: Point,
        p2: Point,
        p3: Point,
        name: Optional[str] = None,
        add_to_scene_as_type: Optional[str] = "RightIsoscelesTriangle",
    ):
        if name is None:
            name = f"RightIsoscelesTriangle_{p1.name}_{p2.name}_{p3.name}"
        # Call the constructor of the first parent in MRO (RightTriangle), passing the specific type
        super().__init__(
            geoscene, p1, p2, p3, name=name, add_to_scene_as_type=add_to_scene_as_type
        )

        # Manually apply the isosceles constraint, as IsoscelesTriangle's __init__ wasn't directly called by super()
        a, b, _ = [side.length for side in self.sides]
        geoscene.constraint.eq(a, b, f"RightIsosceles Triangle : {self.name} a=b")

    @property
    def circumcenter(self):
        A, B, C = self.points
        return Point(f"{self.name}_circumcenter", x=(A.x + C.x) / 2, y=(A.y + C.y) / 2)

    @property
    def circumradius(self) -> Coordinate:
        # For right isosceles triangle with AB=BC=a, hypotenuse=AC=a*sqrt(2), circumradius=a/sqrt(2)
        a = self.sides[0].length  # AB=BC=a
        return a / sympy.sqrt(2)


@typechecked
class ObtuseTriangle(Triangle):
    """
    Triangle with one angle > 90 degrees.
    """

    def __init__(
        self,
        geoscene,
        p1: Point,
        p2: Point,
        p3: Point,
        name: Optional[str] = None,
        add_to_scene_as_type: Optional[str] = "ObtuseTriangle",
    ):
        if name is None:
            name = f"ObtuseTriangle_{p1.name}_{p2.name}_{p3.name}"
        super().__init__(
            geoscene, p1, p2, p3, name=name, add_to_scene_as_type=add_to_scene_as_type
        )
        # Constraint: At least one angle must be greater than 90 degrees
        # This loop will add a constraint for the first angle found to be > 90.
        for i, angle in enumerate(self.internal_angles):
            geoscene.constraint.gt(angle, 90, f"Obtuse: {self.name} angle {i} > 90")
            break  # Only one angle needs to be obtuse for an Obtuse Triangle


@typechecked
class AcuteTriangle(Triangle):
    """
    Triangle with all angles < 90 degrees.
    """

    def __init__(
        self,
        geoscene,
        p1: Point,
        p2: Point,
        p3: Point,
        name: Optional[str] = None,
        add_to_scene_as_type: Optional[str] = "AcuteTriangle",
    ):
        if name is None:
            name = f"AcuteTriangle_{p1.name}_{p2.name}_{p3.name}"
        super().__init__(
            geoscene, p1, p2, p3, name=name, add_to_scene_as_type=add_to_scene_as_type
        )
        # Constraint: All angles must be less than 90 degrees
        for i, angle in enumerate(self.internal_angles):
            geoscene.constraint.lt(angle, 90, f"Acute: {self.name} angle {i} < 90")


@typechecked
class ScaleneTriangle(Triangle):
    """
    Triangle with all sides of different lengths.
    """

    def __init__(
        self,
        geoscene,
        p1: Point,
        p2: Point,
        p3: Point,
        name: Optional[str] = None,
        add_to_scene_as_type: Optional[str] = "ScaleneTriangle",
    ):
        if name is None:
            name = f"ScaleneTriangle_{p1.name}_{p2.name}_{p3.name}"
        super().__init__(
            geoscene, p1, p2, p3, name=name, add_to_scene_as_type=add_to_scene_as_type
        )
        a, b, c = [side.length for side in self.sides]
        geoscene.constraint.neq(a, b, f"Scalene: {self.name} a!=b")
        geoscene.constraint.neq(b, c, f"Scalene: {self.name} b!=c")
        geoscene.constraint.neq(a, c, f"Scalene: {self.name} a!=c")


## Quadrilaterals


@typechecked
class Quadrilateral(Polygon):
    """
    Represents a general quadrilateral defined by four Point objects.
    Inherits area, perimeter, sides, and internal_angles from Polygon.
    """

    def __init__(
        self,
        geoscene,
        p1: Point,
        p2: Point,
        p3: Point,
        p4: Point,
        name: Optional[str] = None,
        add_to_scene_as_type: Optional[str] = "Quadrilateral",
        force_convex: bool = False,
    ):
        if name is None:
            name = f"Quadrilateral_{p1.name}_{p2.name}_{p3.name}_{p4.name}"
        super().__init__(
            geoscene,
            p1,
            p2,
            p3,
            p4,
            name=name,
            add_to_scene_as_type=add_to_scene_as_type,
            force_convex=force_convex,
        )

    @property
    def midsegments(self):
        """
        Returns the two midsegments (lines connecting midpoints of opposite sides) of the quadrilateral.

        Returns:
            list[LineSegment]: The two midsegments.
        """
        sides = self.sides  # Assuming self.sides returns [AB, BC, CD, DA]
        mids = [side.midpoint for side in sides]
        return [LineSegment(mids[0], mids[2]), LineSegment(mids[1], mids[3])]

    @property
    def diagonals(self):
        return [
            LineSegment(self.points[0], self.points[2]),
            LineSegment(self.points[1], self.points[3]),
        ]

    def __repr__(self):
        point_names = ", ".join([p.name for p in self.points])
        return f"Quadrilateral({self.name}, points=[{point_names}])"


@typechecked
class Parallelogram(Quadrilateral):
    """
    Quadrilateral with both pairs of opposite sides parallel.
    """

    def __init__(
        self,
        geoscene,
        p1,
        p2,
        p3,
        p4,
        name=None,
        add_to_scene_as_type="Parallelogram",
        force_convex: bool = False,
    ):
        if name is None:
            name = f"Parallelogram_{p1.name}_{p2.name}_{p3.name}_{p4.name}"
        super().__init__(
            geoscene,
            p1,
            p2,
            p3,
            p4,
            name=name,
            add_to_scene_as_type=add_to_scene_as_type,
            force_convex=force_convex,
        )
        # Add parallel constraints
        geoscene.relate.parallel(self.sides[0], self.sides[2])
        geoscene.relate.parallel(self.sides[1], self.sides[3])

    @property
    def area(self) -> Coordinate:
        # Vector cross product of adjacent sides
        A, B, D = self.points[0], self.points[1], self.points[3]
        return abs((B.x - A.x) * (D.y - A.y) - (B.y - A.y) * (D.x - A.x))


@typechecked
class Rectangle(Parallelogram):
    """
    Parallelogram with all angles 90 degrees.
    """

    def __init__(
        self,
        geoscene,
        p1,
        p2,
        p3,
        p4,
        name=None,
        add_to_scene_as_type="Rectangle",
        force_convex: bool = False,
    ):
        if name is None:
            name = f"Rectangle_{p1.name}_{p2.name}_{p3.name}_{p4.name}"
        super().__init__(
            geoscene,
            p1,
            p2,
            p3,
            p4,
            name=name,
            add_to_scene_as_type=add_to_scene_as_type,
            force_convex=force_convex,
        )
        # Add right angle constraints
        for i in range(4):
            geoscene.relate.perpendicular(self.sides[i], self.sides[(i + 1) % 4], plot = False)

    @property
    def area(self) -> Coordinate:
        # Adjacent sides
        return self.sides[0].length * self.sides[1].length

    @property
    def perimeter(self) -> Coordinate:
        return 2 * (self.sides[0].length + self.sides[1].length)

    @property
    def diagonals(self):
        return [
            LineSegment(self.points[0], self.points[2]),
            LineSegment(self.points[1], self.points[3]),
        ]

    @property
    def circumcenter(self):
        # Intersection of diagonals (midpoint for rectangle)
        x = (self.points[0].x + self.points[2].x) / 2
        y = (self.points[0].y + self.points[2].y) / 2
        return Point(f"{self.name}_circumcenter", x, y)

    @property
    def circumradius(self) -> Coordinate:
        # Half the diagonal length
        d = self.diagonals[0].length
        return d / 2

@typechecked
class Square(Rectangle):
    """
    Rectangle with all sides equal.
    """

    def __init__(
        self,
        geoscene,
        p1,
        p2,
        p3,
        p4,
        name=None,
        add_to_scene_as_type="Square",
        force_convex: bool = False,
    ):
        if name is None:
            name = f"Square_{p1.name}_{p2.name}_{p3.name}_{p4.name}"
        super().__init__(
            geoscene,
            p1,
            p2,
            p3,
            p4,
            name=name,
            add_to_scene_as_type=add_to_scene_as_type,
            force_convex=force_convex,
        )
        # Add equal side constraints
        a, b, c, d = [side.length for side in self.sides]
        geoscene.constraint.eq(a, b, f"Square: {self.name} a=b")
        geoscene.constraint.eq(b, c, f"Square: {self.name} b=c")
        geoscene.constraint.eq(c, d, f"Square: {self.name} c=d")

    @property
    def area(self) -> Coordinate:
        a = self.sides[0].length
        return a**2

    @property
    def perimeter(self) -> Coordinate:
        a = self.sides[0].length
        return 4 * a

    @property
    def circumcenter(self):
        return self.centroid

    @property
    def incenter(self):
        return self.centroid

    @property
    def inradius(self) -> Coordinate:
        a = self.sides[0].length
        return a / 2

    @property
    def circumradius(self) -> Coordinate:
        a = self.sides[0].length
        return a / sympy.sqrt(2)


@typechecked
class Rhombus(Parallelogram):
    """
    Parallelogram with all sides equal.
    """

    def __init__(
        self,
        geoscene,
        p1,
        p2,
        p3,
        p4,
        name=None,
        add_to_scene_as_type="Rhombus",
        force_convex: bool = False,
    ):
        if name is None:
            name = f"Rhombus_{p1.name}_{p2.name}_{p3.name}_{p4.name}"
        super().__init__(
            geoscene,
            p1,
            p2,
            p3,
            p4,
            name=name,
            add_to_scene_as_type=add_to_scene_as_type,
            force_convex=force_convex,
        )
        a, b, c, d = [side.length for side in self.sides]
        geoscene.constraint.eq(a, b, f"Rhombus: {self.name} a=b")
        geoscene.constraint.eq(b, c, f"Rhombus: {self.name} b=c")
        geoscene.constraint.eq(c, d, f"Rhombus: {self.name} c=d")
        geoscene.constraint.eq(d, a, f"Rhombus: {self.name} a=d")

    @property
    def area(self) -> Coordinate:
        d1 = LineSegment(self.points[0], self.points[2]).length
        d2 = LineSegment(self.points[1], self.points[3]).length
        return (d1 * d2) / 2

    @property
    def incenter(self):
        # Center of rhombus (intersection of diagonals)
        x = (self.points[0].x + self.points[2].x) / 2
        y = (self.points[1].y + self.points[3].y) / 2
        return Point(f"{self.name}_incenter", x, y)

    @property
    def inradius(self) -> Coordinate:
        # Area divided by half the perimeter
        return self.area / (0.5 * self.perimeter)


@typechecked
class Trapezoid(Quadrilateral):
    """
    Quadrilateral with at least one pair of parallel sides. Trapezoid(A,B,C,D) means AB is parallel to CD.
    """

    def __init__(
        self,
        geoscene,
        p1,
        p2,
        p3,
        p4,
        name=None,
        add_to_scene_as_type="Trapezoid",
        force_convex: bool = False,
    ):
        if name is None:
            name = f"Trapezoid_{p1.name}_{p2.name}_{p3.name}_{p4.name}"
        super().__init__(
            geoscene,
            p1,
            p2,
            p3,
            p4,
            name=name,
            add_to_scene_as_type=add_to_scene_as_type,
            force_convex=force_convex,
        )
        # Add parallel constraint for one pair (e.g., sides 0 and 2)
        geoscene.relate.parallel(self.sides[0], self.sides[2])


@typechecked
class IsoscelesTrapezoid(Trapezoid):
    """
    Trapezoid with non-parallel sides equal in length. IsoscelesTrapezoid(A,B,C,D) means AB is parallel to CD and BC and DA have same lenght.
    """

    def __init__(
        self,
        geoscene,
        p1,
        p2,
        p3,
        p4,
        name=None,
        add_to_scene_as_type="IsoscelesTrapezoid",
        force_convex: bool = False,
    ):
        if name is None:
            name = f"IsoscelesTrapezoid_{p1.name}_{p2.name}_{p3.name}_{p4.name}"
        super().__init__(
            geoscene,
            p1,
            p2,
            p3,
            p4,
            name=name,
            add_to_scene_as_type=add_to_scene_as_type,
            force_convex=force_convex,
        )
        # Add equal non-parallel side constraint (e.g., sides 1 and 3)
        geoscene.constraint.eq(
            self.sides[1].length,
            self.sides[3].length,
            f"IsoscelesTrapezoid: {self.name} non-parallel sides equal",
        )


@typechecked
class RightTrapezoid(Trapezoid):
    """
    Trapezoid with non-parallel sides equal in length. RightTrapezoid(A,B,C,D) means AB is parallel to CD and angle A and D are right angles.
    """

    def __init__(
        self,
        geoscene,
        p1,
        p2,
        p3,
        p4,
        name=None,
        add_to_scene_as_type="IsoscelesTrapezoid",
        force_convex: bool = False,
    ):
        if name is None:
            name = f"RightTrapezoid_{p1.name}_{p2.name}_{p3.name}_{p4.name}"
        super().__init__(
            geoscene,
            p1,
            p2,
            p3,
            p4,
            name=name,
            add_to_scene_as_type=add_to_scene_as_type,
            force_convex=force_convex,
        )
        # Add equal non-parallel side constraint (e.g., sides 1 and 3)
        geoscene.relate.perpendicular(self.sides[2], self.sides[3])
        geoscene.relate.perpendicular(self.sides[3], self.sides[0])


@typechecked
class Kite(Quadrilateral):
    """
    Quadrilateral with two distinct pairs of adjacent sides equal.
    """

    def __init__(
        self,
        geoscene,
        p1,
        p2,
        p3,
        p4,
        name=None,
        add_to_scene_as_type="Kite",
        force_convex: bool = False,
    ):
        if name is None:
            name = f"Kite_{p1.name}_{p2.name}_{p3.name}_{p4.name}"
        super().__init__(
            geoscene,
            p1,
            p2,
            p3,
            p4,
            name=name,
            add_to_scene_as_type=add_to_scene_as_type,
            force_convex=force_convex,
        )
        # Add equal adjacent side constraints
        geoscene.constraint.eq(
            self.sides[0].length, self.sides[1].length, f"Kite: {self.name} sides 0=1"
        )
        geoscene.constraint.eq(
            self.sides[2].length, self.sides[3].length, f"Kite: {self.name} sides 2=3"
        )

    @property
    def area(self) -> Coordinate:
        d1 = LineSegment(self.points[0], self.points[2]).length
        d2 = LineSegment(self.points[1], self.points[3]).length
        return (d1 * d2) / 2


@typechecked
class Rhomboid(Parallelogram):
    """
    Parallelogram with adjacent sides of unequal lengths and angles not right.
    """

    def __init__(
        self,
        geoscene,
        p1,
        p2,
        p3,
        p4,
        name=None,
        add_to_scene_as_type="Rhomboid",
        force_convex: bool = False,
    ):
        if name is None:
            name = f"Rhomboid_{p1.name}_{p2.name}_{p3.name}_{p4.name}"
        super().__init__(
            geoscene,
            p1,
            p2,
            p3,
            p4,
            name=name,
            add_to_scene_as_type=add_to_scene_as_type,
            force_convex=force_convex,
        )
        # Adjacent sides not equal
        geoscene.constraint.neq(
            self.sides[0].length,
            self.sides[1].length,
            f"Rhomboid: {self.name} adjacent sides 0!=1",
        )
        geoscene.constraint.neq(
            self.sides[1].length,
            self.sides[2].length,
            f"Rhomboid: {self.name} adjacent sides 1!=2",
        )
        # No right angles
        for i in range(4):
            angle = self.points[i].angle(self.points[i - 1], self.points[(i + 1) % 4])
            geoscene.constraint.neq(angle, 90, f"Rhomboid: {self.name} angle {i} != 90")
