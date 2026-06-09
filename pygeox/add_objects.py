
"""
Factory namespace for creating and adding geometric objects to a scene.

This module defines the AddNamespace class, which provides convenient constructors
to create geometric objects such as points, lines, arcs, polygons, and various
special polygons (triangles, quadrilaterals, regular polygons, etc.).

Each method creates the object, registers it in the scene, and adds necessary
constraints automatically (e.g., equal radii for arcs, collinearity for semicircles).

The namespace facilitates easy and consistent object creation for geometric problem
modeling, with support for copying existing objects and symbolic coordinate handling.
"""

from typing import Union
from typing_extensions import deprecated

from .basic_objects import Circle, Line, LineSegment, MajorArc, MinorArc, Point, Ray, Angle
from .custom_objects import *

Coordinate = Union[int, float, sympy.Expr]


@typechecked
class AddNamespace:
    """Factory namespace for adding geometric objects to a scene.

    This class exposes convenience constructors that create objects,
    register them in the scene, and add any required constraints.
    """

    def __init__(self, scene):
        """Initialize the namespace.

        Args:
            scene: The owning scene that tracks objects, parameters, and constraints.
        """
        self.scene = scene

    def _same_coords(self, p1: Union[Point, List[Point]], p2: Union[Point, List[Point]]):
        """Overwrite coordinates of ``p1`` with those of ``p2``.

        This is a hard replace (no constraints are added to the scene).

        Args:
            p1: Point or list of points whose coordinates will be overwritten.
            p2: Point or list of points providing the coordinates.
        """
        if isinstance(p1, Point) and isinstance(p2, Point):
            p1.x = p2.x
            p1.y = p2.y
        else:
            for p1x, p2x in zip(p1, p2):
                self._same_coords(p1x, p2x)

    def point(
        self,
        name: str = None,
        x: Optional[Coordinate] = None,
        y: Optional[Coordinate] = None,
        copy: Optional["Point"] = None,
    ) -> "Point":
        """Add a point or copy an existing one.

        If ``copy`` is provided, its coordinates and name are used unless
        overridden. If ``x`` or ``y`` is ``None`` (and not copying), a
        :class:`sympy.Symbol` is created for that coordinate, using any
        domain set on the scene.

        Args:
            name: Optional name for the point.
            x: X coordinate (number or SymPy expression). If ``None``, a symbol is created.
            y: Y coordinate (number or SymPy expression). If ``None``, a symbol is created.
            copy: Existing point to duplicate.

        Returns:
            Point: The created point, registered in the scene.
        """
        if copy is not None:
            if name is None:
                name = f"{copy.name}"
            if x is None:
                x = copy.x
            if y is None:
                y = copy.y
            point = Point(name, x, y)
        else:
            x_domain = self.scene._domain_consts["x"]
            x_sym = sympy.Symbol(
                f"{name}_x", domain=sympy.Interval(*x_domain) if x_domain else None
            )
            if x is not None:
                self.scene.constraint.eq(x_sym, x)
            y_domain = self.scene._domain_consts["y"]
            y_sym = sympy.Symbol(
                f"{name}_y", domain=sympy.Interval(*y_domain) if y_domain else None
            )
            if y is not None:
                self.scene.constraint.eq(y_sym, y)

            point = Point(name, x_sym, y_sym)

        self.scene.add_object("Point", name, point)

        # Add symbolic coordinates to full_parameters set
        reduced_parameters = [
            info["symbol"]
            for _, info in self.scene.parameters.items()
            if not info["full"]
        ]

        if isinstance(point.x, sympy.Symbol) and point.x not in reduced_parameters:
            self.scene.add_parameter(
                symbol=point.x, type="x", full=True, origin=name, bound=x_domain
            )
        if isinstance(point.y, sympy.Symbol) and point.y not in reduced_parameters:
            self.scene.add_parameter(
                symbol=point.y, type="y", full=True, origin=name, bound=y_domain
            )
        return point

    def line(
        self,
        point1: Optional[Point] = None,
        point2: Optional[Point] = None,
        name: Optional[str] = None,
        copy: Optional["Line"] = None,
    ) -> "Line":
        """Add a line or copy an existing one.

        If ``copy`` is provided and ``point1``/``point2`` are given,
        their coordinates are overwritten to match the copy. A default name
        is inferred if none is given.

        Args:
            point1: First point on the line.
            point2: Second point on the line.
            name: Optional name for the line.
            copy: Existing line to duplicate.

        Returns:
            Line: The created line, registered in the scene.

        Raises:
            ValueError: If not copying and fewer than two points are provided.
        """
        if copy is not None:
            if point1 and point2:
                self._same_coords([point1, point2], [copy.point1, copy.point2])
            if name is None:
                name = copy.name
        elif point1 is None or point2 is None:
            raise ValueError("You must specify two points to define a line.")

        if name is None:
            name = f"Line_{point1.name}_{point2.name}"

        line_obj = Line(point1, point2, name=name)
        self.scene.add_object("Line", line_obj.name, line_obj)
        return line_obj

    def line_segment(
        self,
        point1: Optional[Point] = None,
        point2: Optional[Point] = None,
        name: Optional[str] = None,
        copy: Optional["LineSegment"] = None,
    ) -> "LineSegment":
        """Add a line segment or copy an existing one.

        Args:
            point1: First endpoint.
            point2: Second endpoint.
            name: Optional name for the segment.
            copy: Existing segment to duplicate.

        Returns:
            LineSegment: The created segment, registered in the scene.

        Raises:
            ValueError: If not copying and fewer than two points are provided.
        """
        if copy is not None:
            point1 = copy.point1
            point2 = copy.point2
            if name is None:
                name = copy.name
        elif point1 is None or point2 is None:
            raise ValueError("You must specify two points to define a line segment.")

        if name is None:
            name = f"LineSegment_{point1.name}_{point2.name}"

        segment_obj = LineSegment(point1, point2, name=name)
        self.scene.add_object("LineSegment", segment_obj.name, segment_obj)
        return segment_obj

    def ray(
        self,
        start_point: Optional[Point] = None,
        direction_point: Optional[Point] = None,
        name: Optional[str] = None,
        copy: Optional["Ray"] = None,
    ) -> "Ray":
        """Add a ray or copy an existing one.

        Args:
            start_point: Ray origin.
            direction_point: Second point indicating direction.
            name: Optional name for the ray.
            copy: Existing ray to duplicate.

        Returns:
            Ray: The created ray, registered in the scene.

        Raises:
            ValueError: If not copying and a start/direction point is missing.
        """
        if copy is not None:
            start_point = copy.start_point
            direction_point = copy.direction_point
            if name is None:
                name = copy.name
        elif start_point is None or direction_point is None:
            raise ValueError(
                "You must specify a start point and a direction point to define a ray."
            )

        if name is None:
            name = f"Ray_{start_point.name}_to_{direction_point.name}"

        ray_obj = Ray(start_point, direction_point, name=name)
        self.scene.add_object("Ray", ray_obj.name, ray_obj)
        return ray_obj

    def circle(
        self,
        center: Optional[Point] = None,
        radius: Optional[Coordinate] = None,
        name: Optional[str] = None,
        copy: Optional["Circle"] = None,
    ) -> "Circle":
        """Add a circle or copy an existing one.

        If not copying and ``radius`` is ``None``, a :class:`sympy.Symbol`
        is created for the radius using the scene's radius domain.

        Args:
            center: Center point of the circle.
            radius: Radius value or SymPy expression. If ``None``, a symbol is created.
            name: Optional name for the circle.
            copy: Existing circle to duplicate.

        Returns:
            Circle: The created circle, registered in the scene.

        Raises:
            ValueError: If not copying and ``center`` is not provided.
        """
        r_domain = self.scene._domain_consts["r"]

        if copy:
            center = copy.center
            radius = copy.radius
            if name is None:
                name = copy.name

        elif center is None:
            raise ValueError("You must specify a center for the circle.")
        if copy is None and radius is None:
            radius = sympy.Symbol(f"{center.name}_r", domain=sympy.Interval(*r_domain))

        if name is None:
            name = f"Circle_{center.name}"

        circle = Circle(center, radius, name=name)
        self.scene.add_object("Circle", circle.name, circle)

        reduced_parameters = [
            info["symbol"]
            for _, info in self.scene.parameters.items()
            if not info["full"]
        ]
        if (
            isinstance(circle.radius, sympy.Symbol)
            and circle.radius not in reduced_parameters
        ):
            self.scene.add_parameter(
                symbol=circle.radius, type="r", full=True, origin=name, bound=r_domain
            )
        return circle

    def minor_arc(
        self,
        center: "Point",
        start_point: "Point",
        end_point: "Point",
        name: Optional[str] = None,
        on_circle: Optional[Circle] = None,
    ):
        """Create a :class:`MinorArc` and register it in the scene.

        Adds equal-radius constraints and optionally enforces that the arc
        lies on a given circle (same center and radius).

        Args:
            center: Center of the circle.
            start_point: Start point of the arc.
            end_point: End point of the arc.
            name: Optional name. If ``None``, one is generated.
            on_circle: If provided, enforces arc radius equals the circle's radius.

        Returns:
            MinorArc: The created arc.
        """
        if name is None:
            name = f"MinorArc_{center.name}_{start_point.name}_{end_point.name}"

        arc = MinorArc(center, start_point, end_point, name=name)
        self.scene.add.line_segment(copy=arc.start_radius_line)
        self.scene.add.line_segment(copy=arc.end_radius_line)
        desc = f"Equal radii for {name}: {center.name} to {start_point.name} and {center.name} to {end_point.name}"
        self.scene.constraint.eq(
            center.distance_squared(start_point),
            center.distance_squared(end_point),
            desc,
        )
        self.scene.add_object("MinorArc", arc.name, arc)

        if on_circle:
            if center.name != on_circle.center.name:
                raise ValueError(
                    f"For arc to belong on circle {on_circle.name} they must have the same center"
                )
            self.scene.constraint.eq(
                arc.radius, on_circle.radius, "Arc radius = circle radius"
            )

        return arc

    def major_arc(
        self,
        center: Point,
        start_point: Point,
        end_point: Point,
        name: Optional[str] = None,
        on_circle: Optional[Circle] = None,
    ):
        """Create a :class:`MajorArc` and register it in the scene.

        Adds equal-radius constraints and optionally enforces that the arc
        lies on a given circle (same center and radius).

        Args:
            center: Center of the circle.
            start_point: Start point of the arc.
            end_point: End point of the arc.
            name: Optional name. If ``None``, one is generated.
            on_circle: If provided, enforces arc radius equals the circle's radius.

        Returns:
            MajorArc: The created arc.
        """
        if name is None:
            name = f"MajorArc_{center.name}_{start_point.name}_{end_point.name}"

        arc = MajorArc(center, start_point, end_point, name=name)
        self.scene.add.line_segment(copy=arc.start_radius_line)
        self.scene.add.line_segment(copy=arc.end_radius_line)
        desc = f"Equal radii for {name}: {center.name} to {start_point.name} and {center.name} to {end_point.name}"
        self.scene.constraint.eq(
            center.distance_squared(start_point),
            center.distance_squared(end_point),
            desc,
        )
        self.scene.add_object("MajorArc", arc.name, arc)

        if on_circle:
            if center.name != on_circle.center.name:
                raise ValueError(
                    f"For arc to belong on circle {on_circle.name} they must have the same center"
                )
            self.scene.constraint.eq(
                arc.radius, on_circle.radius, "Arc radius = circle radius"
            )

        return arc

    def points(self, names: list[str]) -> list[Point]:
        """Add multiple points by name.

        Args:
            names: List of point names.

        Returns:
            list[Point]: The created points.
        """
        return [self.point(name) for name in names]

    def polygon(
        self,
        *points: Point,
        name: Optional[str] = None,
        copy: Optional["Polygon"] = None,
    ) -> Polygon:
        """Add a general polygon or return a copy.

        Assumptions & constraints (when creating):
        at least three points, given in order. If ``copy`` is provided and the
        number of points matches, provided points are overwritten to match.

        Args:
            *points: Vertices in order.
            name: Optional name.
            copy: Existing polygon to reuse.

        Returns:
            Polygon: The polygon instance (copy if provided)."""
        if copy:
            if len(points) == len(copy.points):
                self._same_coords(list(points), copy.points)
            copy.name = name if name else copy.name
            return copy
        return Polygon(self.scene, *points, name=name)

    def regular_polygon(
        self,
        *points: Point,
        name: Optional[str] = None,
        copy: Optional["RegularPolygon"] = None,
    ) -> RegularPolygon:
        """Add a regular polygon or return a copy.

        Assumptions & constraints: at least three ordered points; all sides and
        angles are constrained equal; polygon is inscribed in a circle.

        Args:
            *points: Vertices in order.
            name: Optional name.
            copy: Existing regular polygon to reuse.

        Returns:
            RegularPolygon: The polygon instance (copy if provided)."""
        if copy:
            copy.name = name if name else copy.name
            return copy
        return RegularPolygon(self.scene, *points, name=name)

    def regular_pentagon(
        self,
        *points: Point,
        name: Optional[str] = None,
        copy: Optional["RegularPentagon"] = None,
    ) -> RegularPentagon:
        """Add a regular pentagon or return a copy.

        Assumptions & constraints: exactly five ordered points; all sides and
        angles equal; inscribed in a circle.

        Args:
            *points: Five vertices in order.
            name: Optional name.
            copy: Existing pentagon to reuse.

        Returns:
            RegularPentagon: The polygon instance (copy if provided)."""
        if copy:
            if len(points) == 5:
                self._same_coords(list(points), copy.points)
            copy.name = name if name else copy.name
            return copy
        return RegularPentagon(self.scene, *points, name=name)

    def regular_hexagon(
        self,
        *points: Point,
        name: Optional[str] = None,
        copy: Optional["RegularHexagon"] = None,
    ) -> RegularHexagon:
        """Add a regular hexagon or return a copy.

        Assumptions & constraints: exactly six ordered points; all sides and
        angles equal; inscribed in a circle.

        Args:
            *points: Six vertices in order.
            name: Optional name.
            copy: Existing hexagon to reuse.

        Returns:
            RegularHexagon: The polygon instance (copy if provided)."""
        if copy:
            if len(points) == 6:
                self._same_coords(list(points), copy.points)
            copy.name = name if name else copy.name
            return copy
        return RegularHexagon(self.scene, *points, name=name)

    def regular_heptagon(
        self,
        *points: Point,
        name: Optional[str] = None,
        copy: Optional["RegularHeptagon"] = None,
    ) -> RegularHeptagon:
        """Add a regular heptagon or return a copy.

        Assumptions & constraints: exactly seven ordered points; all sides and
        angles equal.

        Args:
            *points: Seven vertices in order.
            name: Optional name.
            copy: Existing heptagon to reuse.

        Returns:
            RegularHeptagon: The polygon instance (copy if provided)."""
        if copy:
            if len(points) == 7:
                self._same_coords(list(points), copy.points)
            copy.name = name if name else copy.name
            return copy
        return RegularHeptagon(self.scene, *points, name=name)

    def regular_octagon(
        self,
        *points: Point,
        name: Optional[str] = None,
        copy: Optional["RegularOctagon"] = None,
    ) -> RegularOctagon:
        """Add a regular octagon or return a copy.

        Assumptions & constraints: exactly eight ordered points; all sides and
        angles equal.

        Args:
            *points: Eight vertices in order.
            name: Optional name.
            copy: Existing octagon to reuse.

        Returns:
            RegularOctagon: The polygon instance (copy if provided)."""
        if copy:
            if len(points) == 8:
                self._same_coords(list(points), copy.points)
            copy.name = name if name else copy.name
            return copy
        return RegularOctagon(self.scene, *points, name=name)

    def triangle(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        name: Optional[str] = None,
        copy: Optional[Triangle] = None,
    ) -> Triangle:
        """Add a general triangle or return a copy.

        Args:
            p1: Vertex A.
            p2: Vertex B.
            p3: Vertex C.
            name: Optional name.
            copy: Existing triangle to reuse.

        Returns:
            Triangle: The triangle instance (copy if provided).

        Notes:
            Three distinct points are required when creating.
        """
        if copy:
            if p1 and p2 and p3:
                self._same_coords([p1, p2, p3], copy.points)
            copy.name = name if name else copy.name
            return copy

        return Triangle(self.scene, p1, p2, p3, name=name)

    def equilateral_triangle(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        name: Optional[str] = None,
        copy: Optional["EquilateralTriangle"] = None,
    ) -> EquilateralTriangle:
        """Add an equilateral triangle or return a copy.

        Args:
            p1: Vertex A.
            p2: Vertex B.
            p3: Vertex C.
            name: Optional name.
            copy: Existing triangle to reuse.

        Returns:
            EquilateralTriangle: The triangle instance (copy if provided).

        Notes:
            Three distinct points are required; all sides are constrained equal.
        """
        if copy:
            if p1 and p2 and p3:
                self._same_coords([p1, p2, p3], copy.points)
            copy.name = name if name else copy.name
            return copy
        return EquilateralTriangle(self.scene, p1, p2, p3, name=name)

    def isosceles_triangle(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        name: Optional[str] = None,
        copy: Optional["IsoscelesTriangle"] = None,
    ) -> IsoscelesTriangle:
        """Add an isosceles triangle or return a copy.

        Args:
            p1: Vertex A.
            p2: Vertex B (apex for equal sides AB and BC).
            p3: Vertex C.
            name: Optional name.
            copy: Existing triangle to reuse.

        Returns:
            IsoscelesTriangle: The triangle instance (copy if provided).

        Notes:
            Sides AB and BC are constrained equal (order matters).
        """
        if copy:
            if p1 and p2 and p3:
                self._same_coords([p1, p2, p3], copy.points)
            copy.name = name if name else copy.name
            return copy
        return IsoscelesTriangle(self.scene, p1, p2, p3, name=name)

    def right_triangle(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        name: Optional[str] = None,
        copy: Optional["RightTriangle"] = None,
    ) -> RightTriangle:
        """Add a right triangle or return a copy.

        Args:
            p1: Vertex A.
            p2: Vertex B (right angle at B).
            p3: Vertex C.
            name: Optional name.
            copy: Existing triangle to reuse.

        Returns:
            RightTriangle: The triangle instance (copy if provided)."""
        if copy:
            if p1 and p2 and p3:
                self._same_coords([p1, p2, p3], copy.points)
            copy.name = name if name else copy.name
            return copy
        return RightTriangle(self.scene, p1, p2, p3, name=name)

    def right_isosceles_triangle(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        name: Optional[str] = None,
        copy: Optional["RightIsoscelesTriangle"] = None,
    ) -> RightIsoscelesTriangle:
        """Add a right isosceles triangle or return a copy.

        Args:
            p1: Vertex A.
            p2: Vertex B (right angle at B).
            p3: Vertex C.
            name: Optional name.
            copy: Existing triangle to reuse.

        Returns:
            RightIsoscelesTriangle: The triangle instance (copy if provided).

        Notes:
            Sides AB and BC are constrained equal (order matters).
        """
        if copy:
            if p1 and p2 and p3:
                self._same_coords([p1, p2, p3], copy.points)
            copy.name = name if name else copy.name
            return copy
        return RightIsoscelesTriangle(self.scene, p1, p2, p3, name=name)

    def obtuse_triangle(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        name: Optional[str] = None,
        copy: Optional["ObtuseTriangle"] = None,
    ) -> ObtuseTriangle:
        """Add an obtuse triangle or return a copy.

        Args:
            p1: Vertex A.
            p2: Vertex B.
            p3: Vertex C.
            name: Optional name.
            copy: Existing triangle to reuse.

        Returns:
            ObtuseTriangle: The triangle instance (copy if provided).

        Notes:
            At least one internal angle is constrained > 90°.
        """
        if copy:
            if p1 and p2 and p3:
                self._same_coords([p1, p2, p3], copy.points)
            copy.name = name if name else copy.name
            return copy
        return ObtuseTriangle(self.scene, p1, p2, p3, name=name)

    def acute_triangle(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        name: Optional[str] = None,
        copy: Optional["AcuteTriangle"] = None,
    ) -> AcuteTriangle:
        """Add an acute triangle or return a copy.

        Args:
            p1: Vertex A.
            p2: Vertex B.
            p3: Vertex C.
            name: Optional name.
            copy: Existing triangle to reuse.

        Returns:
            AcuteTriangle: The triangle instance (copy if provided).

        Notes:
            All internal angles are constrained < 90°.
        """
        if copy:
            if p1 and p2 and p3:
                self._same_coords([p1, p2, p3], copy.points)
            copy.name = name if name else copy.name
            return copy
        return AcuteTriangle(self.scene, p1, p2, p3, name=name)

    def scalene_triangle(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        name: Optional[str] = None,
        copy: Optional["ScaleneTriangle"] = None,
    ) -> ScaleneTriangle:
        """Add a scalene triangle or return a copy.

        Args:
            p1: Vertex A.
            p2: Vertex B.
            p3: Vertex C.
            name: Optional name.
            copy: Existing triangle to reuse.

        Returns:
            ScaleneTriangle: The triangle instance (copy if provided).

        Notes:
            All side lengths are constrained to be distinct.
        """
        if copy:
            if p1 and p2 and p3:
                self._same_coords([p1, p2, p3], copy.points)
            copy.name = name if name else copy.name
            return copy
        return ScaleneTriangle(self.scene, p1, p2, p3, name=name)

    def quadrilateral(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        p4: Optional[Point] = None,
        name: Optional[str] = None,
        force_convex: bool = False,
        copy: Optional["Quadrilateral"] = None,
    ) -> Quadrilateral:
        """Add a quadrilateral or return a copy.

        Args:
            p1: Vertex A.
            p2: Vertex B.
            p3: Vertex C.
            p4: Vertex D.
            name: Optional name.
            force_convex: If ``True``, enforces convexity and simplicity.
            copy: Existing quadrilateral to reuse.

        Returns:
            Quadrilateral: The quadrilateral instance (copy if provided)."""
        if copy:
            if p1 and p2 and p3 and p4:
                self._same_coords([p1, p2, p3, p4], copy.points)
            copy.name = name if name else copy.name
            return copy
        return Quadrilateral(
            self.scene, p1, p2, p3, p4, name=name, force_convex=force_convex
        )

    def parallelogram(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        p4: Optional[Point] = None,
        name: Optional[str] = None,
        force_convex: bool = False,
        copy: Optional["Parallelogram"] = None,
    ) -> Parallelogram:
        """Add a parallelogram or return a copy.

        Opposite sides are constrained parallel. Optional convexity enforcement.

        Args:
            p1: Vertex A.
            p2: Vertex B.
            p3: Vertex C.
            p4: Vertex D.
            name: Optional name.
            force_convex: If ``True``, enforces convexity and simplicity.
            copy: Existing parallelogram to reuse.

        Returns:
            Parallelogram: The quadrilateral instance (copy if provided)."""
        if copy:
            if p1 and p2 and p3 and p4:
                self._same_coords([p1, p2, p3, p4], copy.points)
            copy.name = name if name else copy.name
            return copy
        return Parallelogram(
            self.scene, p1, p2, p3, p4, name=name, force_convex=force_convex
        )

    def rectangle(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        p4: Optional[Point] = None,
        name: Optional[str] = None,
        force_convex: bool = False,
        copy: Optional["Rectangle"] = None,
    ) -> Rectangle:
        """Add a rectangle or return a copy.

        All angles are constrained 90°; opposite sides parallel. Optional
        convexity enforcement.

        Args:
            p1: Vertex A.
            p2: Vertex B.
            p3: Vertex C.
            p4: Vertex D.
            name: Optional name.
            force_convex: If ``True``, enforces convexity and simplicity.
            copy: Existing rectangle to reuse.

        Returns:
            Rectangle: The quadrilateral instance (copy if provided)."""
        if copy:
            if p1 and p2 and p3 and p4:
                self._same_coords([p1, p2, p3, p4], copy.points)
            copy.name = name if name else copy.name
            return copy
        return Rectangle(
            self.scene, p1, p2, p3, p4, name=name, force_convex=force_convex
        )

    def square(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        p4: Optional[Point] = None,
        name: Optional[str] = None,
        copy: Optional["Square"] = None,
    ) -> Square:
        """Add a square or return a copy.

        All sides equal, all angles 90°, opposite sides parallel.

        Args:
            p1: Vertex A.
            p2: Vertex B.
            p3: Vertex C.
            p4: Vertex D.
            name: Optional name.
            copy: Existing square to reuse.

        Returns:
            Square: The quadrilateral instance (copy if provided)."""
        if copy:
            if p1 and p2 and p3 and p4:
                self._same_coords([p1, p2, p3, p4], copy.points)
            copy.name = name if name else copy.name
            return copy
        return Square(self.scene, p1, p2, p3, p4, name=name)

    def rhombus(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        p4: Optional[Point] = None,
        name: Optional[str] = None,
        force_convex: bool = False,
        copy: Optional["Rhombus"] = None,
    ) -> Rhombus:
        """Add a rhombus or return a copy.

        All sides equal; opposite sides parallel; angles not necessarily 90°.
        Optional convexity enforcement.

        Args:
            p1: Vertex A.
            p2: Vertex B.
            p3: Vertex C.
            p4: Vertex D.
            name: Optional name.
            force_convex: If ``True``, enforces convexity and simplicity.
            copy: Existing rhombus to reuse.

        Returns:
            Rhombus: The quadrilateral instance (copy if provided)."""
        if copy:
            if p1 and p2 and p3 and p4:
                self._same_coords([p1, p2, p3, p4], copy.points)
            copy.name = name if name else copy.name
            return copy
        return Rhombus(self.scene, p1, p2, p3, p4, name=name, force_convex=force_convex)

    def trapezoid(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        p4: Optional[Point] = None,
        name: Optional[str] = None,
        force_convex: bool = False,
        copy: Optional["Trapezoid"] = None,
    ) -> Trapezoid:
        """Add a trapezoid or return a copy.

        Constrains AB ∥ CD. Optional convexity enforcement.

        Args:
            p1: Vertex A.
            p2: Vertex B.
            p3: Vertex C.
            p4: Vertex D.
            name: Optional name.
            force_convex: If ``True``, enforces convexity and simplicity.
            copy: Existing trapezoid to reuse.

        Returns:
            Trapezoid: The quadrilateral instance (copy if provided)."""
        if copy:
            if p1 and p2 and p3 and p4:
                self._same_coords([p1, p2, p3, p4], copy.points)
            copy.name = name if name else copy.name
            return copy
        return Trapezoid(
            self.scene, p1, p2, p3, p4, name=name, force_convex=force_convex
        )

    def isosceles_trapezoid(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        p4: Optional[Point] = None,
        name: Optional[str] = None,
        force_convex: bool = False,
        copy: Optional["IsoscelesTrapezoid"] = None,
    ) -> IsoscelesTrapezoid:
        """Add an isosceles trapezoid or return a copy.

        Constrains AB ∥ CD and BC = DA. Optional convexity enforcement.

        Args:
            p1: Vertex A.
            p2: Vertex B.
            p3: Vertex C.
            p4: Vertex D.
            name: Optional name.
            force_convex: If ``True``, enforces convexity and simplicity.
            copy: Existing trapezoid to reuse.

        Returns:
            IsoscelesTrapezoid: The quadrilateral instance (copy if provided)."""
        if copy:
            if p1 and p2 and p3 and p4:
                self._same_coords([p1, p2, p3, p4], copy.points)
            copy.name = name if name else copy.name
            return copy
        return IsoscelesTrapezoid(
            self.scene, p1, p2, p3, p4, name=name, force_convex=force_convex
        )

    def right_trapezoid(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        p4: Optional[Point] = None,
        name: Optional[str] = None,
        force_convex: bool = False,
        copy: Optional["RightTrapezoid"] = None,
    ) -> RightTrapezoid:
        """Add a right trapezoid or return a copy.

        Constrains AB ∥ CD and angles A, D = 90°. Optional convexity enforcement.

        Args:
            p1: Vertex A.
            p2: Vertex B.
            p3: Vertex C.
            p4: Vertex D.
            name: Optional name.
            force_convex: If ``True``, enforces convexity and simplicity.
            copy: Existing trapezoid to reuse.

        Returns:
            RightTrapezoid: The quadrilateral instance (copy if provided)."""
        if copy:
            if p1 and p2 and p3 and p4:
                self._same_coords([p1, p2, p3, p4], copy.points)
            copy.name = name if name else copy.name
            return copy
        return RightTrapezoid(
            self.scene, p1, p2, p3, p4, name=name, force_convex=force_convex
        )

    def kite(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        p4: Optional[Point] = None,
        name: Optional[str] = None,
        force_convex: bool = False,
        copy: Optional["Kite"] = None,
    ) -> Kite:
        """Add a kite or return a copy.

        Constrains AB = BC and CD = DA. Optional convexity enforcement.

        Args:
            p1: Vertex A.
            p2: Vertex B.
            p3: Vertex C.
            p4: Vertex D.
            name: Optional name.
            force_convex: If ``True``, enforces convexity and simplicity.
            copy: Existing kite to reuse.

        Returns:
            Kite: The quadrilateral instance (copy if provided)."""
        if copy:
            if p1 and p2 and p3 and p4:
                self._same_coords([p1, p2, p3, p4], copy.points)
            copy.name = name if name else copy.name
            return copy
        return Kite(self.scene, p1, p2, p3, p4, name=name, force_convex=force_convex)

    def rhomboid(
        self,
        p1: Optional[Point] = None,
        p2: Optional[Point] = None,
        p3: Optional[Point] = None,
        p4: Optional[Point] = None,
        name: Optional[str] = None,
        force_convex: bool = False,
        copy: Optional["Rhomboid"] = None,
    ) -> Rhomboid:
        """Add a rhomboid or return a copy.

        Opposite sides parallel; adjacent sides unequal; no right-angle
        constraint. Optional convexity enforcement.

        Args:
            p1: Vertex A.
            p2: Vertex B.
            p3: Vertex C.
            p4: Vertex D.
            name: Optional name.
            force_convex: If ``True``, enforces convexity and simplicity.
            copy: Existing rhomboid to reuse.

        Returns:
            Rhomboid: The quadrilateral instance (copy if provided)."""
        if copy:
            if p1 and p2 and p3 and p4:
                self._same_coords([p1, p2, p3, p4], copy.points)
            copy.name = name if name else copy.name
            return copy
        return Rhomboid(
            self.scene, p1, p2, p3, p4, name=name, force_convex=force_convex
        )

    @deprecated("Instead of using this function, you should create a line segment and then use scene.relate.is_chord()")
    def chord(self, circle: Circle, A: Point, B: Point, name: Optional[str] = None) -> LineSegment:
        """Add a chord between two points on a circle.

        Points ``A`` and ``B`` are constrained to lie on ``circle``.

        Args:
            circle: The circle on which the points lie.
            A: First point on the circle.
            B: Second point on the circle.
            name: Optional name. Defaults to ``chord_A_B_on_<circle>``.

        Returns:
            LineSegment: The created chord, registered in the scene.
        """
        if name is None:
            name = f"chord_{A.name}_{B.name}_on_{circle.name}"
        chord_line = LineSegment(A, B, name=name)
        self.scene.add_object("Line", name, chord_line)
        self.scene.relate.point_lies_on(A, circle)
        self.scene.relate.point_lies_on(B, circle)
        return chord_line

    def semicircle(self, center: Point, A: Point, B: Point, name: Optional[str] = None):
        """Add a semicircle (180° major arc) from A to B with a given center.

        Enforces collinearity of ``center``, ``A``, and ``B`` (direction not forced).

        Args:
            center: Center of the circle.
            A: First endpoint on the circle.
            B: Second endpoint on the circle.
            name: Optional name. Defaults to ``semicircle_A_B_center_<center>``.

        Returns:
            MajorArc: The created semicircle as a major arc.
        """
        if name is None:
            name = f"semicircle_{A.name}_{B.name}_center_{center.name}"
        arc = self.scene.add.major_arc(center, A, B, name)
        self.scene.relate.collinear(center, A, B, force_direction=False)
        return arc
        # self.scene.constraint.eq(arc.central_angle, 180, f"Semicircle: arc {arc.name} = 180°")

    def angle(self, A: Point, vertex: Point, B:Point, plot_sign: Optional[bool] = True, plot_text: Optional[bool] = False, name: Optional[str] = None):
        """
        Add an angle object defined by three points: A, vertex, and B.

        The angle is formed at the `vertex` point between points `A` and `B`.

        Args:
            A (Point): One point defining the angle.
            vertex (Point): The vertex point where the angle is measured.
            B (Point): The other point defining the angle.
            name (Optional[str], optional): Name for the angle object.

        Returns:
            The measure of the angle at `vertex` between points `A` and `B`.
            This is typically a symbolic expression (e.g., a SymPy Expr) if coordinates
            are symbolic or unknown.
        """

        if name is None:
            name = f"angle_{A.name}_{vertex.name}_{B.name}"

        angle = Angle(A, vertex, B, plot_sign, plot_text, name)
        self.scene.add_object("Angle", name, angle)
        
        return angle