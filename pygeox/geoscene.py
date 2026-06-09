"""
GeoScene module: Core framework for geometric scene management, constraint handling,
and problem solving.

This module defines the GeoScene class, which serves as the central manager for
geometric objects (points, lines, circles, polygons, arcs) and constraints. It
provides a unified API to add, retrieve, and manipulate geometric elements, define
symbolic constraints between them, and invoke solvers to find solutions to geometric
problems.

Key features include:
- Automatic management and storage of geometric objects.
- Symbolic constraint definition with human-readable descriptions.
- Integration of namespaces for relationships, constraint addition, solving,
  animation, and geometric proofs.
- Sketching capability by auto-creating points when referenced by name.
- Utilities for calculating angles, plotting scenes, and proof management.

The GeoScene class acts as the main entry point for users to programmatically
create, analyze, and visualize geometry problems and their solutions.
"""

from typing import Any, Dict, List, Optional, Union
from typing_extensions import deprecated

import sympy
from typeguard import typechecked

from .add_objects import AddNamespace
from .animate import AnimateNamespace
from .basic_objects import Point, LineLike, Circle, BaseArc, Angle
from .custom_objects import Polygon
from .constraint import ConstraintNamespace
from .relationship import RelationshipNamespace
from .solver import SolverNamespace
from .reward import RewardNamespace
from .utils import normalize_domain, plot_scene, plot_scene_step_by_step
from .prover import prover_func
from time import time


# Type alias for coordinate values
Coordinate = Union[int, float, sympy.Expr]


@typechecked
class GeoScene:
    """
    Manages all geometric objects and constraints, and solves for unknown variables.
    This class is the main entry point for defining, manipulating, and solving
    geometric problems. It provides sketching capabilities by automatically creating
    points when referenced by name.
    """

    def __init__(self, domain: Optional[Any] = 50) -> None:
        self._domain_consts = normalize_domain(domain)

        # A unified dictionary for all geometric objects
        self.objects: Dict[str, Dict[str, Any]] = {
            "Point": {},
            "Line": {},
            "Circle": {},
            "MinorArc": {},
            "MajorArc": {},
            "Polygon": {},
        }
        self.objects_solved: Dict[str, Dict[str, Any]] = {} #with floats

        self._relationships: List[Dict[str, Any]] = []
        self.constraints: List[Dict[str, Any]] = []
        self.mapping_constraints: List[Dict[str, Any]] = []
        self.parameters = {}
        self.solution = None
        self._timings = {"scene_init": time()}

        self._ordered_objects_names = [] #names of objects by their order of addition
        self._moving_objects: List[Point] = []
        self._relationship_num = 0
        self._object_num = 0

        # tracked quantities
        self.track_unsolved = {} 
        self.track_solved = {}
        
        # Namespaces for a cleaner API
        self.relate = RelationshipNamespace(self)
        self.add = AddNamespace(self)
        self.constraint = ConstraintNamespace(self)
        self.solver = SolverNamespace(self)
        self.reward = RewardNamespace(self)
        self.animate = AnimateNamespace(self)

        # dealing with proofs
        self.proof_statements = []
        self._in_proving_mode = False


    def _check_name_exists(self, name: str, obj: Any, obj_type: str):
        """
        Check if an object name already exists in the scene.

        Returns a pointer to the existing object if found (except for circles),
        otherwise raises an error for duplicate circle names.

        Args:
            name: Name to check.
            obj: The new object.
            obj_type: Type of the new object.

        Returns:
            Tuple of (name, object) to use.
        """
        for existing_category_name, objects_in_category in self.objects.items():
            if name in objects_in_category:
                existing_object_with_name = objects_in_category[name]
                if obj_type == "Circle":
                    raise ValueError(
                        f"Error: Circle with name '{name}' already exists."
                        f"Please choose another name for your Circle."
                    )
                elif obj_type == "Point":
                    raise ValueError(
                        f"Error: Point with name '{name}' already exists. "
                        f"Please choose another name for your point or remove previous point addition."
                    )      
                else:
                    obj = existing_object_with_name
                    print(
                        f"[INFO] Object with name {name} already exists in scene. Passing a pointer to the existing object."
                    )
                    return name, obj

        return name, obj

    def add_object(self, obj_type: str, name: str, obj: Any) -> None:
        """
        Add a geometric object to the scene under the given type and name.

        Args:
            obj_type: Type of the object (e.g., 'Point', 'Circle').
            name: Name of the object.
            obj: The geometric object instance.
        """
        if obj_type not in self.objects:
            self.objects[obj_type] = {}
        name, obj = self._check_name_exists(name, obj, obj_type)

        self.objects[obj_type][name] = obj
        self._ordered_objects_names.append({"name": name, "obj_type": obj_type})

    def get_object(self, name: str, solved: bool = False) -> Optional[Any]:
        """
        Retrieve a specific object by name, searching across all object types.
        Prefers solved objects, falls back to unsolved objects if the solved
        one is missing or empty.

        Args:
            name: Name of the object to retrieve.

        Returns:
            The geometric object or None if not found in any object type.
        """
        if solved:
            for obj_type in self.objects_solved:
                obj = self.objects_solved[obj_type].get(name)
                if obj not in (None, [], {}):
                    return obj
        else:
            for obj_type in self.objects:
                obj = self.objects[obj_type].get(name)
                if obj not in (None, [], {}):
                    return obj
        return None 


    def get_all_objects(self, obj_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve all objects of a given type or all objects if no type is specified,
        preferring solved objects. Falls back to unsolved objects if solved ones
        are missing or empty.

        Args:
            obj_type: Optional type of objects to retrieve.

        Returns:
            Dictionary of object names to objects.
        """
        if obj_type:
            objs = self.objects_solved.get(obj_type, {})
            if objs not in (None, {}, []):
                return objs
            return self.objects.get(obj_type, {})

        # Merge across all types
        all_objs = {}
        for obj_dict in self.objects_solved.values():
            if obj_dict not in (None, {}, []):
                all_objs.update(obj_dict)

        if all_objs:
            return all_objs

        # fallback to unsolved
        for obj_dict in self.objects.values():
            all_objs.update(obj_dict)
        return all_objs

    def get_points(self) -> dict:
        """
        Get dictionary of all points in the scene with their (x, y) coordinates.

        Returns:
            Dict mapping point names to (x, y) coordinate tuples.
        """
        point_data = {}
        for name, point in self.get_all_objects("Point").items():
            point_data[name] = (point.x, point.y)
        return point_data

    def add_parameter(
        self,
        symbol: sympy.Symbol,
        type: str = "x",
        full: bool = True,
        origin: str = "Problem",
        bound=None,
    ):
        """
        Add a symbolic parameter with metadata to the scene.

        Args:
            symbol: The sympy.Symbol representing the parameter.
            type: Parameter type (default "x").
            full: Whether parameter is fully active.
            origin: Origin description.
            bound: Optional numeric bounds [min, max].

        Returns:
            The symbol added.
        """
        if bound is None:
            bound = [-float("inf"), float("inf")]
        self.parameters[symbol.name] = {
            "symbol": symbol,
            "bound": bound,
            "type": type,
            "full": full,
            "origin": origin,
        }
        return symbol
        
    def add_constraint(self, equation: sympy.Basic, description: str) -> None:
        """
        Add a symbolic equation constraint with a description.

        If in proving mode, stores constraint for proof; else adds to main constraints.

        Args:
            equation: Sympy symbolic equation.
            description: Human-readable description.
        """
        dic_to_add = {"equation": equation, "description": description, "relationship_id": self._relationship_num if self.relate._inside else None}
        
        if self._in_proving_mode:
            self.proof_statements.append(dic_to_add)
        else:
            self.constraints.append(dic_to_add)
    
    def _add_relationship(self, cathegory: str, args: dict) -> None:
        self._relationship_num += 1
        self._relationships.append({"id": self._relationship_num, "relationship": cathegory, "args": args,  "to_prove": self._in_proving_mode})


    def add_mapping_constraint(self, equation: sympy.Basic, description: str) -> None:
        """
        Add a mapping constraint equation used for parameter mapping.

        Args:
            equation: Sympy symbolic equation.
            description: Human-readable description.
        """
        self.mapping_constraints.append(
            {"equation": equation, "description": description}
        )

    @deprecated("Instead of using this function, you should use scene.add.angle")
    def angle(self, point1: Point, vertex: Point, point2: Point, signed: bool = False) -> Coordinate:
        """
        Calculate the angle (degrees between 0 and 180) at vertex between point1 and point2.

        Args:
            point1: First point defining the angle.
            vertex: Vertex point of the angle.
            point2: Second point defining the angle.
            signed: If True, returns signed angle (CCW positive).

        Returns:
            Angle in degrees.
        """
        return vertex.angle(point1, point2, signed)

    def __getitem__(self, name: str) -> Any:
        """
        Get an object by name via dictionary-style access.

        Args:
            name: Object name.

        Returns:
            The object matching the name.

        Raises:
            KeyError if object not found.
        """
        for obj_dict in self.objects.values():
            if name in obj_dict:
                return obj_dict[name]
        raise KeyError(f"Object with name '{name}' not found in the scene.")

    def __repr__(self) -> str:
        """
        Return string summary of the GeoScene.

        Returns:
            String showing number of objects and constraints.
        """
        num_objs = sum(len(v) for v in self.objects.values())
        return (
            f"GeoScene with {num_objs} objects and {len(self.constraints)} constraints"
        )

    def plot(
        self,
        show_names: bool = True,
        show_axis_grid: bool = True,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        highlight_tracked_objects: bool = True,
        return_fig: bool = False,
    ):
        """
        Plot all geometric objects in the scene.

        Args:
            show_names: Show object names on plot.
            show_axis_grid: Show axis grid.
            xlim: X-axis limits.
            ylim: Y-axis limits.
            return_fig: If True, return matplotlib figure object.

        Returns:
            Matplotlib figure if return_fig is True; otherwise None.
        """
        fig = plot_scene(
            self,
            show_names=show_names,
            show_axis_grid=show_axis_grid,
            xlim=xlim,
            ylim=ylim,
            highlight_tracked_objects=highlight_tracked_objects
        )

        if return_fig:
            return fig
    
    def plot_step_by_step(
        self,
        show_names: bool = True,
        show_axis_grid: bool = False,
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        return_fig: bool = False,
    ):
        """
        Plot all geometric objects in the scene, step-by-step. Returns a list of matplotlib figures.

        Args:
            show_names: Show object names on plot.
            show_axis_grid: Show axis grid.
            xlim: X-axis limits.
            ylim: Y-axis limits.
            return_fig: If True, return matplotlib figure object.

        Returns:
            Matplotlib figure if return_fig is True; otherwise None.
        """
        fig = plot_scene_step_by_step(
            self,
            show_names=show_names,
            show_axis_grid=show_axis_grid,
            xlim=xlim,
            ylim=ylim,
            return_figures = return_fig
        )

        if return_fig:
            return fig
    

    class ProvingContext:
        """
        Initialize proving context manager.

        Args:
            scene_instance: GeoScene instance.
        """
        def __init__(self, scene_instance):
            """
            Initialize proving context manager.

            Args:
                scene_instance: GeoScene instance.
            """
            self.scene = scene_instance

        def __enter__(self):
            """
            Enter proving mode.

            Returns:
                RelationshipNamespace for defining constraints in proof.
            """
            self.scene._in_proving_mode = True
            self.scene._proof_statements = []
            return self.scene.relate

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Exit proving mode.
            """
            self.scene._in_proving_mode = False

    def proving(self):
        """
        Create a context manager for defining proof constraints.

        Returns:
            ProvingContext instance.
        """
        return self.ProvingContext(self)

    def prove(self):
        """
        Run the prover function on the accumulated proof statements and constraints.
        Updates the proof_statements with prover results.
        """

        self._timings["prover.start"] = time()
        self.proof_statements = prover_func(self.constraints, self.proof_statements)
        self._timings["prover.end"] = time()


    def track(self, name: str = "str", quantity: sympy.Expr = sympy.Symbol('q'), 
              objects: List[Union[Polygon, Point, LineLike, Circle, BaseArc, Angle]] = None) -> None:
            """
            Tracks a quantity associated with a list of geometric objects.
            """
            if objects is None:
                objects = [] # Use an empty list as a default value instead of the mutable object in the signature
            
            new_objects = []
            for object in objects:
                if isinstance(object, Polygon):
                    lines = object.sides
                    new_objects.extend(lines)
                else:
                    new_objects.append(object)

            self.track_unsolved[name] = {"quantity": quantity, "objects": new_objects}

