import io
import itertools
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import sympy
from PIL import Image
from tqdm import tqdm
from typeguard import typechecked
from typing import Optional, List, Union, Literal

from .basic_objects import LineSegment, Point, Circle, MajorArc, MinorArc
from .utils import crossfade_hard_threshold

NumericExpr = Union[int, float, sympy.Expr, sympy.Symbol]


@typechecked
class AnimateNamespace:
    """
    Namespace to manage animation parameters and generate animations
    by varying parameters within a geometric scene.

    Provides functionality to add animation parameters, run animations
    by solving constraints at discrete parameter values, and export
    animations as GIFs.
    """
    def __init__(self, scene):
        self.scene = scene
        self.anim_scenes = []

    def add_parameter(self, name: str, total_time: NumericExpr = 1):
        """
        Add a new symbolic animation parameter to the scene.

        Args:
            name: The name of the animation parameter.

        Returns:
            The sympy.Symbol representing the animation parameter.
        """
        time_param = sympy.Symbol(name)

        self.scene.add_parameter(
            symbol=time_param,
            type="lambda",
            full=True,
            origin="animation",
            bound=[0, total_time],
        )

        return time_param
    
    def get_time_params(self):
        """
        Retrieve all animation parameters currently registered in the scene.

        Returns:
            List of sympy.Symbol animation parameters.
        """

        time_params = []
        for param in self.scene.parameters.values():
            if param["origin"] == "animation":
                time_params.append(param)
        return time_params

    def _create_time_param_name(self):        
        base_name = "time"
        counter = 0
        name = base_name
        while name in self.scene.parameters:
            counter += 1
            name = f"{base_name}_{counter}"

        return name

    
    def eval_piecewise_at_zero(self, expr, anim_syms):
        """
        If `expr` contains Piecewise, pick the first branch whose condition is:
        - literally True, or
        - an And(..., alpha >= 0, ...) (ignoring other bounds), or
        - becomes True after substituting alpha -> 0.
        Otherwise (no Piecewise), return expr unchanged.
        """

        subs0 = {s: sympy.Integer(0) for s in anim_syms}

        if not expr.has(sympy.Piecewise):
            return expr.xreplace(subs0)  # do nothing

        def has_alpha_ge0(cond):
            if not isinstance(cond, sympy.And):
                return False
            for a in cond.args:
                if isinstance(a, sympy.core.relational.Relational):
                    # alpha >= 0  or  0 <= alpha
                    if (a.rel_op in (">=", ">") and a.lhs in anim_syms and a.rhs == 0) \
                    or (a.rel_op in (">=", ">") and a.lhs == 0 and a.rhs in anim_syms):
                        return True
            return False

        def pick_branch(*args):
            # args is tuple of ( (val1, cond1), (val2, cond2), ... )
            for val, cond in args:
                if cond is True or cond is sympy.S.true:
                    return val.xreplace(subs0)
                if has_alpha_ge0(cond):
                    return val.xreplace(subs0)
                cond0 = cond.xreplace(subs0)
                if cond0 is True or cond0 is sympy.S.true:
                    return val.xreplace(subs0)
            # fallback: return Piecewise unchanged but with subs applied
            return sympy.Piecewise(*[(v.xreplace(subs0), c.xreplace(subs0)) for v, c in args])

        # Transform all nested Piecewise nodes only
        return expr.replace(sympy.Piecewise, pick_branch)

    #TODO: 
    # fix self.scenes -> make sure animation params show at moving points (express moving points in terms of known coords and anim_params -> use simplify on new constraints) xxx
    # test with other methods not moving points (later)
    # do not run solver.numerical() -> instead build your optimization function and run it only once (later)
    # add more run_types, f.i. sequentially (later, of needed)
    def run(
        self,
        moving_objects: List[Point],
        frames: int = 20, 
        opt_method: str = "dual_annealing", 
        distance_penalty: Optional[float] = 1, 
        min_dist: Optional[float] = None,
        run_type: Literal["independent", "together"] = "independent"  #TODO:maybe add sequentially aftter
    ):
        """
        Generate a sequence of solved scenes by interpolating animation parameters
        from 0 to 1, effectively creating frames of the animation.

        Points not listed in `moving_objects` are fixed to their initial positions.

        Args:
            moving_objects: A list of Point objects that are allowed to move during animation.
                        Points not in this list will remain fixed.
            frames: The number of frames to generate in the animation.
            opt_method: The numerical solver method to use for constraint solving. 
                        Refer to `scipy.optimize.minimize` for available options.
            distance_penalty: The penalty weight to avoid degenerate cases related to
                            area and point distances. This is used in the first frame
                            and as a penalty constraint. It is removed from the moving points.
            min_dist: The minimum allowed distance between points. This is only used for
                    the first frame.

        Raises:
            ValueError: If no animation parameters are defined in the scene.
        """

        #0. PREPROCESSING --------------------------------------------------------------
        #this will be used to build penalty -> avoid setting penalty on missing points   
        self.scene._moving_objects = moving_objects
        self._run_type = run_type

        #make two deepcopies -> you can remove
        original_scene = deepcopy(self.scene) #before running anything
        first_scene = deepcopy(self.scene) # fist frame

        #check if animation parameters exist
        time_params_info = first_scene.animate.get_time_params()
        time_params = set([time_param["symbol"] for time_param in time_params_info])
        if not time_params_info:
            raise ValueError(
                "[solver.analytical] No animation parameters found. Please use the 'add_parameter' function from the 'animate' namespace."
            )

        # 1. For first frame force animation parameters to start at 0 -------------------
        # scan constraint and force time_param to be 0 (sympy trick to deal with variable piecewise bounds)
        for c in first_scene.constraints:
            eq = c["equation"]
            if (eq.free_symbols & time_params):
                ans = first_scene.animate.eval_piecewise_at_zero(eq, time_params)
                c["equation"] = ans 
        # remove time_params from parameters
        to_delete = [key for key, param in first_scene.parameters.items() if param.get("origin") == "animation"]
        for key in to_delete:
            del first_scene.parameters[key]

        # 2. Solve numerically -----------------------------------------------------------
        print("[INFO] Obtaining numerical solution first for animation params = 0")
        first_scene.solver.numerical(distance_penalty = distance_penalty, min_dist = min_dist)

        # check if first frame failed
        if first_scene.solver.status != "success":
            raise RuntimeError("Optimization for the first frame failed!")

        # 2. update animation parameter bounds ------------------------------------------- 
        # this is needed for discritizing time_param, and gif
        # right bounds can depend on other parameters
        self_time_params_info = self.get_time_params()
        time_params_original = original_scene.animate.get_time_params()
        for anim_og, anim_self in zip(time_params_original, self_time_params_info):
            if not (isinstance(anim_og["bound"][0], float) or  isinstance(anim_og["bound"][0], int) ) :
                right_value = float(anim_og["bound"][0].subs(first_scene.solution))
                anim_og["bound"][0] = right_value
                anim_self["bound"][0] = right_value
            if not (isinstance(anim_og["bound"][1], float) or  isinstance(anim_og["bound"][1], int) ) :
                right_value = float(anim_og["bound"][1].subs(first_scene.solution))
                anim_og["bound"][1] = right_value
                anim_self["bound"][1] = right_value

        # 3. Create new solution dictionary ------------------------------------------------
        # contains only fixed point coordinates and reduced var (no MP and anim params)
        sol_no_moving_objects = first_scene.solution
        vars_to_remove = []
        for mp in original_scene._moving_objects:
            for attr_value in vars(mp).values():
                if isinstance(attr_value, sympy.Symbol): #remove all sympy symbols fields of moving objects ("x","y","radius",etc)
                    if attr_value not in vars_to_remove:
                        vars_to_remove.append(attr_value)
        vars_to_remove.extend(time_params)
        for var in vars_to_remove:
            sol_no_moving_objects.pop(var, None)
        # 4. Replace objects with new values --------------------------------------------------
        for obj in original_scene.get_all_objects().values():
            original_scene.solver.substitute_in_obj(obj, sol_no_moving_objects, numeric=True)    
        #do this for original_scene and the self.scene object!
        for obj in self.scene.get_all_objects().values():
            self.scene.solver.substitute_in_obj(obj, sol_no_moving_objects, numeric=True)

        # 5. Replace constraints with new values --------------------------------------------------
        # also remove constraints with no variables 
        new_constraints = []
        for constraint in original_scene.constraints:
            new_eq = constraint["equation"].xreplace(sol_no_moving_objects)
            # Keep only if symbols remain
            if new_eq.free_symbols:
                constraint["equation"] = new_eq
                new_constraints.append(constraint)
        original_scene.constraints = new_constraints


        # 5. Replace all params except moving points and new vars----------------------------------
        vars_to_solve = {p.name: p for p in vars_to_remove}
        original_scene.parameters = {
            k: v for k, v in first_scene.parameters.items()
            if k in vars_to_solve
        }

        print(f"[INFO] Reduced number of parameters to solve: {list(original_scene.parameters.keys())}")
        print(f"[INFO] Reduced number of constraints to solve: {original_scene.constraints}")

        # 6. Build grid of animation param values to scan------------------------------------------
        anim_scenes_data = [] 
        num_time_params = len(time_params)
        total_times = [time_param["bound"][1] for time_param in time_params_original]
        
        if run_type == "together": 
            # Global timeline from 0 to the slowest parameter's total time
            total_duration = float(max(total_times))
            t_grid = np.linspace(0.0, total_duration, num=frames)

            # For each frame time t, lock-step values are min(t, total_times[i])
            # i.e., parameters that finish earlier stay at their final value.
            iterable = (tuple(float(min(t, total_times[k])) for k in range(num_time_params))
                        for t in t_grid)

            total_frames_to_generate = frames

            print(
                f"[INFO] Successful. Now computing {frames} total frames "
                f"in 'together' mode over total_duration={total_duration:.6g}s "
                f"with {num_time_params} animation parameters."
            )

        else:
            # separate: full Cartesian product (aim for ~frames total)
            print("[INFO] Since run_type = 'independent' (points move independently), 'speed' argument is ignored.")
            steps_per_param = int(round(frames ** (1 / max(1, num_time_params))))
            if steps_per_param < 1:
                steps_per_param = 3

            total_time = max(total_times)
            time_param_ranges = [
                np.linspace(0.0, total_time, num=steps_per_param)
                for i in range(num_time_params)
            ]
            total_frames_to_generate = steps_per_param ** num_time_params
            iterable = itertools.product(*time_param_ranges)

            print(
                f"[INFO] Successful. Now computing {total_frames_to_generate} total frames "
                f"with {num_time_params} animation parameters"
                f"(steps_per_param={steps_per_param}) in 'separate' mode."
            )
        # 7. Starting loop now - scaning for animation parameter grid--------------------
        initial_value = None
        time_params_names = [anim.name for anim in time_params]
        for i, time_param_vals_tuple in enumerate(
            tqdm(iterable, total=total_frames_to_generate, desc="Generating Animation Frames")
        ):

            #temporary scene is copy of original scene <- with specific time_param values
            temp_scene = deepcopy(original_scene)

            #use previously found animation value
            if initial_value is not None:
                temp_scene.solver._internal_sol = initial_value

            # Create the dictionary for the current frame (anim param values)
            frame_data = {}
            for j, time_param_val in enumerate(time_param_vals_tuple):
                frame_data[time_params_info[j]["symbol"]] = time_param_val
            
            #replace in equation
            new_constraints = []
            for constraint in temp_scene.constraints:
                constraint["equation"] = constraint["equation"].xreplace(frame_data)

            # remove all other params except moving points and nims
            temp_scene.parameters = {
                k: v for k, v in temp_scene.parameters.items()
                if k not in time_params_names
            }

            #numerically solve
            temp_scene.solver.numerical(
                distance_penalty=0, verbose=False, method=opt_method
            )

            # set new init_values for the next iteration
            initial_value = temp_scene.solver._internal_sol
            
            # Add the solved scene to the dictionary
            frame_data["scene"] = temp_scene
            anim_scenes_data.append(frame_data)

        self.anim_scenes = anim_scenes_data 
        self.scene.solver.status = "success" #consider success



    def gif(self, save_loc: str = "animation.gif") -> None: #, interp_per_gap = 5
        """
        Generate and save an animated GIF from the current list of solved scenes.

        Args:
            save_loc: File path to save the GIF.
            total_time: Total duration of the GIF in seconds.

        Raises:
            ValueError: If there are no frames to generate.
        """
        if not self.anim_scenes:
            raise ValueError("animate.anim_scenes is empty. No frames to generate GIF.")

        num_frames = len(self.anim_scenes)
        time_params = self.get_time_params()
        num_time_params = len(time_params)
        total_times = [time_param["bound"][1] for time_param in time_params]

        # Calculate duration per frame in milliseconds
        if self._run_type == "together":
            total_time = max(total_times)
        else:
            total_time = max(total_times)*num_frames**(1/num_time_params) if num_time_params > 1 else max(total_times)
        
        frame_duration_ms = int((total_time * 1000) / num_frames)
        frames = []

        initial_fig = self.anim_scenes[0]["scene"].plot(return_fig=True)
        initial_ax = initial_fig.gca()

        consistent_xlim = initial_ax.get_xlim()
        consistent_ylim = initial_ax.get_ylim()

        print(
            f"[INFO] Generating GIF with {num_frames} frames over {total_time} seconds." # (x{interp_per_gap})
        )

        for i, scene_obj in enumerate(self.anim_scenes):
            if scene_obj["scene"].solver.status != "success":
                print(f"[WARNING] Error plotting scene {i}: optimization not sucessfull")
                continue
            try:
                fig = scene_obj["scene"].plot(
                    return_fig=True, xlim=consistent_xlim, ylim=consistent_ylim
                )

                # Save the current figure to a BytesIO object (in-memory)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
                buf.seek(0)  # Rewind the buffer to the beginning

                # Open the image from BytesIO using Pillow and append to frames list
                frames.append(Image.open(buf))

                # Close the figure immediately after saving to free up memory
                plt.close(fig)

            except Exception as e:
                print(f"[WARNING] Error plotting scene {i}: {e}")
                continue  # Skip to the next scene

        # Save the frames as a GIF
        if frames:

            #all_new_frames = [] # frames 
            #for a, b in zip(frames, frames[1:]):
            #    all_new_frames.extend(crossfade_hard_threshold(a, b, num_inbetweens=interp_per_gap)[:-1])
            #all_new_frames.append(frames[-1])

            frames[0].save(
                save_loc,
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration_ms,#/interp_per_gap,
                loop=0,  # 0 means loop forever (standard for continuous animation)
            )
            print(f"[INFO] GIF saved successfully to '{save_loc}'")
        
            return 
        
        raise RuntimeError("No valid frames were generated to create the GIF.")

    def point_sweep_line(self, point: Point, start_point: Point, end_point: Point, total_time: Optional[NumericExpr] = 2, speed: Optional[NumericExpr] = None):
        """
        Animate a point moving along a line segment by directly setting its x and y coordinates parametrically.
        """
        self.scene._add_relationship("point_sweep_line_direct", {
            "point": point, "start_point": start_point, "end_point": end_point, "total_time": total_time, "speed": speed
        })

        if speed:
            if total_time != 2:
                print("[WARNING] Total time argument ignored. Speed argument is being taken into account")
            total_time = start_point.distance(end_point) / speed

        # Create animation parameter lambda from 0 to total_time
        time_param = self.add_parameter(self._create_time_param_name(), total_time)

        # Calculate delta_x and delta_y between start and end points
        delta_x = end_point.x - start_point.x
        delta_y = end_point.y - start_point.y

        # Add constraints for point.x and point.y directly
        self.scene.constraint.eq(
            point.x,
            start_point.x + delta_x * (time_param / total_time),
            description=f"Point {point.name} x-coordinate sweep"
        )
        self.scene.constraint.eq(
            point.y,
            start_point.y + delta_y * (time_param / total_time),
            description=f"Point {point.name} y-coordinate sweep"
        )

        
    def point_sweep_circle(self, point: Point, circle: Circle, counterclockwise: bool = True, total_time: Optional[NumericExpr] = 2, speed: Optional[NumericExpr] = None):
        """
        Animate a point moving around a circle by directly constraining its x and y coordinates.
        Point starts at the top of the circle and moves counterclockwise by default.
        """
        self.scene._add_relationship(
            "point_sweep_circle_direct",
            {"point": point, "circle": circle, "counterclockwise": counterclockwise, "total_time": total_time, "speed": speed},
        )

        if speed:
            if total_time != 2:
                print("[WARNING] Total time argument ignored. Speed argument is being taken into account")
            total_time = circle.perimeter / speed

        time_param = self.add_parameter(self._create_time_param_name(), total_time)

        sign = 1 if counterclockwise else -1

        two_pi = 2 * sympy.pi
        start_angle = sympy.pi / 2  # start at top of circle

        theta = start_angle + sign * (time_param / total_time) * two_pi

        cx = circle.center.x
        cy = circle.center.y
        r = circle.radius

        self.scene.constraint.eq(
            point.x,
            cx + r * sympy.cos(theta),
            f"Point {point.name} x-coordinate sweeping circle {circle.name}"
        )
        self.scene.constraint.eq(
            point.y,
            cy + r * sympy.sin(theta),
            f"Point {point.name} y-coordinate sweeping circle {circle.name}"
        )

    def point_sweep_arc(self, point: Point, arc: Union[MinorArc, MajorArc], counterclockwise: bool = True, total_time: Optional[NumericExpr] = 2, speed: Optional[NumericExpr] = None):
        """
        Animate a point moving along an arc by directly constraining its x and y coordinates.
        Point moves from arc.start_point to arc.end_point over the arc length.
        """
        self.scene._add_relationship(
            "point_sweep_arc_direct",
            {"point": point, "arc": arc, "counterclockwise": counterclockwise, "total_time": total_time, "speed": speed},
        )

        if speed:
            if total_time != 2:
                print("[WARNING] Total time argument ignored. Speed argument is being taken into account")
            total_time = arc.length / speed

        time_param = self.add_parameter(self._create_time_param_name(), total_time)

        sign = 1 if counterclockwise else -1

        cx = arc.center.x
        cy = arc.center.y
        r = arc.radius

        dx_start = arc.start_point.x - cx
        dy_start = arc.start_point.y - cy
        theta0 = sympy.atan2(dy_start, dx_start)

        delta_theta = sympy.rad(arc.central_angle)  # Convert degrees to radians

        theta = theta0 + sign * (time_param / total_time) * delta_theta

        self.scene.constraint.eq(
            point.x,
            cx + r * sympy.cos(theta),
            f"Point {point.name} x-coordinate sweeping arc {arc.name}"
        )
        self.scene.constraint.eq(
            point.y,
            cy + r * sympy.sin(theta),
            f"Point {point.name} y-coordinate sweeping arc {arc.name}"
        )
    
        
    def point_sweep_polyline(
        self,
        point: Point,
        path_points: List[Point],
        total_time: Optional[NumericExpr] = 5,
        speed: Optional[List[NumericExpr]] = None
    ):
        """
        Animates a point moving smoothly along a polyline defined by a sequence of points.

        A single parameter `time_param` in [0, 1] controls the point's position.
        As `time_param` increases from 0 to 1, the point traverses each segment of the polyline
        in proportion to segment length. A single sympy.Piecewise constraint ensures that the point's
        coordinates match the appropriate interpolated position along the active segment.

        Args:
            point (Point): The Point object to animate.
            path_points (List[Point]): Ordered list of points defining the polyline path.
        """
        self.scene._add_relationship(
            "point_sweep_polyline",
            {
                "point": point,
                "path_points": path_points,
            },
        )

        # build path
        num_segments = len(path_points) - 1
        if num_segments < 1:
            raise ValueError("Need at least start and end to define a sweep polyline.")

        # compute segment lengths (symbolic OK)
        segment_lengths = []
        for i in range(num_segments):
            A = path_points[i]
            B = path_points[i + 1]
            dx = B.x - A.x
            dy = B.y - A.y
            length = sympy.sqrt(dx * dx + dy * dy)
            segment_lengths.append(length)

        total_length = sum(segment_lengths)
        if total_length == 0:
            raise ValueError("Total polyline length is zero; cannot define sweep.")

        # --- speed/time handling ---
        durations = []  # per-segment time windows

        if speed is not None:
            if len(speed) != num_segments:
                raise ValueError("Size of speed must match number of segments in the path.")
            # Validate speeds > 0 when numeric
            for idx, v in enumerate(speed):
                if float(v) <= 0:
                    raise ValueError(f"speed[{idx}] must be > 0 (got {v}).")
            # time on each segment = length / speed
            durations = [segment_lengths[i] / speed[i] for i in range(num_segments)]
            total_time = sum(durations)
            if total_time == 5:  # unlikely but keeps your previous message style
                print("[WARNING] Total time argument ignored because 'speed' is provided.")
        else:
            # No per-segment speeds: move at constant global speed.
            # Allocate time to segments proportional to their lengths
            if float(total_time) <= 0:
                raise ValueError("total_time must be > 0.")
            durations = [total_time * (segment_lengths[i] / total_length) for i in range(num_segments)]

        # parameter: now time_param is *time* in [0, total_time]
        time_param = self.add_parameter(self._create_time_param_name(), total_time)

        # cumulative time thresholds (segment start times)
        cumulative = []
        acc = sympy.Integer(0)
        for d in durations:
            cumulative.append(acc)      # start time for this segment
            acc = acc + d  # end time accumulates to total_time

        # Build piecewise x and y using time windows
        pieces_x = []
        pieces_y = []
        for i in range(num_segments):
            A = path_points[i]
            B = path_points[i + 1]
            dur_i = durations[i]
            t0_i = cumulative[i]                 # segment start time
            # local fraction along segment based on time on this segment
            t_i = (time_param - t0_i) / dur_i
            # domain for this segment in time
            cond_i = sympy.And(time_param >= t0_i, time_param <= t0_i + dur_i)
            interp_x = A.x + t_i * (B.x - A.x)
            interp_y = A.y + t_i * (B.y - A.y)
            pieces_x.append((interp_x, cond_i))
            pieces_y.append((interp_y, cond_i))

        # Fallback — ensure end position outside exact domains
        pieces_x.append((path_points[-1].x, True))
        pieces_y.append((path_points[-1].y, True))

        piecewise_x = sympy.Piecewise(*pieces_x)
        piecewise_y = sympy.Piecewise(*pieces_y)

        # Single constraint: squared distance between actual point and piecewise target is zero
        self.scene.constraint.eq(
            point.x,
            piecewise_x,
            description="Point sweeps along polyline - x coord",
        )
        self.scene.constraint.eq(
            point.y,
            piecewise_y,
            description="Point sweeps along polyline - y coord",
        )


    def point_sweep_polyline_v2(
            self,
            point: Point,
            path_points: List[Point],
            radius: Optional[List[NumericExpr]] = None,
            arc_centers: Optional[List[Optional[Point]]] = None, # <-- NEW PARAMETER
            total_time: Optional[NumericExpr] = 5,
            speed: Optional[List[NumericExpr]] = None
        ):
            """
            Animates a point along a path of lines and arcs.

            This version allows specifying explicit centers for arc segments, which is
            essential for complex paths like a circle rolling around a polygon.

            Args:
                point (Point): The Point object to animate.
                path_points (List[Point]): Ordered list of points defining the path.
                radius (Optional[List[NumericExpr]]): List of radii for each segment.
                arc_centers (Optional[List[Optional[Point]]]): A list specifying the center
                    for each segment. If a segment is a line, its entry should be None.
                    If a segment is an arc, its entry is the Point object for its center.
                    If an arc segment's center is None, it will be calculated automatically.
                total_time (Optional[NumericExpr]): Total animation duration.
                speed (Optional[List[NumericExpr]]): Per-segment speeds.
            """
            num_segments = len(path_points) - 1
            if num_segments < 1:
                raise ValueError("Need at least two points to define a path.")

            if radius is not None and len(radius) != num_segments:
                raise ValueError(f"The 'radius' list must have {num_segments} elements.")
            
            if arc_centers is not None and len(arc_centers) != num_segments:
                raise ValueError(f"The 'arc_centers' list must have {num_segments} elements.")

            segment_lengths = []
            segment_geometries = []

            for i in range(num_segments):
                A = path_points[i]
                B = path_points[i + 1]
                r_i = 0 if radius is None else radius[i]

                if r_i == 0:
                    dx, dy = B.x - A.x, B.y - A.y
                    length = sympy.sqrt(dx**2 + dy**2)
                    segment_lengths.append(length)
                    segment_geometries.append({'type': 'line'})
                else:
                    abs_r_i = sympy.Abs(r_i)
                    sign_r_i = sympy.sign(r_i)
                    
                    # --- MODIFICATION START ---
                    # Prioritize the provided center over calculation
                    custom_center = arc_centers[i] if arc_centers else None
                    
                    if custom_center:
                        Cx, Cy = custom_center.x, custom_center.y
                    else:
                        # Fallback to the original calculation if no center is provided
                        print(f"[Warning] Calculating center for segment {i} automatically.")
                        dx, dy = B.x - A.x, B.y - A.y
                        d_sq = dx**2 + dy**2
                        d = sympy.sqrt(d_sq)
                        if r_i.is_number and d.is_number and abs(float(r_i)) < float(d)/2:
                            raise ValueError(f"Radius for segment {i} is too small.")
                        h = sympy.sqrt(abs_r_i**2 - d_sq / 4)
                        Mx, My = (A.x + B.x) / 2, (A.y + B.y) / 2
                        Cx = Mx + sign_r_i * h * (-dy / d)
                        Cy = My + sign_r_i * h * (dx / d)

                    # --- MODIFICATION END ---
                    start_angle = sympy.atan2(A.y - Cy, A.x - Cx)
                    end_angle = sympy.atan2(B.y - Cy, B.x - Cx)
                    
                    # Ensure correct rotation direction
                    delta_angle = sympy.Mod(end_angle - start_angle, 2 * sympy.pi)
                    if sign_r_i < 0:
                        delta_angle -= 2 * sympy.pi

                    arc_length = abs_r_i * sympy.Abs(delta_angle)
                    segment_lengths.append(arc_length)

                    segment_geometries.append({
                        'type': 'arc', 'center_x': Cx, 'center_y': Cy,
                        'radius': abs_r_i, 'start_angle': start_angle, 'delta_angle': delta_angle,
                    })

            total_length = sum(segment_lengths)
            if total_length == 0:
                raise ValueError("Total polyline length is zero.")

            # --- Phase 2 & 3 are unchanged ---
            durations = []
            if speed is not None:
                if len(speed) == 1:
                    durations = [length / speed[0] for length in segment_lengths]
                else:
                    if len(speed) != num_segments: raise ValueError("Speed list size mismatch.")
                    durations = [segment_lengths[i] / speed[i] for i in range(num_segments)]
                total_time = sum(durations)
            else:
                durations = [total_time * (l/total_length) for l in segment_lengths]

            time_param = self.add_parameter(f"t_sweep_{point.name}", total_time)
            
            cumulative_time = [0]
            for d in durations[:-1]: cumulative_time.append(cumulative_time[-1] + d)

            pieces_x, pieces_y = [], []
            for i in range(num_segments):
                geom = segment_geometries[i]
                t_i = sympy.Piecewise((0, sympy.Eq(durations[i], 0)), ((time_param - cumulative_time[i]) / durations[i], True))
                cond_i = sympy.And(time_param >= cumulative_time[i], time_param <= cumulative_time[i] + durations[i])

                if geom['type'] == 'line':
                    A, B = path_points[i], path_points[i+1]
                    interp_x = A.x + t_i * (B.x - A.x)
                    interp_y = A.y + t_i * (B.y - A.y)
                else: # 'arc'
                    angle = geom['start_angle'] + t_i * geom['delta_angle']
                    interp_x = geom['center_x'] + geom['radius'] * sympy.cos(angle)
                    interp_y = geom['center_y'] + geom['radius'] * sympy.sin(angle)
                
                pieces_x.append((interp_x, cond_i))
                pieces_y.append((interp_y, cond_i))

            pieces_x.append((path_points[-1].x, True))
            pieces_y.append((path_points[-1].y, True))
            
            self.scene.constraint.eq(point.x, sympy.Piecewise(*pieces_x))
            self.scene.constraint.eq(point.y, sympy.Piecewise(*pieces_y))