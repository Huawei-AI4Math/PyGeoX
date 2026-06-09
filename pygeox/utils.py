import math
from typing import Optional, List
from collections import Counter

from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import sympy
from copy import deepcopy
from .basic_objects import Point

def normalize_domain(domain):
    """
    Normalize domain input to a standardized dict with keys 'x', 'y', 'r'.
    - If domain is None: all are None (unbounded, i.e., Real).
    - If domain is a number: x and y are [-domain, domain], r is [0, domain].
    - If domain is [xsize, ysize]: x is [-xsize/2, xsize/2], y is [-ysize/2, ysize/2], r is [0, max(xsize, ysize)].
    - If domain is [[xmin, xmax], [ymin, ymax]]: x is [xmin, xmax], y is [ymin, ymax], r is [0, max(xmax-xmin, ymax-ymin)].
    """
    if domain is None:
        return {"x": None, "y": None, "r": [0, np.inf]}
    elif isinstance(domain, (int, float)):
        return {"x": [-domain, domain], "y": [-domain, domain], "r": [0, domain]}
    elif isinstance(domain, (list, tuple)) and len(domain) == 2:
        if all(isinstance(x, (int, float)) for x in domain):
            xsize, ysize = domain
            rmax = max(xsize, ysize)
            return {
                "x": [-xsize / 2, xsize / 2],
                "y": [-ysize / 2, ysize / 2],
                "r": [0, rmax],
            }
        elif all(isinstance(x, (list, tuple)) and len(x) == 2 for x in domain):
            xmin, xmax = domain[0]
            ymin, ymax = domain[1]
            rmax = max(xmax - xmin, ymax - ymin)
            return {"x": [xmin, xmax], "y": [ymin, ymax], "r": [0, rmax]}
    raise ValueError(
        "Invalid domain format. Use None, a number, [xsize,ysize], or [[xmin,xmax],[ymin,ymax]]"
    )


def check_symbol_in_constraints(symbol, constraints) -> bool:
    """
    Checks if the given anim_lambda_symbol exists in any of the equations
    within a list of constraints.

    Args:
        constraints: A list of dictionaries, where each dictionary
                     is expected to have an 'equation' key whose value
                     is a SymPy expression (sympy.Expr or sympy.Relational).
        anim_lambda_symbol: The sympy.Symbol object representing anim_lambda.

    Returns:
        True if anim_lambda_symbol is found in any equation, False otherwise.
    """
    for constraint_dict in constraints:
        equation = constraint_dict["equation"]
        if symbol in equation.free_symbols:
            return True  # Found anim_lambda_symbol in this equation
    return False

def draw_perpendicular_sign(ax, p1, vertex, p2, size=0.1, color='black', linewidth=1.5, zorder=10):
    """
    Draw a small perpendicular (right angle) marker at 'vertex' formed by points p1 and p2.

    Args:
        ax (matplotlib.axes.Axes): The axes to draw on.
        p1, vertex, p2: Points with .x and .y attributes (coordinates).
        size (float): Length of each leg of the perpendicular sign.
        color (str): Color of the marker.
        linewidth (float): Thickness of the lines.
        zorder (int): Drawing order.
    """
    # Convert points to numpy arrays
    v = np.array([vertex.x, vertex.y])
    p1_vec = np.array([p1.x, p1.y]) - v
    p2_vec = np.array([p2.x, p2.y]) - v

    # Normalize vectors
    p1_norm = p1_vec / np.linalg.norm(p1_vec)
    p2_norm = p2_vec / np.linalg.norm(p2_vec)

    # Compute points for the perpendicular marker
    # Move along p1 and p2 by 'size'
    point_a = v + p1_norm * size
    point_b = v + p2_norm * size
    # The corner of the right angle square
    corner = point_a + p2_norm * size

    # Draw the three segments forming the right angle
    #ax.plot([point_a[0], v[0]], [point_a[1], v[1]], color=color, linewidth=linewidth, zorder=zorder)
    #ax.plot([point_b[0], v[0]], [point_b[1], v[1]], color=color, linewidth=linewidth, zorder=zorder)
    ax.plot([point_a[0], corner[0]], [point_a[1], corner[1]], color=color, linewidth=linewidth, zorder=zorder)
    ax.plot([point_b[0], corner[0]], [point_b[1], corner[1]], color=color, linewidth=linewidth, zorder=zorder)

def plot_scene_step_by_step(
    scene,  # Use the actual type hint for GeoScene
    show_names: bool = True,
    show_axis_grid: bool = False,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    return_figures: bool = True,  # Add option to control return behavior
) -> List[plt.Figure]:
    
    if scene.solver.status == "not solved":
        raise ValueError(
            "Please perform geometru constraint solving with solver.solve to solve the geometric system."
        )
    elif scene.solver.status == "failure":
        raise ValueError(
            "Geometry constraint solving failed. Please obtain a valid solution with solver.solve"
        )


    # Store original interactive state
    was_interactive = plt.isinteractive()
    
    try:
        # Turn off interactive mode to prevent auto-display
        plt.ioff()
        
        # Plot first object to get axis limits
        last_frame = scene.plot(
                show_names=show_names,
                show_axis_grid=show_axis_grid,
                xlim=xlim,
                ylim=ylim,
                return_fig=True
            )
        
        ax = last_frame.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        plt.close(last_frame)  # Close the reference plot

        allowed_objects = []
        current_allowed = []
        
        # Build progressive lists of allowed objects
        for obj_dict in scene._ordered_objects_names:
            obj_name = obj_dict["name"]
            obj_type = obj_dict["obj_type"]

            if obj_type == 'Point':
                continue

            obj = scene.objects_solved[obj_type][obj_name]
            
            # Find associated points for this object
            associated_points = []
            
            if hasattr(obj, '__dict__'):
                for attr_name, attr_value in obj.__dict__.items():
                    if isinstance(attr_value, Point):
                        associated_points.append(attr_value.name)
                    elif isinstance(attr_value, (list, tuple)):
                        for item in attr_value:
                            if isinstance(item, Point):
                                associated_points.append(item.name)
            
            # Add current object and its points to current allowed list
            step_objects = current_allowed.copy()
            step_objects.append(obj_name)
            step_objects.extend(associated_points)
            
            # Remove duplicates while preserving order
            step_objects = list(dict.fromkeys(step_objects))
            
            current_allowed = step_objects
            allowed_objects.append(step_objects)
        
        # Generate plots
        figures = []
        
        for allowed_list in allowed_objects:
            # Make a deep copy of the scene
            new_scene = deepcopy(scene)
            
            # Filter objects to keep only allowed ones
            filtered_objects = {}
            
            for obj_type, objects_dict in new_scene.objects_solved.items():
                filtered_type_dict = {}
                
                for obj_name, obj_data in objects_dict.items():
                    if obj_name in allowed_list:
                        filtered_type_dict[obj_name] = obj_data
                
                if filtered_type_dict:
                    filtered_objects[obj_type] = filtered_type_dict
            
            # Update the scene with filtered objects
            new_scene.objects_solved = filtered_objects
            
            # Generate the plot
            fig = new_scene.plot(
                show_names=show_names,
                show_axis_grid=show_axis_grid,
                xlim=xlim,
                ylim=ylim,
                return_fig=True
            )
            
            if return_figures:
                figures.append(fig)
            else:
                # If not returning figures, display them
                plt.show()
                plt.close(fig)
        
        return figures if return_figures else []
        
    finally:
        # Restore original interactive state
        if was_interactive:
            plt.ion()
        else:
            plt.ioff()

def plot_scene(
    scene,  # Use the actual type hint for GeoScene
    show_names: bool = True,
    show_axis_grid: bool = False,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    highlight_tracked_objects: bool = True,
) -> plt.Figure:
    """
    Plots all geometric objects in a GeoScene instance using matplotlib.

    Args:
        scene: The GeoScene object to plot.
        show_names: If True, displays the names of the objects on the plot.
        show_axis_grid: If True, shows axis ticks, labels, and grid.
        xlim: Optional tuple (min, max) for x-axis limits.
        ylim: Optional tuple (min, max) for y-axis limits.
        highlight_tracked_objects: If True, highlights objects in track_unsolved with dark red.

    Returns:
        The matplotlib Figure object containing the plot.
    """
    if scene.solver.status == "not solved":
        raise ValueError(
            "Please perform geometru constraint solving with solver.solve to solve the geometric system."
        )
    elif scene.solver.status == "failure":
        raise ValueError(
            "Geometry constraint solving failed. Please obtain a valid solution with solver.solve"
        )
    
    # Initial styling -------------------------------------------------------------------------------------------------------------------

    style = "light"
    if style == "light":
        color_background = "#f3f4f9"
        color_spine = "#b2b6d7"
        color_axes = "white"
        color_linelike = "#3a3640"        # Desaturated deep purple-gray
        color_point = "#2e2e2e"           # Slightly lighter, still neutral dark gray
        color_circle = "#5a3f3f"          # Muted, dusty reddish-brown
        color_arc = "#2e2e2e"             # Black color for arcs
        color_ray = "#5a5740"             # Muted yellow-gray, low contrast
        color_infinite_line = "#5a4343"   # Muted reddish-gray, slightly lighter
        colot_text = "black"
        
        # Dark red colors for highlighting track_unsolved objects
        color_highlight_linelike = "#8B0000"    # Dark red for line segments
        color_highlight_point = "#8B0000"      # Dark red for points
        color_highlight_circle = "#8B0000"     # Dark red for circles
        color_highlight_arc = "#8B0000"        # Dark red for arcs
        color_highlight_ray = "#8B0000"        # Dark red for rays
        color_highlight_infinite_line = "#8B0000"  # Dark red for infinite lines
    else:
        color_background = "#0c0b06"        # Inverted from #f3f4f9 → very dark background
        color_spine = "#4d4928"             # Inverted from #b2b6d7 → warm gray-brown
        color_axes = "#000000"              # Inverted from white → black
        color_linelike = "#c5c9bf"          # Inverted from #3a3640 → pale grayish green
        color_point = "#d1d1d1"             # Inverted from #2e2e2e → medium-light gray
        color_circle = "#a5c0c0"            # Inverted from #5a3f3f → muted teal
        color_arc = "#d1d1d1"               # Inverted from #2e2e2e → light gray
        color_ray = "#a5a8bf"               # Inverted from #5a5740 → pale steel blue
        color_infinite_line = "#a5bcbc"     # Inverted from #5a4343 → soft cyan-gray
        colot_text = "white"

    # Identify objects that should be highlighted (from track_unsolved)
    highlighted_objects = set()
    if highlight_tracked_objects and hasattr(scene, 'track_unsolved') and scene.track_unsolved:
        for track_name, track_info in scene.track_unsolved.items():
            if "objects" in track_info:
                for obj in track_info["objects"]:
                    if hasattr(obj, 'name'):
                        highlighted_objects.add(obj.name)

    # Create the figure with a higher DPI and set the figure's background color to white
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    fig.set_facecolor(color_axes)

    # Set the axes' background color to grey-purple
    ax.set_facecolor(color_background)

    # Customize the frame (spines)
    for spine in ax.spines.values():
        spine.set_edgecolor(color_spine)  # Set spine color to purple-black
        spine.set_linewidth(1.5)       # Make the lines thicker

    # --- Collect all valid coordinates for initial auto-scaling --------------------------------------------------------------------------------
    # This part can be reused for determining overall plot range.
    all_x_coords = []
    all_y_coords = []

    def add_coords(x, y):
        try:
            fx, fy = float(x), float(y)
            all_x_coords.append(fx)
            all_y_coords.append(fy)
            return fx, fy
        except (TypeError, ValueError):  # Catches symbolic values
            return None, None

    # Plot Points------------------------------------------------------------------------------------------------------------------------------
    for name, point in scene.get_all_objects("Point").items():
        x, y = add_coords(point.x, point.y)
        if x is not None and y is not None:
            # Use highlight color if this point is in track_unsolved
            point_color = color_highlight_point if name in highlighted_objects else color_point
            ax.plot(x, y, "o", color=point_color, markersize=4, zorder=10)
            if show_names:
                ax.text(
                    x, y, f" {name}", color=colot_text, verticalalignment="bottom", fontsize=9, zorder=15
                )

    # Plot LineSegments--------------------------------------------------------------------------------------------------------------------------
    for name, line_segment in scene.get_all_objects("LineSegment").items():
        x1, y1 = add_coords(line_segment.point1.x, line_segment.point1.y)
        x2, y2 = add_coords(line_segment.point2.x, line_segment.point2.y)
        if all(c is not None for c in [x1, y1, x2, y2]):
            # Use highlight color if this line segment is in track_unsolved
            line_color = color_highlight_linelike if name in highlighted_objects else color_linelike
            ax.plot([x1, x2], [y1, y2], color=line_color, linestyle="-", zorder=5) # Purple-black line

    # Plot Circles--------------------------------------------------------------------------------------------------------------------------------
    for name, circle in scene.get_all_objects("Circle").items():
        cx, cy = add_coords(circle.center.x, circle.center.y)
        try:
            r = float(circle.radius)
        except (TypeError, ValueError):
            r = None  # Handle symbolic radius

        if cx is not None and cy is not None and r is not None and r > 0:
            # Use highlight color if this circle is in track_unsolved
            circle_color = color_highlight_circle if name in highlighted_objects else color_circle
            circle_patch = patches.Circle(
                (cx, cy), r, edgecolor=circle_color, facecolor="none", zorder=5  # Red-black circle
            )
            ax.add_patch(circle_patch)
            # Add circle's bounding box to coordinates for limits
            all_x_coords.extend([cx - r, cx + r])
            all_y_coords.extend([cy - r, cy + r])

    # Plot Minor and Major Arcs----------------------------------------------------------------------------------------------------------------
    for arc_type in ["MinorArc", "MajorArc"]:
        for name, arc in scene.get_all_objects(arc_type).items():
            cx, cy = add_coords(arc.center.x, arc.center.y)
            try:
                r = float(arc.radius)
                start_x, start_y = float(arc.start_point.x), float(arc.start_point.y)
                end_x, end_y = float(arc.end_point.x), float(arc.end_point.y)
            except (TypeError, ValueError):
                continue  # Skip if any part is symbolic

            if cx is None or cy is None or r is None or r <= 0:
                continue

            angle1 = np.rad2deg(np.arctan2(start_y - cy, start_x - cx))
            angle2 = np.rad2deg(np.arctan2(end_y - cy, end_x - cx))

            determinant = (start_x - cx) * (end_y - cy) - (end_x - cx) * (start_y - cy)

            is_minor = arc_type == "MinorArc"

            if determinant > 0:
                t1, t2 = (angle1, angle2) if is_minor else (angle2, angle1)
            else:
                t1, t2 = (angle2, angle1) if is_minor else (angle1, angle2)

            # Use highlight color if this arc is in track_unsolved
            arc_color = color_highlight_arc if name in highlighted_objects else color_arc
            arc_patch = patches.Arc(
                (cx, cy),
                2 * r,
                2 * r,
                angle=0,
                theta1=t1,
                theta2=t2,
                edgecolor=arc_color,  # Green-black arc
                lw=1.5,
                zorder=5,
            )
            ax.add_patch(arc_patch)
            # Add arc's bounding box (approximate) to coordinates for limits
            all_x_coords.extend([cx - r, cx + r])
            all_y_coords.extend([cy - r, cy + r])



    # Calculate x and y limits -----------------------------------------------------------------------------------------------------------
    
    # Set aspect ratio, and then apply user-defined or auto-calculated limits
    ax.set_aspect("equal", adjustable="box")

    pad_x = 0.3 * (np.max(all_x_coords) - np.min(all_x_coords))
    pad_y = 0.3 * (np.max(all_y_coords) - np.min(all_y_coords))
    if (not xlim) and (not ylim):
        #ax.autoscale_view()
        #ax.relim()
        ax.set_xlim(np.min(all_x_coords) - pad_x, np.max(all_x_coords) + pad_x)
        ax.set_ylim(np.min(all_y_coords) - pad_y, np.max(all_y_coords) + pad_y)
    else:
        if xlim:
            ax.set_xlim(xlim)
        elif all_x_coords:
            ax.set_xlim(np.min(all_x_coords) - pad_x, np.max(all_x_coords) + pad_x)

        if ylim:
            ax.set_ylim(ylim)
        elif all_y_coords:
            ax.set_ylim(np.min(all_y_coords) - pad_y, np.max(all_y_coords) + pad_y)

    # Get current plot limits (either user-defined or auto-scaled) for lines/rays
    current_xlim = ax.get_xlim()
    current_ylim = ax.get_ylim()

    # Helper to get plot points for infinite lines/rays----------------------------------------------------------------------------------
    def _get_plot_segment_for_line_like(
        line_like_obj,
        x_limits: tuple[float, float],
        y_limits: tuple[float, float],
        is_ray: bool = False,
    ) -> Optional[tuple[float, float, float, float]]:
        """
        Calculates the segment of an infinite line or ray that spans the plot limits.
        Returns (x_start, y_start, x_end, y_end) or None if the line is outside limits or symbolic.
        """
        p1 = line_like_obj.point1
        p2 = line_like_obj.point2

        try:
            x1, y1 = float(p1.x), float(p1.y)
            x2, y2 = float(p2.x), float(p2.y)
        except (TypeError, ValueError):
            return None  # Cannot plot if points have symbolic coordinates

        m = line_like_obj.slope
        c = line_like_obj.intercept

        # Define the bounding box corners
        x_min, x_max = x_limits
        y_min, y_max = y_limits

        EPS = 1e-9  # Small epsilon for float comparisons

        intersections = []

        def add_intersection(x_intersect, y_intersect):
            if (
                x_min - EPS <= x_intersect <= x_max + EPS
                and y_min - EPS <= y_intersect <= y_max + EPS
            ):
                intersections.append((x_intersect, y_intersect))

        if m == sympy.oo:  # Vertical line (x = x1)
            x_val = x1
            add_intersection(x_val, y_min)
            add_intersection(x_val, y_max)
        elif m == 0:  # Horizontal line (y = y1)
            y_val = y1
            add_intersection(x_min, y_val)
            add_intersection(x_max, y_val)
        elif m is not None:  # Diagonal line (y = mx + c) - m is not symbolic
            # Intersect with x_min boundary: y = m*x_min + c
            add_intersection(x_min, m * x_min + c)
            # Intersect with x_max boundary: y = m*x_max + c
            add_intersection(x_max, m * x_max + c)
            # Intersect with y_min boundary: x = (y_min - c) / m
            add_intersection((y_min - c) / m, y_min)
            # Intersect with y_max boundary: x = (y_max - c) / m
            add_intersection((y_max - c) / m, y_max)
        else:  # m is None (symbolic slope)
            return None

        # Filter unique intersection points
        unique_intersections = []
        for ix, iy in intersections:
            is_unique = True
            for uix, uiy in unique_intersections:
                if abs(ix - uix) < EPS and abs(iy - uiy) < EPS:
                    is_unique = False
                    break
            if is_unique:
                unique_intersections.append((ix, iy))

        if not unique_intersections:
            return None  # Line is entirely outside the plot area

        if is_ray:
            ray_start_x, ray_start_y = (
                float(line_like_obj.start_point.x),
                float(line_like_obj.start_point.y),
            )
            dx_ray, dy_ray = line_like_obj.direction_vector

            furthest_intersect = None
            max_proj_dist = -float("inf")

            all_potential_points = [(ray_start_x, ray_start_y)] + unique_intersections

            for ix, iy in all_potential_points:
                vec_to_intersect_x = ix - ray_start_x
                vec_to_intersect_y = iy - ray_start_y

                proj_dist = vec_to_intersect_x * dx_ray + vec_to_intersect_y * dy_ray

                if proj_dist > max_proj_dist:
                    max_proj_dist = proj_dist
                    furthest_intersect = (ix, iy)

            if furthest_intersect and (max_proj_dist >= -EPS):
                return (
                    ray_start_x,
                    ray_start_y,
                    furthest_intersect[0],
                    furthest_intersect[1],
                )
            else:
                return None

        else:  # For infinite Line
            unique_intersections.sort()

            if len(unique_intersections) < 2:
                # Fallback to extend line across the entire view if few intersections found
                if m == sympy.oo:  # Vertical
                    x_plot = x1
                    y_plot_start = (
                        y_limits[0] - abs(y_limits[1] - y_limits[0]) * 0.1
                    )  # Extend slightly
                    y_plot_end = y_limits[1] + abs(y_limits[1] - y_limits[0]) * 0.1
                    return (x_plot, y_plot_start, x_plot, y_plot_end)
                elif m is not None:  # Horizontal or Diagonal
                    x_plot_start = x_limits[0] - abs(x_limits[1] - x_limits[0]) * 0.1
                    x_plot_end = x_limits[1] + abs(x_limits[1] - x_limits[0]) * 0.1
                    y_plot_start = m * x_plot_start + c
                    y_plot_end = m * x_plot_end + c
                    return (x_plot_start, y_plot_start, x_plot_end, y_plot_end)
                else:  # Symbolic slope, cannot plot line
                    return None

            x_start, y_start = unique_intersections[0]
            x_end, y_end = unique_intersections[-1]
            return (x_start, y_start, x_end, y_end)

    # Plot Lines (infinite)----------------------------------------------------------------------------------------
    for name, line in scene.get_all_objects("Line").items():
        # Pass the current effective plot limits to the helper
        plot_coords = _get_plot_segment_for_line_like(
            line, current_xlim, current_ylim, is_ray=False
        )
        if plot_coords:
            x_start, y_start, x_end, y_end = plot_coords
            # Use highlight color if this line is in track_unsolved
            line_color = color_highlight_infinite_line if name in highlighted_objects else color_infinite_line
            # Use color and linestyle arguments for better control
            ax.plot([x_start, x_end], [y_start, y_end], color=line_color, linestyle="--", zorder=5)

    # Plot Rays---------------------------------------------------------------------------------------------------
    for name, ray in scene.get_all_objects("Ray").items():
        # Pass the current effective plot limits to the helper
        plot_coords = _get_plot_segment_for_line_like(
            ray, current_xlim, current_ylim, is_ray=True
        )
        if plot_coords:
            x_start, y_start, x_end, y_end = plot_coords
            # Use highlight color if this ray is in track_unsolved
            ray_color = color_highlight_ray if name in highlighted_objects else color_ray
            # Use color and linestyle arguments for better control
            ax.plot([x_start, x_end], [y_start, y_end], color=ray_color, linestyle="-.", zorder=5)
            dx_arr = x_end - x_start
            dy_arr = y_end - y_start
            length_arr = np.sqrt(dx_arr**2 + dy_arr**2)

            if length_arr > 0:
                x_span = current_xlim[1] - current_xlim[0]
                y_span = current_ylim[1] - current_ylim[0]
                avg_span = (x_span + y_span) / 2

                head_width = min(0.02 * avg_span, 0.5)
                head_length = min(0.03 * avg_span, 0.7)

                ax.arrow(
                    x_start,
                    y_start,
                    dx_arr,
                    dy_arr,
                    head_width=head_width,
                    head_length=head_length,
                    fc=ray_color,  # Fill color of the arrow head
                    ec=ray_color,  # Edge color of the arrow head
                    zorder=5,
                    length_includes_head=True,
                )

    # Add perpendicular and angle sign and write angle value ------------------------------------------------------
    if "Angle" in scene.objects_solved:
        for angle_name, angle in scene.objects_solved["Angle"].items():
            if angle.plot_sign or angle.plot_text:
                A = angle.A
                B = angle.vertex
                C = angle.B
                angle_value = angle.value
                # Get numeric coordinates
                bx, by = add_coords(B.x, B.y)
                ax_, ay_ = add_coords(A.x, A.y)
                cx, cy = add_coords(C.x, C.y)
                
                if None in (bx, by, ax_, ay_, cx, cy):
                    continue  # skip if any coordinate is symbolic
                
                # Use highlight color if this angle is in track_unsolved
                angle_color = color_highlight_arc if angle_name in highlighted_objects else color_arc
                
                # Vectors from vertex B
                vec_ba = np.array([ax_ - bx, ay_ - by])
                vec_bc = np.array([cx - bx, cy - by])
                
                # Normalize vectors
                vec_ba_norm = vec_ba / np.linalg.norm(vec_ba)
                vec_bc_norm = vec_bc / np.linalg.norm(vec_bc)
                
                # Angles relative to x-axis (degrees)
                angle_ba = np.degrees(np.arctan2(vec_ba_norm[1], vec_ba_norm[0]))
                angle_bc = np.degrees(np.arctan2(vec_bc_norm[1], vec_bc_norm[0]))
                
                # Determine arc start and end angles (CCW)
                start_angle = angle_ba
                end_angle = angle_bc
                angle_diff = (end_angle - start_angle) % 360
                
                # Draw smaller arc if angle_diff > 180
                if angle_diff > 180:
                    start_angle, end_angle = end_angle, start_angle
                    angle_diff = 360 - angle_diff
                
                radius = 0.04*(A.distance(B) + B.distance(C))  # adjust radius for scale
                
                if angle.plot_sign:
                    # Create the arc patch
                    arc = patches.Arc(
                        (bx, by),
                        width=2*radius,
                        height=2*radius,
                        angle=0,
                        theta1=start_angle,
                        theta2=end_angle,
                        color=angle_color,
                        linewidth=2,
                        zorder=8,
                    )
                    ax.add_patch(arc)
                
                # Position text label at arc midpoint
                mid_angle = (start_angle + angle_diff / 2) % 360
                mid_angle_rad = np.radians(mid_angle)
                text_x = bx + 1.5*radius * np.cos(mid_angle_rad)
                text_y = by + 1.5*radius * np.sin(mid_angle_rad)
                
                if angle.plot_text:
                    ax.text(
                        text_x,
                        text_y,
                        f"{angle_value:.1f}°",
                        color=angle_color,
                        fontsize=8,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        zorder=12,
                    )

    #ADD PERPENDICULAR SYMBOL ONLY IF RELEVANT LINES EXIST-------------------
    #warning points in rel are not solved yet
    perp_rel = [rel for rel in scene._relationships if rel["relationship"] == "perpendicular"]
    # Get names of all recorded lines from the three object types
    recorded_lines = []
    if "Line" in scene.objects_solved:
        recorded_lines.extend(scene.objects_solved["Line"].keys())
    if "LineSegment" in scene.objects_solved:
        recorded_lines.extend(scene.objects_solved["LineSegment"].keys())
    if "Ray" in scene.objects_solved:
        recorded_lines.extend(scene.objects_solved["Ray"].keys())

    # Filter perp_rel to only include relationships where both lines are in recorded_lines
    filtered_perp_rel = []
    for rel in perp_rel:
        line1_name = rel["args"]["line1"].name
        line2_name = rel["args"]["line2"].name
        
        if line1_name in recorded_lines and line2_name in recorded_lines:
            filtered_perp_rel.append(rel)

    # Extract the required information from filtered relationships
    perp_rel = [(rel["args"]["line1"].point1, 
                rel["args"]["line1"].point2, 
                rel["args"]["line2"].point1, 
                rel["args"]["line2"].point2,  
                rel["args"]["foot"], 
                rel["args"]["plot"]) for rel in filtered_perp_rel]
    #FINISH DEALING WITH PERPENDOCULAR SYMBOLS------------------------------------

    for A, B, C, D, foot, plot in perp_rel:
        if plot:
            if not foot:
                counter = Counter((A, B, C, D))
                common_point = [key for key, value in counter.items() if value == 2]
                if len(common_point) == 0:
                    continue
                common_point = common_point[0]
                first_point = B if common_point == A else A
                second_point = C if common_point == D else D
            else:
                first_point = B if foot == A else A
                second_point = C if foot == D else D
                common_point = foot

            #extracting correct points
            first_point = scene.objects_solved["Point"][first_point.name]
            second_point = scene.objects_solved["Point"][second_point.name]
            common_point = scene.objects_solved["Point"][common_point.name]

            # Draw the perpendicular sign on your matplotlib axes 'ax'
            draw_perpendicular_sign(ax, first_point, common_point, second_point, size=0.03*(first_point.distance(common_point) + common_point.distance(second_point)), color=color_arc)

    # Final plot styling-------------------------------------------------------------------------------------------
    ax.set_title("")  # Set title to empty string

    if not show_axis_grid:
        ax.grid(False)  # Disable grid
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(True, zorder=-10)

    return fig


def rotate_polygon_parameters(
    x_center_old: float,
    y_center_old: float,
    alpha_old_deg: float,  # Original angle with horizontal in degrees
    x_pivot: float,
    y_pivot: float,
    theta_deg: float,  # Rotation angle in degrees (positive for CCW)
) -> tuple[float, float, float]:
    """
    Calculates the new center and orientation angle of a regular polygon
    after rotating it around a specified pivot point.

    Args:
        x_center_old (float): The original X-coordinate of the polygon's center.
        y_center_old (float): The original Y-coordinate of the polygon's center.
        alpha_old_deg (float): The original angle (in degrees) of the polygon's
                               orientation relative to the horizontal axis.
        x_pivot (float): The X-coordinate of the rotation pivot point.
        y_pivot (float): The Y-coordinate of the rotation pivot point.
        theta_deg (float): The rotation angle in degrees. Positive values indicate
                           counter-clockwise rotation.

    Returns:
        tuple[float, float, float]: A tuple containing:
            - x_center_new (float): The new X-coordinate of the polygon's center.
            - y_center_new (float): The new Y-coordinate of the polygon's center.
            - alpha_new_deg (float): The new angle (in degrees) of the polygon's
                                     orientation relative to the horizontal axis.

    Note:
        The polygon's circumradius (and thus its size) does NOT change during a
        rotation, as rotation is a rigid body transformation that preserves distances.
    """

    # 1. Convert rotation angle to radians for trigonometric functions
    theta_rad = math.radians(theta_deg)

    # 2. Calculate New Center Coordinates
    # a. Translate the old center relative to the pivot (make pivot the temporary origin)
    center_rel_x = x_center_old - x_pivot
    center_rel_y = y_center_old - y_pivot

    # b. Rotate these relative coordinates
    rotated_rel_x = center_rel_x * math.cos(theta_rad) - center_rel_y * math.sin(
        theta_rad
    )
    rotated_rel_y = center_rel_x * math.sin(theta_rad) + center_rel_y * math.cos(
        theta_rad
    )

    # c. Translate back (add pivot's coordinates to get absolute position)
    x_center_new = rotated_rel_x + x_pivot
    y_center_new = rotated_rel_y + y_pivot

    # 3. Calculate New Angle with Horizontal Line
    # The polygon's orientation simply changes by the rotation angle
    alpha_new_deg = alpha_old_deg + theta_deg

    # Optional: Normalize the angle to be within [0, 360) degrees for consistency
    alpha_new_deg = alpha_new_deg % 360
    if alpha_new_deg < 0:
        alpha_new_deg += 360

    return x_center_new, y_center_new, alpha_new_deg

def crossfade_hard_threshold(im1, im2, num_inbetweens=2, threshold=50):
    im1 = np.array(im1).astype(np.float32)
    im2 = np.array(im2).astype(np.float32)
    frames = [Image.fromarray(im1.astype(np.uint8))]
    for k in range(1, num_inbetweens + 1):
        t = k / (num_inbetweens + 1)
        blended = (1 - t) * im1 + t * im2
        # After blending, threshold: any value below threshold (near black) → black
        blended = np.where(blended < threshold, 0, blended)
        frames.append(Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)))
    frames.append(Image.fromarray(im2.astype(np.uint8)))
    return frames