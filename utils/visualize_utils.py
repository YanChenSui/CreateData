import os
import math
import pandas as pd
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import box
from shapely.geometry import LineString, Point

try:
    from trajdata import MapAPI
    from utils.trajdata_utils import DataFrameCache, get_agent_states
except ImportError:  # WOMD renderer does not require the trajdata stack.
    MapAPI = None
    DataFrameCache = None
    get_agent_states = None
# from extract_utils import *


# Rendering one frame used to reload the map, dataframe cache, and all agent
# states. Keep these immutable scene-window objects across frames and rows.
_DRAW_CONTEXT_CACHE = {}



def extend_line(line, distance):
    """
    Extends a line to a specified length if the original line is shorter.

    Args:
        line (LineString): A LineString object representing the trajectory or path.
        distance (float): The desired length to extend the line to.

    Returns:
        LineString: A new LineString object with the extended length.
    """
    # Extract the coordinates of the line and define the start and end points of the last segment
    coords = list(line.coords)
    start, end = Point(coords[-2]), Point(coords[-1])

    # Calculate the direction vector (dx, dy) of the last segment
    dx = end.x - start.x
    dy = end.y - start.y
    segment_length = math.sqrt(dx ** 2 + dy ** 2)

    # Calculate the scaling factor needed to extend the line to the desired distance
    factor = (distance - line.length) / segment_length
    new_x = end.x + dx * factor
    new_y = end.y + dy * factor

    # Add the new point to the coordinates list
    extended_coords = coords + [(new_x, new_y)]

    # Create and return a new LineString with the extended coordinates
    return LineString(extended_coords)

    
def extract_segment_by_distance(line, distance):
    """
    Extracts a segment from the starting point of the LineString up to a specified distance. 
    If the distance exceeds the length of the line, it extends the line to meet the required distance.

    Args:
        line (LineString): The input LineString representing the trajectory.
        distance (float): The desired distance from the start point to extract the segment.

    Returns:
        LineString: A new LineString object representing the extracted or extended segment. 
                    Returns None if the distance is less than or equal to zero.
    """
    if distance <= 0:
        return None  # Return None if the distance is zero or negative

    # Get the total length of the LineString
    total_length = line.length

    # If the distance exceeds the total length, extend the line
    if distance > total_length:
        return extend_line(line, distance)

    # Find the point at the specified distance along the line
    cut_point = line.interpolate(distance)

    # Extract coordinates up to the cut_point
    coords = list(line.coords)
    new_coords = []

    current_length = 0.0
    for i in range(1, len(coords)):
        segment = LineString([coords[i - 1], coords[i]])
        segment_length = segment.length
        if current_length + segment_length > distance:
            # Calculate the split point in the final segment
            remaining_distance = distance - current_length
            split_point = segment.interpolate(remaining_distance)
            new_coords.append((split_point.x, split_point.y))
            break
        new_coords.append(coords[i - 1])
        current_length += segment_length

    new_coords.append((cut_point.x, cut_point.y))

    # Create a new LineString object using the extracted coordinates
    segment = LineString(new_coords)

    return segment

def process_tracks_single(considertime, all_tracks, track_id_index, timestamp_index, timerange, position_index, v_index):
    """Build a future trajectory while filtering invalid XY rows along time."""
    v_x, v_y = all_tracks[track_id_index, timestamp_index, v_index]
    v = float(np.sqrt(v_x ** 2 + v_y ** 2))
    distance = v * considertime
    timerange = int(timerange)
    timestamp_index = int(timestamp_index)
    track = np.asarray(
        all_tracks[track_id_index, timestamp_index:timestamp_index + timerange, position_index],
        dtype=float,
    )
    if track.ndim != 2:
        return None
    if track.shape[1] != 2 and track.shape[0] == 2:
        track = track.T
    if track.shape[1] != 2:
        return None
    valid = np.all(np.isfinite(track), axis=1) & np.any(np.abs(track) > 1e-8, axis=1)
    filtered_track = track[valid]
    if filtered_track.shape[0] < 2:
        return None
    step_lengths = np.linalg.norm(np.diff(filtered_track, axis=0), axis=1)
    filtered_track = filtered_track[np.concatenate(([True], step_lengths > 1e-8))]
    if filtered_track.shape[0] < 2:
        return None
    line0 = LineString(filtered_track)
    line = extract_segment_by_distance(line0, distance) if distance > 0 else line0
    return {'line': line, 'velocity': v, 'line0': line0}

def is_line_in_view(line_points, view_rect):
    """
    Checks if a line segment intersects with the rectangular view window.

    Args:
        line_points (ndarray): An array of points representing the line segment, typically of shape (N, 2).
        view_rect (shapely.geometry.Polygon): A rectangular view window represented as a Shapely Polygon object.

    Returns:
        bool: True if the line segment intersects with the view rectangle, False otherwise.
    """
    # Waymo may contain degenerate lane boundaries with only one point.
    # They cannot form a Shapely LineString and should simply be ignored.
    line_points = np.asarray(line_points)
    if line_points.ndim != 2 or line_points.shape[0] < 2:
        return False

    line = LineString(line_points)

    # Check if the line intersects with the view rectangle
    return line.intersects(view_rect)



def rotate_around_center(pts, center, yaw):
    """
    Rotates a set of points around a given center by a specified angle.

    Args:
        pts (ndarray): An array of shape (N, 2) representing N points (x, y) to be rotated.
        center (ndarray): An array of shape (2,) representing the center (x, y) around which to rotate the points.
        yaw (float): The rotation angle in radians (positive for counter-clockwise rotation).

    Returns:
        ndarray: The rotated points as an array of shape (N, 2).
    """
    # Subtract the center point from all points to shift them to the origin
    shifted_pts = pts - center

    # Create the rotation matrix based on the yaw angle
    rotation_matrix = np.array([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)]
    ])

    # Apply the rotation matrix and shift the points back to the original center
    rotated_pts = np.dot(shifted_pts, rotation_matrix) + center

    return rotated_pts

def polygon_xy_from_motionstate(x, y, psi_rad, width=1.6, length=4.0):
    """
    Generates the coordinates of a rectangle (polygon) representing a vehicle based on its position, orientation, and dimensions.

    Args:
        x (float): The x-coordinate of the vehicle's center.
        y (float): The y-coordinate of the vehicle's center.
        psi_rad (float): The orientation angle of the vehicle in radians.
        width (float): The width of the vehicle. Default is 1.6 meters.
        length (float): The length of the vehicle. Default is 4.0 meters.

    Returns:
        ndarray: The coordinates of the vehicle's four corners after rotation based on its orientation.
    """
    # Define the four corners of the vehicle before rotation
    lowleft = (x - length / 2.0, y - width / 2.0)
    lowright = (x + length / 2.0, y - width / 2.0)
    upright = (x + length / 2.0, y + width / 2.0)
    upleft = (x - length / 2.0, y + width / 2.0)

    # Create a numpy array with the coordinates of the corners
    corners = np.array([lowleft, lowright, upright, upleft])

    # Rotate the polygon around the vehicle's center (x, y) by the specified angle (psi_rad)
    rotated_corners = rotate_around_center(corners, np.array([x, y]), yaw=psi_rad - math.pi / 180)

    return rotated_corners


def get_map_and_kdtrees(dataset, desired_scene):
    """
    Retrieves the vector map and KD-trees for the lanes based on the given dataset and scene.

    Args:
        dataset: The dataset object containing information about the scenes and cache paths.
        desired_scene: The scene object for which the map and KD-trees need to be retrieved.

    Returns:
        tuple: A tuple containing:
            - vec_map: The vector map for the given scene's environment and location.
            - lane_kd_tree: The KD-tree for lane information in the scene.
    """
    # Initialize the MapAPI with the cache path from the dataset
    map_api = MapAPI(dataset.cache_path)
    
    # Retrieve the environment name and set up the scene cache
    env_name = desired_scene.env_name
    scene_cache = DataFrameCache(cache_path=dataset.cache_path, scene=desired_scene)
    
    # Load KD-trees for the scene and extract the lane KD-tree
    scene_cache.load_kdtrees()
    lane_kd_tree = scene_cache.get_kdtrees(True)[1]
    
    # Get the vector map for the environment and location specified in the scene
    vec_map = map_api.get_map(f"{env_name}:{desired_scene.location}")
    
    return vec_map, lane_kd_tree



def setup_plot_bounds(interact_ids, mean_time, sc, column_dict, vec_map, all_agents, all_timesteps, dt, considertime):
    """
    Sets up the plot boundaries for visualizing the interaction of agents within the scene.

    Args:
        interact_ids (list): List of IDs for interacting agents.
        mean_time (int): The central timestamp used for determining the plot center.
        sc (DataFrameCache): Cache object for accessing scene data.
        column_dict (dict): Dictionary mapping column names to their indices in raw state data.
        vec_map (VectorMap): The vector map of the environment.
        all_agents (list): List of all agents in the scene.
        all_timesteps (list): List of all timesteps in the scene.
        dt (float): Timestep duration.
        considertime (float): The duration to be considered for plotting.

    Returns:
        tuple: A tuple containing the minimum and maximum bounds for x and y coordinates.
    """
    try:
        # Attempt to get the center coordinates (x, y) based on the first interacting agent at the mean time
        center_x = sc.get_raw_state(agent_id=interact_ids[0], scene_ts=mean_time)[column_dict['x']]
        center_y = sc.get_raw_state(agent_id=interact_ids[0], scene_ts=mean_time)[column_dict['y']]
    except:
        # If the first agent fails, use the last agent in the list
        center_x = sc.get_raw_state(agent_id=interact_ids[-1], scene_ts=mean_time)[column_dict['x']]
        center_y = sc.get_raw_state(agent_id=interact_ids[-1], scene_ts=mean_time)[column_dict['y']]

    # Define the radius for the view area
    r = 100
    x_min, x_max = center_x - r, center_x + r
    y_min, y_max = center_y - r, center_y + r

    view_rect = box(x_min, y_min, x_max, y_max)
    lane_ids = vec_map.lanes

    return x_min, x_max, y_min, y_max


def plot_lanes(plt, vec_map, view_rect):
    """
    Plots the lanes within the given view rectangle on the map.

    Args:
        plt (matplotlib.pyplot): The matplotlib plotting object.
        vec_map (VectorMap): The vector map containing lane information.
        view_rect (shapely.geometry.box): The view rectangle defining the area to display lanes.

    Returns:
        matplotlib.pyplot: The updated plotting object with lanes plotted.
    """
    for laneid in vec_map.lanes:
        # Access the left and right edges of the lane
        left_lane = laneid.left_edge
        right_lane = laneid.right_edge

        # Plot the left lane edge if it exists and is within the view rectangle
        if left_lane is not None and is_line_in_view(left_lane.points, view_rect):
            plt.plot(left_lane.points[:, 0], left_lane.points[:, 1], '-', markersize=1, color='#969696', linewidth=0.4)

        # Plot the right lane edge if it exists and is within the view rectangle
        if right_lane is not None and is_line_in_view(right_lane.points, view_rect):
            plt.plot(right_lane.points[:, 0], right_lane.points[:, 1], '-', markersize=1, color='#969696', linewidth=0.4)

    return plt


def _get_vehicle_dimensions(agent_metadata, scene_ts):
    """Return the current vehicle length/width for dynamic labels and polygons."""
    default_length, default_width = 4.0, 1.6
    if agent_metadata is None:
        return default_length, default_width
    try:
        extent = np.asarray(
            agent_metadata.extent.get_extents(scene_ts, scene_ts)[0],
            dtype=float,
        )
        if extent.size >= 2:
            length = float(extent[0]) if np.isfinite(extent[0]) and extent[0] > 0 else default_length
            width = float(extent[1]) if np.isfinite(extent[1]) and extent[1] > 0 else default_width
            return length, width
    except (AttributeError, IndexError, TypeError, ValueError, NotImplementedError):
        pass
    return default_length, default_width


def _labels_overlap(candidate, occupied):
    """Use simple world-coordinate boxes to avoid overlapping dynamic labels."""
    x, y, half_width, half_height = candidate
    for other_x, other_y, other_half_width, other_half_height in occupied:
        if (
            abs(x - other_x) < half_width + other_half_width
            and abs(y - other_y) < half_height + other_half_height
        ):
            return True
    return False


def _draw_key_agent_body_marker(marker, current_x, current_y, vehicle_length, vehicle_width):
    """Draw the stable key-agent order number at the current vehicle center."""
    ax = plt.gca()
    # Estimate a fontsize in points based on the smaller vehicle dimension.
    # This keeps the label visually contained within the vehicle bounds.
    min_dim = min(vehicle_length, vehicle_width)
    fontsize = max(6, min(12, int(min_dim * 4)))
    plt.text(
        current_x,
        current_y,
        marker,
        color="black",
        fontsize=fontsize,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=31,
        clip_on=True,
        transform=ax.transData,
    )


def _format_current_speed(agent_id, all_agents, agent_states, timestamp_index, column_dict):
    """Format the current speed of an agent for the fixed legend."""
    try:
        agent_index = all_agents.index(agent_id)
        vx = float(agent_states[agent_index, timestamp_index, column_dict['vx']])
        vy = float(agent_states[agent_index, timestamp_index, column_dict['vy']])
        speed = math.hypot(vx, vy)
        if np.isfinite(speed):
            return f"{speed:.1f} m/s"
    except (KeyError, TypeError, ValueError, IndexError):
        pass
    return "speed unavailable"


def plot_agent_trajectory(agent_id, participating_ids, key_agents, ego_id, all_agents, agent_metadata_by_id,
                          agent_states, timestamp_index, scene_ts, all_timesteps, dt,
                          vec_map, column_dict, lane_id, lane_kd_tree, sc, agent_lane_ids,
                          considertime, context_only=False):
    """Plot one agent using participation/key-agent color rules."""
    id_index = all_agents.index(agent_id)
    timerange = int(considertime / dt)
    position_index = [column_dict['x'], column_dict['y']]
    v_index = [column_dict['vx'], column_dict['vy']]

    processed_tracks = process_tracks_single(
        considertime, agent_states, id_index, timestamp_index,
        timerange, position_index, v_index
    )
    if not processed_tracks:
        # A stationary vehicle, or a vehicle with only one usable point in
        # the local window, has no LineString to draw.  It must still remain
        # visible as a vehicle marker; otherwise target vehicles can appear
        # to lose their highlight in individual GIF frames.
        current_x = agent_states[id_index, timestamp_index, column_dict['x']]
        current_y = agent_states[id_index, timestamp_index, column_dict['y']]
        psi_rad = agent_states[id_index, timestamp_index, column_dict['heading']]
        if not (np.isfinite(current_x) and np.isfinite(current_y)):
            return None
        if agent_id in key_agents:
            vehicle_length, vehicle_width = _get_vehicle_dimensions(
                agent_metadata_by_id.get(agent_id), scene_ts
            )
            color = '#F28E2B' if agent_id == ego_id else '#2CA02C'
            plot_interacting_agent(
                None, None, current_x, current_y, psi_rad,
                color=color, vehicle_length=vehicle_length,
                vehicle_width=vehicle_width,
            )
        else:
            plot_non_interacting_agent(
                None, None, current_x, current_y, psi_rad
            )
        return {'line': None, 'velocity': 0.0}

    line, v = processed_tracks['line'], processed_tracks['velocity']
    line_x, line_y = line.xy
    current_x = agent_states[id_index, timestamp_index, column_dict['x']]
    current_y = agent_states[id_index, timestamp_index, column_dict['y']]
    psi_rad = agent_states[id_index, timestamp_index, column_dict['heading']]

    if agent_id in key_agents:
        color = '#F28E2B' if agent_id == ego_id else '#2CA02C'
        vehicle_length, vehicle_width = _get_vehicle_dimensions(
            agent_metadata_by_id.get(agent_id), scene_ts
        )
        plot_interacting_agent(
            line_x, line_y, current_x, current_y, psi_rad,
            color=color, vehicle_length=vehicle_length, vehicle_width=vehicle_width,
        )
        if len(key_agents) == 2 and ego_id is None:
            _draw_key_agent_body_marker(
                str(key_agents.index(agent_id) + 1),
                current_x,
                current_y,
                vehicle_length,
                vehicle_width,
            )
    elif context_only and agent_id in participating_ids:
        plot_non_interacting_agent(line_x, line_y, current_x, current_y, psi_rad)
    elif agent_id in participating_ids:
        plot_interacting_agent(line_x, line_y, current_x, current_y, psi_rad, color='#3868A6')
    else:
        plot_non_interacting_agent(line_x, line_y, current_x, current_y, psi_rad)


def plot_interacting_agent(line_x, line_y, x, y, angle, color='#3868A6',
                         vehicle_length=4.0, vehicle_width=1.6):
    """Plot the trajectory in its original time order and draw the current vehicle."""
    if line_x is not None and line_y is not None and len(line_x) >= 2:
        plt.plot(line_x, line_y, '-', color=color, zorder=10, linewidth=1)
    rect = patches.Polygon(
        polygon_xy_from_motionstate(
            x, y, angle, width=vehicle_width, length=vehicle_length
        ),
        closed=True,
        zorder=20,
        color=color,
    )
    plt.gca().add_patch(rect)

def plot_non_interacting_agent(line_x, line_y, x, y, angle):
    """Plot a non-interacting vehicle with its trajectory in time order."""
    if line_x is not None and line_y is not None and len(line_x) >= 2:
        plt.plot(line_x, line_y, '--', color='#E9E9E9', zorder=5, linewidth=0.8)
    rect = patches.Polygon(polygon_xy_from_motionstate(x, y, angle), closed=True, zorder=6, color='#C9CACA')
    plt.gca().add_patch(rect)

def draw_pic(desired_scene, all_agents, all_timesteps, dataset, id_rawid, raw_scene_id,
             path_relation, participating_ids, key_agents, timestamp, start, end, dt, considertime,
             save_path="./", ego_id=None, dpi=600, render_agent_ids=None,
             context_only=False, output_stem=None, title_prefix=None):
    """
    Draws a scene showing the interaction trajectories of selected agents within a given scene.

    Args:
        desired_scene: The scene object containing information about the scenario.
        all_agents (list): List of all agent IDs present in the scene.
        all_timesteps (range): Range of timesteps within the scene.
        dataset: The dataset object containing information about the scenes.
        id_rawid (dict): Dictionary mapping scene IDs to raw data indices.
        raw_scene_id (int): Raw scene ID to retrieve the desired scene.
        participating_ids (list): IDs from track_id that participate in the interaction.
        key_agents (list): Final selected key-agent IDs.
        start (int): Start timestep for the interaction.
        end (int): End timestep for the interaction.
        dt (float): Timestep duration.
        considertime (int): Time range to consider around the interaction.
        save_path (str): Path to save the generated image.

    Returns:
        None
    """
    timestamp_index = all_timesteps.index(timestamp)
    cache_key = (
        id(dataset),
        getattr(desired_scene, "raw_data_idx", desired_scene.name),
        tuple(all_timesteps),
    )
    context = _DRAW_CONTEXT_CACHE.get(cache_key)
    if context is None:
        # Retrieve the vector map and KD-trees for the lanes once per
        # scene/window instead of once per rendered frame.
        vec_map, lane_kd_tree = get_map_and_kdtrees(dataset, desired_scene)
        lane_id = [lane.id for lane in vec_map.lanes]
        scene_cache = DataFrameCache(cache_path=dataset.cache_path, scene=desired_scene)
        column_dict = scene_cache.column_dict
        agent_states, agent_lane_ids = get_agent_states(
            participating_ids,
            all_agents,
            vec_map,
            lane_kd_tree,
            scene_cache,
            desired_scene,
            column_dict,
            all_timesteps,
        )
        context = {
            "vec_map": vec_map,
            "lane_kd_tree": lane_kd_tree,
            "lane_id": lane_id,
            "scene_cache": scene_cache,
            "column_dict": column_dict,
            "agent_states": agent_states,
            "agent_lane_ids": agent_lane_ids,
        }
        _DRAW_CONTEXT_CACHE[cache_key] = context
    vec_map = context["vec_map"]
    lane_kd_tree = context["lane_kd_tree"]
    lane_id = context["lane_id"]
    scene_cache = context["scene_cache"]
    column_dict = context["column_dict"]
    agent_states = context["agent_states"]
    agent_lane_ids = context["agent_lane_ids"]

    # Set up plot bounds based on the mean time and interaction details
    mean_time = int((start + end) / 2)
    x_min, x_max, y_min, y_max = setup_plot_bounds(
        participating_ids, mean_time, scene_cache, column_dict, vec_map, all_agents,
        all_timesteps, dt, considertime
    )

    # Enable interactive plotting
    plt.ion()

    # Plot lanes within the view rectangle
    plot_lanes(plt, vec_map, box(x_min, y_min, x_max, y_max))

    # Get current agents present in the scene at the given timestamp
    current_ids = [
        all_agents[index] 
        for index, state in enumerate(agent_states[:, timestamp_index, :])
        if (
            state[column_dict['vx']] != 0
            or state[column_dict['vy']] != 0
            or state[column_dict['x']] != 0
            or state[column_dict['y']] != 0
        )
    ]
    if render_agent_ids is not None:
        allowed_ids = {str(agent_id) for agent_id in render_agent_ids}
        current_ids = [agent_id for agent_id in current_ids if str(agent_id) in allowed_ids]

    # Plot all current agents. Participation and key-agent status are separate.
    agent_metadata_by_id = {agent.name: agent for agent in desired_scene.agents}
    for agent_id in current_ids:
        plot_agent_trajectory(
            agent_id, participating_ids, key_agents, ego_id, all_agents, agent_metadata_by_id,
            agent_states, timestamp_index, timestamp, all_timesteps, dt,
            vec_map, column_dict, lane_id, lane_kd_tree, scene_cache,
            agent_lane_ids, considertime, context_only=context_only,
        )

    # Set up the plot appearance
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    title = title_prefix or f'Track {participating_ids} at Timestamp {timestamp}_{path_relation}'
    plt.title(title)
    plt.xlim(x_min, x_max)  # Set X axis range
    plt.ylim(y_min, y_max)  # Set Y axis range

    plt.gca().set_facecolor('xkcd:white')
    plt.gca().margins(0)
    plt.gca().set_aspect('equal')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)

    # Keep the legend anchored to the axes, so it remains fixed in the upper-left
    # corner while the world-coordinate view changes from frame to frame.  The
    # legend and body marker use the same stable 1/2 order.
    legend_handles = []
    for index, key_agent in enumerate(key_agents, start=1):
        key_color = '#F28E2B' if key_agent == ego_id else '#2CA02C'
        speed_text = _format_current_speed(
            key_agent, all_agents, agent_states, timestamp_index, column_dict
        )
        legend_handles.append(
            Line2D(
                [0], [0], marker='s', linestyle='None',
                markerfacecolor=key_color, markeredgecolor=key_color,
                markersize=7,
                label=f'Vehicle {index}: {key_agent} | {speed_text}',
            )
        )
    if legend_handles:
        plt.legend(
            handles=legend_handles,
            loc='upper left',
            bbox_to_anchor=(0.02, 0.98),
            bbox_transform=plt.gca().transAxes,
            frameon=True,
            framealpha=0.9,
            fontsize=8,
            borderpad=0.6,
            handletextpad=0.4,
        )
    plt.tight_layout()

    # Create directory if it doesn't exist and save the plot
    os.makedirs(save_path, exist_ok=True)
    interact_ids_str = output_stem or '_'.join(key_agents) or 'scene'
    plt.savefig(f'{save_path}/{interact_ids_str}_{timestamp}.png', dpi=dpi)

    #  Clear the plot
    plt.clf()


def plot_womd_lanes(lane_polylines, view_rect=None):
    """Render WOMD lane-center polylines using the existing plot style."""
    for polyline in lane_polylines or []:
        points = np.asarray(polyline, dtype=float)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
            continue
        if view_rect is not None:
            try:
                if not is_line_in_view(points[:, :2], view_rect):
                    continue
            except (TypeError, ValueError):
                pass
        plt.plot(
            points[:, 0], points[:, 1], "-", color="#969696",
            linewidth=0.4, zorder=1,
        )


def draw_womd_pic(
    all_agents,
    agent_states,
    timestamp_index,
    all_timesteps,
    dt,
    lane_polylines,
    target_vehicle_id,
    behavior_segments=None,
    supporting_frame_ranges=None,
    save_path="./",
    dpi=160,
    considertime=5.0,
    output_stem=None,
    title_prefix=None,
):
    """Render a WOMD scene through the existing InterHub trajectory renderer.

    This is an input adapter, not an interaction extractor.  ``all_agents``
    and ``agent_states`` are built directly from WOMD tracks; every non-target
    vehicle is neutral context and only ``target_vehicle_id`` is highlighted.
    Behavior/evidence ranges are visual annotations over the current frame,
    never new trajectories or detector outputs.
    """
    os.makedirs(save_path, exist_ok=True)
    timestamp_index = int(timestamp_index)
    all_timesteps = list(all_timesteps)
    if timestamp_index < 0 or timestamp_index >= len(all_timesteps):
        raise IndexError("timestamp_index is outside all_timesteps")

    x_col, y_col = 0, 1
    valid_points = []
    for agent_index in range(agent_states.shape[0]):
        states = agent_states[agent_index]
        valid = np.isfinite(states[:, x_col]) & np.isfinite(states[:, y_col])
        if np.any(valid):
            valid_points.append(states[valid][:, [x_col, y_col]])
    if not valid_points:
        raise ValueError("WOMD scene contains no finite vehicle positions")
    points = np.concatenate(valid_points, axis=0)
    x_low, y_low = np.min(points, axis=0)
    x_high, y_high = np.max(points, axis=0)
    margin = max(20.0, 0.08 * max(x_high - x_low, y_high - y_low))
    x_min, x_max = float(x_low - margin), float(x_high + margin)
    y_min, y_max = float(y_low - margin), float(y_high + margin)

    figure, axis = plt.subplots(figsize=(10, 8), dpi=dpi)
    plot_womd_lanes(lane_polylines)

    target_id = str(target_vehicle_id)
    current_ids = []
    for index, agent_id in enumerate(all_agents):
        state = agent_states[index, timestamp_index]
        if np.isfinite(state[x_col]) and np.isfinite(state[y_col]):
            current_ids.append(str(agent_id))

    column_dict = {
        "x": x_col,
        "y": y_col,
        "vx": 2,
        "vy": 3,
        "heading": 4,
    }
    agent_metadata_by_id = {}
    for agent_id in current_ids:
        plot_agent_trajectory(
            agent_id,
            [],
            [target_id],
            None,
            [str(value) for value in all_agents],
            agent_metadata_by_id,
            agent_states,
            timestamp_index,
            all_timesteps[timestamp_index],
            all_timesteps,
            float(dt),
            None,
            column_dict,
            [],
            None,
            None,
            None,
            max(float(considertime), float(dt)),
            context_only=True,
        )

    def _ranges(value):
        return [item for item in (value or []) if isinstance(item, dict)]

    active_segments = [
        item for item in _ranges(behavior_segments)
        if int(item.get("start_frame", -1)) <= timestamp_index
        <= int(item.get("end_frame", -2))
    ]
    active_evidence = [
        item for item in _ranges(supporting_frame_ranges)
        if int(item.get("start_frame", -1)) <= timestamp_index
        <= int(item.get("end_frame", -2))
    ]

    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal")
    axis.set_axis_off()
    if active_segments:
        axis.set_facecolor("#eef6ff")
        segment_color = "#2563eb"
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(segment_color)
            spine.set_linewidth(3.0)
    else:
        axis.set_facecolor("white")
        for spine in axis.spines.values():
            spine.set_visible(False)

    if active_evidence:
        axis.add_patch(patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            fill=False, edgecolor="#f59e0b", linewidth=2.0,
            linestyle="--", zorder=40,
        ))

    segment_text = " | ".join(
        str(item.get("description", "")).strip()
        for item in active_segments
        if str(item.get("description", "")).strip()
    )
    evidence_text = (
        "evidence " + ", ".join(
            f"{item.get('start_frame')}-{item.get('end_frame')}"
            for item in active_evidence
        )
        if active_evidence else ""
    )
    title = title_prefix or f"WOMD Vehicle {target_id} | Frame {all_timesteps[timestamp_index]}"
    if segment_text:
        title += f"\nBehavior segment: {segment_text}"
    if evidence_text:
        title += f"\n{evidence_text}"
    axis.set_title(title, fontsize=11)
    legend_handles = [
        Line2D([0], [0], color="#2CA02C", linewidth=2.4, label=f"Target vehicle {target_id}"),
        Line2D([0], [0], color="#E9E9E9", linewidth=1.2, linestyle="--", label="Other vehicles"),
    ]
    if active_segments:
        legend_handles.append(
            Line2D([0], [0], color="#2563eb", linewidth=3.0, label="Behavior segment window")
        )
    if active_evidence:
        legend_handles.append(
            Line2D([0], [0], color="#f59e0b", linewidth=2.0, linestyle="--", label="Supporting evidence range")
        )
    axis.legend(handles=legend_handles, loc="upper left", framealpha=0.9, fontsize=8)
    figure.tight_layout(pad=0.5)
    output_path = os.path.join(
        save_path, f"{output_stem or 'womd_scene'}_{all_timesteps[timestamp_index]}.png"
    )
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path
