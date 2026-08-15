import os
import argparse
import pandas as pd
import imageio
from trajdata import UnifiedDataset
from utils.visualize_utils import draw_pic

# define parameters
parser = argparse.ArgumentParser(description="Visualize extracted InterHub interactions.")
parser.add_argument("--cache_location", required=True, help="trajdata cache root or its waymo_train subdirectory")
parser.add_argument("--interaction_idx_info", required=True, help="extracted interaction CSV")
parser.add_argument(
    "--selection-mode",
    choices=("row", "top"),
    default="row",
    help="row: CSV order and row-based filenames; top: intensity-sorted Top-N filenames",
)
parser.add_argument("--top_n", type=int, default=10, help="number of top-intensity cases to visualize")
parser.add_argument(
    "--interaction-row",
    type=int,
    help="zero-based CSV row to visualize instead of selecting top-intensity cases",
)
parser.add_argument(
    "--all-rows",
    action="store_true",
    help="visualize every CSV row; cannot be combined with --interaction-row",
)
parser.add_argument("--start-row", type=int, help="inclusive CSV row for resumable batches")
parser.add_argument("--end-row", type=int, help="exclusive CSV row for resumable batches")
parser.add_argument(
    "--rank-offset",
    type=int,
    default=0,
    help="add to Top rank in filenames; used only with --selection-mode top",
)
parser.add_argument("--gif_path", default="figs/case", help="directory for generated frames/GIFs")
parser.add_argument(
    "--gif-loop",
    type=int,
    default=0,
    help="GIF loop count; 0 means infinite looping",
)
parser.add_argument("--gif-fps", type=float, default=10.0)
parser.add_argument("--dpi", type=int, default=600)
args = parser.parse_args()

starting_extension_time = 3.0  # starting extension time in seconds
ending_extension_time = 3.0   # ending extension time in seconds
considertime = 5  # future time of trajectories in seconds
gif_path = args.gif_path
cache_location = os.path.normpath(args.cache_location)
interaction_idx_info = args.interaction_idx_info
top_n = args.top_n


# read the csv file containing interaction information
extract_df = pd.read_csv(interaction_idx_info)


def parse_agent_ids(value):
    """Parse the semicolon-separated agent IDs used by results.csv."""
    if pd.isna(value):
        return []
    text = str(value).strip().strip("[]")
    return [agent_id.strip().strip("'\"") for agent_id in text.split(';') if agent_id.strip()]

# Select rows. The default is CSV order; top mode retains the original Top-N behavior.
if args.interaction_row is not None:
    if args.all_rows or args.start_row is not None or args.end_row is not None:
        parser.error("--interaction-row cannot be combined with row-range options")
    if args.interaction_row < 0 or args.interaction_row >= len(extract_df):
        parser.error(
            f"--interaction-row must be in 0..{len(extract_df) - 1}"
        )
    selected_rows = extract_df.iloc[[args.interaction_row]]
elif args.selection_mode == "row":
    selected_rows = extract_df
else:
    selected_rows = extract_df.nlargest(top_n, 'intensity')

if args.start_row is not None or args.end_row is not None:
    start_row = 0 if args.start_row is None else args.start_row
    end_row = len(extract_df) if args.end_row is None else args.end_row
    if start_row < 0 or end_row < start_row or end_row > len(extract_df):
        parser.error(
            f"row slice must satisfy 0 <= start <= end <= {len(extract_df)}"
        )
    if args.selection_mode == "row":
        selected_rows = extract_df.iloc[start_row:end_row]
    else:
        selected_rows = selected_rows.iloc[start_row:end_row]

# Keep one UnifiedDataset/index per dataset while rendering many rows.
dataset_cache = {}

# iterate through the selected rows
for rank, (idx, row) in enumerate(
    selected_rows.iterrows(), start=1 + args.rank_offset
):
    # extract information from the selected row
    desired_data = row['dataset']
    raw_scene_id = int(row['scenario_idx'])
    start = int(row['start'])
    end = int(row['end'])
    participating_ids = parse_agent_ids(row['track_id'])
    key_agents = parse_agent_ids(row.get('key_agents', row['track_id']))
    if not key_agents:
        key_agents = participating_ids.copy()
    ego_id = 'ego' if 'ego' in key_agents else None
    intensity = row['intensity']
    path_relation = row['path_relation']

    # Keep output names stable across resumable batches and selection modes.
    if args.selection_mode == "row":
        output_stem = f"row_{int(idx):04d}"
        progress_label = output_stem
    else:
        output_stem = f"Top_{rank:04d}"
        progress_label = output_stem

    # Print the CSV row/rank that is actually being rendered.
    print(f"Processing {progress_label}: {raw_scene_id}, {start}, {end}, participants={participating_ids}, key_agents={key_agents}")

    if desired_data not in dataset_cache:
        # Accept either the cache root or the dataset-specific .../waymo_train path.
        dataset_cache_location = cache_location
        if os.path.basename(cache_location) == str(desired_data):
            dataset_cache_location = os.path.dirname(cache_location)
        dataset = UnifiedDataset(
            desired_data=[desired_data],
            standardize_data=False,
            rebuild_cache=False,  # do not rebuild cache
            rebuild_maps=False,   # do not rebuild maps
            centric="scene",
            verbose=True,
            cache_location=dataset_cache_location,
            num_workers=os.cpu_count(),
            incl_vector_map=True,
            data_dirs={desired_data: ''}
        )
        id_rawid = {
            desired_scene.raw_data_idx: idx
            for idx, desired_scene in enumerate(dataset.scenes())
        }
        dataset_cache[desired_data] = (dataset, id_rawid)
    dataset, id_rawid = dataset_cache[desired_data]

    # retrieve the desired scene based on the raw scene ID
    desired_scene = dataset.get_scene(id_rawid[raw_scene_id])

    # extract the time step and agent information
    dt = desired_scene.dt
    agents = {agent.name: agent for agent in desired_scene.agents}
    all_agents = list(agents.keys())
    first, last = 99999, 0

    # determine the first and last time steps for the interacting agents
    for agent in participating_ids:
        first = min(first, agents[agent].first_timestep)
        last = max(last, agents[agent].last_timestep)

    print(row, first, last)

    # Generate the complete scene timeline.  The interaction start/end frames
    # are still passed to draw_pic for highlighting, but no longer crop the GIF.
    pic_list = []
    plot_start = 0
    plot_end = desired_scene.length_timesteps - 1
    all_timesteps = range(plot_start, plot_end + 1)

    # loop through each timestamp to generate images
    for timestamp in all_timesteps:
        draw_pic(
            desired_scene, all_agents, all_timesteps, dataset, id_rawid, raw_scene_id,path_relation,
            participating_ids, key_agents, timestamp, start, end, dt, considertime, gif_path,
            ego_id=ego_id,
            dpi=args.dpi,
        )
        
        interact_ids_str = '_'.join(key_agents)
        pic_list.append(f'{gif_path}/{interact_ids_str}_{timestamp}.png')

    # create gif from the generated images
    gif_file = f"{gif_path}/gif/{output_stem}_{raw_scene_id}_{start}-{end}_{path_relation}.gif"
    os.makedirs(os.path.dirname(gif_file), exist_ok=True)
    images = [imageio.imread(image_file) for image_file in pic_list]
    imageio.mimsave(
        gif_file,
        images,
        duration=1000.0 / args.gif_fps,
        loop=args.gif_loop,
    )

    # optionally remove individual image files after creating the gif
    for image_file in pic_list:
        os.remove(image_file)
