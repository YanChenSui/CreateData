"""Compatibility import for the archived normalized-record loader.

The canonical facts builder contains the raw Scenario protobuf loader and map
matching needed by the active workflow.  This module is retained only for
legacy callers that still consume ``scene_motion_v3`` records.
"""

from .waymo_scenario_loader_scene_motion_v1 import *  # noqa: F401,F403
