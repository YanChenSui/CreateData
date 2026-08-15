"""Compatibility import for the archived scene-motion timeline.

The active pair pipeline is :mod:`scripts.facts.build_pair_timeline`, which reads a
raw Waymo Scenario and emits physical facts only.  This module remains only so
old unit tests and historical notebooks can still import the previous
``scene_motion_v3`` implementation; it is not part of the active pipeline.
"""

from ..legacy.pair_behavior_timeline_v1 import (  # noqa: F401
    TimelineConfig,
    build_pair_behavior_timeline,
)

__all__ = ["TimelineConfig", "build_pair_behavior_timeline"]
