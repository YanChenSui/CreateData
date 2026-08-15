"""
定位并读取原始 Waymo Motion scenario proto。

关键点：
- tfrecord 分片内的下标不稳定（不同预处理脚本可能重新切分/打乱），
  必须以 scenario.scenario_id 为准建立索引，索引建好后落盘缓存。
- Track.id 是 proto 内的字段值，不等于 tracks 列表下标，
  get_track_by_id 必须显式按 id 查找。
"""

from __future__ import annotations
import glob
import os
import pickle
from typing import Dict, Tuple, List, Optional

import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2

import config


ScenarioIndex = Dict[str, Tuple[str, int]]   # scenario_id -> (file_path, record_offset_in_file)


def build_scenario_index(tfrecord_glob: str, cache_path: str = None) -> ScenarioIndex:
    """
    线性扫描一遍所有 tfrecord 分片，记录每个 scenario_id 出现在哪个文件的第几条 record。
    只需运行一次，结果落盘缓存（pickle），后续直接加载。
    """
    cache_path = cache_path or config.SCENARIO_INDEX_CACHE
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    index: ScenarioIndex = {}
    files = sorted(glob.glob(tfrecord_glob))
    if not files:
        raise FileNotFoundError(f"no tfrecord files matched: {tfrecord_glob}")

    for fp in files:
        dataset = tf.data.TFRecordDataset(fp, compression_type="")
        for offset, raw in enumerate(dataset):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(raw.numpy())
            index[scenario.scenario_id] = (fp, offset)

    with open(cache_path, "wb") as f:
        pickle.dump(index, f)
    return index


def _read_record_at(file_path: str, offset: int) -> scenario_pb2.Scenario:
    dataset = tf.data.TFRecordDataset(file_path, compression_type="")
    for i, raw in enumerate(dataset):
        if i == offset:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(raw.numpy())
            return scenario
    raise IndexError(f"offset {offset} out of range in {file_path}")


def load_scenario_by_id(
    scenario_id: str,
    index: ScenarioIndex,
) -> scenario_pb2.Scenario:
    """按 scenario_id 精确定位并读取（依赖预先构建好的 index）。"""
    if scenario_id not in index:
        raise KeyError(f"scenario_id not found in index: {scenario_id}")
    file_path, offset = index[scenario_id]
    return _read_record_at(file_path, offset)


def get_track_by_id(scenario: scenario_pb2.Scenario, track_id: int) -> "scenario_pb2.Track":
    """
    Track 是 repeated 字段，proto 列表下标 != Track.id，必须显式查找。
    """
    for track in scenario.tracks:
        if track.id == track_id:
            return track
    raise KeyError(f"track_id {track_id} not found in scenario {scenario.scenario_id}")


def get_map_features(scenario: scenario_pb2.Scenario) -> List["scenario_pb2.MapFeature"]:
    return list(scenario.map_features)


def track_valid_frame_range(track: "scenario_pb2.Track") -> Optional[Tuple[int, int]]:
    """返回该 track 的 valid=True 的最早/最晚帧索引，全程无效则返回 None。"""
    valid_idx = [i for i, s in enumerate(track.states) if s.valid]
    if not valid_idx:
        return None
    return min(valid_idx), max(valid_idx)
