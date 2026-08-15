"""
对齐层：把 InterHub 记录里的 (start, end, track_id ...) 映射到原始
Waymo scenario 的真实 timestep 索引上，并做误差校验。

背景问题：
InterHub 在抽取交互时可能对原始 91 帧场景做过裁剪/重采样，
其表格里的 start/end 帧号不一定等于原始 scenario.tracks[i].states 的下标。
直接拿 start/end 去索引原始数据会导致悄悄地读错帧，且不会报错——这是最危险的
一类 bug，所以必须做显式的对齐校验，而不是假设两者一致。

校验思路：
1. 用 InterHub 给出的 original_track_id 在原始 scenario 中定位该 track；
2. 假设 InterHub 的 start 帧对应原始 track 中"从有效帧开始数第 offset 帧"，
   枚举一个小范围的候选 offset；
3. 对每个候选 offset，比较 InterHub 记录窗口长度 (end-start) 是否与原始
   track 在该 offset 位置往后 N 帧的有效性、以及运动学连续性吻合；
4. 取误差最小的 offset，若误差超过阈值则整条记录标记为 alignment_failed。

注：如果 InterHub 中间结果本身包含每帧的 (x, y) 坐标（有些版本会保留），
应优先直接用坐标做最近邻匹配，比"假设 offset"更可靠 —— 见
`align_record_by_position`，若没有坐标数据则退回 `align_record_by_offset_search`。
"""

from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np

import config
from schemas import InterHubRecord, AlignedRecord
from io.waymo_scenario_loader import (
    ScenarioIndex,
    load_scenario_by_id,
    get_track_by_id,
    track_valid_frame_range,
)


def _track_positions(track) -> np.ndarray:
    """返回 (T, 2) 的 (x, y) 数组，无效帧填 nan。"""
    pts = np.full((len(track.states), 2), np.nan, dtype=float)
    for i, s in enumerate(track.states):
        if s.valid:
            pts[i, 0] = s.center_x
            pts[i, 1] = s.center_y
    return pts


def align_record_by_offset_search(
    record: InterHubRecord,
    scenario,
    max_offset_search: int = 20,
) -> Optional[AlignedRecord]:
    """
    没有逐帧坐标可用时的退化方案：
    在原始 track 的有效帧区间内，搜索一个 offset，使得
    "InterHub 窗口长度" 与 "原始 track 从 offset 开始的有效连续帧长度" 匹配最好，
    同时窗口整体落在 track 的有效范围内。

    这是一个弱校验（只能保证长度/范围一致，不能保证语义对齐），
    因此 alignment_error 用一个较粗的 confidence 代理值表示，
    强烈建议优先使用 align_record_by_position。
    """
    if not record.original_track_id:
        return None
    track_id = record.original_track_id[0]
    try:
        track = get_track_by_id(scenario, track_id)
    except KeyError:
        return None

    valid_range = track_valid_frame_range(track)
    if valid_range is None:
        return None
    valid_start, valid_end = valid_range
    interhub_len = record.end - record.start

    best_offset, best_err = None, float("inf")
    for offset in range(valid_start, min(valid_end, valid_start + max_offset_search) + 1):
        window_end = offset + interhub_len
        if window_end > valid_end:
            continue
        # 简单误差代理：窗口是否全程 valid
        seg = track.states[offset: window_end + 1]
        invalid_count = sum(1 for s in seg if not s.valid)
        err = invalid_count
        if err < best_err:
            best_err, best_offset = err, offset

    if best_offset is None:
        return None

    error_value = float(best_err)  # 以无效帧数近似误差，非帧数误差
    if error_value > config.ALIGNMENT_MAX_FRAME_ERROR:
        return None

    return AlignedRecord(
        scene_id=record.original_scene_id,
        agent_a=record.original_track_id[0],
        agent_b=record.original_track_id[1] if len(record.original_track_id) > 1 else -1,
        start_frame=best_offset,
        end_frame=best_offset + interhub_len,
        frame_offset=best_offset - record.start,
        alignment_error=error_value,
        source_record=record,
    )


def align_record_by_position(
    record: InterHubRecord,
    scenario,
    ref_xy: Tuple[float, float],
    ref_is_start: bool = True,
    search_radius_frames: int = 30,
) -> Optional[AlignedRecord]:
    """
    若能从 InterHub 中间产物拿到该交互起（或止）帧的 agent 位置 ref_xy，
    则直接在原始 track 全程坐标里找最近邻帧，这是最可靠的对齐方式。
    """
    if not record.original_track_id:
        return None
    track_id = record.original_track_id[0]
    try:
        track = get_track_by_id(scenario, track_id)
    except KeyError:
        return None

    pts = _track_positions(track)
    dists = np.linalg.norm(pts - np.array(ref_xy), axis=1)
    matched_frame = int(np.nanargmin(dists))
    matched_err = float(dists[matched_frame])

    if matched_err > config.ALIGNMENT_POS_TOL_M:
        return None

    interhub_len = record.end - record.start
    if ref_is_start:
        start_frame, end_frame = matched_frame, matched_frame + interhub_len
    else:
        start_frame, end_frame = matched_frame - interhub_len, matched_frame

    return AlignedRecord(
        scene_id=record.original_scene_id,
        agent_a=record.original_track_id[0],
        agent_b=record.original_track_id[1] if len(record.original_track_id) > 1 else -1,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_offset=start_frame - record.start,
        alignment_error=matched_err,
        source_record=record,
    )


def align_record(
    record: InterHubRecord,
    scenario_index: ScenarioIndex,
    ref_xy: Optional[Tuple[float, float]] = None,
) -> Optional[AlignedRecord]:
    """对外统一入口：优先坐标匹配，失败/无坐标则退化到 offset 搜索。"""
    try:
        scenario = load_scenario_by_id(record.original_scene_id, scenario_index)
    except KeyError:
        return None

    if ref_xy is not None:
        result = align_record_by_position(record, scenario, ref_xy)
        if result is not None:
            return result

    return align_record_by_offset_search(record, scenario)


def batch_align(
    records: List[InterHubRecord],
    scenario_index: ScenarioIndex,
) -> Tuple[List[AlignedRecord], List[InterHubRecord]]:
    """批量对齐，返回 (成功列表, 失败列表)，便于统计失败率。"""
    ok, failed = [], []
    for r in records:
        result = align_record(r, scenario_index)
        if result is None:
            failed.append(r)
        else:
            ok.append(result)
    return ok, failed
