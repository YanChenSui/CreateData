"""
逐帧把 agent 的 (x, y, heading) 分配到地图 lane_id 上。

Waymo 原始数据里车辆没有标注"当前车道"，这里必须重新计算：
1. KD-tree 粗筛半径内的候选车道；
2. 把点投影到每条候选车道 polyline 上，得到 Frenet (s, d)；
3. 用 |d| 和航向差过滤掉不合理候选；
4. 加权打分选最优；
5. 对整条 lane_id 序列做 hysteresis 去抖，避免路口/多重车道导致的帧间抖动。
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

import config
from schemas import LaneAssignment
from mapping.lane_graph import LaneGraph, LaneRecord


def _project_point_to_polyline(
    point: np.ndarray, polyline: np.ndarray, arc_length: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """
    把 point 投影到 polyline（逐段线段）上，返回 (s, d, tangent_vector)。
    d 的符号约定：以车道行进方向为正前方，左侧为正，右侧为负
    （即 tangent 逆时针旋转90度的方向为 d 正方向）。
    """
    best_dist = np.inf
    best_s = 0.0
    best_d = 0.0
    best_tangent = np.array([1.0, 0.0])

    for i in range(len(polyline) - 1):
        p0, p1 = polyline[i], polyline[i + 1]
        seg = p1 - p0
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-6:
            continue
        seg_dir = seg / seg_len
        t = np.clip(np.dot(point - p0, seg_dir), 0.0, seg_len)
        proj = p0 + t * seg_dir
        dist = np.linalg.norm(point - proj)
        if dist < best_dist:
            best_dist = dist
            best_s = arc_length[i] + t
            # 带符号横向距离：法向量 = seg_dir 逆时针旋转90度
            normal = np.array([-seg_dir[1], seg_dir[0]])
            best_d = float(np.dot(point - proj, normal))
            best_tangent = seg_dir

    return best_s, best_d, best_tangent


def _lane_half_width(lane: LaneRecord) -> float:
    # Waymo LaneCenter 本身不直接给宽度，这里用默认值兜底；
    # 如果项目里另外解析了 road_edge / road_line 可在此替换为真实宽度估计。
    return config.LANE_HALF_WIDTH_DEFAULT_M


def project_to_lane(
    point: Tuple[float, float],
    heading_rad: float,
    lane: LaneRecord,
) -> Optional[Tuple[float, float, float]]:
    """
    返回 (s, d, heading_diff_deg)；若不满足横向/航向约束返回 None。
    """
    s, d, tangent = _project_point_to_polyline(
        np.array(point, dtype=float), lane.polyline, lane.arc_length
    )
    if s < 0 or s > lane.arc_length[-1]:
        return None

    half_width = _lane_half_width(lane)
    if abs(d) > half_width + config.LANE_D_MARGIN_M:
        return None

    lane_heading = np.arctan2(tangent[1], tangent[0])
    heading_diff = np.degrees(
        np.abs(np.arctan2(np.sin(heading_rad - lane_heading), np.cos(heading_rad - lane_heading)))
    )
    if heading_diff > config.LANE_HEADING_DIFF_MAX_DEG:
        return None

    return s, d, heading_diff


def assign_lane_single_frame(
    point: Tuple[float, float],
    heading_rad: float,
    lane_graph: LaneGraph,
) -> Optional[LaneAssignment]:
    candidate_ids = lane_graph.candidate_lanes_near(point, config.LANE_SEARCH_RADIUS_M)
    best = None
    best_score = -np.inf

    for lane_id in candidate_ids:
        lane = lane_graph.lanes[lane_id]
        proj = project_to_lane(point, heading_rad, lane)
        if proj is None:
            continue
        s, d, heading_diff = proj

        d_score = 1.0 - min(abs(d) / (_lane_half_width(lane) + config.LANE_D_MARGIN_M), 1.0)
        h_score = 1.0 - min(heading_diff / config.LANE_HEADING_DIFF_MAX_DEG, 1.0)
        score = (
            config.LANE_SCORE_WEIGHT_D * d_score
            + config.LANE_SCORE_WEIGHT_HEADING * h_score
        )

        if score > best_score:
            best_score = score
            best = (lane_id, s, d, heading_diff)

    if best is None:
        return None

    lane_id, s, d, heading_diff = best
    return LaneAssignment(
        frame=-1,  # 由调用方填充
        lane_id=lane_id,
        s=s,
        d=d,
        heading_diff_deg=heading_diff,
        score=best_score,
    )


def assign_lane_per_frame(
    states: List[Tuple[float, float, float, bool]],  # (x, y, heading, valid)
    lane_graph: LaneGraph,
) -> List[LaneAssignment]:
    """对整条轨迹逐帧分配车道（未去抖的原始结果）。"""
    results: List[LaneAssignment] = []
    for frame, (x, y, heading, valid) in enumerate(states):
        if not valid:
            results.append(LaneAssignment(frame=frame, lane_id=None, s=0.0, d=0.0,
                                            heading_diff_deg=0.0, score=0.0))
            continue
        assignment = assign_lane_single_frame((x, y), heading, lane_graph)
        if assignment is None:
            results.append(LaneAssignment(frame=frame, lane_id=None, s=0.0, d=0.0,
                                            heading_diff_deg=0.0, score=0.0))
        else:
            assignment.frame = frame
            results.append(assignment)
    return results


def smooth_lane_sequence(
    assignments: List[LaneAssignment],
    min_dwell_frames: int = None,
) -> List[LaneAssignment]:
    """
    Hysteresis 去抖：新的 lane_id 必须连续出现 >= min_dwell_frames 次才确认切换，
    否则沿用上一个已确认的车道（None 值同样需要连续确认才会被采纳，
    避免遮挡/短暂丢检导致的车道置空）。
    """
    min_dwell_frames = min_dwell_frames or config.LANE_MIN_DWELL_FRAMES
    if not assignments:
        return assignments

    smoothed: List[LaneAssignment] = []
    confirmed_lane = assignments[0].lane_id
    pending_lane = None
    pending_count = 0

    for a in assignments:
        if a.lane_id == confirmed_lane:
            pending_lane, pending_count = None, 0
        else:
            if a.lane_id == pending_lane:
                pending_count += 1
            else:
                pending_lane, pending_count = a.lane_id, 1

            if pending_count >= min_dwell_frames:
                confirmed_lane = pending_lane
                pending_lane, pending_count = None, 0

        smoothed.append(
            LaneAssignment(
                frame=a.frame,
                lane_id=confirmed_lane,
                s=a.s,
                d=a.d,
                heading_diff_deg=a.heading_diff_deg,
                score=a.score,
            )
        )
    return smoothed
