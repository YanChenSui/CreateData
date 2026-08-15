"""
从 Waymo map_features 构建车道图。

要点：
- LaneCenter.left_neighbors / right_neighbors 不是"整条车道全程平行"，
  而是带 (self_start_index, self_end_index, neighbor_start_index, neighbor_end_index)
  的区间关系 —— 两条车道可能只在一段范围内相邻（比如匝道汇入前半段不相邻）。
  所有邻接查询都必须带上纵向位置 s，不能只看 lane_id 是否互为邻居。
- entry_lanes / exit_lanes 是拓扑前驱后继（车头到车尾方向），
  用于区分"变道"（neighbor 关系）和"沿车道继续走/路口转弯"（entry/exit 关系）。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class NeighborSegment:
    neighbor_id: int
    self_start_index: int
    self_end_index: int
    neighbor_start_index: int
    neighbor_end_index: int
    boundary_type: Optional[str] = None  # 若有车道线类型（是否允许变道）可存这里


@dataclass
class LaneRecord:
    lane_id: int
    polyline: np.ndarray              # (N, 2) 的 (x, y)
    arc_length: np.ndarray            # (N,) 每个点到起点的累积弧长
    entry_lanes: List[int] = field(default_factory=list)
    exit_lanes: List[int] = field(default_factory=list)
    left_neighbors: List[NeighborSegment] = field(default_factory=list)
    right_neighbors: List[NeighborSegment] = field(default_factory=list)
    lane_type: Optional[int] = None


@dataclass
class LaneGraph:
    lanes: Dict[int, LaneRecord]
    kdtree: cKDTree                   # 所有车道所有采样点的最近邻索引
    kdtree_lookup: List[Tuple[int, int]]  # kdtree 第 i 个点 -> (lane_id, point_index_in_polyline)

    def candidate_lanes_near(self, xy: Tuple[float, float], radius: float) -> List[int]:
        idxs = self.kdtree.query_ball_point(xy, r=radius)
        lane_ids = {self.kdtree_lookup[i][0] for i in idxs}
        return list(lane_ids)

    def neighbors_at_s(self, lane_id: int, s: float, side: str) -> List[int]:
        """返回 lane_id 在纵向位置 s 处、side ('left'/'right') 方向上真正相邻的车道 id。"""
        lane = self.lanes.get(lane_id)
        if lane is None:
            return []
        segs = lane.left_neighbors if side == "left" else lane.right_neighbors
        result = []
        for seg in segs:
            s_start = lane.arc_length[min(seg.self_start_index, len(lane.arc_length) - 1)]
            s_end = lane.arc_length[min(seg.self_end_index, len(lane.arc_length) - 1)]
            if s_start <= s <= s_end:
                result.append(seg.neighbor_id)
        return result

    def is_neighbor_relation(self, lane_id_a: int, lane_id_b: int, s_a: float) -> bool:
        left = self.neighbors_at_s(lane_id_a, s_a, "left")
        right = self.neighbors_at_s(lane_id_a, s_a, "right")
        return lane_id_b in left or lane_id_b in right

    def is_topo_relation(self, lane_id_a: int, lane_id_b: int) -> bool:
        """判断 a、b 是否存在 entry/exit（前后拓扑连接）关系，用于区分变道 vs 转弯/直行。"""
        lane = self.lanes.get(lane_id_a)
        if lane is None:
            return False
        return lane_id_b in lane.entry_lanes or lane_id_b in lane.exit_lanes


def _arc_length(polyline: np.ndarray) -> np.ndarray:
    if len(polyline) < 2:
        return np.zeros(len(polyline))
    diffs = np.diff(polyline, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_len)])


def build_lane_graph(map_features) -> LaneGraph:
    lanes: Dict[int, LaneRecord] = {}

    for feat in map_features:
        if not feat.HasField("lane"):
            continue
        lane_proto = feat.lane
        polyline = np.array(
            [[p.x, p.y] for p in lane_proto.polyline], dtype=float
        )
        if len(polyline) == 0:
            continue

        left_neighbors = [
            NeighborSegment(
                neighbor_id=n.feature_id,
                self_start_index=n.self_start_index,
                self_end_index=n.self_end_index,
                neighbor_start_index=n.neighbor_start_index,
                neighbor_end_index=n.neighbor_end_index,
            )
            for n in lane_proto.left_neighbors
        ]
        right_neighbors = [
            NeighborSegment(
                neighbor_id=n.feature_id,
                self_start_index=n.self_start_index,
                self_end_index=n.self_end_index,
                neighbor_start_index=n.neighbor_start_index,
                neighbor_end_index=n.neighbor_end_index,
            )
            for n in lane_proto.right_neighbors
        ]

        lanes[feat.id] = LaneRecord(
            lane_id=feat.id,
            polyline=polyline,
            arc_length=_arc_length(polyline),
            entry_lanes=list(lane_proto.entry_lanes),
            exit_lanes=list(lane_proto.exit_lanes),
            left_neighbors=left_neighbors,
            right_neighbors=right_neighbors,
            lane_type=lane_proto.type,
        )

    # 构建全局 KD-tree：把所有车道所有采样点拍平
    all_points = []
    lookup: List[Tuple[int, int]] = []
    for lane_id, lane in lanes.items():
        for pt_idx, pt in enumerate(lane.polyline):
            all_points.append(pt)
            lookup.append((lane_id, pt_idx))

    if not all_points:
        raise ValueError("no lane polylines found in map_features")

    kdtree = cKDTree(np.array(all_points))
    return LaneGraph(lanes=lanes, kdtree=kdtree, kdtree_lookup=lookup)
