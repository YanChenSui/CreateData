"""
贯穿全流程的强类型数据结构。
各模块之间只通过这些结构传递数据，避免字段名拼错 / 类型漂移。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Literal


# ---------------------------------------------------------------------------
# 1. InterHub 原始记录（对应你贴的表格一行）
# ---------------------------------------------------------------------------
@dataclass
class InterHubRecord:
    dataset: str
    folder: str
    scenario_idx: str
    track_ids: List[int]              # 原表 "306;330" 类似的多个 id
    start: int
    end: int
    intensity: float
    pet: Optional[float]
    interaction_type: str             # "two" / "multi"
    vehicle_type: List[str]
    av_included: str
    key_agents: List[int]
    pre_int_i: int
    post_int_i: int
    pre_int_j: int
    post_int_j: int
    path_category: str
    path_relation: str
    turn_label: str
    priority_label: str
    original_data_file: str
    original_scene_id: str
    original_track_id: List[int]


# ---------------------------------------------------------------------------
# 2. 对齐后的记录 (io/alignment.py 输出)
# ---------------------------------------------------------------------------
@dataclass
class AlignedRecord:
    scene_id: str
    agent_a: int
    agent_b: int
    start_frame: int                  # 已换算为原始 scenario 的 timestep 索引
    end_frame: int
    frame_offset: int                 # InterHub 帧号 -> 原始帧号 的偏移量
    alignment_error: float            # 估计误差（米或帧，見 alignment.py）
    source_record: InterHubRecord


# ---------------------------------------------------------------------------
# 3. 车道分配结果（逐帧）
# ---------------------------------------------------------------------------
@dataclass
class LaneAssignment:
    frame: int
    lane_id: Optional[int]            # None 表示该帧未能分配车道
    s: float                          # Frenet 纵向坐标
    d: float                          # Frenet 横向坐标（带符号，左正右负，约定见 lane_assignment.py）
    heading_diff_deg: float
    score: float


# ---------------------------------------------------------------------------
# 4. 中间事实层
# ---------------------------------------------------------------------------
@dataclass
class Facts:
    same_lane: bool
    adjacent_lane: bool
    lane_id_changed_a: bool
    lane_id_changed_b: bool
    a_is_behind_b: bool
    a_is_behind_b_before: Optional[bool]
    a_is_behind_b_after: Optional[bool]
    distance_min_m: float
    ttc_min_s: float
    pet_s: Optional[float]
    pet_mismatch: Optional[bool]
    a_decelerates: bool
    b_decelerates: bool
    trajectory_conflict: bool
    conflict_geom: bool
    conflict_topo: bool
    lane_seq_a: List[Optional[int]]
    lane_seq_b: List[Optional[int]]
    s_gap_m: Optional[float]


# ---------------------------------------------------------------------------
# 5. 证据层
# ---------------------------------------------------------------------------
@dataclass
class Evidence:
    lane_change_support: float
    merge_support: float
    overtake_support: float
    yield_support: float


# ---------------------------------------------------------------------------
# 6. 决策结果
# ---------------------------------------------------------------------------
Status = Literal["ambiguous", "candidate", "candidate_multi", "alignment_failed"]


@dataclass
class CandidateResult:
    scene_id: str
    agent_a: int
    agent_b: int
    window: Tuple[int, int]
    facts: Facts
    evidence: Evidence
    candidate_label: Optional[List[str]]   # 单个或 top-2
    confidence: Optional[float]
    status: Status
    rule_trace: List[str] = field(default_factory=list)
