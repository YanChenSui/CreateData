"""
全局配置：所有可调阈值、打分权重集中在此，方便调参和消融实验。
"""

# ---------------------------------------------------------------------------
# 采样与平滑
# ---------------------------------------------------------------------------
SIM_HZ = 10.0                     # Waymo Motion 采样频率 (Hz)
DT = 1.0 / SIM_HZ

SAVGOL_WINDOW = 5                 # Savitzky-Golay 平滑窗口（帧数，需为奇数）
SAVGOL_POLYORDER = 2

# ---------------------------------------------------------------------------
# 对齐校验 (io/alignment.py)
# ---------------------------------------------------------------------------
ALIGNMENT_MAX_FRAME_ERROR = 1     # 估计 offset 与实际匹配允许的最大帧误差
ALIGNMENT_POS_TOL_M = 0.5         # 位置匹配容差 (米)

# ---------------------------------------------------------------------------
# 车道分配 (mapping/lane_assignment.py)
# ---------------------------------------------------------------------------
LANE_SEARCH_RADIUS_M = 15.0       # KD-tree 粗筛半径
LANE_HALF_WIDTH_DEFAULT_M = 1.75  # 默认车道半宽 (无边界信息时兜底)
LANE_D_MARGIN_M = 0.5             # 横向偏移容差
LANE_HEADING_DIFF_MAX_DEG = 30.0  # 航向与车道切线夹角阈值
LANE_SCORE_WEIGHT_D = 0.6         # 打分权重：横向偏移
LANE_SCORE_WEIGHT_HEADING = 0.4   # 打分权重：航向差
LANE_MIN_DWELL_FRAMES = 3         # hysteresis 去抖：新车道最少连续驻留帧数

# ---------------------------------------------------------------------------
# 风险/几何事实 (facts/risk_facts.py)
# ---------------------------------------------------------------------------
COLLISION_RADIUS_M = 1.0          # TTC 判定用的碰撞半径 margin（两车半宽和之外再加）
TTC_INF = float("inf")
TTC_RISK_HORIZON_S = 3.0          # risk(ttc) 归一化上界

PET_MISMATCH_TOL_S = 0.3          # 自算 PET 与 InterHub 给定 PET 允许的最大误差

# ---------------------------------------------------------------------------
# 运动学事实 (facts/kinematic_facts.py)
# ---------------------------------------------------------------------------
DECEL_THRESHOLD_MS2 = -0.5        # 判定"减速"的纵向加速度阈值 (m/s^2)
DECEL_MIN_CONSEC_FRAMES = 5       # 连续满足阈值的最少帧数 (0.5s @ 10Hz)

# ---------------------------------------------------------------------------
# 证据打分权重 (evidence/scoring.py)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "lane_change": {
        "lane_id_changed_a": 2.5,
        "adjacent_lane_before": 1.5,
        "not_same_lane_before": 1.0,
        "trajectory_conflict": 0.8,
        "bias": -1.0,
    },
    "merge": {
        "lane_id_changed_a": 2.0,
        "same_lane_after": 1.5,
        "conflict_topo": 1.5,
        "risk_ttc": 1.0,
        "bias": -1.2,
    },
    "overtake": {
        "lane_id_changed_a": 1.5,
        "a_is_behind_b_before": 1.5,
        "a_is_ahead_b_after": 1.5,
        "b_not_decelerates": 1.0,
        "bias": -1.2,
    },
    "yield": {
        "b_decelerates": 2.0,
        "trajectory_conflict": 1.0,
        "risk_ttc": 1.0,
        "not_a_behind_b": 0.5,
        "bias": -1.0,
    },
}

# ---------------------------------------------------------------------------
# 决策层 (decision/rule_engine.py)
# ---------------------------------------------------------------------------
DECISION_MIN_SCORE = 0.5          # 低于此分数 -> ambiguous
DECISION_TOP2_MARGIN = 0.15       # top1 - top2 < margin -> candidate_multi

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
SCENARIO_INDEX_CACHE = "scenario_index.pkl"
