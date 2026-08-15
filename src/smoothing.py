"""平滑与微分工具：Waymo 自带速度是差分估计，噪声较大，facts 层计算前先平滑。"""

from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter

import config


def savgol_smooth(series: np.ndarray, window: int = None, polyorder: int = None) -> np.ndarray:
    """
    对 1D 或 (T, D) 序列做 Savitzky-Golay 平滑。
    序列长度不足 window 时自动退化为原始序列（避免报错中断批处理）。
    """
    window = window or config.SAVGOL_WINDOW
    polyorder = polyorder or config.SAVGOL_POLYORDER
    n = series.shape[0]
    if n < window:
        return series
    w = window if window % 2 == 1 else window - 1
    return savgol_filter(series, window_length=w, polyorder=polyorder, axis=0)


def central_diff(series: np.ndarray, dt: float = None) -> np.ndarray:
    """中心差分求导（首尾用前向/后向差分），用于由速度求加速度。"""
    dt = dt or config.DT
    n = series.shape[0]
    deriv = np.zeros_like(series, dtype=float)
    if n < 2:
        return deriv
    deriv[1:-1] = (series[2:] - series[:-2]) / (2 * dt)
    deriv[0] = (series[1] - series[0]) / dt
    deriv[-1] = (series[-1] - series[-2]) / dt
    return deriv


def longitudinal_component(vec: np.ndarray, heading_rad: np.ndarray) -> np.ndarray:
    """把 (T, 2) 的向量（如速度）投影到各帧航向方向，得到纵向分量 (T,)。"""
    dirs = np.stack([np.cos(heading_rad), np.sin(heading_rad)], axis=1)
    return np.sum(vec * dirs, axis=1)
