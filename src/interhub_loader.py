"""
读取 InterHub 交互抽取结果表 (csv/tsv)，转成 schemas.InterHubRecord 列表。

InterHub 表格中部分字段是用分号分隔的多值字符串（如 track_id 列 "256;306;330"，
key_agents 列 "['HV', 'HV', 'HV']" 这种 python-repr 字符串），需要专门解析。
"""

from __future__ import annotations
import ast
from typing import Iterator, List, Optional
import pandas as pd

from schemas import InterHubRecord


def _parse_semicolon_ints(raw: str) -> List[int]:
    """ '256;306;330' -> [256, 306, 330] """
    if pd.isna(raw) or raw == "":
        return []
    return [int(x) for x in str(raw).split(";") if x.strip() != ""]


def _parse_pylist(raw: str) -> List[str]:
    """ "['HV', 'HV', 'HV']" -> ['HV', 'HV', 'HV']；解析失败则退化为按逗号切分 """
    if pd.isna(raw) or raw == "":
        return []
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, (list, tuple)):
            return [str(v) for v in val]
        return [str(val)]
    except (ValueError, SyntaxError):
        return [s.strip().strip("[]'\"") for s in str(raw).split(",")]


def load_interhub_table(path: str, sep: str = "\t") -> pd.DataFrame:
    """读取原始表格文件。列名需与 InterHub 输出一致（见模块顶部注释的字段列表）。"""
    df = pd.read_csv(path, sep=sep, dtype=str)
    return df


def _row_to_record(row: "pd.Series") -> InterHubRecord:
    key_agents_raw = row.get("key_agents", "")
    key_agents = _parse_semicolon_ints(key_agents_raw) if ";" in str(key_agents_raw) else \
        [int(x) for x in _parse_pylist(key_agents_raw) if str(x).lstrip("-").isdigit()]

    return InterHubRecord(
        dataset=row["dataset"],
        folder=row["folder"],
        scenario_idx=row["scenario_idx"],
        track_ids=_parse_semicolon_ints(row["track_id"]),
        start=int(row["start"]),
        end=int(row["end"]),
        intensity=float(row["intensity"]),
        pet=float(row["PET"]) if pd.notna(row.get("PET")) and str(row.get("PET")) != "" else None,
        interaction_type=row["two/multi"],
        vehicle_type=_parse_pylist(row["vehicle_type"]),
        av_included=row["AV_included"],
        key_agents=key_agents,
        pre_int_i=int(row["pre_int_i"]),
        post_int_i=int(row["post_int_i"]),
        pre_int_j=int(row["pre_int_j"]),
        post_int_j=int(row["post_int_j"]),
        path_category=row["path_category"],
        path_relation=row["path_relation"],
        turn_label=row["turn_label"],
        priority_label=row["priority_label"],
        original_data_file=row["original_data_file"],
        original_scene_id=row["original_scene_id"],
        original_track_id=_parse_semicolon_ints(row["original_track_id"]),
    )


def iter_interaction_records(df: "pd.DataFrame") -> Iterator[InterHubRecord]:
    """逐行转换。转换失败的行会被跳过并打印警告（不中断批处理）。"""
    for idx, row in df.iterrows():
        try:
            yield _row_to_record(row)
        except (KeyError, ValueError) as e:
            print(f"[interhub_loader] skip row {idx}: {e}")
            continue
