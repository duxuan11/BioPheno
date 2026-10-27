"""Stage 6：跨天配准、tracklet 匹配与 SPOT 专用 uid（Spec §6-Stage6）。

这份数据是 3 天（day0/day3/day5）× 3 视野的静态切片，没有真正的视频；motion 特征被有意跳过，
所以这里的"跟踪"只为下游服务：每簇占比随时间、簇间转移矩阵、表型轨迹都需要知道
"哪些实例是同一条 organoid 在不同天的样子"。

**坐标与偏移约定（最容易搞反的一处）**

`register_frame(prev, cur, max_shift)` 返回的 `(dy, dx)` 是**可以直接加到 cur 坐标上**、
把 cur 变到 prev 坐标系里的平移量（质心与 bbox 都适用）。它由
`SPOT.Registration.registration_2D.translation_register_correlation` 的位移换算而来：
那个函数返回的是 `(x, y)` 采样位移，内容会朝反方向移动，所以
`offset = (-round(disps[1]), -round(disps[0]))`。`test_register_frame_recovers_known_shift`
用已知平移钉住了这个符号。

参考系固定为每个视野的**第一帧**（day0）：先求 day3→day0 与 day5→day0，再把 day3→day5 表达成
两者的复合 `O5 - O3`（同一参考系下的平移可加减），而不是重新配准 day3 与 day5 的原始图。

**SPOT 对 uid 的硬要求（Spec D13 + `construct_obj_traj_from_uniq_obj_ids` 源码）**

SPOT 内部对 uid 做 `idd.split('_')` 后取**最后一段** `int()` 当帧号，再用单一标量 `d_Frame`
判连续性。所以 `spot_uid` 只能是 `f"{fov}_{track_id}_{frame_idx}"`（恰好 3 段、帧号在最后、
都是整数），帧轴必须是 0/1/2 连续整数，真实小时数另存 `time_hours`。`spot_uid_from` 在拼接时
逐条校验并在违反时抛含上下文的 `ValueError`，`assign_tracks` 再兜一道"uid 不许重复"。

**merge / split 的语义与轨迹继承**

- 候选对：bbox IoU ≥ `MIN_BBOX_IOU` **或** 质心距离 ≤ `MAX_CENT_DIST_PX`（在 prev 坐标系里算）。
- 贪心：候选对按 `(-iou, prev_uid, cur_uid)` 排序后一对一分配，**只有一对一的行会延续轨迹**。
- 残余多对一 → `merge`：多颗上一帧实例都指向同一颗下一帧实例。幸存父实例取**面积最大者**
  （同面积取 uid 小者），它会以 `one_to_one` 行出现；其余被吸收的实例记 `merge` 行。
  被吸收的实例**不**继承幸存者的 track_id —— 否则 `(fov, track_id, frame_idx)` 会重复，
  SPOT 会把同一帧的两条实例当成两段轨迹。它们的旧轨迹到此为止（自己成为长度 1 的孤立 track）。
- 残余一对多 → `split`：上一帧实例已经有一对一的继承人，同时又冒出新的下一帧实例。
  新实例记 `split` 行，但同样**不**继承父 track（同一帧里两个实例共享一个 track_id 一样会重复）；
  它按"新生成"处理，另起一条新轨迹。真实数据里 merge 常见（1145/984/917 → 697/768/582 → 581/614/453，
  类器官长大并融合）、split 极少，所以这条取舍对占比/转移统计几乎没有影响。

**配准用的是"对齐后的标签图"**

Stage 1 的 meta（bbox/质心）是从 `build_catalog` 里 `align_mask_to_image` 之后的标签图算出来的，
原始标签图可能整体偏了 `align_offset=(dy, dx)`（±`ALIGN_MAX_SHIFT_PX` 内）。如果配准直接拿原始标签图，
帧对偏移就会多出 `align_ref - align_cur` 这一项（最多 ±14 px，远大于实例尺度）。
所以 `build_tracks` 接受 `align_offsets`（`"day0/A1" → (dy, dx)`，run_stage6 从 Stage 1 的
`stage1_qc.json` 读），先用 `instances._place` 把原始掩码搬到与 meta 一致的坐标系，再做配准；
逐 (day, fov) 的偏移会回显进 `match_qc.json` 的 `registration.align_offsets`。

产物（`output/analysis/`）：`tracks.csv`（`instance_uid, spot_uid, track_id, fov, frame_idx,
time_hours, day`）、`matches.csv`（**逐帧对的完整匹配表**：`fov, pair, prev_uid, cur_uid, iou,
cent_dist, event`，用于事后审计"谁并进了谁"、合并父实例的选择、分裂父实例，不必重跑本阶段）与
`match_qc.json`（每个视野的配准偏移/对齐偏移、每个帧对的候选/一对一/合并/分裂/未匹配数与匹配表行数、
每个视野的 track 长度直方图、配准告警（字符串 + 按类型计数）与配准几何核验）。`run_stage6` 末尾硬断言
"至少存在一条长度 ≥ 2 的 track"，防止帧轴编码退化（Spec §9 风险表）。

**"配准真的没生效吗"怎么自己核验**：复合偏移全为 (0, 0) 既可能是"掩码本来就对齐"，也可能是"相位相关
估计被上限拒收后退回零"。`check_registration_geometry` 不看相位相关，只在二值掩码上扫零偏移附近的 IoU，
给出 `zero_offset_iou / best_iou / best_offset / aligned_at_zero`，落进
`match_qc.json` 的 `registration_geometry`（旧产物没有这一块时，可由
`scripts/check_registration_geometry.py` 单独跑一次并写 `logs/registration_geometry_check.json`）。
"""

import contextlib
import io
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from biopheno import config as cfg
from biopheno import instances as inst
from SPOT.Registration import registration_2D as r2d

# —— 上游产物（Stage 1 的汇总表；掩码图仍从 DATA_ROOT 按约定路径读）——
_INSTANCES_SUBDIR: str = "instances"
_ALL_INSTANCES_FILE: str = "all_instances.csv"

# —— 本阶段产物（Spec §6-Stage6）——
_ANALYSIS_SUBDIR: str = "analysis"
_TRACKS_FILE: str = "tracks.csv"
_MATCHES_FILE: str = "matches.csv"
_QC_FILE: str = "match_qc.json"

# 匹配表的列顺序（下游 QC/报告按这个顺序读）。
_MATCH_COLUMNS: tuple[str, ...] = ("prev_uid", "cur_uid", "iou", "cent_dist", "event")

# 落盘 matches.csv 的列顺序：在逐帧对匹配表前面加 fov 与 pair（如 "day0->day3" 的帧号形式 "0->1"）。
_MATCH_OUTPUT_COLUMNS: tuple[str, ...] = ("fov", "pair", *_MATCH_COLUMNS)

# 匹配表只允许这三种事件；`merge`/`split` 是"附加信息行"，不参与轨迹继承。
_EVENT_VALUES: tuple[str, ...] = ("one_to_one", "merge", "split")

# 事件在表里的排列顺序（同一个 cur 先列一对一，再列被吸收的 parent 们，最后列分裂出来的新实例）。
_EVENT_RANK: dict[str, int] = {"one_to_one": 0, "merge": 1, "split": 2}

# —— 配准告警的机器可读类型 ——
# `registration.warning_kinds` 按类型计数（类型 → 条数），字符串告警仍逐条留在 `registration.warnings` 里给人读。
# 只有"相位相关估计被 ±上限拒收"才是"被拒的估计"；空掩码、上游异常、复合偏移越界都不是同一件事，
# 所以报告里"被拒次数"只能数 `WARNING_KIND_OFFSET_EXCEEDS_LIMIT`（不能数 `len(warnings)`）。
WARNING_KIND_EMPTY_MASK: str = "empty_mask"
WARNING_KIND_UPSTREAM_ERROR: str = "upstream_error"
WARNING_KIND_OFFSET_EXCEEDS_LIMIT: str = "offset_exceeds_limit"
WARNING_KIND_COMPOUND_OFFSET_EXCEEDS_LIMIT: str = "compound_offset_exceeds_limit"

# —— 配准几何核验（"掩码本来是否就对齐"）——
# 直接平移**全分辨率**二值掩码，在 ±max_shift **像素**内逐像素扫 IoU：真实数据 6 个帧对约十几秒。
# 不在这里下采样：一旦把掩码缩到网格上再平移，偏移就变成了"网格步"，写进 JSON / 报告的 `max_shift_px`
# 与 `best_offset` 都会少报一个步长因子（真实 8 px 的平移曾被报成 2）。
# `tolerance` 是"零偏移 IoU 与最佳 IoU 的差"的容差：差值不超过它就算 `aligned_at_zero`。
GEOMETRY_ALIGN_TOLERANCE: float = 0.02
# 偏移与扫描窗口的单位标记（写进 QC / 核验 JSON，让"这是像素还是网格步"不再靠猜）。
GEOMETRY_OFFSET_UNIT: str = "px"
GEOMETRY_SHIFT_STEP_PX: int = 1

# tracks.csv 的列顺序（ruling：至少这 7 列；label 作为可追溯的附加列）。
_TRACK_COLUMNS: tuple[str, ...] = (
    "instance_uid",
    "spot_uid",
    "track_id",
    "fov",
    "frame_idx",
    "time_hours",
    "day",
    "label",
)

# Stage 1 的 meta 里匹配/建轨所必需的列（缺任何一列都不该静默降级）。
_META_REQUIRED: tuple[str, ...] = (
    "instance_uid",
    "day",
    "fov",
    "frame_idx",
    "time_hours",
    "r0",
    "c0",
    "r1",
    "c1",
    "centroid_row",
    "centroid_col",
)

# 配准告警的兜底出口：`register_frame` 的调用者没给 sink 时写到这里，由 `take_registration_warnings`
# 取走并落进 match_qc.json（模块级列表只作为"没人接收时也不静默丢"的保险）。
_REGISTRATION_WARNINGS: list[str] = []
_REGISTRATION_WARNING_KINDS: dict[str, int] = {}


# ---------------------------------------------------------------- 配准


def take_registration_warnings() -> list[str]:
    """取走并清空兜底的配准告警列表（run_stage6/build_tracks 之外的调用者用）。"""
    warnings = list(_REGISTRATION_WARNINGS)
    _REGISTRATION_WARNINGS.clear()
    _REGISTRATION_WARNING_KINDS.clear()
    return warnings


def _record_warning(
    sink: list[str] | None,
    message: str,
    kind: str,
    kinds_out: dict[str, int] | None = None,
) -> None:
    """把配准告警写进调用者给的 sink，并按 `kind` 计数（QC 已经承载告警，这里不再 print）。

    字符串列表是给人读的；`kind` 才是机器可读的那一半：报告只数 `WARNING_KIND_OFFSET_EXCEEDS_LIMIT`，
    不会把空掩码/上游异常/复合偏移越界也算成"被拒的相位相关估计"。
    """
    (sink if sink is not None else _REGISTRATION_WARNINGS).append(message)
    target = kinds_out if kinds_out is not None else _REGISTRATION_WARNING_KINDS
    target[kind] = target.get(kind, 0) + 1


def _read_binary(day: str, fov: str) -> np.ndarray:
    """读某天某视野的标签图并二值化（`labels > 0`）。

    Stage 1（`instances.run_stage1`）不落盘原图/标签图（每张约 15 MB），只落 meta/轮廓/裁剪图，
    所以配准需要重新按 Spec §2 的路径读标签图。这里复用 `instances._read_labels`：它是同包内
    唯一知道 `<DATA_ROOT>/<day>/Output/<fov>{suffix}.tif` 约定、且会跟着 cfg.DATA_ROOT 走的地方，
    不在本模块重复一份路径拼接。
    """
    return np.asarray(inst._read_labels(day, fov)) > 0


def _aligned_mask(mask: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    """把原始标签二值图按 Stage 1 的 `align_offset` 搬到 meta 所在的坐标系。

    `instances.align_mask_to_image` 返回的 (dy, dx) 就是"把原始标签图平移这么多"后的那份标签图
    （`_place` 的 dy>0 = 内容下移）；Stage 1 的 bbox/质心来自那份对齐后的图，所以配准也必须用它。
    偏移为 (0, 0) 时原样返回（不复制、不改 dtype），既有调用方与测试拿到的还是同一个对象。
    """
    dy, dx = int(offset[0]), int(offset[1])
    if dy == 0 and dx == 0:
        return mask
    array = np.asarray(mask)
    shape = (int(array.shape[0]), int(array.shape[1]))
    return inst._place(array, dy, dx, shape)


def _iou_at_offset(prev_binary: np.ndarray, cur_binary: np.ndarray, dy: int, dx: int) -> float:
    """把 `cur` 平移 `(dy, dx)` 后与 `prev` 的 IoU（只在两者重叠的区域内算）。

    偏移的语义与 `register_frame` 一致：`(dy, dx)` 加到 cur 坐标上就落到 prev 坐标系，
    所以 cur 的像素 (r, c) 对齐到 prev 的 (r + dy, c + dx)。
    """
    height, width = prev_binary.shape
    prev_rows = slice(max(0, dy), height + min(0, dy))
    cur_rows = slice(max(0, -dy), height + min(0, -dy))
    prev_cols = slice(max(0, dx), width + min(0, dx))
    cur_cols = slice(max(0, -dx), width + min(0, -dx))
    prev_window = prev_binary[prev_rows, prev_cols]
    cur_window = cur_binary[cur_rows, cur_cols]
    union = int(np.count_nonzero(prev_window | cur_window))
    if union == 0:
        return 0.0
    return int(np.count_nonzero(prev_window & cur_window)) / union


def check_frame_pair_geometry(
    prev: np.ndarray,
    cur: np.ndarray,
    max_shift: int,
    *,
    tolerance: float = GEOMETRY_ALIGN_TOLERANCE,
) -> dict:
    """对一个帧对自问："掩码在零偏移下本来就已经对齐了吗？"

    做法（廉价、不依赖相位相关）：在**全分辨率**二值掩码上，于 ±`max_shift` **像素**内逐像素（步长 1 px）
    算 IoU，记下零偏移的 IoU、扫描范围内的最佳 IoU 与达成它的偏移（单位：像素）。零偏移 IoU 与最佳 IoU
    的差不超过 `tolerance` 时判 `aligned_at_zero=True`：此时"配准失败退回 (0, 0)"与"不做平移"在几何上等价。

    为什么不在平移前下采样：缩到网格上再按网格步平移会让 `best_offset` 少报一个步长因子、扫描窗口
    比 `max_shift` 大一个步长因子（真实 12 px 的平移曾被报成 3 个“像素”）；直接平移全分辨率掩码，
    `best_offset` 与 `max_shift` 才是真实的像素数。

    并列最佳（例如一块均匀平移的掩码在多个偏移上 IoU 都是 1）时取**离零最近**的那个偏移，
    所以 `best_offset` 不会因为遍历顺序而随机跳到一个远处的等价峰。
    """
    prev_binary = np.asarray(prev) > 0
    cur_binary = np.asarray(cur) > 0
    height = min(prev_binary.shape[0], cur_binary.shape[0])
    width = min(prev_binary.shape[1], cur_binary.shape[1])
    prev_full = prev_binary[:height, :width]
    cur_full = cur_binary[:height, :width]
    zero_iou = _iou_at_offset(prev_full, cur_full, 0, 0)
    best_iou = zero_iou
    best_offset = (0, 0)
    # 由近及远地扫：先扫小偏移，严格大于才替换，于是并列最佳时留下的是离零最近的偏移
    offsets = sorted(
        (
            (dy, dx)
            for dy in range(-int(max_shift), int(max_shift) + 1)
            for dx in range(-int(max_shift), int(max_shift) + 1)
        ),
        key=lambda offset: (max(abs(offset[0]), abs(offset[1])), abs(offset[0]) + abs(offset[1]), offset),
    )
    for dy, dx in offsets:
        iou = _iou_at_offset(prev_full, cur_full, dy, dx)
        if iou > best_iou:
            best_iou = iou
            best_offset = (dy, dx)
    return {
        "zero_offset_iou": round(float(zero_iou), 6),
        "best_iou": round(float(best_iou), 6),
        # `best_offset` 的单位是像素（与 `max_shift_px` / `align_offsets` / `pair_offsets` 一致）
        "best_offset": [int(best_offset[0]), int(best_offset[1])],
        "offset_unit": GEOMETRY_OFFSET_UNIT,
        "iou_gain": round(float(best_iou - zero_iou), 6),
        "aligned_at_zero": bool(best_iou - zero_iou <= float(tolerance)),
    }


def check_registration_geometry(
    masks: dict[str, np.ndarray],
    frames: dict[str, dict[int, str]],
    align_offsets: dict[str, tuple[int, int]] | None = None,
    *,
    max_shift: int | None = None,
    tolerance: float = GEOMETRY_ALIGN_TOLERANCE,
) -> dict:
    """逐视野、逐相邻帧对做 `check_frame_pair_geometry`，返回可落进 QC / 核验 JSON 的汇总。

    输入是**原始**标签二值图（`masks[key]`，key = `"day/fov"`）与 Stage 1 的对齐偏移：内部先按偏移把掩码
    搬到 meta 坐标系（与配准同一套输入），再在零偏移下互相比较。因此这里回答的是"掩码本来就对齐吗"，
    与相位相关估计是否成功完全独立。

    `max_shift_px` 是**真实**的扫描窗口，`pairs[*][*]["best_offset"]` 的单位也是像素；
    `offset_unit` / `shift_step_px` 把单位写死进 JSON，报告与核验脚本不必再猜。

    `frames` 是 `{fov: {frame_idx: key}}`（`build_tracks` 里已有这份索引），帧对按帧号相邻取。
    """
    limit = int(cfg.ALIGN_MAX_SHIFT_PX if max_shift is None else max_shift)
    offsets_map = dict(align_offsets or {})
    pairs: dict[str, dict] = {}
    n_pairs = 0
    n_aligned = 0
    for fov in sorted(frames):
        ordered = sorted(frames[fov])
        fov_pairs: dict[str, dict] = {}
        for prev_frame, cur_frame in pairwise(ordered):
            prev_key = frames[fov][prev_frame]
            cur_key = frames[fov][cur_frame]
            prev_mask = _aligned_mask(masks[prev_key], offsets_map.get(prev_key, (0, 0)))
            cur_mask = _aligned_mask(masks[cur_key], offsets_map.get(cur_key, (0, 0)))
            entry = check_frame_pair_geometry(prev_mask, cur_mask, limit, tolerance=float(tolerance))
            fov_pairs[f"{prev_frame}->{cur_frame}"] = entry
            n_pairs += 1
            n_aligned += int(entry["aligned_at_zero"])
        pairs[fov] = fov_pairs
    return {
        "offset_unit": GEOMETRY_OFFSET_UNIT,
        "shift_step_px": GEOMETRY_SHIFT_STEP_PX,
        "max_shift_px": limit,
        "tolerance": float(tolerance),
        "pairs": pairs,
        "n_pairs": n_pairs,
        "n_aligned_at_zero": n_aligned,
        "all_aligned_at_zero": bool(n_pairs > 0 and n_aligned == n_pairs),
    }


def register_frame(
    prev: np.ndarray,
    cur: np.ndarray,
    max_shift: int,
    warnings_out: list[str] | None = None,
    kinds_out: dict[str, int] | None = None,
) -> tuple[int, int]:
    """在掩码二值上用相位相关求 prev→cur 的平移，返回"加到 cur 坐标上"的 `(dy, dx)`。

    用 `labels > 0` 得到的二值掩码而不是强度图：本数据的强度图在跨天之间几乎不可配准（Spec §6-Stage6-1
    实测），而掩码的布局是稳的。`impute_pixels=False` 是必需的：SPOT 那个函数默认会把 0 像素填成
    "非零像素均值"，二值掩码的非零值全是同一个数，填完就成了一张常数图，相位相关直接失真。

    异常、空掩码、或偏移超过 `max_shift`（任一轴）时退回 `(0, 0)` 并把原因写进 `warnings_out`
    （不抛错：一张配不准的视野不该中断整轮 Stage 6）。
    """
    prev_binary = np.asarray(prev) > 0
    cur_binary = np.asarray(cur) > 0
    # 掩码画布可能差几行几列：裁到公共最小尺寸，免得 cv2.remap 形状不匹配直接抛异常
    height = min(prev_binary.shape[0], cur_binary.shape[0])
    width = min(prev_binary.shape[1], cur_binary.shape[1])
    prev_binary = prev_binary[:height, :width]
    cur_binary = cur_binary[:height, :width]
    if not prev_binary.any() or not cur_binary.any():
        _record_warning(
            warnings_out,
            f"空掩码（前景像素 prev={int(prev_binary.sum())}、cur={int(cur_binary.sum())}），偏移退回 (0, 0)",
            WARNING_KIND_EMPTY_MASK,
            kinds_out,
        )
        return (0, 0)

    # SPOT 的配准按 (n_frames, rows, cols) 的视频处理；这里只喂两帧，取回唯一一段位移
    stack = np.stack([prev_binary, cur_binary]).astype(np.uint8) * np.uint8(255)
    try:
        # tqdm 会对 1 次迭代也画进度条；redirect_stderr 只为消掉这行噪声，异常照常上抛
        with contextlib.redirect_stderr(io.StringIO()):
            _registered, displacements = r2d.translation_register_correlation(
                stack, stack.copy(), sub_pix_res=1, impute_pixels=False
            )
    except Exception as exc:  # noqa: BLE001 - 上游任何异常都只降级、不中断
        _record_warning(
            warnings_out,
            f"配准失败（{type(exc).__name__}: {exc}），偏移退回 (0, 0)",
            WARNING_KIND_UPSTREAM_ERROR,
            kinds_out,
        )
        return (0, 0)

    # disps 是 (x, y) 采样位移：内容朝反方向移动，所以取负号再换成 (dy, dx)
    offsets = displacements[0]
    dy = -round(float(offsets[1]))
    dx = -round(float(offsets[0]))
    if max(abs(dy), abs(dx)) > int(max_shift):
        _record_warning(
            warnings_out,
            f"配准偏移 (dy={dy}, dx={dx}) 超过 ±{int(max_shift)} px 上限，偏移退回 (0, 0)",
            WARNING_KIND_OFFSET_EXCEEDS_LIMIT,
            kinds_out,
        )
        return (0, 0)
    return (dy, dx)


# ---------------------------------------------------------------- 匹配


def _empty_match_table() -> pd.DataFrame:
    """空匹配表（列名与顺序和正常表一致，下游不用特判）。"""
    return pd.DataFrame(
        {
            "prev_uid": pd.Series(dtype=object),
            "cur_uid": pd.Series(dtype=object),
            "iou": pd.Series(dtype=float),
            "cent_dist": pd.Series(dtype=float),
            "event": pd.Series(dtype=object),
        }
    )


def _validate_frame_meta(meta: pd.DataFrame, label: str) -> None:
    """校验一帧 meta 有匹配所需的列（bbox/质心/uid）；缺列直接给中文上下文。"""
    missing = [name for name in _META_REQUIRED if name not in meta.columns]
    if missing:
        raise ValueError(f"{label} meta 缺少列 {missing}：Stage 6 需要 Stage 1 all_instances.csv 的完整列")


def _bbox_area(meta: pd.DataFrame) -> np.ndarray:
    """面积数组：优先用 Stage 1 的 `area_px`（真实像素数），缺列时退回 bbox 面积。"""
    if "area_px" in meta.columns:
        return meta["area_px"].to_numpy(dtype=float)
    return ((meta["r1"] - meta["r0"] + 1) * (meta["c1"] - meta["c0"] + 1)).to_numpy(dtype=float)


def _candidate_pairs(
    prev: pd.DataFrame,
    cur: pd.DataFrame,
    offset: tuple[int, int],
    min_iou: float,
    max_cent_dist: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """候选对：bbox IoU ≥ min_iou **或** 质心距离 ≤ max_cent_dist（都在 prev 坐标系里算）。

    返回按 `(-iou, prev 行号, cur 行号)` 排好序的 (prev_idx, cur_idx, iou, cent_dist)；
    prev/cur 两个表都由调用方按 instance_uid 排过序，所以行号顺序就是 uid 顺序。
    """
    dy, dx = float(offset[0]), float(offset[1])
    prev_r0 = prev["r0"].to_numpy(dtype=float)
    prev_c0 = prev["c0"].to_numpy(dtype=float)
    prev_r1 = prev["r1"].to_numpy(dtype=float)
    prev_c1 = prev["c1"].to_numpy(dtype=float)
    # 下一帧的 bbox 先搬到上一帧坐标系再比较
    cur_r0 = cur["r0"].to_numpy(dtype=float) + dy
    cur_c0 = cur["c0"].to_numpy(dtype=float) + dx
    cur_r1 = cur["r1"].to_numpy(dtype=float) + dy
    cur_c1 = cur["c1"].to_numpy(dtype=float) + dx

    # 闭区间 bbox（Stage 1 的 r1/c1 是含端点的），所以交叠长度要 +1
    inter_h = np.clip(
        np.minimum(prev_r1[:, None], cur_r1[None, :]) - np.maximum(prev_r0[:, None], cur_r0[None, :]) + 1.0, 0.0, None
    )
    inter_w = np.clip(
        np.minimum(prev_c1[:, None], cur_c1[None, :]) - np.maximum(prev_c0[:, None], cur_c0[None, :]) + 1.0, 0.0, None
    )
    inter = inter_h * inter_w
    area_prev = (prev_r1 - prev_r0 + 1.0) * (prev_c1 - prev_c0 + 1.0)
    area_cur = (cur_r1 - cur_r0 + 1.0) * (cur_c1 - cur_c0 + 1.0)
    union = area_prev[:, None] + area_cur[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    row_delta = prev["centroid_row"].to_numpy(dtype=float)[:, None] - (
        cur["centroid_row"].to_numpy(dtype=float)[None, :] + dy
    )
    col_delta = prev["centroid_col"].to_numpy(dtype=float)[:, None] - (
        cur["centroid_col"].to_numpy(dtype=float)[None, :] + dx
    )
    cent_dist = np.hypot(row_delta, col_delta)

    is_candidate = (iou >= float(min_iou)) | (cent_dist <= float(max_cent_dist))
    prev_idx, cur_idx = np.nonzero(is_candidate)
    iou_values = iou[prev_idx, cur_idx]
    dist_values = cent_dist[prev_idx, cur_idx]
    # 贪心的排序契约：按 IoU 降序；并列时依次按 prev_uid、cur_uid（= 行号）定序，绝不依赖输入顺序
    order = np.lexsort((cur_idx, prev_idx, -iou_values))
    return prev_idx[order], cur_idx[order], iou_values[order], dist_values[order]


def _match_frames_with_stats(
    prev_meta: pd.DataFrame,
    cur_meta: pd.DataFrame,
    offset: tuple[int, int],
    min_iou: float,
    max_cent_dist: float,
) -> tuple[pd.DataFrame, dict]:
    """`match_frames` 的实现 + 逐帧对统计（build_tracks 要写进 QC）。"""
    _validate_frame_meta(prev_meta, "上一帧")
    _validate_frame_meta(cur_meta, "下一帧")
    # 行序无关的前提：先按 uid 排序，之后的贪心与并列定序都只依赖 uid
    prev = prev_meta.sort_values("instance_uid", kind="mergesort").reset_index(drop=True)
    cur = cur_meta.sort_values("instance_uid", kind="mergesort").reset_index(drop=True)
    prev_uid = prev["instance_uid"].astype(str).to_numpy()
    cur_uid = cur["instance_uid"].astype(str).to_numpy()
    stats: dict = {
        "n_prev": len(prev),
        "n_cur": len(cur),
        "n_candidates": 0,
        "n_one_to_one": 0,
        "n_merge": 0,
        "n_merge_targets": 0,
        "n_split": 0,
        "n_split_parents": 0,
        "n_prev_unmatched": 0,
        "n_cur_unmatched": 0,
        "prev_link_rate": 0.0,
        "cur_link_rate": 0.0,
        "offset": [int(offset[0]), int(offset[1])],
    }
    if len(prev) == 0 or len(cur) == 0:
        stats["n_prev_unmatched"] = len(prev)
        stats["n_cur_unmatched"] = len(cur)
        return _empty_match_table(), stats

    prev_idx, cur_idx, iou_values, dist_values = _candidate_pairs(prev, cur, offset, min_iou, max_cent_dist)
    stats["n_candidates"] = len(prev_idx)

    def _record(k: int, event: str) -> dict:
        """把第 k 个候选对打包成一条匹配记录（event 为 one_to_one / merge 等），键名即落盘列名。"""
        return {
            "prev_idx": int(prev_idx[k]),
            "cur_idx": int(cur_idx[k]),
            "iou": float(iou_values[k]),
            "cent_dist": float(dist_values[k]),
            "event": event,
        }

    # —— 第一轮：一对一贪心（候选对已按 (-iou, prev_uid, cur_uid) 排好，先到先得）——
    used_prev = np.zeros(len(prev), dtype=bool)
    used_cur = np.zeros(len(cur), dtype=bool)
    records: list[dict] = []
    one_to_one_by_cur: dict[int, dict] = {}
    for k in range(len(prev_idx)):
        p, c = int(prev_idx[k]), int(cur_idx[k])
        if used_prev[p] or used_cur[c]:
            continue
        used_prev[p] = True
        used_cur[c] = True
        record = _record(k, "one_to_one")
        records.append(record)
        one_to_one_by_cur[c] = record

    # —— 第二轮：残余多对一 → merge ——
    # 贪心跑完后，任何候选对至少有一端已被占用，所以"没匹配上的上一帧实例"的候选下一帧实例
    # 一定已经名花有主 —— 这正是"多对一"的残余：多颗上一帧实例都指向同一颗下一帧实例。
    best_merge: dict[int, int] = {}
    for k in range(len(prev_idx)):
        p, c = int(prev_idx[k]), int(cur_idx[k])
        if not used_prev[p] and used_cur[c]:
            # 已按 (-iou, prev_uid, cur_uid) 排序 → 该 prev 首次出现的那条就是它的最佳候选
            best_merge.setdefault(p, k)
    merge_by_cur: dict[int, list[int]] = {}
    for p, k in best_merge.items():
        merge_by_cur.setdefault(int(cur_idx[k]), []).append(p)
    if merge_by_cur:
        area = _bbox_area(prev)

        # 幸存父实例取面积最大者、同面积取 uid 小者：合并后的实例形态主要由大的那颗决定
        def _parent_key(index: int) -> tuple[float, str]:
            """幸存父实例的排序键：面积降序、同面积取 uid 升序（合并后形态主要由面积大的那颗决定）。"""
            return (-float(area[index]), str(prev_uid[index]))

        pair_index: dict[tuple[int, int], int] = {
            (int(prev_idx[k]), int(cur_idx[k])): k for k in range(len(prev_idx)) if int(cur_idx[k]) in merge_by_cur
        }
        for c, absorbed in sorted(merge_by_cur.items()):
            one_to_one = one_to_one_by_cur[c]
            candidates = [one_to_one["prev_idx"], *absorbed]
            parent = min(candidates, key=_parent_key)
            if parent != one_to_one["prev_idx"]:
                # 贪心赢家面积不是最大：让面积最大的那颗延续轨迹，原赢家降级为被吸收
                one_to_one["event"] = "merge"
                records.append(_record(pair_index[(parent, c)], "one_to_one"))
            for p in absorbed:
                if p != parent:
                    records.append(_record(pair_index[(p, c)], "merge"))

    # —— 第三轮：残余一对多 → split ——
    # 本轮所有候选对都是"上一帧已占用、下一帧还没着落"：上一帧实例已经有一对一的继承人，
    # 同时又冒出新的下一帧实例（新生成）。按 (-iou, prev_uid, cur_uid) 取每个新实例的最佳父实例。
    best_split: dict[int, int] = {}
    for k in range(len(prev_idx)):
        p, c = int(prev_idx[k]), int(cur_idx[k])
        if used_prev[p] and not used_cur[c]:
            best_split.setdefault(c, k)
    for c in sorted(best_split):
        records.append(_record(best_split[c], "split"))

    # —— 汇总成表：同一 cur 先一对一、再被吸收、最后分裂，组内按 prev_uid 排序 ——
    rows = [
        {
            "prev_uid": prev_uid[record["prev_idx"]],
            "cur_uid": cur_uid[record["cur_idx"]],
            "iou": record["iou"],
            "cent_dist": record["cent_dist"],
            "event": record["event"],
        }
        for record in records
    ]
    table = pd.DataFrame(rows, columns=list(_MATCH_COLUMNS))
    table = (
        table.assign(_rank=table["event"].map(_EVENT_RANK))
        .sort_values(["cur_uid", "_rank", "prev_uid"], kind="mergesort")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )

    n_one_to_one = int((table["event"] == "one_to_one").sum())
    stats.update(
        {
            "n_one_to_one": n_one_to_one,
            "n_merge": int((table["event"] == "merge").sum()),
            "n_merge_targets": len(merge_by_cur),
            "n_split": int((table["event"] == "split").sum()),
            "n_split_parents": len({int(prev_idx[k]) for k in best_split.values()}),
            "n_prev_unmatched": int(len(prev) - len(set(table["prev_uid"]))),
            "n_cur_unmatched": int(len(cur) - len(set(table["cur_uid"]))),
            # "匹配率"只算真正延续了轨迹的一对一（merge/split 的附加行不延续）
            "prev_link_rate": float(n_one_to_one / len(prev)),
            "cur_link_rate": float(n_one_to_one / len(cur)),
        }
    )
    return table, stats


def match_frames(
    prev_meta: pd.DataFrame,
    cur_meta: pd.DataFrame,
    offset: tuple[int, int],
    min_iou: float,
    max_cent_dist: float,
) -> pd.DataFrame:
    """把上一帧与下一帧的实例配起来，返回 `prev_uid, cur_uid, iou, cent_dist, event`。

    `offset` 是加到下一帧坐标上、把它搬进上一帧坐标系的 `(dy, dx)`（`register_frame` 的返回值）。
    `event` 只有三种：`one_to_one`（唯一会延续轨迹的行）、`merge`（多颗上一帧实例被吸收进同一颗
    下一帧实例）、`split`（上一帧实例另外冒出一颗新实例）。merge/split 行仍带着两端 uid，
    方便 QC 与人工核对"谁并进了谁、谁从谁分裂出来"，但 `assign_tracks` 不会用它们连轨迹。
    """
    table, _stats = _match_frames_with_stats(prev_meta, cur_meta, offset, min_iou, max_cent_dist)
    return table


# ---------------------------------------------------------------- SPOT uid


def _require_int(value, field: str) -> int:
    """把 uid 字段转成 int；非整数（如 't-0'、1.5、NaN）直接抛中文 `ValueError`。"""
    if isinstance(value, (bool, np.bool_)):
        # 契约：所有非法值统一抛 ValueError，调用方只需捕获一种异常（不必同时接 TypeError）
        raise ValueError(  # noqa: TRY004
            f"spot_uid 的 {field} 必须是整数，不能是布尔值（拿到 {value!r}）"
        )
    if isinstance(value, (float, np.floating)):
        if not float(value).is_integer():
            raise ValueError(f"spot_uid 的 {field} 必须是整数，拿到 {value!r}（SPOT 会用 int() 解析最后一段）")
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(
            f"spot_uid 的 {field} 必须是整数，拿到 {value!r}（SPOT 内部用 int(uid.split('_')[-1]) 取帧号，"
            "非整数会直接崩）"
        ) from None


def spot_uid_from(meta: pd.DataFrame) -> np.ndarray:
    """由 `fov / track_id / frame_idx` 列拼出 SPOT 专用 uid，并逐条校验格式。

    SPOT 的 `construct_obj_traj_from_uniq_obj_ids` 写死了 `split('_')` 且把**最后一段** `int()` 当帧号，
    所以这里保证：恰好 3 段、帧号在最后且是整数、fov 里不含 `_`（多一段就会串位）、非负。
    任一违反都抛带上下文的 `ValueError` —— 让错误在这里炸，而不是在 SPOT 解析轨迹时炸。
    """
    required = ("fov", "track_id", "frame_idx")
    missing = [name for name in required if name not in meta.columns]
    if missing:
        raise ValueError(f"spot_uid_from 需要 fov/track_id/frame_idx 三列，缺少 {missing}")

    uids: list[str] = []
    for fov, track_id, frame_idx in zip(meta["fov"], meta["track_id"], meta["frame_idx"]):
        fov_text = str(fov)
        if not fov_text or "_" in fov_text:
            raise ValueError(f"fov 不能是空串或含下划线（拿到 {fov!r}）：SPOT 按 '_' 切 uid，多一段就会串位")
        track = _require_int(track_id, "track_id")
        frame = _require_int(frame_idx, "frame_idx")
        if track < 0 or frame < 0:
            raise ValueError(f"spot_uid 的 track_id/frame_idx 必须非负（拿到 track_id={track}, frame_idx={frame}）")
        uid = f"{fov_text}_{track}_{frame}"
        parts = uid.split("_")
        if len(parts) != 3 or not parts[-1].isdigit():
            raise ValueError(f"拼出的 spot_uid {uid!r} 不是 'fov_track_frame' 三段式（{len(parts)} 段）")
        uids.append(uid)
    return np.asarray(uids, dtype=object)


# ---------------------------------------------------------------- 建轨


def _validate_all_meta(all_meta: pd.DataFrame) -> None:
    """校验汇总 meta：必需列齐全、非空、instance_uid 唯一。"""
    missing = [name for name in _META_REQUIRED if name not in all_meta.columns]
    if missing:
        raise ValueError(f"Stage 6 的 meta 缺少列 {missing}：需要 Stage 1 all_instances.csv 的完整列")
    if all_meta.empty:
        raise RuntimeError("Stage 6 的 meta 为空：没有任何实例可以建 track（先确认 Stage 1 跑出了实例）")
    duplicated = all_meta["instance_uid"][all_meta["instance_uid"].duplicated()]
    if len(duplicated):
        raise RuntimeError(
            f"instance_uid 重复 {len(duplicated)} 行（例如 {sorted(set(duplicated.astype(str)))[:5]}）："
            "建轨前必须保证每个实例只有一行"
        )


def _validate_match_table(table: pd.DataFrame) -> None:
    """校验匹配表的列与 event 取值（防止手改/旧版本的 CSV 悄悄混进来）。"""
    missing = [name for name in _MATCH_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(f"匹配表缺少列 {missing}：只接受 match_frames 的输出（{list(_MATCH_COLUMNS)}）")
    unknown = sorted(set(table["event"].astype(str)) - set(_EVENT_VALUES))
    if unknown:
        raise ValueError(f"匹配表出现未知 event {unknown}：只允许 {list(_EVENT_VALUES)}")


def assign_tracks(matches: list[pd.DataFrame], all_meta: pd.DataFrame) -> pd.DataFrame:
    """给每个实例分配整数 `track_id`，并拼出 SPOT 专用 `spot_uid`。

    轨迹靠并查集串起来：**只有 `one_to_one` 行会合并两个实例的集合**，所以
    frame0→frame1→frame2 的链自然变成同一条 track；merge 的附加行不连（被吸收的实例自己
    成为长度 1 的孤立 track）、split 的附加行也不连（新生成的实例另起一条 track）。
    这正是"同一 (fov, track_id, frame_idx) 不许重复"的落地方式，函数末尾还会硬校验 uid 唯一。

    `track_id` 的编号是确定性的：先按视野名，再按该集合里最早的 (frame_idx, instance_uid) 排序，
    全局连续编号（不同视野不重号，`groupby("track_id")` 才等于"一条轨迹"）。
    """
    _validate_all_meta(all_meta)
    uids = all_meta["instance_uid"].astype(str).to_numpy()
    position = {uid: index for index, uid in enumerate(uids)}
    parent = np.arange(len(uids))

    def find(index: int) -> int:
        """并查集查找 + 路径减半：返回 index 所在集合的根，即该轨迹的合并代表。"""
        while parent[index] != index:
            parent[index] = parent[parent[index]]  # 路径减半
            index = int(parent[index])
        return index

    def union(a: int, b: int) -> None:
        """并查集合并：把两个根里编号大的挂到编号小的下面，保证同一批 match 表合并结果确定。"""
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)  # 小根当根，保证确定性

    for table in matches:
        _validate_match_table(table)
        links = table[table["event"] == "one_to_one"]
        for prev_uid, cur_uid in zip(links["prev_uid"], links["cur_uid"]):
            if pd.isna(prev_uid) or pd.isna(cur_uid):
                continue  # 允许用 NaN 表示"这一端没有对应实例"的附加行（本模块不产生，但别炸）
            left = position.get(str(prev_uid))
            right = position.get(str(cur_uid))
            if left is None or right is None:
                missing = str(prev_uid) if left is None else str(cur_uid)
                raise RuntimeError(f"匹配表里的实例 {missing} 不在 meta 里：建轨要求两端都能查到实例")
            union(left, right)

    # 每个并查集集合的排序键 = (fov, 最早帧号, 最早帧里的最小 uid)，再校验不会跨视野
    root_first: dict[int, tuple[str, int, str]] = {}
    root_fov: dict[int, str] = {}
    fov_values = all_meta["fov"].astype(str).to_numpy()
    frame_values = all_meta["frame_idx"].astype(int).to_numpy()
    for index, uid in enumerate(uids):
        root = find(index)
        fov = fov_values[index]
        if root_fov.setdefault(root, fov) != fov:
            raise RuntimeError(
                f"track 跨越了视野 {root_fov[root]} 与 {fov}：跨天匹配必须限制在同一 fov 内（uid={uid}）"
            )
        key = (fov, int(frame_values[index]), uid)
        if root not in root_first or key < root_first[root]:
            root_first[root] = key

    ordered_roots = sorted(root_first, key=lambda root: root_first[root])
    track_id_of_root = {root: track_id for track_id, root in enumerate(ordered_roots)}
    track_ids = np.asarray([track_id_of_root[find(index)] for index in range(len(uids))], dtype=np.int64)

    tracks = pd.DataFrame(
        {
            "instance_uid": uids,
            "track_id": track_ids,
            "fov": fov_values,
            "frame_idx": frame_values,
            "time_hours": all_meta["time_hours"].to_numpy(dtype=float),
            "day": all_meta["day"].astype(str).to_numpy(),
        }
    )
    if "label" in all_meta.columns:
        tracks["label"] = all_meta["label"].to_numpy()
    tracks["spot_uid"] = spot_uid_from(tracks)

    duplicated = tracks["spot_uid"][tracks["spot_uid"].duplicated()]
    if len(duplicated):
        examples = sorted({str(value) for value in duplicated})[:5]
        raise RuntimeError(
            f"spot_uid 重复 {len(duplicated)} 行（例如 {examples}）：同一 (fov, track_id, frame_idx) 出现两次，"
            "SPOT 的 construct_obj_traj_from_uniq_obj_ids 会把它解析成两段轨迹"
        )

    columns = [name for name in _TRACK_COLUMNS if name in tracks.columns]
    return tracks.loc[:, columns].reset_index(drop=True)


# ---------------------------------------------------------------- 汇总建轨


def _frame_identity(key: str, table: pd.DataFrame) -> tuple[str, str, int]:
    """从一个 meta 帧（或空 meta 的 key）取出 (day, fov, frame_idx)。"""
    if len(table):
        row = table.iloc[0]
        return str(row["day"]), str(row["fov"]), int(row["frame_idx"])
    day, separator, fov = str(key).partition("/")
    if not separator or day not in cfg.DAYS:
        raise RuntimeError(
            f"无法从 key={key!r} 推断 day/fov/frame_idx：空 meta 的 key 必须写成 'day0/A1' 这种 'day/fov' 形式"
        )
    return day, fov, int(cfg.DAYS[day])


def _track_length_stats(tracks: pd.DataFrame) -> dict:
    """每个视野的 track 数、长度 ≥2 的 track 数与长度直方图（Spec §6-Stage6-6 的 QC）。"""
    summary: dict[str, dict] = {}
    for fov, sub in tracks.groupby("fov", sort=True):
        sizes = sub.groupby("track_id").size()
        histogram = {str(int(length)): int(count) for length, count in sizes.value_counts().sort_index().items()}
        summary[str(fov)] = {
            "n_instances": len(sub),
            "n_tracks": len(sizes),
            "n_tracks_len_ge2": int((sizes >= 2).sum()),
            "track_length_histogram": histogram,
        }
    return summary


def build_tracks(
    metas: dict[str, pd.DataFrame],
    masks: dict[str, np.ndarray] | None = None,
    qc: dict | None = None,
    align_offsets: dict[str, tuple[int, int]] | None = None,
    matches_out: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """把各 (day, fov) 的 meta 串成 track 表；`masks` 缺省时按 `_read_binary` 从磁盘读标签图。

    `metas` 的 key 约定为 `"day0/A1"` 形式（只用于报错上下文与空 meta 时推断 day/fov），
    每个 value 必须是**同一帧**的 meta（同一 (fov, frame_idx) 只允许出现一次）。帧号必须在同一视野内
    从 0 开始连续（SPOT 用单一标量 `d_Frame` 判连续性，缺帧会被拆成几段轨迹），否则直接报错。
    `qc` 给了就原地填好 match_qc.json 的内容（配准偏移/告警、每个帧对的匹配统计、track 长度直方图）。
    `align_offsets` 是 Stage 1 的逐 (day, fov) 对齐偏移 `"day0/A1" → (dy, dx)`：配准前先用它把原始掩码
    搬到 meta 所在的坐标系（缺省 (0, 0)）。`matches_out` 给了就把它填成带 `fov/pair` 列的逐帧对匹配表
    （run_stage6 用它落盘 matches.csv，便于事后审计合并/分裂的归属）。
    """
    entries: list[tuple[str, str, str, int, pd.DataFrame]] = []  # (day, fov, key, frame_idx, meta)
    for key, table in metas.items():
        day, fov, frame_idx = _frame_identity(key, table)
        if len(table) and (table["fov"].nunique() != 1 or table["frame_idx"].nunique() != 1):
            raise RuntimeError(
                f"meta {key} 里混了多个 (fov, frame_idx)（fov={sorted(table['fov'].unique())}, "
                f"frame_idx={sorted(table['frame_idx'].unique())}）：一个 meta 必须只对应一帧"
            )
        entries.append((day, fov, str(key), frame_idx, table))
    if not entries:
        raise RuntimeError("Stage 6 没有输入 meta：build_tracks 至少需要一个 (day, fov) 的实例表")

    # 每个 (fov, frame_idx) 只能有一份 meta；重复说明调用方把同一帧传了两次
    seen: dict[tuple[str, int], str] = {}
    for _day, fov, key, frame_idx, _table in entries:
        if (fov, frame_idx) in seen:
            raise RuntimeError(
                f"视野 {fov} 的 frame_idx={frame_idx} 出现了两份 meta（{seen[(fov, frame_idx)]} 与 {key}）："
                "同一帧只能有一份"
            )
        seen[(fov, frame_idx)] = key

    all_meta = (
        pd.concat([entry[4] for entry in entries], ignore_index=True) if any(len(e[4]) for e in entries) else None
    )
    if all_meta is None or all_meta.empty:
        raise RuntimeError(f"Stage 6 的 {len(entries)} 个 (day, fov) 都没有实例：没有任何 track 可建")
    _validate_all_meta(all_meta)

    if masks is None:
        masks = {key: _read_binary(day, fov) for day, fov, key, _frame, _table in entries}
    else:
        missing_masks = [key for _day, _fov, key, _frame, _table in entries if key not in masks]
        if missing_masks:
            raise RuntimeError(f"masks 缺少 {missing_masks} 的掩码：配准需要每个 (day, fov) 的标签二值图")

    # 参考系 = 每个视野的第一帧；其余帧都先配准到参考系，再复合出相邻帧对之间的偏移
    by_fov: dict[str, dict[int, tuple[str, pd.DataFrame]]] = {}
    for _day, fov, key, frame_idx, table in entries:
        by_fov.setdefault(fov, {})[frame_idx] = (key, table)

    tables: list[pd.DataFrame] = []
    fov_qc: dict[str, dict] = {}
    all_warnings: list[str] = []
    align_offsets = dict(align_offsets or {})
    for fov in sorted(by_fov):
        frames = sorted(by_fov[fov])
        # 帧号必须从 0 开始连续：SPOT 的 construct_obj_traj_from_uniq_obj_ids 用单一标量 d_Frame
        # 判连续性，缺一帧（例如只有 day0 与 day5 → 0 和 2）就把它当成两段轨迹，
        # 而本阶段的"至少一条长度 ≥2 的 track"照样能成立 —— 两边会对不上，所以建轨前先拦住。
        expected = list(range(frames[0], frames[0] + len(frames)))
        if frames[0] != 0 or frames != expected:
            missing = sorted(set(expected) - set(frames)) if frames[0] == 0 else [0]
            raise RuntimeError(
                f"视野 {fov} 的帧号 {frames} 不连续（缺少帧 {missing}）：SPOT 的 "
                "construct_obj_traj_from_uniq_obj_ids 用单一标量 d_Frame 判连续性，缺帧会被拆成几段轨迹；"
                "请把帧轴编码成从 0 开始的连续整数（cfg.DAYS 应为 0/1/2）再重跑"
            )
        reference = frames[0]
        reference_key = by_fov[fov][reference][0]
        warnings: list[str] = []
        warning_kinds: dict[str, int] = {}
        frame_align = {frame: align_offsets.get(by_fov[fov][frame][0], (0, 0)) for frame in frames}
        offsets: dict[int, tuple[int, int]] = {reference: (0, 0)}
        for frame_idx in frames[1:]:
            offsets[frame_idx] = register_frame(
                _aligned_mask(masks[reference_key], frame_align[reference]),
                _aligned_mask(masks[by_fov[fov][frame_idx][0]], frame_align[frame_idx]),
                int(cfg.ALIGN_MAX_SHIFT_PX),
                warnings_out=warnings,
                kinds_out=warning_kinds,
            )
        pair_offsets: dict[str, list[int]] = {}
        pair_stats: dict[str, dict] = {}
        for prev_frame, cur_frame in pairwise(frames):
            prev_offset = offsets[prev_frame]
            cur_offset = offsets[cur_frame]
            # 同一参考系下的平移可以相减：cur→prev = (cur→ref) - (prev→ref)
            pair_offset = (cur_offset[0] - prev_offset[0], cur_offset[1] - prev_offset[1])
            pair_name = f"{prev_frame}->{cur_frame}"
            pair_offsets[pair_name] = [int(pair_offset[0]), int(pair_offset[1])]
            # 复合偏移可能超出单次配准的上限：两个各自 ±8 px 内接受的对，相减可以是 16 px。
            # 不静默裁掉（裁了匹配会错位），但必须留告警 + 逐帧对标记，让 QC 里看得见。
            limit = int(cfg.ALIGN_MAX_SHIFT_PX)
            exceeds = max(abs(pair_offset[0]), abs(pair_offset[1])) > limit
            if exceeds:
                _record_warning(
                    warnings,
                    f"视野 {fov} 的帧对 {pair_name} 复合偏移 (dy={pair_offset[0]}, dx={pair_offset[1]}) "
                    f"超过 ±{limit} px 上限（两端各自在限内，但相减后越界）：匹配仍按此偏移进行，请人工核对配准",
                    WARNING_KIND_COMPOUND_OFFSET_EXCEEDS_LIMIT,
                    warning_kinds,
                )
            table, stats = _match_frames_with_stats(
                by_fov[fov][prev_frame][1],
                by_fov[fov][cur_frame][1],
                pair_offset,
                float(cfg.MIN_BBOX_IOU),
                float(cfg.MAX_CENT_DIST_PX),
            )
            tables.append(table)
            stats["n_match_rows"] = len(table)
            stats["offset_exceeds_limit"] = bool(exceeds)
            pair_stats[pair_name] = stats
            if matches_out is not None:
                matches_out.append(table.assign(fov=fov, pair=pair_name).reindex(columns=list(_MATCH_OUTPUT_COLUMNS)))
        all_warnings.extend(warnings)
        fov_qc[fov] = {
            "registration": {
                "reference_frame": int(reference),
                "offset_to_reference": {
                    str(frame): [int(offsets[frame][0]), int(offsets[frame][1])] for frame in frames
                },
                "align_offsets": {
                    str(frame): [int(frame_align[frame][0]), int(frame_align[frame][1])] for frame in frames
                },
                "pair_offsets": pair_offsets,
                "warnings": list(warnings),
                # 与 warnings 同源但机器可读：类型 → 条数（报告只把 offset_exceeds_limit 当"被拒的估计"）
                "warning_kinds": dict(warning_kinds),
            },
            "pairs": pair_stats,
        }

    tracks = assign_tracks(tables, all_meta)
    length_stats = _track_length_stats(tracks)
    for fov, stats in length_stats.items():
        fov_qc[fov].update(stats)

    if qc is not None:
        totals = {
            "n_instances": len(tracks),
            "n_tracks": int(tracks["track_id"].nunique()),
            "n_tracks_len_ge2": int((tracks.groupby("track_id").size() >= 2).sum()),
            "n_one_to_one": int(
                sum(stats["n_one_to_one"] for fov in fov_qc for stats in fov_qc[fov]["pairs"].values())
            ),
            "n_merge": int(sum(stats["n_merge"] for fov in fov_qc for stats in fov_qc[fov]["pairs"].values())),
            "n_split": int(sum(stats["n_split"] for fov in fov_qc for stats in fov_qc[fov]["pairs"].values())),
            "n_prev_unmatched": int(
                sum(stats["n_prev_unmatched"] for fov in fov_qc for stats in fov_qc[fov]["pairs"].values())
            ),
            "n_cur_unmatched": int(
                sum(stats["n_cur_unmatched"] for fov in fov_qc for stats in fov_qc[fov]["pairs"].values())
            ),
            "n_match_rows": int(
                sum(stats["n_match_rows"] for fov in fov_qc for stats in fov_qc[fov]["pairs"].values())
            ),
            "n_pairs_offset_exceeding_limit": int(
                sum(stats["offset_exceeds_limit"] for fov in fov_qc for stats in fov_qc[fov]["pairs"].values())
            ),
        }
        qc.clear()
        qc.update(
            {
                "config": {
                    "min_bbox_iou": float(cfg.MIN_BBOX_IOU),
                    "max_cent_dist_px": float(cfg.MAX_CENT_DIST_PX),
                    "align_max_shift_px": int(cfg.ALIGN_MAX_SHIFT_PX),
                    "frame_axis": {day: int(frame) for day, frame in cfg.DAYS.items()},
                    "time_hours": {day: float(hours) for day, hours in cfg.TIME_HOURS.items()},
                },
                "fovs": fov_qc,
                "totals": totals,
                "warnings": all_warnings,
                # "掩码本来就对齐吗"的独立核验：与相位相关估计无关，只扫零偏移附近的 IoU
                "registration_geometry": check_registration_geometry(
                    masks,
                    {fov: {frame: by_fov[fov][frame][0] for frame in by_fov[fov]} for fov in by_fov},
                    align_offsets,
                ),
            }
        )
    return tracks


# ---------------------------------------------------------------- Stage 6 入口


def _load_align_offsets(qc_path: Path) -> tuple[dict[str, tuple[int, int]], str | None]:
    """从 Stage 1 的 stage1_qc.json 读逐 (day, fov) 的对齐偏移。

    配准必须用"对齐后的标签图"（meta 就是从那上面算出来的），所以这里把 `align_dy/align_dx`
    拿给 `build_tracks`。文件缺失时不硬错（配准仍会跑，只是可能多出 `align_ref - align_cur` 的偏差），
    但返回一条告警让 match_qc.json 里看得见。
    """
    if not qc_path.exists():
        return {}, (
            f"找不到 Stage 1 的 {inst._QC_FILE}（{qc_path}）：不知道逐 (day, fov) 的对齐偏移，"
            "配准按 align_offset=(0, 0) 处理，对齐非零时帧对偏移会有偏差"
        )
    data = json.loads(qc_path.read_text(encoding="utf-8"))
    return {
        str(key): (int(entry.get("align_dy", 0)), int(entry.get("align_dx", 0))) for key, entry in data.items()
    }, None


def _empty_matches_table() -> pd.DataFrame:
    """空 matches.csv（列名与顺序与正常表一致，下游不用特判）。"""
    return pd.DataFrame(
        {
            "fov": pd.Series(dtype=object),
            "pair": pd.Series(dtype=object),
            "prev_uid": pd.Series(dtype=object),
            "cur_uid": pd.Series(dtype=object),
            "iou": pd.Series(dtype=float),
            "cent_dist": pd.Series(dtype=float),
            "event": pd.Series(dtype=object),
        }
    )


def run_stage6() -> None:
    """跑完 Stage 6：读 Stage 1 汇总表 → 建 track → 落盘 tracks.csv / matches.csv / match_qc.json。

    硬断言：至少存在一条长度 ≥ 2 的 track。否则说明帧轴编码或匹配整体失效
    （Spec §9 风险表：track 全退化成长度 1 是时间编码错误，不是数据问题），直接中文报错停下。
    断言放在落盘**之前**，避免留下"看起来跑成功了"的半套产物。

    `matches.csv` 是逐帧对的完整匹配表（带 `fov` / `pair` 列）：候选规则（IoU 或质心距离）
    是整个阶段最启发式的一环，"谁并进了谁"、合并父实例的选择、分裂父实例必须能从磁盘直接审，
    而不能靠重跑本阶段反推。match_qc.json 里额外记录它的行数与列名。
    """
    instances_dir = Path(cfg.OUTPUT_ROOT) / _INSTANCES_SUBDIR
    instances_path = instances_dir / _ALL_INSTANCES_FILE
    if not instances_path.exists():
        raise RuntimeError(f"找不到 Stage 1 汇总表 {instances_path}：请先跑 run_stage1（Stage 1）")
    all_meta = pd.read_csv(instances_path)
    metas = {f"{day}/{fov}": table for (day, fov), table in all_meta.groupby(["day", "fov"], sort=True)}
    if not metas:
        raise RuntimeError(f"{instances_path} 里没有任何 (day, fov) 的实例：先确认 Stage 1 的产物完整")

    align_offsets, align_warning = _load_align_offsets(instances_dir / inst._QC_FILE)
    qc: dict = {}
    matches: list[pd.DataFrame] = []
    tracks = build_tracks(metas, qc=qc, align_offsets=align_offsets, matches_out=matches)
    if align_warning is not None:
        qc["warnings"].append(align_warning)

    n_long = int((tracks.groupby("track_id").size() >= 2).sum())
    if n_long == 0:
        raise RuntimeError(
            f"Stage 6 的 {len(tracks)} 个实例（{len(metas)} 个 day/fov）全部是长度为 1 的 track："
            "帧轴编码（应为 0/1/2 连续）或跨天匹配（配准偏移/IoU 阈值）失效，先查这几处再重跑"
        )

    match_table = pd.concat(matches, ignore_index=True) if matches else _empty_matches_table()
    qc["match_table"] = {
        "file": _MATCHES_FILE,
        "n_rows": len(match_table),
        "columns": list(_MATCH_OUTPUT_COLUMNS),
    }

    outdir = Path(cfg.OUTPUT_ROOT) / _ANALYSIS_SUBDIR
    outdir.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(outdir / _TRACKS_FILE, index=False)
    match_table.to_csv(outdir / _MATCHES_FILE, index=False)
    (outdir / _QC_FILE).write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Stage6 完成：{len(tracks)} 个实例 → {qc['totals']['n_tracks']} 条 track，"
        f"其中长度 ≥2 的 {n_long} 条，匹配表 {len(match_table)} 行，产物 → {outdir}"
    )
