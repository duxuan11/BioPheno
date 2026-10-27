"""配准几何核验：回答"掩码本来就对齐，还是配准失败后退回零偏移"。

    uv run python scripts/check_registration_geometry.py

为什么要有这个脚本：Stage 6 的相位相关估计在本数据上全部超出 ±`ALIGN_MAX_SHIFT_PX` 被拒、退回 `(0, 0)`。
"复合偏移全为 0" 本身无法区分两种情况：

1. 掩码在零偏移下本来就已经对齐（退回零偏移无害）；
2. 配准锁到了错误的峰（退回零偏移是掩盖了失败）。

这个脚本不看相位相关：直接读 `config.DATA_ROOT` 下的标签图，按 Stage 1 的对齐偏移搬到 meta 坐标系，
然后对每个相邻帧对在**全分辨率**二值掩码上逐像素横扫 ±`ALIGN_MAX_SHIFT_PX` **像素**内的平移算 IoU，
记下零偏移 IoU、最佳 IoU、最佳偏移与 `aligned_at_zero`（零偏移 IoU 与最佳 IoU 的差不超过
`GEOMETRY_ALIGN_TOLERANCE`）。偏移与窗口都是像素（JSON 里的 `offset_unit="px"` /
`shift_step_px=1` 把这件事写死），可以与 `align_offsets` / `pair_offsets` 直接对比；
不要在下采样后的网格上平移，那会把偏移变成“网格步”（真实 12 px 会被报成 3）。

结果写进 `output/logs/registration_geometry_check.json`，报告只在 `source` 等于本脚本且带 `checked_at`
时才引用它（手写的"自证"JSON 会被当成未核验）。它**不改任何匹配结果**：`tracks.csv` / `matches.csv`
一个字节都不动，只是给已有的零偏移几何补一条独立证据。

Stage 6 自己也会算同一份诊断并写进 `analysis/match_qc.json` 的 `registration_geometry`（新产物优先）；
本脚本是给"不想重跑 Stage 6"的那条路准备的。
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# scripts/ 没有 __init__.py，直接跑脚本时把项目根目录加进 sys.path 才能 import biopheno
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from biopheno import config as cfg
from biopheno import instances as inst
from biopheno import report as report_mod
from biopheno import temporal as temp

# 写进核验 JSON 的 `source`：报告据此判断"这份核验是谁做的"（常量定义在 report，避免两处漂移）。
_SOURCE: str = report_mod.GEOMETRY_CHECK_TOOL

_INSTANCES_SUBDIR = "instances"


def _frames() -> dict[str, dict[int, str]]:
    """`{fov: {frame_idx: "day/fov"}}`：帧号来自 `config.DAYS`，与 Stage 6 的帧轴编码一致。

    用 `instances.available_frames()` 而不是 `DAYS × FOVS` 的笛卡尔积：命令行 `--images/--masks`
    给的输入清单不保证"每个视野每天都有图"，缺的格子不该被造出来当空掩码。
    """
    frames: dict[str, dict[int, str]] = {}
    for day, fov in inst.available_frames():
        frames.setdefault(str(fov), {})[int(cfg.DAYS[day])] = f"{day}/{fov}"
    return frames


def check(output_root: Path | None = None) -> dict:
    """算一遍配准几何核验并落盘 `logs/registration_geometry_check.json`，返回该 JSON 的内容。

    Parameters
    ----------
    output_root : Path | None
        `output/` 根；默认取 `config.OUTPUT_ROOT`（与其他阶段一致，在调用时读取）。
    """
    root = Path(cfg.OUTPUT_ROOT) if output_root is None else Path(output_root)
    qc_path = root / _INSTANCES_SUBDIR / inst._QC_FILE
    if not qc_path.exists():
        raise SystemExit(f"找不到 Stage 1 的 {qc_path}：不知道逐 (day, fov) 的对齐偏移，请先跑 Stage 1")
    align_offsets, align_warning = temp._load_align_offsets(qc_path)
    if align_warning is not None:
        print(f"警告：{align_warning}", file=sys.stderr)

    frames = _frames()
    masks: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    read_seconds: dict[str, float] = {}
    for by_frame in frames.values():
        for key in by_frame.values():
            day, _, fov_name = key.partition("/")
            start = time.perf_counter()
            masks[key] = temp._read_binary(day, fov_name)
            read_seconds[key] = round(time.perf_counter() - start, 3)
    t_read = time.perf_counter() - t0

    t1 = time.perf_counter()
    geometry = temp.check_registration_geometry(masks, frames, align_offsets)
    elapsed = time.perf_counter() - t1

    checked_at = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = {
        "source": _SOURCE,
        "checked_at": checked_at,
        **geometry,
        "align_offsets": {key: [int(value[0]), int(value[1])] for key, value in sorted(align_offsets.items())},
        "config": {
            "data_root": str(cfg.DATA_ROOT),
            "image_dir": str(cfg.IMAGE_DIR) if cfg.IMAGE_DIR is not None else None,
            "mask_dir": str(cfg.MASK_DIR) if cfg.MASK_DIR is not None else None,
            "output_root": str(root),
            "align_max_shift_px": int(cfg.ALIGN_MAX_SHIFT_PX),
            "days": dict(cfg.DAYS),
            "fovs": list(cfg.FOVS),
        },
        "inputs": {"stage1_qc": str(qc_path)},
        "read_seconds": read_seconds,
        "elapsed_seconds": {"read_masks": round(t_read, 3), "geometry": round(elapsed, 3)},
        "method": (
            "读 DATA_ROOT 下的标签图并二值化，按 Stage 1 的 align_offset 搬到 meta 坐标系；对每个相邻帧对在"
            f"全分辨率二值掩码上以 {geometry['shift_step_px']} px 为步长横扫 ±{geometry['max_shift_px']} px 内的"
            "平移算 IoU，记录零偏移 IoU、最佳 IoU、最佳偏移（单位 px）、以及 aligned_at_zero（零偏移与最佳 IoU 的"
            f"差不超过 {geometry['tolerance']}）。刻意不用相位相关，因此与 Stage 6 的估计失败无关；也不改任何匹配结果。"
        ),
    }
    out_path = root / report_mod.LOG_SUBDIR / report_mod.GEOMETRY_CHECK_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    zero = [entry["zero_offset_iou"] for pairs in geometry["pairs"].values() for entry in pairs.values()]
    best = [entry["best_iou"] for pairs in geometry["pairs"].values() for entry in pairs.values()]
    verdict = "全部零偏移已对齐" if geometry["all_aligned_at_zero"] else "存在未对齐的帧对"
    print(
        f"配准几何核验：{geometry['n_aligned_at_zero']}/{geometry['n_pairs']} 个帧对零偏移已对齐（{verdict}），"
        f"零偏移 IoU {min(zero):.3f}–{max(zero):.3f}、最佳 IoU {min(best):.3f}–{max(best):.3f}，"
        f"几何扫描耗时 {elapsed:.1f}s → {out_path}"
    )
    return payload


if __name__ == "__main__":
    import argparse

    from biopheno import cli

    _parser = argparse.ArgumentParser(
        prog="check_registration_geometry.py",
        description="配准几何核验（读 Stage1 的对齐偏移 + 掩码图）；输入目录可用 --images/--masks 指定",
    )
    cli.add_input_args(_parser)
    _args = _parser.parse_args()
    cli.configure_from_args(_parser, _args)
    _result = check(Path(_args.output) if _args.output is not None else None)
    sys.exit(0 if _result["all_aligned_at_zero"] else 1)
