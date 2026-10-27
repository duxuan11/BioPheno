"""UMAP 可复现性核验：重跑一次嵌入，与已落盘的 `analysis/umap_2d.csv` 逐字节/逐值比较。

    uv run python scripts/check_umap_repeat.py

做三件事：

1. 读 `output/preprocessed/X.npy`（Stage 3 的预处理矩阵）与 `output/analysis/kmeans_metrics.json` 里记录的
   UMAP 参数，并与 `config.UMAP_PARAMS` 对照——`landscape.embed` 实际用的正是后者；
2. 用**同一组参数**重跑一次 `landscape.embed`，按 Stage 5 完全相同的写法（`float_format="%.10g"`）写一份
   临时 CSV，与已落盘的 `analysis/umap_2d.csv` 比 md5，并逐值算最大绝对差；
3. 把两个 md5、最大绝对差、参数、时间戳写进 `output/logs/umap_repeat_check.json`，并打印结论。

为什么要有这个脚本：之前那份 `umap_repeat_check.json` 是手工写的（"两次跑出来一致"），仓库里没有任何
代码能重现或核对它，第一次运行的文件也没留——那就是**自证**。现在报告只在 JSON 的 `source` 等于本脚本
且带 `checked_at` 时才写"已核验"，手写的 JSON 会被当成"未核验"。

核验失败（md5 不同/点数不同）也照样落盘：报告会照实写"不一致"，脚本以非零码退出。
"""

import hashlib
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# scripts/ 没有 __init__.py，直接跑脚本时把项目根目录加进 sys.path 才能 import biopheno
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from biopheno import config as cfg
from biopheno import landscape
from biopheno import report as report_mod

# 写进核验 JSON 的 `source`：报告据此判断"这份核验是谁做的"（常量定义在 report，避免两处漂移）。
_SOURCE: str = report_mod.REPEAT_CHECK_TOOL

_PREPROCESSED_SUBDIR = "preprocessed"
_ANALYSIS_SUBDIR = "analysis"
_X_FILE = "X.npy"
_UMAP_FILE = "umap_2d.csv"
_METRICS_FILE = "kmeans_metrics.json"
_UMAP_COLUMNS = ("instance_uid", "umap_1", "umap_2")


def _md5(path: Path) -> str:
    """文件 md5（核验的是**字节**，所以 CSV 的浮点格式也必须与 Stage 5 一致）。"""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _shipped_frame(path: Path) -> pd.DataFrame:
    """读已落盘的 `umap_2d.csv`（行序即 Stage 5 的实例序，用来重写一份同格式 CSV 比 md5）。"""
    if not path.exists():
        raise SystemExit(
            f"找不到已落盘的 {path}：没有可比对的基准。请先跑 Stage 5"
            '（uv run python -c "from biopheno.landscape import run_stage5; run_stage5()"）'
        )
    frame = pd.read_csv(path)
    missing = [name for name in _UMAP_COLUMNS if name not in frame.columns]
    if missing:
        raise SystemExit(f"{path} 缺少列 {missing}：产物格式与 Stage 5 不符，请重跑 Stage 5")
    return frame


def check(output_root: Path | None = None) -> dict:
    """重跑一次 UMAP，把结果写进 `logs/umap_repeat_check.json` 并返回该 JSON 的内容。

    Parameters
    ----------
    output_root : Path | None
        `output/` 根；默认取 `config.OUTPUT_ROOT`（与各阶段一致，在调用时读取）。
    """
    root = Path(cfg.OUTPUT_ROOT) if output_root is None else Path(output_root)
    x_path = root / _PREPROCESSED_SUBDIR / _X_FILE
    metrics_path = root / _ANALYSIS_SUBDIR / _METRICS_FILE
    shipped_path = root / _ANALYSIS_SUBDIR / _UMAP_FILE
    for path, what in ((x_path, "Stage 3 的预处理矩阵"), (metrics_path, "Stage 5 的 K/UMAP 指标")):
        if not path.exists():
            raise SystemExit(f"找不到 {what} {path}：请先按顺序跑完 Stage 3 与 Stage 5")
    X = np.load(x_path)
    if X.ndim != 2:
        raise SystemExit(f"{x_path} 的形状是 {X.shape}，不是二维矩阵：请重跑 Stage 3")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    recorded_params = dict(metrics.get("umap_params", {}))
    # embed() 内部用的是 cfg.UMAP_PARAMS，所以这里必须核对"记录下来的参数 == 实际要用的参数"
    params_match = recorded_params == dict(cfg.UMAP_PARAMS)
    if not params_match:
        print(
            f"警告：kmeans_metrics.json 记的 UMAP 参数 {recorded_params} 与 config.UMAP_PARAMS "
            f"{dict(cfg.UMAP_PARAMS)} 不同；本次重跑用的是 config（与 Stage 5 一致）",
            file=sys.stderr,
        )

    shipped = _shipped_frame(shipped_path)
    shipped_md5 = _md5(shipped_path)

    t0 = time.perf_counter()
    embedding = landscape.embed(X)
    elapsed = time.perf_counter() - t0

    # 按 Stage 5 的写法（同列、同序、同 float_format）写一份临时 CSV：md5 才有可比性
    rerun_frame = pd.DataFrame(
        {"instance_uid": shipped["instance_uid"], "umap_1": embedding[:, 0], "umap_2": embedding[:, 1]}
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        rerun_path = Path(tmpdir) / _UMAP_FILE
        rerun_frame.to_csv(rerun_path, index=False, float_format="%.10g")
        rerun_md5 = _md5(rerun_path)

    n_points = int(embedding.shape[0])
    shipped_n = len(shipped)
    if shipped_n != n_points:
        max_abs_diff = None
        row_note = f"点数不同：重跑 {n_points} 个，落盘 {shipped_n} 个（X.npy 与 umap_2d.csv 不是同一次运行的产物）"
    else:
        diff = np.abs(
            np.asarray(rerun_frame[["umap_1", "umap_2"]], dtype=np.float64)
            - np.asarray(shipped[["umap_1", "umap_2"]], dtype=np.float64)
        )
        max_abs_diff = float(diff.max()) if diff.size else 0.0
        row_note = None

    identical = rerun_md5 == shipped_md5 and row_note is None
    precision_note = None
    if identical and max_abs_diff:
        precision_note = (
            f"md5 相同说明两份 CSV 字节完全一致；max_abs_diff={max_abs_diff:.3g} 只是因为比对时用的是“未舍入的"
            "新坐标 vs 已按 %.10g 舍入写入的落盘 CSV”，不是坐标真的变了。"
        )
    checked_at = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = {
        "source": _SOURCE,
        "checked_at": checked_at,
        "identical": identical,
        "n_points": n_points,
        "n_features": int(X.shape[1]),
        "shipped_md5": shipped_md5,
        "rerun_md5": rerun_md5,
        "max_abs_diff": max_abs_diff,
        "umap_params": recorded_params,
        "umap_params_used": dict(cfg.UMAP_PARAMS),
        "umap_params_match_config": params_match,
        "random_state": cfg.RANDOM_STATE,
        "inputs": {
            "X": str(x_path),
            "umap_csv": str(shipped_path),
            "kmeans_metrics": str(metrics_path),
        },
        "elapsed_seconds": round(elapsed, 3),
        "method": (
            "读 preprocessed/X.npy，用 config.UMAP_PARAMS 重跑一次 landscape.embed；按 Stage 5 的写法"
            "（DataFrame.to_csv(index=False, float_format='%.10g')）重写一份 CSV，与已落盘的 analysis/umap_2d.csv"
            " 比 md5 与逐值最大绝对差。第一次运行的中间文件没有保留，所以这不是两次独立全流程的比对。"
        ),
    }
    if row_note:
        payload["note"] = row_note
    elif precision_note:
        payload["note"] = precision_note

    out_path = root / report_mod.LOG_SUBDIR / report_mod.REPEAT_CHECK_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    verdict = "一致" if identical else "不一致"
    print(
        f"UMAP 重跑核验：{verdict}（{n_points} 个点，md5 重跑 {rerun_md5} / 落盘 {shipped_md5}，"
        f"最大绝对差 {max_abs_diff}，重跑耗时 {elapsed:.1f}s）→ {out_path}"
    )
    return payload


if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(
        prog="check_umap_repeat.py",
        description="UMAP 可复现性核验（只读产物，不需要输入图像）：可用 --output 指向别的输出根",
    )
    _parser.add_argument("--output", type=Path, metavar="目录", help="产物根目录（默认 config.OUTPUT_ROOT）")
    _args = _parser.parse_args()
    _result = check(Path(_args.output) if _args.output is not None else None)
    sys.exit(0 if _result["identical"] else 1)
