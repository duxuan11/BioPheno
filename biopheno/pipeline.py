"""一键运行（Stage 7）：`run_all()` 顺序跑完 Stage1→2→3→4→5→6→6b→静态图→时间图→报告。

设计约束（Spec §10 验收标准 1）：

- **顺序固定**、**逐阶段写日志**（`output/logs/<阶段名>.log`，stdout 同时进控制台与日志），
  所以长跑中途出错时能直接从日志看出停在哪一步、最后几行是什么；
- **失败就停**：任一阶段抛错立刻把异常再抛出（`scripts/run_all.py` 因此非零退出），
  后面的阶段一个都不跑、报告也不写——不产出"看起来跑成功了"的半套结果；
- **耗时落盘**：每阶段耗时与总 wall-clock 写进 `output/logs/run_all_timings.json`，
  报告（`report.write_report`）读它来填"运行耗时"一节，事后也能核对哪一阶段变慢了。

阶段的输出目录一律来自 `config.OUTPUT_ROOT`（在**调用时**读取，不在 import 时缓存），
这样测试可以把整棵树 monkeypatch 到 `tmp_path`。
"""

import json
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager, redirect_stdout
from datetime import datetime
from pathlib import Path

from biopheno import config as cfg
from biopheno import report as report_mod
from biopheno.features import run_stage2
from biopheno.figures import render_static, render_temporal
from biopheno.instances import run_stage1
from biopheno.landscape import run_stage5
from biopheno.modules import run_stage4
from biopheno.preprocess import run_stage3
from biopheno.report import write_report
from biopheno.temporal import run_stage6
from biopheno.temporal_stats import run_stage6b


class _Tee:
    """把 stdout 同时写进控制台与日志文件（逐阶段日志的关键：长跑时能 `tail` 一份完整记录）。"""

    def __init__(self, *streams):
        """记下要同时写入的多个流（控制台 stdout + 该阶段日志文件）。"""
        self._streams = streams

    def write(self, text: str) -> int:
        """把一段文本依次写进每个流并立即 flush，返回写入长度（满足 TextIO 的 write 约定）。"""
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        """刷新所有流，避免程序异常退出时日志尾部还留在缓冲区里。"""
        for stream in self._streams:
            stream.flush()


@contextmanager
def _tee_to(path: Path) -> Iterator[None]:
    """把 stdout 复制到 `path`（覆盖写）；异常时调用方再追加堆栈。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as log, redirect_stdout(_Tee(sys.stdout, log)):
        yield


def _stage_specs(root: Path) -> tuple[tuple[str, Callable[[], object]], ...]:
    """按顺序返回 (日志/耗时用的阶段名, 无参调用)。名字在函数内查找模块全局，测试可 monkeypatch。"""
    return (
        ("stage1_instances", lambda: run_stage1()),
        ("stage2_features", lambda: run_stage2()),
        ("stage3_preprocess", lambda: run_stage3()),
        ("stage4_modules", lambda: run_stage4()),
        ("stage5_landscape", lambda: run_stage5()),
        ("stage6_tracks", lambda: run_stage6()),
        ("stage6b_temporal_stats", lambda: run_stage6b()),
        # 静态图吃 output/ 根（它自己派生 analysis/ modules/ instances/ preprocessed/）
        ("stage7_static_figures", lambda: render_static(root, root / report_mod.FIGURES_SUBDIR)),
        # 时间图的输入全在 analysis/ 里，显式指到子目录，免得依赖"两种写法都接受"的兜底
        ("stage7_temporal_figures", lambda: render_temporal(root / "analysis", root / report_mod.FIGURES_SUBDIR)),
    )


def _write_timings(path: Path, timings: dict) -> None:
    """耗时 JSON 落盘（含失败信息）；写的是 dict，报告与事后核查都直接读它。"""
    timings["total_seconds"] = round(timings.pop("_total", 0.0), 3)
    path.write_text(json.dumps(timings, ensure_ascii=False, indent=2), encoding="utf-8")


def run_all() -> Path:
    """跑完整条流水线，返回写出的 `output/report.md` 路径；任一阶段失败都会把异常再抛出。

    顺序：Stage1 → Stage2 → Stage3 → Stage4 → Stage5 → Stage6 → Stage6b → 静态图 F1–F6b →
    时间图 F7–F11 → 报告。每个阶段一份 `<阶段名>.log`，耗时写进 `logs/run_all_timings.json`
    （每跑完一个阶段就刷新一次，所以中途崩掉也看得到已经花了多久）。
    """
    root = Path(cfg.OUTPUT_ROOT)
    log_dir = root / report_mod.LOG_SUBDIR
    log_dir.mkdir(parents=True, exist_ok=True)
    timings: dict = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "output_root": str(root),
        "stages": {},
        "_total": 0.0,
    }
    timings_path = log_dir / report_mod.TIMING_FILE
    t_all = time.perf_counter()
    current = "（尚未开始）"
    try:
        for name, run in _stage_specs(root):
            current = name
            print(f"==> {name}")
            t0 = time.perf_counter()
            try:
                with _tee_to(log_dir / f"{name}.log"):
                    run()
            except BaseException as error:
                # 日志里留完整堆栈：控制台的 stderr 只有最后几十行，事后排查要靠这份文件
                with (log_dir / f"{name}.log").open("a", encoding="utf-8") as log:
                    traceback.print_exc(file=log)
                # 把阶段名带进异常里：否则 `run_stage3` 抛的原始错误看起来和流水线没关系
                raise RuntimeError(f"{name} 阶段失败：{error}（完整堆栈见 {log_dir / f'{name}.log'}）") from error
            finally:
                timings["stages"][name] = round(time.perf_counter() - t0, 3)
                print(f"<== {name} 完成，用时 {timings['stages'][name]}s")
            timings["_total"] = time.perf_counter() - t_all
            _write_timings(timings_path, dict(timings))
        # 报告最后写：它要读前面所有阶段的产物，也要读本文件里的耗时。
        # 先把 `current` 改掉：否则报告写盘失败时 `failed_stage` 会指到上一阶段（时间图）。
        current = "report"
        path = write_report(root, source="run_all")
    except BaseException as error:
        timings["failed_stage"] = current
        timings["error"] = f"{type(error).__name__}: {error}"
        timings["_total"] = time.perf_counter() - t_all
        _write_timings(timings_path, dict(timings))
        print(f"阶段 {current} 失败：{timings['error']}（日志 {log_dir / f'{current}.log'}）", file=sys.stderr)
        raise
    timings["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    timings["_total"] = time.perf_counter() - t_all
    _write_timings(timings_path, dict(timings))
    total = round(time.perf_counter() - t_all, 3)
    print(f"全流程完成：总耗时 {total}s（{total / 60.0:.1f} 分钟）→ 报告 {path}")
    return path
