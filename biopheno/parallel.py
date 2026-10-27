"""跨 (天, 视野) 的进程级并行：Stage 1 / Stage 2 的每个网格单元彼此独立。

为什么用**进程**而不是线程：这两个阶段的开销大头是 Python 层的逐标签/逐实例循环与大量临时数组，
线程会被 GIL 卡住；而每个单元只读自己那一对图、只写自己的目录，天然适合多进程。

为什么子进程里要重设 `cfg`：Windows 用 spawn 启动子进程，子进程会**重新 import** 模块，拿到的是
`config.py` 的默认值（`OUTPUT_ROOT` 指向项目默认目录、`INPUT_FILES` 为空），不重新配置就会读错
输入、写错产物。所以用 initializer 把运行时的配置显式传进去，并把子进程的 `cfg.N_JOBS` 按
worker 数摊小，避免 `N_JOBS=-1` × 多进程把 CPU 超订成几百个线程。

**顺序路径优先**：`jobs <= 1`（或只有一个视野）时完全不建池、不 spawn，行为与没加并行时一模一样；
`jobs > 1` 时结果仍按"提交顺序"取回，所以汇总表的行序、QC 的键序、日志的行序都与顺序执行相同。

并行只在"每个单元只写自己目录"的地方使用：Stage 3–7 有全局拟合（power transform、等权、UMAP、
模块选择），一律保持单进程，否则数值口径会变。

**实测结论（9 帧、每帧 ~1100 实例、2048×2448）：P0–P3 之后并行不赚。** Stage 2 单独计时：
顺序 41.8 s、jobs=3 87.1 s、jobs=6 61.6 s、jobs=9 42.8 s。原因是单帧成本已经被压到几秒，而每个
子进程都要重新 import numpy/skimage 并解码自己的大图，开销与收益同量级。所以 `config.PARALLEL_JOBS`
默认是 1（顺序），只有帧数很多（≳24）或单帧很重时才用 `--jobs N` 打开。并行本身的正确性已有测试
与逐字节对比保证（jobs=1 与 jobs=6 的 65 个产物完全一致）。
"""

import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Any

from biopheno import config as cfg

# 子进程必须继承的 config 项：阶段函数会读、且可能在运行时被改写（CLI 改 OUTPUT_ROOT/INPUT_FILES，
# 测试改 DATA_ROOT）。漏一个就会在子进程里拿默认值去读错目录——这类健忘最难查，所以列成显式清单。
_CHILD_CONFIG_KEYS: tuple[str, ...] = (
    "OUTPUT_ROOT",
    "INPUT_FILES",
    "DATA_ROOT",
    "IMAGE_DIR",
    "MASK_DIR",
    "ALIGN_MAX_SHIFT_PX",
    "DAYS",
    "FOVS",
    "TIME_HOURS",
)


def resolve_jobs(n_frames: int) -> int:
    """算出实际要用的 worker 数（`cfg.PARALLEL_JOBS`：0 = 自动，1 = 顺序，>1 = 固定上限）。

    自动值取 `min(视野数, max(1, 逻辑核 // 2))`。为什么按 2 个核一个 worker 摊：子进程里的
    `cfg.N_JOBS` 固定为 1（见 `_init_worker`），所以一个 worker 基本只吃一个核，留一半余量给
    主进程的导入、落盘与操作系统调度就够。
    """
    n_frames = max(int(n_frames), 1)
    requested = int(getattr(cfg, "PARALLEL_JOBS", 0))
    if requested == 1:
        return 1  # 明确要求顺序执行（对比/调试用）
    if requested > 1:
        return min(requested, n_frames)
    if requested < 0:
        return 1  # 负值同样表示"关掉并行"：与 N_JOBS=-1（用满全部核）的语义区分开
    cpus = os.cpu_count() or 1
    return min(n_frames, max(1, cpus // 2))


# BLAS/OpenMP 的线程数：子进程必须钉成 1，否则每个子进程里的 numpy 默认会开满全部核，
# 多个 worker × 28 线程互相抢核，实测比单进程还慢。这些变量必须在 numpy 被 import **之前**生效，
# 所以只能在父进程 spawn 前设环境变量（子进程会继承）；在子进程 initializer 里设已经来不及。
_THREAD_ENV_KEYS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextmanager
def _single_threaded_env() -> Iterator[None]:
    """临时把 BLAS 线程数钉成 1（供 spawn 子进程继承），退出时原样恢复。"""
    saved = {key: os.environ.get(key) for key in _THREAD_ENV_KEYS}
    try:
        for key in _THREAD_ENV_KEYS:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _init_worker(overrides: dict) -> None:
    """子进程初始化：把运行时的 config 覆盖成父进程的值（spawn 出来的子进程默认是模块默认值）。

    两层注意：

    1. `overrides` 由 `_child_overrides()` 从 `_CHILD_CONFIG_KEYS` 里抽，**不能只传 OUTPUT_ROOT**：
       测试/CLI 会在运行时改写 `DATA_ROOT`（旧目录约定）或 `INPUT_FILES`（--images/--masks），
       子进程只拿默认值就会读错地方（表现为“旧约定的原图不存在”这类谜之报错）。
    2. `N_JOBS` 在子进程里**固定为 1**：SPOT 的 `n_jobs` 走的是 `joblib(backend='loky')`，是**进程池**，
       在 6 个 worker 里各开 4 进程 = 24 个进程同时 import skimage/mahotas，实测比单进程慢 10 倍
       （340 s vs 32 s）。跨视野并行已经把核吃满，不需要再嵌套一层进程池。
    """
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.N_JOBS = 1


def _child_overrides() -> dict:
    """要带给子进程的 config 项（阶段函数会读、且可能在运行时被改写的那几个）。"""
    return {key: getattr(cfg, key) for key in _CHILD_CONFIG_KEYS}


def map_frames(task: Callable[..., Any], frames: Sequence[tuple[str, str]], jobs: int | None = None) -> list[Any]:
    """按 (天, 视野) 逐个调用 `task(day, fov)`，`jobs > 1` 时用多进程，结果按输入顺序返回。

    `task` 必须是**模块级函数**（要能被 pickle 到子进程）。`jobs=None` 时按 `cfg.PARALLEL_JOBS`
    自动决定；`jobs=1` 时走纯顺序路径。任何一个视野抛错：立即向上抛（与"失败就停"一致），
    并取消还没开始的任务，不再等其余视野跑完。
    """
    frames = [(str(day), str(fov)) for day, fov in frames]
    if not frames:
        return []
    workers = resolve_jobs(len(frames)) if jobs is None else min(int(jobs), len(frames))
    if workers <= 1:
        return [task(day, fov) for day, fov in frames]

    future_list = []
    with _single_threaded_env():  # 必须在建池（spawn）之前设，子进程才会继承
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(_child_overrides(),),
        )
        try:
            future_list = [pool.submit(task, day, fov) for day, fov in frames]
            # 按提交顺序取结果：汇总行序/QC 键序/日志行序都与顺序执行一致
            return [future.result() for future in future_list]
        except BaseException:
            for future in future_list:
                future.cancel()  # 还没开跑的直接取消；已经跑着的等进程池收敛
            raise
        finally:
            pool.shutdown(wait=True)
