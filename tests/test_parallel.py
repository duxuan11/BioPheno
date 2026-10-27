"""跨视野并行（`biopheno.parallel`）的测试：并行度解析、顺序/并行路径一致、错误与配置传递。

这里只测"调度层"：真正跑 Stage 1/2 的端到端一致性由 test_features / test_stage1_cli 与
一次真实数据对比保证。任务函数必须是模块级（要能被 pickle 到子进程），所以放在文件顶部。
"""

import os

import pytest

from biopheno import config as cfg
from biopheno import parallel


def _tag(day: str, fov: str) -> dict:
    """任务替身：返回 (day, fov, pid) —— pid 用来分辨是同一进程还是子进程跑的。"""
    return {"day": day, "fov": fov, "pid": os.getpid()}


def _echo_config(day: str, fov: str) -> dict:
    """任务替身：把子进程里的 cfg 读出来，验证 initializer 是否把运行时配置传进去了。"""
    return {
        "output_root": str(cfg.OUTPUT_ROOT),
        "input_files": {f"{d}/{f}": [str(i), str(m)] for (d, f), (i, m) in cfg.INPUT_FILES.items()},
        "n_jobs": int(cfg.N_JOBS),
    }


def _boom(day: str, fov: str) -> dict:
    """任务替身：只对 day3/A1 抛错，其余正常返回——用来验证"第二个视野失败也会整轮失败"。"""
    if (day, fov) == ("day3", "A1"):
        raise RuntimeError(f"{day}/{fov} 故意失败")
    return {"day": day, "fov": fov}


def test_resolve_jobs_honours_config(monkeypatch):
    """守口径：PARALLEL_JOBS 的 0/1/>1/负数 四种取值各自对应什么 worker 数。"""
    monkeypatch.setattr(cfg, "PARALLEL_JOBS", 1)
    assert parallel.resolve_jobs(9) == 1  # 1 = 顺序执行

    monkeypatch.setattr(cfg, "PARALLEL_JOBS", 3)
    assert parallel.resolve_jobs(9) == 3  # 固定上限
    assert parallel.resolve_jobs(2) == 2  # 不超过视野数

    monkeypatch.setattr(cfg, "PARALLEL_JOBS", -1)
    assert parallel.resolve_jobs(9) == 1  # 负值同样表示关掉并行

    monkeypatch.setattr(cfg, "PARALLEL_JOBS", 0)
    cpus = os.cpu_count() or 1
    assert parallel.resolve_jobs(1000) == max(1, cpus // 2)  # 自动：按核数摊
    assert parallel.resolve_jobs(1) == 1  # 只有一个视野就不并行


def test_map_frames_sequential_stays_in_process():
    """守口径：jobs=1 时不建进程池、结果按输入顺序返回（顺序路径是数值一致的基准）。"""
    frames = [("day0", "A1"), ("day3", "A1"), ("day5", "A1")]
    results = parallel.map_frames(_tag, frames, jobs=1)
    assert [(r["day"], r["fov"]) for r in results] == frames
    assert {r["pid"] for r in results} == {os.getpid()}


def test_map_frames_parallel_preserves_order_and_uses_workers():
    """守口径：jobs>1 时用子进程跑，但结果仍按输入顺序返回（汇总行序/QC 键序靠这个保证）。"""
    frames = [("day0", "A1"), ("day3", "A1"), ("day5", "A1"), ("day0", "A2")]
    results = parallel.map_frames(_tag, frames, jobs=2)
    assert [(r["day"], r["fov"]) for r in results] == frames
    assert os.getpid() not in {r["pid"] for r in results}  # 真的在子进程里跑的


def test_map_frames_propagates_config_to_workers(monkeypatch, tmp_path):
    """守契约：子进程必须拿到父进程的 OUTPUT_ROOT / INPUT_FILES / 摊小后的 N_JOBS。

    Windows 用 spawn，子进程会重新 import 模块并拿 config 的默认值；不传配置就会读错输入、
    写错产物，或者在子进程里用满全部核把 CPU 超订。
    """
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(cfg, "INPUT_FILES", {("day0", "A1"): (tmp_path / "a.png", tmp_path / "a.tif")})
    monkeypatch.setattr(cfg, "N_JOBS", -1)
    results = parallel.map_frames(_echo_config, [("day0", "A1"), ("day3", "A1")], jobs=2)
    for result in results:
        assert result["output_root"] == str(tmp_path)
        assert result["input_files"] == {"day0/A1": [str(tmp_path / "a.png"), str(tmp_path / "a.tif")]}
        # 子进程里固定 1：跨视野并行已经吃满核，再嵌套 loky 进程池实测慢 10 倍
        assert result["n_jobs"] == 1


def test_map_frames_raises_from_worker():
    """守契约：任一视野抛错就整轮失败（与"失败就停"一致），异常类型与消息原样传回。"""
    with pytest.raises(RuntimeError, match="day3/A1 故意失败"):
        parallel.map_frames(_boom, [("day0", "A1"), ("day3", "A1")], jobs=1)
    with pytest.raises(RuntimeError, match="day3/A1 故意失败"):
        parallel.map_frames(_boom, [("day0", "A1"), ("day3", "A1")], jobs=2)


def test_map_frames_empty_is_noop():
    """守边界：空视野清单返回空列表（不建进程池、不报错）。"""
    assert parallel.map_frames(_tag, [], jobs=4) == []
