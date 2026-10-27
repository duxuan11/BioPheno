"""Stage 1 批处理（run_stage1）的端到端测试：全部用 tmp_path 伪造 DATA_ROOT，不碰真实数据。"""

import json

import pandas as pd
import pytest

from biopheno import config as cfg
from biopheno import instances as inst


def test_run_stage1_writes_catalogs_and_summary(monkeypatch, fake_dataset, tmp_path):
    monkeypatch.setattr(cfg, "DATA_ROOT", fake_dataset)
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    meta = inst.run_stage1(days=["day0", "day3"], fovs=["A1"])
    assert (tmp_path / "instances" / "day0" / "A1" / "instances.csv").exists()
    assert (tmp_path / "instances" / "all_instances.csv").exists()
    assert set(meta["day"]) == {"day0", "day3"}
    assert meta["instance_uid"].is_unique


def test_run_stage1_returns_dataframe_with_required_columns(monkeypatch, fake_dataset, tmp_path):
    monkeypatch.setattr(cfg, "DATA_ROOT", fake_dataset)
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    meta = inst.run_stage1(days=["day0", "day3"], fovs=["A1"])
    assert isinstance(meta, pd.DataFrame)
    # 汇总表必须带下游（Stage 2/5/6）依赖的这几列
    required = {"instance_uid", "day", "fov", "frame_idx", "time_hours", "label", "area_px", "is_small"}
    assert required <= set(meta.columns)
    # 汇总表落盘的内容与返回值一致
    back = pd.read_csv(tmp_path / "instances" / "all_instances.csv")
    assert len(back) == len(meta)
    assert list(back["instance_uid"]) == list(meta["instance_uid"])


def test_run_stage1_writes_qc_summary(monkeypatch, fake_dataset, tmp_path):
    monkeypatch.setattr(cfg, "DATA_ROOT", fake_dataset)
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    meta = inst.run_stage1(days=["day0", "day3"], fovs=["A1"])
    qc = json.loads((tmp_path / "instances" / "stage1_qc.json").read_text(encoding="utf-8"))
    assert set(qc) == {"day0/A1", "day3/A1"}
    assert sum(row["n_instances"] for row in qc.values()) == len(meta)
    for row in qc.values():
        assert row["n_instances"] == 2  # fixture 每个视野造了 2 个类器官
        # 对齐偏移必须落在 ±ALIGN_MAX_SHIFT_PX 内，且以整数记录
        assert abs(row["align_dy"]) <= cfg.ALIGN_MAX_SHIFT_PX
        assert abs(row["align_dx"]) <= cfg.ALIGN_MAX_SHIFT_PX
        assert row["n_labels_discarded_contours"] == 0
        assert row["n_is_small"] == 0


def test_run_stage1_stops_when_alignment_exceeds_limit(monkeypatch, fake_dataset, tmp_path):
    # 偏移搜索上限固定在 ±8，跑出来不可能 > 8，只能注入一个越界偏移来验证"停下并报错"这条风险兜底。
    monkeypatch.setattr(cfg, "DATA_ROOT", fake_dataset)
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(inst, "align_mask_to_image", lambda image, labels, max_shift: (labels, (12, 0)))
    with pytest.raises(RuntimeError, match="day0/A1"):
        inst.run_stage1(days=["day0"], fovs=["A1"])


def test_run_stage1_stops_when_alignment_hits_search_boundary(monkeypatch, fake_dataset_boundary_shift, tmp_path):
    # 真实搜索路径（不注入偏移）：掩码上移正好 8 行，最优解落在 ±8 的边界上。
    # 边界解意味着"真正的最优解可能在边界之外、被搜索范围截断了"（如真需 12 px 时只能给出 (8, 3)），
    # 所以要判定为需人工确认，而不是静默通过。
    monkeypatch.setattr(cfg, "DATA_ROOT", fake_dataset_boundary_shift)
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    # 先钉住"这个 fixture 确实把搜索顶到边界"，否则下面的 RuntimeError 可能来自别的原因
    catalog = inst.build_catalog("day0", "A1")
    assert max(abs(v) for v in catalog["align_offset"]) == cfg.ALIGN_MAX_SHIFT_PX
    with pytest.raises(RuntimeError, match="day0/A1") as excinfo:
        inst.run_stage1(days=["day0", "day3"], fovs=["A1"])
    # 消息要说清是撞到了搜索边界（而不是只给一个看不懂的偏移数字）
    assert "边界" in str(excinfo.value)
    assert f"±{cfg.ALIGN_MAX_SHIFT_PX}" in str(excinfo.value)


def test_run_stage1_clears_previous_summary_before_running(monkeypatch, fake_dataset, tmp_path):
    # 第一次成功跑完：汇总表 + QC 都在。
    monkeypatch.setattr(cfg, "DATA_ROOT", fake_dataset)
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    inst.run_stage1(days=["day0", "day3"], fovs=["A1"])
    summary = tmp_path / "instances" / "all_instances.csv"
    qc = tmp_path / "instances" / "stage1_qc.json"
    assert summary.exists() and qc.exists()

    # 第二次在第二个视野上失败：day0 的目录已落盘，但汇总/QC 必须先被删掉，
    # 否则上一次成功留下的旧汇总表会冒充本次结果，下游无从分辨它是陈的。
    calls = {"n": 0}

    def _fail_on_second_fov(image, labels, max_shift):
        calls["n"] += 1
        return (labels, (0, 0)) if calls["n"] == 1 else (labels, (12, 0))

    monkeypatch.setattr(inst, "align_mask_to_image", _fail_on_second_fov)
    with pytest.raises(RuntimeError, match="day3/A1"):
        inst.run_stage1(days=["day0", "day3"], fovs=["A1"])
    assert calls["n"] == 2  # 确实跑进了第二个视野才中断
    assert (tmp_path / "instances" / "day0" / "A1" / "instances.csv").exists()  # 前 N−1 个目录已落盘
    assert not summary.exists()
    assert not qc.exists()
