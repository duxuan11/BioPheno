"""Stage 2（shape + appearance）的测试：全部用合成图与伪造目录，不依赖 1.2 GB 真实数据。

motion 特征被有意不计算（人类决定），所以这里只验证 NaN 占位列的存在与命名可解析，
不验证任何运动学数值。
"""

import json

import numpy as np
import pandas as pd
import pytest

from biopheno import config as cfg
from biopheno import features as feat
from biopheno import instances as inst


def _write_stage1_summary(root, day_to_fov_uids):
    """按 Stage 1 的落盘格式写 all_instances.csv（只写 Stage 2 交叉核对需要的三列）。"""
    outdir = root / "instances"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"day": day, "fov": fov, "instance_uid": uid} for (day, fov), uids in day_to_fov_uids.items() for uid in uids
    ]
    pd.DataFrame(rows, columns=["day", "fov", "instance_uid"]).to_csv(outdir / "all_instances.csv", index=False)


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    # SPOT 的 n_jobs=-1 会为 2–3 个实例也起一堆 loky 子进程（每个都要重新 import skimage/mahotas），
    # 单测里没有任何并行收益。这里只是把并行度钉成 1，测试体本身保持与计划逐字一致。
    monkeypatch.setattr(cfg, "N_JOBS", 1)


def test_shape_features_have_expected_families(make_mask_and_image):
    _image, labels = make_mask_and_image(n_blobs=3)
    contours, _ = inst.extract_contours(labels, 100)
    X, names = feat.compute_shape_features(list(contours.values()), labels.shape, pixel_res=1.0)
    assert X.shape == (3, len(names))
    assert np.isfinite(X).all()
    assert any(n.startswith("ECT_") for n in names)
    assert any(n == "area" or n.startswith("area") for n in names)


def test_appearance_features_are_finite(make_mask_and_image):
    image, labels = make_mask_and_image(n_blobs=2)
    contours, _ = inst.extract_contours(labels, 100)
    X, names = feat.compute_appearance_features(image, list(contours.values()), pixel_res=1.0)
    assert X.shape[0] == 2 and len(names) > 0
    assert np.isfinite(X).all()


def test_ecc_reduction_shrinks_ect_family():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 200)).astype(np.float64)
    names = [f"ECT_{i + 1}" for i in range(150)] + [f"other_{i}" for i in range(50)]
    Xr, nr = feat.reduce_ecc_features(X, names, n_dim=10)
    assert Xr.shape == (20, 60)  # 20 行不变：50 个非 ECT 列 + 10 个核投影列
    assert sum(n.startswith("ECT_") for n in nr) <= 10
    assert sum(n.startswith("other_") for n in nr) == 50


def test_motion_placeholder_names_are_unique_and_prefixed():
    # NaN 占位列的名字必须是可辨认的 motion 名，且不能重复（重复列名会让下游按名取列时静默取错）。
    names = feat.motion_placeholder_names()
    assert len(names) > 0
    assert len(set(names)) == len(names)
    # 冒烟：默认参数下应覆盖全局运动量、按区域的直方图与 SIFT 家族
    assert "mean_speed_global" in names
    assert any(n.startswith("hist_bin_1_region_") for n in names)
    assert any(n.startswith("flow_vector_sift_") for n in names)


def test_run_stage2_writes_features_and_nan_motion_columns(monkeypatch, fake_dataset, tmp_path):
    monkeypatch.setattr(cfg, "DATA_ROOT", fake_dataset)
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    df = feat.run_stage2(days=["day0", "day3"], fovs=["A1"])

    outdir = tmp_path / "features"
    for name in ("sam_shape.npz", "sam_appearance.npz", "feature_names.json", "all_features.csv"):
        assert (outdir / name).exists()

    # fixture 每个视野 2 个类器官 → 2 天 × 2 个 = 4 行，行数不能因为任何一步而变
    assert len(df) == 4
    assert df.index.name == "instance_uid"
    names = json.loads((outdir / "feature_names.json").read_text(encoding="utf-8"))
    assert len(names) == df.shape[1]  # 列名清单与矩阵列一一对应
    assert list(df.columns) == names

    # motion 列必须整列为 NaN（有意置空，下游看到 NaN 就知道不是"算失败"）
    motion = feat.motion_placeholder_names()
    assert motion, "本环境下静态解析应能给出 motion 名"
    assert df[motion].isna().all().all()
    # shape/appearance 列不应有 NaN：合成图很干净，出现 NaN 说明算错了
    assert np.isfinite(df[[c for c in names if c not in set(motion)]].to_numpy(dtype=np.float64)).all()

    # ECT 家族被压到 ECC_N_DIM 维，其它 shape 家族原样保留
    assert sum(n.startswith("ECT_") for n in names) <= cfg.ECC_N_DIM
    assert "area" in names

    qc = json.loads((outdir / "stage2_qc.json").read_text(encoding="utf-8"))
    assert set(qc) == {"day0/A1", "day3/A1"}
    assert sum(row["n_instances_computed"] for row in qc.values()) == len(df)
    assert all(row["n_skipped_no_contour"] == 0 for row in qc.values())
    assert qc["day0/A1"]["n_motion_placeholder"] == len(motion)
    # 每家族的非有限值计数必须落进 QC（合成图很干净，应为 0），Task 5 的 NaN 处理才有据可查
    assert qc["day0/A1"]["n_nonfinite_shape"] == 0
    assert qc["day0/A1"]["n_nonfinite_appearance"] == 0
    assert qc["day0/A1"]["n_nonfinite_ect_filled"] == 0
    # 没有 Stage 1 汇总表时交叉核对无法进行，必须是可见的"未核对"而不是假装通过
    assert qc["day0/A1"]["uid_cross_check"] == "not_checked_no_all_instances"
    assert all(row["skipped_reason"] is None for row in qc.values())


def test_run_stage2_skips_instances_without_contour(monkeypatch, make_mask_and_image, tmp_path):
    # Task 2 的已知契约：没有可用轮廓的标签在 meta 里有行、但 contours 里没有键。
    # Stage 2 必须只算有轮廓的实例，并把跳过数记进 QC —— 不崩，也不悄悄改行数。
    image, labels = make_mask_and_image(n_blobs=2)
    contours, _ = inst.extract_contours(labels, cfg.N_CONTOUR_POINTS)
    catalog = inst.build_catalog("day0", "A1", image=image, labels=labels)
    # 造一个"只有 meta 行、没有轮廓"的标签 3
    extra = catalog["meta"].iloc[[0]].copy()
    extra["label"] = 3
    extra["instance_uid"] = "A1_3_0"
    catalog["meta"] = pd.concat([catalog["meta"], extra], ignore_index=True)

    monkeypatch.setattr(cfg, "DATA_ROOT", tmp_path / "unused")
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(inst, "build_catalog", lambda day, fov: catalog)

    df = feat.run_stage2(days=["day0"], fovs=["A1"])
    assert len(df) == len(contours) == 2  # 只保留有轮廓的两行
    assert "A1_3_0" not in df.index
    qc = json.loads((tmp_path / "features" / "stage2_qc.json").read_text(encoding="utf-8"))
    assert qc["day0/A1"]["n_instances_meta"] == 3
    assert qc["day0/A1"]["n_skipped_no_contour"] == 1


def test_as_boundaries_rejects_non_uniform_contours(make_mask_and_image):
    # 轮廓点数不一致时 np.stack 抛的是裸 numpy 错误（看不出是谁的锅）。必须先自查形状，
    # 给出一句带上下文的 ValueError。
    contours, _ = inst.extract_contours(make_mask_and_image(n_blobs=2)[1], 100)
    first, second = contours.values()
    uneven = [first, second[:90]]
    with pytest.raises(ValueError, match="轮廓点数不一致"):
        feat.compute_shape_features(uneven, (240, 320), pixel_res=1.0)


def test_nonfinite_ect_fill_uses_substring_predicate():
    # 上游用 `'ECT_' in name`（SPOT/Analysis/preprocessing.py:245）选 ECT 列，这里必须逐字对齐：
    # 前缀匹配会漏掉被上游选中的列，让 NaN 直接进 Nystroem。
    X = np.array([[np.nan, 1.0], [np.inf, 2.0]])
    assert feat._fill_nonfinite_ect(X, ["prefix_ECT_1", "other"]) == 2
    assert np.isfinite(X).all()


def test_ecc_reduction_picks_up_non_prefixed_ect_columns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 30)).astype(np.float64)
    names = [f"some_prefix_ECT_{i}" for i in range(10)] + [f"other_{i}" for i in range(20)]
    Xr, nr = feat.reduce_ecc_features(X, names, n_dim=5)
    assert Xr.shape == (20, 25)
    assert not any(n.startswith("some_prefix_ECT_") for n in nr)
    assert sum(n.startswith("other_") for n in nr) == 20


def test_motion_placeholder_count_is_pinned_to_360(monkeypatch):
    # 描述子维度 / bin 数漂移时，静态展开的个数会变；个数不再是 360 就退化成"没有 motion 列"，
    # 而不是给出一份可能错位的清单。
    assert len(feat.motion_placeholder_names()) == feat._MOTION_N_PLACEHOLDER_COLS == 360
    monkeypatch.setattr(feat, "_MOTION_N_PLACEHOLDER_COLS", 359)
    with pytest.warns(UserWarning, match="motion"):
        assert feat.motion_placeholder_names() == []


@pytest.mark.parametrize("empty_first", [True, False])
def test_run_stage2_skips_fov_with_zero_contours(monkeypatch, make_mask_and_image, tmp_path, empty_first):
    # 一个视野一个可用轮廓都没有时，它的特征块宽度是 0。若照样 append，最后 np.vstack 会崩
    # （"0 列块 + 正常块"），而且是跑完全部视野之后才崩、看不出是哪个视野。
    image, labels = make_mask_and_image(n_blobs=2)
    normal = inst.build_catalog("day0", "A1", image=image, labels=labels)
    blank = inst.build_catalog("day3", "A1", image=image, labels=labels)
    blank["contours"] = {}  # meta 有 2 行，但一个轮廓都没有
    catalogs = {("day0", "A1"): normal, ("day3", "A1"): blank}
    monkeypatch.setattr(inst, "build_catalog", lambda day, fov: catalogs[(day, fov)])
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)

    order = ["day3", "day0"] if empty_first else ["day0", "day3"]
    df = feat.run_stage2(days=order, fovs=["A1"])

    # 正常视野的 2 行必须完好，空视野不贡献任何行
    assert len(df) == 2
    assert all(uid.startswith("A1_") and uid.endswith("_0") for uid in df.index)
    qc = json.loads((tmp_path / "features" / "stage2_qc.json").read_text(encoding="utf-8"))
    assert qc["day3/A1"]["skipped_reason"] == "no_usable_contour"
    assert qc["day3/A1"]["n_instances_meta"] == 2
    assert qc["day3/A1"]["n_instances_computed"] == 0
    assert qc["day3/A1"]["n_skipped_no_contour"] == 2
    assert qc["day0/A1"]["skipped_reason"] is None
    assert qc["day0/A1"]["n_instances_computed"] == 2


def test_run_stage2_skips_fov_with_empty_feature_block(monkeypatch, make_mask_and_image, tmp_path):
    # 第二个变体：视野有轮廓，但 SPOT 给出的特征块宽度是 0（列全退化）。
    # 同样必须整帧跳过并记账，而不是把一个 (n, 0) 块拼进去。
    image, labels = make_mask_and_image(n_blobs=2)
    normal = inst.build_catalog("day0", "A1", image=image, labels=labels)
    image1, labels1 = make_mask_and_image(n_blobs=1)
    degenerate = inst.build_catalog("day3", "A1", image=image1, labels=labels1)
    catalogs = {("day0", "A1"): normal, ("day3", "A1"): degenerate}
    monkeypatch.setattr(inst, "build_catalog", lambda day, fov: catalogs[(day, fov)])
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)

    real_shape = feat.compute_shape_features

    def _shape(contours, imshape, pixel_res):
        if len(contours) == 1:  # day3/A1：模拟"全部轮廓退化 → (n, 0) 块"
            return np.zeros((1, 0), dtype=np.float64), []
        return real_shape(contours, imshape, pixel_res)

    monkeypatch.setattr(feat, "compute_shape_features", _shape)
    df = feat.run_stage2(days=["day0", "day3"], fovs=["A1"])

    assert len(df) == 2
    qc = json.loads((tmp_path / "features" / "stage2_qc.json").read_text(encoding="utf-8"))
    assert qc["day3/A1"]["skipped_reason"] == "empty_feature_block"
    assert qc["day3/A1"]["n_shape_features_raw"] == 0
    assert qc["day0/A1"]["skipped_reason"] is None


def test_run_stage2_rejects_block_width_that_disagrees_with_names(monkeypatch, make_mask_and_image, tmp_path):
    # 宽度与名数不一致时列的含义无法确定，必须报错而不是拼进去
    image, labels = make_mask_and_image(n_blobs=2)
    catalog = inst.build_catalog("day0", "A1", image=image, labels=labels)
    monkeypatch.setattr(inst, "build_catalog", lambda day, fov: catalog)
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        feat, "compute_shape_features", lambda c, imshape, pixel_res: (np.zeros((len(c), 5)), ["a", "b"])
    )

    with pytest.raises(ValueError, match="不匹配"):
        feat.run_stage2(days=["day0"], fovs=["A1"])


def test_run_stage2_cross_checks_stage1_uids(monkeypatch, make_mask_and_image, tmp_path):
    # Stage 2 重新 build_catalog，若 cfg（N_CONTOUR_POINTS / ALIGN_MAX_SHIFT_PX）或 instances.py
    # 在两次运行之间被改动，特征会悄悄挂到与 Stage 1 落盘的轮廓不同的实例上。
    image, labels = make_mask_and_image(n_blobs=2)
    catalog = inst.build_catalog("day0", "A1", image=image, labels=labels)
    monkeypatch.setattr(inst, "build_catalog", lambda day, fov: catalog)
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    uids = [str(u) for u in catalog["meta"]["instance_uid"]]
    assert len(uids) == 2

    _write_stage1_summary(tmp_path, {("day0", "A1"): uids})
    df = feat.run_stage2(days=["day0"], fovs=["A1"])
    assert len(df) == 2
    qc = json.loads((tmp_path / "features" / "stage2_qc.json").read_text(encoding="utf-8"))
    assert qc["day0/A1"]["uid_cross_check"] == "ok"

    # 删掉一行：uid 集合对不上 → 必须带着视野名与个数差异拒绝继续
    _write_stage1_summary(tmp_path, {("day0", "A1"): uids[:1]})
    with pytest.raises(RuntimeError, match="day0/A1") as excinfo:
        feat.run_stage2(days=["day0"], fovs=["A1"])
    assert "2" in str(excinfo.value) and "1" in str(excinfo.value)


def test_run_stage2_rejects_duplicate_feature_names(monkeypatch, make_mask_and_image, tmp_path):
    # 列名重复会让下游按名取列静默取错（df[motion] 会得到两列），必须在落盘前拒绝。
    # 用 _fail（ValueError）而不是裸 assert：`python -O` 会把 assert 整条删掉，这道关卡就消失了。
    image, labels = make_mask_and_image(n_blobs=2)
    catalog = inst.build_catalog("day0", "A1", image=image, labels=labels)
    monkeypatch.setattr(inst, "build_catalog", lambda day, fov: catalog)
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(feat, "motion_placeholder_names", lambda: ["ECT_rbf_1"])

    with pytest.raises(ValueError, match="重复"):
        feat.run_stage2(days=["day0"], fovs=["A1"])


def test_run_stage2_tolerates_empty_fov_absent_from_stage1_summary(monkeypatch, make_mask_and_image, tmp_path):
    # Stage 1 用 pd.concat(metas) 写汇总表（instances.py）：某视野 meta 为 0 行时，它在 all_instances.csv
    # 里连一行都没有。"键不存在"正是"零实例"的正常表达，两侧对同一事实达成一致 → ok_empty。
    # 这正是空视野跳过逻辑要容忍的情形，不能因为键缺失就硬失败。
    image, labels = make_mask_and_image(n_blobs=2)
    normal = inst.build_catalog("day0", "A1", image=image, labels=labels)
    blank = inst.build_catalog("day3", "A1", image=image, labels=labels)
    blank["meta"] = blank["meta"].iloc[0:0].copy()  # 分割结果全空：0 个实例
    blank["contours"] = {}
    catalogs = {("day0", "A1"): normal, ("day3", "A1"): blank}
    monkeypatch.setattr(inst, "build_catalog", lambda day, fov: catalogs[(day, fov)])
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    # 汇总表里只有一个视野的行（与 Stage 1 对零实例视野的写法完全一致）
    _write_stage1_summary(tmp_path, {("day0", "A1"): [str(u) for u in normal["meta"]["instance_uid"]]})

    df = feat.run_stage2(days=["day3", "day0"], fovs=["A1"])

    assert len(df) == 2  # 零实例视野不贡献行，正常视野 2 行完好
    qc = json.loads((tmp_path / "features" / "stage2_qc.json").read_text(encoding="utf-8"))
    assert qc["day3/A1"]["uid_cross_check"] == "ok_empty"
    assert qc["day3/A1"]["n_instances_meta"] == 0
    assert qc["day3/A1"]["skipped_reason"] == "no_usable_contour"
    assert qc["day0/A1"]["uid_cross_check"] == "ok"


def test_run_stage2_raises_when_stage1_summary_lacks_nonempty_fov(monkeypatch, make_mask_and_image, tmp_path):
    # 反向：本视野重建出非空的 meta，汇总表里却完全没有这个键 → 不能当作"零实例"放行，
    # 那说明 Stage 1 与本次重建的视野集合不同（days/fovs 被改过，或 Stage 1 没跑过这个视野）。
    image, labels = make_mask_and_image(n_blobs=2)
    catalog = inst.build_catalog("day0", "A1", image=image, labels=labels)
    monkeypatch.setattr(inst, "build_catalog", lambda day, fov: catalog)
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    _write_stage1_summary(tmp_path, {("day3", "A1"): ["A1_9_0"]})  # 只有别的视野的行

    with pytest.raises(RuntimeError, match="day0/A1") as excinfo:
        feat.run_stage2(days=["day0"], fovs=["A1"])
    assert "没有" in str(excinfo.value)


@pytest.mark.parametrize(
    "content",
    [
        "",  # 空文件
        "\n\n",  # 只有空行
        "day,fov\n",  # 缺 instance_uid 列（被截断的表头）
        "day,fov,instance_uid\nday0,A1\n",  # 行内字段数不对（写到一半被杀）
    ],
)
def test_load_stage1_uid_sets_reports_broken_summary(monkeypatch, tmp_path, content):
    # 空 / 被截断的 all_instances.csv 会让 pandas 抛裸 EmptyDataError / ParserError，
    # 看不出是哪个产物坏了；必须换成一句能照着修的中文提示。
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    outdir = tmp_path / "instances"
    outdir.mkdir(parents=True)
    (outdir / "all_instances.csv").write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="Stage1 汇总表"):
        feat._load_stage1_uid_sets()
