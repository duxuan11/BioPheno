"""Stage 3（预处理与等权）的测试：全部用合成数据，不依赖真实 Stage 1/2 产物。

两个契约测试（全局拟合不抹平天间差异、motion NaN 列被剔除且记名）取自任务说明；
其余测试盯住实现里最容易出错的地方：剔除的**顺序**（曲率尺度归必须在 power transform 之前）与
按原因互斥的逐列记账（并集 = 被剔除的全部列）、第 3 步非有限列安全网的幂等性、`cfg` 的 σ 阈值确实
被用上、曲率特征确实乘了等效直径、SAM 特征族等权后每个 feature set 的总方差相等，
以及 `run_stage3` 的输入校验与四份产物（含记录全部剔除名字的 `stage3_qc.json`）。
"""

import json

import numpy as np
import pandas as pd
import pytest

from biopheno import config as cfg
from biopheno import preprocess as prep
from SPOT.Analysis import preprocessing as spot_prep

# 剔除原因类别（与实现的 `_DROP_CATEGORIES` 同名）：QC 的不变量要能被独立复核
_DROP_CATEGORY_KEYS = (
    "dropped_zero",
    "dropped_all_nan",
    "dropped_partial_nan",
    "dropped_non_finite",
    "dropped_high_variance",
)


def test_global_scaling_keeps_between_day_difference():
    """守口径：StandardScaler 只能全局拟合一次，天间均值差不能被按天拆分抹平。"""
    # 若 all_conditions 误传 day，两组会被各自 z-score，天间均值差会被抹平。
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, size=(50, 8)), rng.normal(3, 1, size=(50, 8))])
    names = [f"f{i}" for i in range(8)]
    Xp, kept, info = prep.preprocess(X, names)
    gap = abs(Xp[:50].mean() - Xp[50:].mean())
    # 原始均值差约 3、全局 std 约 1.9，全局拟合后 gap 仍应远大于 0.5；按天拆分则 gap ≈ 0。
    assert gap > 0.5, "天间差异被抹平：all_conditions 很可能被按天拆分了"
    # 全局拟合不会让列数或列名发生变化（只剩等权缩放）
    assert kept == names
    assert info["n_features_out"] == 8


def test_nan_motion_columns_are_dropped_and_recorded():
    """守契约：全 NaN 与部分 NaN 的 motion 列都被剔除，并各自记进对应的剔除原因清单。"""
    # 契约测试（取自任务说明）。构造被刻意加强：a/b/c 三列**真的在变**（若是常数会被第 1 步剔掉，
    # 后面的有限性/无 motion 断言就全是空转了）；两个 motion 列一个是全 NaN、一个是部分 NaN，
    # 这样“只删全 NaN 列”（第 2 步的窄化实现）或“删掉第 3 步”都会被抓住。
    rng = np.random.default_rng(7)
    n = 40
    a, b, c = rng.normal(size=(3, n))
    partial = rng.normal(size=n)
    partial[:4] = np.nan
    X = np.column_stack([a, b, c, np.full(n, np.nan), partial])
    names = ["a", "b", "c", "motion_speed", "motion_flow"]
    Xp, kept, info = prep.preprocess(X, names)

    assert kept == ["a", "b", "c"]  # 非空矩阵：下面的断言才有意义
    assert Xp.shape == (n, 3)
    assert np.isfinite(Xp).all()
    # 全 NaN 与部分 NaN 两条路径各有一列被删，且都记了名字
    assert "motion_speed" in info["dropped_all_nan"]
    assert "motion_flow" in info["dropped_partial_nan"]
    assert all("motion" not in name for name in kept)


def test_drop_categories_partition_the_dropped_columns(monkeypatch):
    """守不变量：五类剔除原因互斥，并集恰好等于被剔除的全部列，且与保留列不交。"""
    # 不变量：五个“原因”清单互斥，并集恰好等于被剔除的全部列；n_dropped_features 只由维度相减得出。
    # 用 σ=0 把高方差阈值钉成“均值 std”，让尖刺列被确定地剔掉（不靠随机数值巧合）。
    monkeypatch.setattr(cfg, "VARIANCE_THRESHOLD_SIGMA", 0.0)
    rng = np.random.default_rng(11)
    n = 60
    partial = rng.normal(size=n)
    partial[:3] = np.nan
    inf_col = rng.normal(size=n)
    inf_col[0] = np.inf
    X = np.column_stack(
        [
            rng.normal(size=n),  # keep_a
            rng.normal(size=n),  # keep_b
            np.zeros(n),  # zero_col：第 1 步（全零）
            np.full(n, 5.0),  # const_col：第 1 步（零方差）
            np.full(n, np.nan),  # motion_placeholder：第 2 步（全 NaN）
            partial,  # partial_nan：第 2 步（部分 NaN）
            inf_col,  # inf_col：第 2 步（std=NaN）
            rng.normal(size=n) * 50.0,  # spiky：第 4 步（高方差）
        ]
    )
    names = [
        "keep_a",
        "keep_b",
        "zero_col",
        "const_col",
        "motion_placeholder",
        "partial_nan",
        "inf_col",
        "spiky",
    ]
    _, kept, info = prep.preprocess(X, names)

    assert kept == ["keep_a", "keep_b"]
    assert info["dropped_zero"] == ["zero_col", "const_col"]
    assert info["dropped_all_nan"] == ["motion_placeholder"]
    assert info["dropped_partial_nan"] == ["partial_nan"]
    assert info["dropped_non_finite"] == ["inf_col"]
    assert info["dropped_high_variance"] == ["spiky"]

    dropped = [name for category in _DROP_CATEGORY_KEYS for name in info[category]]
    assert len(dropped) == len(set(dropped)), "剔除类别之间重叠了：同一列被记进两类（双计）"
    assert set(dropped) & set(kept) == set()
    assert set(dropped) | set(kept) == set(names), "有列既没被保留也没被记进任何剔除类别（漏记）"
    assert info["n_dropped_features"] == len(dropped) == info["n_features_in"] - info["n_features_out"]
    for category in _DROP_CATEGORY_KEYS:
        assert info[f"n_{category}"] == len(info[category])
    # 各步口径与原因口径正交：每列只在某一步被删一次，所以合计也等于实际剔除数
    assert sum(info["n_dropped_per_step"].values()) == info["n_dropped_features"]
    # 全 NaN、部分 NaN、含 Inf 三列都是在第 2 步因 std=NaN 出局，所以这一步恰好剔 3 列。
    assert info["n_dropped_per_step"]["remove_nan_variance_features"] == 3
    assert info["n_dropped_per_step"]["remove_non_finite_features"] == 0


def test_non_finite_guard_is_idempotent_after_step2(monkeypatch):
    """守边界：第 2 步因 std=NaN 已清走非有限列，第 3 步安全网实际剔除 0 列。"""
    # 第 3 步是安全网：含 NaN/±Inf 的列在第 2 步就因为 std=NaN 出局了，所以第 3 步实际剔除 0 列。
    # 用 spy 直接看第 2 步的**输出**：它已经全有限，这就是“第 3 步无活可干”的事实依据。
    seen: dict = {}
    real = spot_prep.remove_nan_variance_features

    def _spy(X, names, return_index=False):
        """包一层记录第 2 步输出的列名与是否全有限，然后原样返回结果。"""
        out = real(X, names, return_index=return_index)
        seen["all_finite"] = bool(np.isfinite(out[0]).all())
        seen["names"] = [str(name) for name in out[1]]
        return out

    monkeypatch.setattr(spot_prep, "remove_nan_variance_features", _spy)
    rng = np.random.default_rng(13)
    n = 60
    inf_col = rng.normal(size=n)
    inf_col[0] = np.inf
    X = np.column_stack([rng.normal(size=(n, 3)), np.full(n, np.nan), inf_col])
    names = ["a", "b", "c", "motion_placeholder", "inf_col"]
    _, kept, info = prep.preprocess(X, names)

    assert seen["all_finite"] is True
    assert seen["names"] == ["a", "b", "c"]
    assert info["n_dropped_per_step"]["remove_non_finite_features"] == 0
    # 含 Inf 的列确实进了“非有限”这个原因类别（虽然物理上是第 2 步删的）
    assert info["dropped_non_finite"] == ["inf_col"]
    assert kept == ["a", "b", "c"]


def test_non_finite_guard_drops_non_finite_columns_when_handed_them():
    """守口径：单独喂给 _drop_non_finite 非有限列时必须真删，且对干净矩阵再跑一次幂等。"""
    # 直接单测第 3 步：正常路径上它没有活可干，但拿到非有限列时它必须真的删。
    X = np.array([[1.0, np.nan, np.inf, 4.0], [2.0, 3.0, 5.0, 6.0]])
    names = np.array(["a", "b", "c", "d"], dtype=str)
    out, out_names, dropped = prep._drop_non_finite(X, names)

    assert list(out_names) == ["a", "d"]
    assert out.shape == (2, 2)
    assert np.isfinite(out).all()
    assert dropped == ["b", "c"]
    # 幂等：对已剔除干净的矩阵再跑一次，什么都不删
    out2, names2, dropped2 = prep._drop_non_finite(out, out_names)
    assert dropped2 == []
    assert np.array_equal(out2, out) and list(names2) == ["a", "d"]


def test_mandated_step_order_is_pinned(monkeypatch):
    """守口径：七步顺序被钉死，尤其是曲率尺度归必须在 power transform 之前。"""
    # 顺序是硬约束（曲率尺度归一必须作用在**原始量纲**上，power transform 之后 diameter 已被 z-score，
    # 再乘就错了）。这里把七步依次包一层记录调用顺序。
    calls: list[str] = []
    # (label, 模块, 属性名)：把上游的五步与我们的两个自家步骤（非有限列安全网、曲率尺度归一）
    # 依次包一层，记录完整调用顺序。
    targets: tuple[tuple[str, object, str], ...] = (
        ("remove_zero_features", spot_prep, "remove_zero_features"),
        ("remove_nan_variance_features", spot_prep, "remove_nan_variance_features"),
        ("drop_non_finite", prep, "_drop_non_finite"),
        ("remove_high_variance_features", spot_prep, "remove_high_variance_features"),
        ("curvature_scale", prep, "_curvature_scale"),
        ("power_transform_and_scale_features", spot_prep, "power_transform_and_scale_features"),
        ("weight_sam_features", spot_prep, "weight_sam_features"),
    )

    def _wrap(label, fn):
        """把 fn 包成「先记录 label 再调用原函数」的记录器。"""

        def _recorder(*args, **kwargs):
            """记录本次调用标签，然后转交原函数。"""
            calls.append(label)
            return fn(*args, **kwargs)

        return _recorder

    for label, module, attribute in targets:
        monkeypatch.setattr(module, attribute, _wrap(label, getattr(module, attribute)))

    rng = np.random.default_rng(17)
    names = ["equivalent_diameter", "mean_curvature", "Fourier_1", "mean_intensity"]
    prep.preprocess(rng.normal(size=(40, 4)), names)

    assert calls == [
        "remove_zero_features",
        "remove_nan_variance_features",
        "drop_non_finite",
        "remove_high_variance_features",
        "curvature_scale",
        "power_transform_and_scale_features",
        "weight_sam_features",
    ]


def test_info_records_dropped_names_and_final_dims():
    """守口径：QC 的剔除名字、个数与最终维度三者自洽，且与 kept 列名一致。"""
    # 零值列被第一步剔除；记账要给出名字与个数，且最终维度与 kept 一致。
    rng = np.random.default_rng(1)
    X = np.hstack([rng.normal(size=(60, 5)), np.zeros((60, 1))])
    names = ["a", "b", "equivalent_diameter", "mean_curvature", "Fourier_1", "zero_col"]
    Xp, kept, info = prep.preprocess(X, names)

    assert "zero_col" in info["dropped_zero"]
    assert info["n_features_in"] == 6
    assert info["n_features_out"] == Xp.shape[1] == len(kept) == 5
    assert info["n_dropped_features"] == 1
    assert info["n_dropped_zero"] == 1
    assert info["n_dropped_high_variance"] == len(info["dropped_high_variance"])
    assert info["n_dropped_all_nan"] == len(info["dropped_all_nan"])
    assert info["n_dropped_partial_nan"] == len(info["dropped_partial_nan"])
    assert info["n_dropped_non_finite"] == len(info["dropped_non_finite"])
    assert kept == ["a", "b", "equivalent_diameter", "mean_curvature", "Fourier_1"]
    assert info["n_objects"] == 60


def test_curvature_features_are_scaled_by_equivalent_diameter():
    """守口径：curvature 列乘等效直径，等效直径与 area 等非 curvature 列原样不动。"""
    # 曲率量纲是 length^-1，SPOT 的尺寸归一就是乘等效直径；非 curvature 列必须原样保留。
    X = np.array([[2.0, 3.0, 5.0], [4.0, 0.5, 7.0]])
    names = np.array(["equivalent_diameter", "mean_curvature", "area"], dtype=str)
    out, out_names = prep._curvature_scale(X, names)

    assert np.allclose(out[:, 1], X[:, 1] * X[:, 0])
    assert np.allclose(out[:, 0], X[:, 0])  # 等效直径自己不动
    assert np.allclose(out[:, 2], X[:, 2])  # area 不带 curvature，不缩放
    assert list(out_names) == list(names)


def test_curvature_scaling_requires_equivalent_diameter():
    """守边界：有 curvature 列却缺 equivalent_diameter 时报带中文上下文的 ValueError。"""
    # 有 curvature 列却没有 equivalent_diameter 时，上游会拿空索引做广播，抛一句看不出原因的
    # numpy 错误；必须先自查并给中文上下文。
    X = np.ones((4, 2))
    names = np.array(["mean_curvature", "area"], dtype=str)
    with pytest.raises(ValueError, match="equivalent_diameter"):
        prep._curvature_scale(X, names)


def test_power_transform_is_fitted_on_one_global_condition(monkeypatch):
    """守口径：传给 power transform 的 all_conditions 必须是单一常量，才算全局拟合一次。"""
    # 与契约测试互补：直接盯住 all_conditions 的取值，单一常量才算"全局拟合一次"。
    seen = {}
    real = spot_prep.power_transform_and_scale_features

    def _spy(X, names, all_conditions):
        """拦截 power transform 调用，抄下传入的 all_conditions 再原样转交。"""
        seen["conditions"] = np.asarray(all_conditions).copy()
        return real(X, names, all_conditions)

    monkeypatch.setattr(spot_prep, "power_transform_and_scale_features", _spy)
    rng = np.random.default_rng(2)
    prep.preprocess(rng.normal(size=(40, 4)), [f"f{i}" for i in range(4)])

    conditions = seen["conditions"]
    assert len(conditions) == 40
    assert np.unique(conditions).size == 1
    # 常量取 0（而不是 day 名或递增值）才说明实现真的只喂了一个 condition。
    assert np.all(conditions == 0)


def test_high_variance_threshold_comes_from_config(monkeypatch):
    """守口径：高方差阈值读 cfg.VARIANCE_THRESHOLD_SIGMA，阈值极低时全部列都被剔除。"""
    # 阈值必须读 cfg（论文用 2 sigma），且要按名字记账；把阈值调到远低于任何 std 时所有列都该被扔掉。
    monkeypatch.setattr(cfg, "VARIANCE_THRESHOLD_SIGMA", -100.0)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 4))
    names = ["a", "b", "c", "d"]
    Xp, kept, info = prep.preprocess(X, names)

    assert Xp.shape == (40, 0)
    assert kept == []
    assert sorted(info["dropped_high_variance"]) == names


def test_weighting_equalises_total_variance_per_feature_set():
    """守口径：SAM 特征族等权后每个 feature set 的总方差都等于 1，与家族列数无关。"""
    # shape 有 4 列、appearance 只有 2 列：等权之后两个 set 的总方差都应等于 1，
    # 这样列多的家族不会仅凭列数主导距离。
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 6)) * np.array([1.0, 3.0, 0.5, 2.0, 1.0, 4.0])
    names = ["Fourier_1", "Fourier_2", "EFD_1", "equivalent_diameter", "mean_intensity", "stddev_intensity"]
    Xp, kept, info = prep.preprocess(X, names)

    assert len(kept) == 6 and info["n_features_out"] == 6  # 这批列的 std 相当，σ=2 不该剔入
    appearance_names = {"mean_intensity", "stddev_intensity"}
    shape_cols = [i for i, n in enumerate(kept) if n not in appearance_names]
    appearance_cols = [i for i, n in enumerate(kept) if n in appearance_names]
    assert np.isclose(Xp[:, shape_cols].var(axis=0).sum(), 1.0, atol=1e-6)
    assert np.isclose(Xp[:, appearance_cols].var(axis=0).sum(), 1.0, atol=1e-6)


def test_preprocess_rejects_names_that_disagree_with_columns():
    """守边界：特征名个数与矩阵列数不一致时立即 ValueError，不进入后续步骤。"""
    with pytest.raises(ValueError, match="特征名"):
        prep.preprocess(np.ones((5, 3)), ["a", "b"])


def _write_features(root, names, X):
    """按 Stage 2 的落盘格式写 features/（特征表 + 列名清单）。"""
    outdir = root / "features"
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, index=pd.Index([f"A1_{i}_0" for i in range(len(X))], name="instance_uid"), columns=names)
    df.to_csv(outdir / "all_features.csv")
    (outdir / "feature_names.json").write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")


def _synthetic_feature_table(seed=0, n=60):
    """造一份形如 Stage 2 产物的合成表：返回 10 个特征名与 (n, 10) 矩阵，末尾 2 列是全 NaN 的 motion 占位。"""
    rng = np.random.default_rng(seed)
    names = [
        "equivalent_diameter",
        "mean_curvature",
        "std_curvature",
        "Fourier_1",
        "Fourier_2",
        "mean_intensity",
        "mean_intensity_region_001",
        "sift_appear_1",
        "motion_speed_global",
        "motion_flow_vector_sift_1",
    ]
    X = np.hstack([rng.normal(size=(n, 8)), np.full((n, 2), np.nan)])
    return names, X


def test_run_stage3_writes_artifacts(monkeypatch, tmp_path):
    """守口径：run_stage3 落盘四份产物，且矩阵维度、QC 记账与 motion 剔除三者一致。"""
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    names, X = _synthetic_feature_table()
    _write_features(tmp_path, names, X)

    prep.run_stage3()

    outdir = tmp_path / "preprocessed"
    for fname in ("X.npy", "keep_features.json", "all_conditions.npy", "stage3_qc.json"):
        assert (outdir / fname).exists()

    Xp = np.load(outdir / "X.npy")
    kept = json.loads((outdir / "keep_features.json").read_text(encoding="utf-8"))
    conditions = np.load(outdir / "all_conditions.npy")
    qc = json.loads((outdir / "stage3_qc.json").read_text(encoding="utf-8"))

    assert Xp.shape == (len(X), len(kept))  # 行数不变、列数 = 保留列名个数
    assert Xp.shape[1] == qc["n_features_out"] == qc["n_features_in"] - qc["n_dropped_features"]
    assert qc["n_features_in"] == len(names)
    assert np.isfinite(Xp).all()
    # 两个 motion 占位列都是全 NaN，名字必须出现在 QC 里（报告里点名的最大风险）
    assert "motion_speed_global" in qc["dropped_all_nan"]
    assert "motion_flow_vector_sift_1" in qc["dropped_all_nan"]
    assert not any("motion" in name for name in kept)
    assert sum(len(qc[category]) for category in _DROP_CATEGORY_KEYS) == qc["n_dropped_features"]
    assert qc["n_dropped_per_step"]["remove_non_finite_features"] == 0
    assert "single global condition" in qc["all_conditions"]
    assert qc["variance_threshold_sigma"] == pytest.approx(cfg.VARIANCE_THRESHOLD_SIGMA)
    assert conditions.shape == (Xp.shape[0],)
    assert np.unique(conditions).size == 1  # 全局拟合：下游能看出没有按天/按视野拆分
    assert Xp.dtype == np.float64


def test_run_stage3_prints_and_persists_high_variance_names(monkeypatch, tmp_path, capsys):
    """守口径：高方差剔除的名字既打印到 stdout，也落进 stage3_qc.json，不能只留一处。"""
    # 高方差剔除删的是“有值且变异大”的列，最容易被忽略，所以名字必须打印出来（至少前 N 个）
    # 并落在 QC 里。σ=0 把阈值钉成“均值 std”，尖刺列被确定地剔掉。
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(cfg, "VARIANCE_THRESHOLD_SIGMA", 0.0)
    rng = np.random.default_rng(5)
    names = [f"Fourier_{i}" for i in range(5)] + ["mean_intensity"]
    X = rng.normal(size=(40, 6)) * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 12.0])
    _write_features(tmp_path, names, X)

    prep.run_stage3()

    qc = json.loads((tmp_path / "preprocessed" / "stage3_qc.json").read_text(encoding="utf-8"))
    assert qc["dropped_high_variance"] == ["mean_intensity"], "构造失败：这个用例应当只剔掉尖刺列"
    out = capsys.readouterr().out
    assert "mean_intensity" in out


def test_run_stage3_reports_degenerate_drops_consistently(monkeypatch, tmp_path):
    """守边界：所有列被剔完时报错，且错误里的合计剔除数与分类记账不自相矛盾。"""
    # 退化：所有列都被剔掉时，错误信息里的各步剔除数必须自洽（不能出现分类重复计数的和）。
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(cfg, "VARIANCE_THRESHOLD_SIGMA", -100.0)
    names, X = _synthetic_feature_table()
    _write_features(tmp_path, names, X)

    with pytest.raises(ValueError, match="没有任何特征剩下") as excinfo:
        prep.run_stage3()
    assert "合计剔除了 10 列" in str(excinfo.value)
    # 10 = 输入 10 列全被剔；这个数必须来自「输入维度 - 输出维度」，而不是把分类计数相加。


def test_run_stage3_rejects_non_list_name_file(monkeypatch, tmp_path):
    """守边界：feature_names.json 不是名字列表时给带上下文的中文 ValueError。"""
    # feature_names.json 被写成对象/标量时，迭代出来的是 dict 的键或直接崩；必须给带上下文的中文错误。
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    names, X = _synthetic_feature_table()
    _write_features(tmp_path, names, X)
    (tmp_path / "features" / "feature_names.json").write_text(json.dumps({"names": names}), encoding="utf-8")

    with pytest.raises(ValueError, match="名字列表"):
        prep.run_stage3()


def test_run_stage3_wraps_csv_decode_error(monkeypatch, tmp_path):
    """守边界：非 UTF-8 的 all_features.csv 要换成中文 ValueError，不能冒出裸 UnicodeDecodeError。"""
    # 非 UTF-8 的 CSV（二进制损坏/被别的编码重写过）不能冒出裸 UnicodeDecodeError。
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    outdir = tmp_path / "features"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "all_features.csv").write_bytes(b"instance_uid,Fourier_1\nA1_0_0,1.0\nA1_1_0,\xff\xfe2.0\n")

    with pytest.raises(ValueError, match="读不出来"):
        prep.run_stage3()


def test_run_stage3_requires_stage2(monkeypatch, tmp_path):
    """守边界：缺 Stage 2 产物时提示先跑 Stage 2，而不是冒出裸 FileNotFoundError。"""
    # 没有 all_features.csv 时不能冒出裸 FileNotFoundError，要说清"先跑 Stage 2"。
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="Stage2") as excinfo:
        prep.run_stage3()
    assert "先跑" in str(excinfo.value)


def test_run_stage3_rejects_stale_name_file(monkeypatch, tmp_path):
    """守边界：列名清单与特征表对不上时拒绝，避免按名取列静默错位。"""
    # 列名清单与特征表的列对不上（Stage 2 中途被改过）时，按名取列会静默错位，必须拒绝。
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    names, X = _synthetic_feature_table()
    _write_features(tmp_path, names, X)
    (tmp_path / "features" / "feature_names.json").write_text(json.dumps(names[:-1]), encoding="utf-8")

    with pytest.raises(ValueError, match="列名"):
        prep.run_stage3()
