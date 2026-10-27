"""Stage 5（UMAP 景观 + k-means 表型簇）的测试：全部合成数据，不依赖真实 Stage 1–4 产物。

两个契约测试（两个 blob 能被切成 2 簇且按 score 重排、UMAP 坐标可复现）取自任务说明；
其余测试盯住实现里最容易悄悄错的地方：

- UMAP 参数必须**照论文原样**传给 `umap.UMAP`（不许为了"图好看"偷偷加 n_jobs/n_components）；
- k-means 必须走 SPOT 的 elbow 扫描且 `ploton=False`（否则每个任务都会弹图、在无头环境里挂住）；
- 行身份：`umap_2d.csv` 必须带 `instance_uid`，Stage 5/6/9 不再靠"猜行序"；
- 簇排序用平均 eccentricity（论文口径），而且必须是**原生量纲**：`all_features.csv` 的
  `moment_eccentricity` → bbox 推导值 →（最后手段）Stage 3 预处理列；三套簇均值与单调性都要落进 QC；
- `per_instance_features.csv` 的 `eccentricity` 列与排序 score 是两件事：它只要能用的原生量纲，
  路径单独记在 `eccentricity_path`（缺列时是 `bbox`，而不是跟着 score 的 fallback 状态乱写）；
- `run_stage5` 的输入自检：缺 `X.npy`、或与 `all_features.csv` 行数不符时给带上下文的中文错误。
"""

import json

import matplotlib

# SPOT 的密度函数里 `plt.show()` 是无条件调用的；交互式后端（本机默认 TkAgg）会真的开窗口。
# 必须在任何 pyplot / matplotlib 相关导入之前设置。
matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

from biopheno import config as cfg
from biopheno import landscape as land
from SPOT.Analysis import sam_analysis as sam

# —— 合成数据的列名（Stage 2 的原始特征表）——
_AREA = "area"
_ECC = "moment_eccentricity"
_INTENSITY = "mean_intensity"
_RAW_COLUMNS: tuple[str, ...] = (_AREA, _ECC, _INTENSITY, "f1", "f2", "f3", "f4")

# 三个 (day, fov) 组合，每个 40 个实例 = 120 行（> cfg.UMAP_PARAMS 的 n_neighbors=100，
# 免得 UMAP 每次都发"n_neighbors 超过样本数"的截断警告）
_GROUPS: tuple[tuple[str, str], ...] = (("day0", "A1"), ("day3", "A2"), ("day5", "A3"))
_PER_GROUP: int = 40


def _uid(fov: str, label: int, day: str) -> str:
    """Stage 1 的 uid 格式（Spec D13）：`{fov}_{label}_{frame_idx}`，frame_idx = cfg.DAYS[day]。"""
    return f"{fov}_{int(label)}_{int(cfg.DAYS[day])}"


def _uids() -> list[str]:
    """按 `_GROUPS` × label 1..40 生成 Stage 1 的 uid 列表（顺序 = 写盘顺序 = X 的行序）。"""
    return [_uid(fov, label, day) for day, fov in _GROUPS for label in range(1, _PER_GROUP + 1)]


def _write_preprocessed(root, names, X):
    """写 Stage 3 的 `X.npy` 与 `keep_features.json`：列序即 UMAP/k-means 的输入列序。"""
    outdir = root / "preprocessed"
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "X.npy", np.asarray(X, dtype=np.float64))
    (outdir / "keep_features.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")


def _write_stage2(root, uids, raw_columns=_RAW_COLUMNS, n_rows=None):
    """写 Stage 2 的 `all_features.csv`（instance_uid 索引 + 原始特征列）。

    `n_rows` 给定时只写前 n 行，用来模拟"X.npy 与特征表行数对不上"。
    """
    n = len(uids) if n_rows is None else int(n_rows)
    rng = np.random.default_rng(11)
    data = rng.normal(size=(n, len(raw_columns)))
    frame = pd.DataFrame(data, index=pd.Index(list(uids)[:n], name="instance_uid"), columns=list(raw_columns))
    # 换成物理量纲，免得"取错了列"这类错误被量纲差异掩盖
    if _AREA in frame:
        frame[_AREA] = rng.uniform(200.0, 3000.0, n)
    if _ECC in frame:
        frame[_ECC] = rng.uniform(0.0, 1.0, n)
    if _INTENSITY in frame:
        frame[_INTENSITY] = rng.uniform(0.0, 255.0, n)
    outdir = root / "features"
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "all_features.csv")
    return frame


def _write_instances(root, uids, wide_first=0):
    """写 Stage 1 的 `all_instances.csv`（bbox 齐全）。

    `wide_first` 个实例给扁长 bbox（高 20 × 宽 100 → bbox 偏心率约 0.98），其余给正方形 bbox
    （偏心率 0）；同时让扁长实例的面积**更小**，这样"按面积排"与"按偏心率排"会给出相反顺序，
    测试才能分辨实现到底用了哪个 score。
    """
    rows = []
    for i, uid in enumerate(uids):
        fov, label, frame = uid.split("_")
        day = next(name for name, idx in cfg.DAYS.items() if int(idx) == int(frame))
        if i < wide_first:
            r0, c0, r1, c1 = 0, 0, 19, 99  # 20 × 100
        else:
            r0, c0, r1, c1 = 0, 0, 39, 39  # 40 × 40
        rows.append(
            {
                "instance_uid": uid,
                "day": day,
                "fov": fov,
                "frame_idx": int(frame),
                "time_hours": cfg.TIME_HOURS[day],
                "label": int(label),
                "centroid_row": (r0 + r1) / 2.0,
                "centroid_col": (c0 + c1) / 2.0,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "area_px": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                "touches_border": False,
                "is_small": False,
                "n_contours_discarded": 0,
            }
        )
    outdir = root / "instances"
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / "all_instances.csv", index=False)


def _build_inputs(root, *, kept_names=None, raw_columns=_RAW_COLUMNS, uids=None, wide_first=0):
    """布线 Stage 1/2/3 的落盘产物，返回 `(uids, X)`（X 的行序 = uids 的顺序 = Stage 2 的行序）。"""
    uids = _uids() if uids is None else list(uids)
    raw_columns = list(raw_columns)
    kept_names = list(raw_columns) if kept_names is None else list(kept_names)
    raw = _write_stage2(root, uids, raw_columns=raw_columns)
    _write_instances(root, uids, wide_first=wide_first)
    X = raw[kept_names].to_numpy(dtype=np.float64)
    _write_preprocessed(root, kept_names, X)
    return uids, X


def _read_json(path):
    """读 UTF-8 JSON（`kmeans_metrics.json` / QC），返回 dict。"""
    return json.loads(path.read_text(encoding="utf-8"))


# —— 任务说明给的两个契约测试 ——


def test_clusters_recover_two_blobs_and_are_ordered_by_score():
    """守口径：两个 blob 被胘部法切成 2 簇，并按平均 score 升序重排成 1..K。"""
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(0, 0.3, (60, 2)), rng.normal(6, 0.3, (60, 2))])
    labels, metrics = land.cluster(emb, k_range=(2, 4))
    assert metrics["k"] == 2
    assert len(np.unique(labels)) == 2
    score = np.where(labels == labels[0], 0.2, 0.9)
    # 0.2 vs 0.9：两簇均值差得够远，并列不会把"高得分簇拿 2 号"的断言搞成玄学
    ordered, mapping = land.order_clusters_by_score(labels, score)
    assert ordered.min() == 1 and ordered[score > 0.5].max() == 2
    # 映射表要和重排结果自洽：平均 score 高的旧簇必须映射到 2
    high_score_old_label = int(np.unique(labels[score > 0.5])[0])
    assert dict(map(tuple, mapping.tolist()))[high_score_old_label] == 2


def test_embedding_is_deterministic():
    """守复现：固定 random_state 下两次 embed 必须给出逐元素一致的 (N, 2) 坐标。"""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 12))
    a = land.embed(X)
    b = land.embed(X)
    assert a.shape == (120, 2)
    # atol=1e-6：允许浮点末位抖动，但不允许坐标整体漂移（离线重画图也必须一致）
    assert np.allclose(a, b, atol=1e-6)


# —— UMAP / k-means 的调用契约 ——


def test_embed_passes_paper_umap_params_verbatim(monkeypatch):
    """UMAP 参数必须原样来自 cfg（一个都不多、一个都不少），否则"论文参数"就名不副实。"""
    recorded: dict = {}
    real_umap = land.umap.UMAP

    class _Spy:
        """spy-stub：记录构造 kwargs，再转给真实 `umap.UMAP` 干活。"""

        def __init__(self, **kwargs):
            """把 kwargs 抄一份给断言，并持有真实 UMAP 实例。"""
            recorded.update(kwargs)
            self._inner = real_umap(**kwargs)

        def fit_transform(self, X):
            """转调真实 UMAP 的 `fit_transform`，保证返回值形状不变。"""
            return self._inner.fit_transform(X)

    monkeypatch.setattr(land.umap, "UMAP", _Spy)
    X = np.random.default_rng(2).normal(size=(120, 6))
    out = land.embed(X)
    # 全等比较：多传 n_jobs/n_components 之类"为了图好看"的参数也会让"论文参数"名不副实
    assert recorded == cfg.UMAP_PARAMS
    assert out.shape == (120, 2)


def test_cluster_uses_spot_scan_with_ploton_false(monkeypatch):
    """必须真的走 SPOT 的 elbow 扫描，且 `ploton=False`（否则每个调用点都会弹图）。"""
    calls: list[dict] = []
    real_scan = sam.KMeans_cluster_low_dimensional_embedding_with_scan

    def _spy(low_dimensional_feats, k_range=(2, 20), ploton=True):
        """spy-stub：记录每次调用的 k_range / ploton 后转交 SPOT 原函数。"""
        calls.append({"k_range": tuple(k_range), "ploton": ploton})
        return real_scan(low_dimensional_feats, k_range=k_range, ploton=ploton)

    monkeypatch.setattr(land.sam, "KMeans_cluster_low_dimensional_embedding_with_scan", _spy)
    rng = np.random.default_rng(3)
    emb = np.vstack([rng.normal(0, 0.3, (60, 2)), rng.normal(6, 0.3, (60, 2))])
    original_scan = sam.chooseBestKforKMeansParallel
    labels, metrics = land.cluster(emb, k_range=(2, 4))
    # 只调一次、k_range 原样透传（上界不收盘，扫描范围完全由 SPOT 决定）
    assert calls == [{"k_range": (2, 4), "ploton": False}]
    # pandas-3 兼容补丁只在调用期间生效，绝不能泄到进程里其他调用点
    assert sam.chooseBestKforKMeansParallel is original_scan
    assert "pandas3_compat" in metrics["elbow_shim"]
    assert metrics["k"] == 2
    assert labels.shape == (120,)


def test_cluster_k_matches_argmin_of_scaled_inertia():
    """钉住 elbow shim 的**选择规则**：选中的 k = 扫描列里 adjusted inertia 最小的 k。"""
    rng = np.random.default_rng(7)
    emb = np.vstack([rng.normal(0, 0.3, (50, 2)), rng.normal(6, 0.3, (50, 2)), rng.normal(12, 0.3, (50, 2))])
    labels, metrics = land.cluster(emb, k_range=(2, 6))
    ks = metrics["k_scanned"]
    # 三个映射必须用同一套 k 键，否则"哪个数对应哪个 k"就靠数位置了
    assert list(metrics["scaled_inertia"]) == [str(k) for k in ks]
    assert list(metrics["inertia"]) == [str(k) for k in ks]
    assert list(metrics["silhouette"]) == [str(k) for k in ks]
    scaled = np.array([metrics["scaled_inertia"][str(k)] for k in ks])
    assert metrics["k"] == ks[int(np.argmin(scaled))]
    # chosen_k 与 k 是同一决定的两种记法（一个给外层 QC、一个给旧调用点），写重了也必须一致
    assert metrics["chosen_k"] == metrics["k"]
    assert len(np.unique(labels)) == metrics["k"]


def test_cluster_restores_spot_function_when_scan_raises(monkeypatch):
    """扫描报错时也必须还原 SPOT 的内部函数（补丁不能因为异常泄漏到进程里）。"""
    original = sam.chooseBestKforKMeansParallel

    def _boom(low_dimensional_feats, k_range):
        """stub：直接抛错，模拟 SPOT 的肘部扫描在补丁生效期间失败。"""
        raise RuntimeError("模拟 elbow 扫描失败")

    monkeypatch.setattr(land, "_choose_best_k_pandas3", _boom)
    emb = np.random.default_rng(8).normal(size=(40, 2))
    with pytest.raises(RuntimeError, match="模拟 elbow 扫描失败"):
        land.cluster(emb, k_range=(2, 4))
    assert sam.chooseBestKforKMeansParallel is original


def test_cluster_reports_inertia_and_silhouette_for_every_scanned_k():
    """守口径：k_range=(2,5) 只扫 2/3/4（SPOT 不含上界），且每个 k 都有 inertia 与 silhouette。"""
    rng = np.random.default_rng(4)
    emb = np.vstack([rng.normal(0, 0.3, (60, 2)), rng.normal(6, 0.3, (60, 2))])
    labels, metrics = land.cluster(emb, k_range=(2, 5))
    # SPOT 的扫描是 `range(k_range[0], k_range[1])`——**不含上界**，k_range=(2,5) 只扫 2/3/4
    assert metrics["k_scanned"] == [2, 3, 4]
    assert len(metrics["inertia"]) == len(metrics["silhouette"]) == 3
    # 用 `{str(k): 值}` 映射钉住"数值 ↔ k"，避免 list 与 k_scanned 错位时静默挂错 k
    inertia = [metrics["inertia"][str(k)] for k in metrics["k_scanned"]]
    silhouette = [metrics["silhouette"][str(k)] for k in metrics["k_scanned"]]
    assert all(np.isfinite(inertia)) and all(np.isfinite(silhouette))
    # 惯性随 k 单调不增（k 越大，簇内平方和越小）
    assert inertia == sorted(inertia, reverse=True)
    assert metrics["chosen_k"] == metrics["k"] == 2
    assert labels.min() == 0 and labels.max() == metrics["k"] - 1


def test_cluster_rejects_degenerate_k_range():
    """守边界：k_range=(3,3) 是空扫描区间（不含上界），必须报"k 范围"错误。"""
    # 数据本身无关紧要：退化的是扫描区间，不是点集
    emb = np.random.default_rng(5).normal(size=(30, 2))
    with pytest.raises(ValueError, match="k 范围"):
        land.cluster(emb, k_range=(3, 3))


def test_order_clusters_mapping_table_is_consistent_with_relabeled_ids():
    """守口径：映射表是 (旧标签, 新标签) 两列、与重排结果逐行自洽，编号 1-based 且按 score 升序。"""
    clusters = np.array([0] * 4 + [1] * 4 + [2] * 4)
    score = np.array([0.5] * 4 + [0.1] * 4 + [0.9] * 4)
    ordered, mapping = land.order_clusters_by_score(clusters, score)
    # 论文口径 1..K：SPOT 原生的 0-based 排序号被 +1，最小簇号必须是 1
    assert ordered.min() == 1
    assert set(np.unique(ordered)) == {1, 2, 3}
    # 映射表是 (旧标签, 新标签) 两列的数组，且与重排结果逐行一致
    assert mapping.shape == (3, 2)
    for old, new in mapping:
        assert np.all(ordered[clusters == old] == new)
    # 升序：score 最小的旧簇 1 拿 1 号，最大的旧簇 2 拿 3 号
    mapping_dict = {int(old): int(new) for old, new in mapping}
    assert mapping_dict == {1: 1, 0: 2, 2: 3}


def test_order_clusters_reports_contextual_error_when_mapping_table_is_truncated(monkeypatch):
    """SPOT 返回的映射表项数与簇数对不上时，要给带上下文的中文错误（而不是 numpy reshape 的英文堆栈）。"""

    def _truncated(clusters, score, order="ascending"):
        """stub：标签不变，但映射表只给 1 项（模拟 SPOT 返回的映射表被截断）。"""
        return np.asarray(clusters), {0: 0}  # 编号合法，但映射表少了两项

    monkeypatch.setattr(land.sam, "reorder_and_relabel_clusters", _truncated)
    clusters = np.array([0] * 4 + [1] * 4 + [2] * 4)
    score = np.array([0.1] * 4 + [0.5] * 4 + [0.9] * 4)
    with pytest.raises(ValueError, match="簇映射表"):
        land.order_clusters_by_score(clusters, score)


def test_numbering_qc_flags_tied_cluster_scores():
    """簇均值并列时编号是任意的：只 flag（不报错），并仍把 score 单调性记为 True。"""
    labels = np.array([1, 1, 2, 2])
    # 均值全为 0.5：单调性平凡成立，但必须 flag 成 tied（编号本身是任意的）
    qc = land._numbering_qc(labels, score=np.full(4, 0.5), native=None, preprocessed=None, source="bbox")
    assert qc["tied_score_means"] is True
    assert qc["monotone_score"] is True
    assert qc["monotone_native"] is None
    assert qc["monotone_preprocessed"] is None


def test_per_instance_eccentricity_path_is_independent_of_score_path():
    """per-instance 的 eccentricity 路径只看"原生列在不在"，不看排序 score 是否 fallback。"""
    bbox = np.array([0.9, 0.8])
    # 原生列与 bbox 都在：必须用原生列（review finding 1 的那个场景），路径不能写成 bbox
    values, present, path = land._per_instance_eccentricity(pd.DataFrame({_ECC: [0.1, 0.2]}), bbox_ecc=bbox)
    assert present is True and path == _ECC
    assert np.allclose(values, [0.1, 0.2])
    # 原生列缺失：用 bbox
    values, present, path = land._per_instance_eccentricity(pd.DataFrame({"x": [0, 0]}), bbox_ecc=bbox)
    assert present is False and path == "bbox"
    assert np.allclose(values, bbox)
    # 两者都没有：NaN + 无路径（QC 的 missing 会记名）
    values, present, path = land._per_instance_eccentricity(pd.DataFrame({"x": [0, 0]}), bbox_ecc=None)
    assert present is False and path is None
    assert np.all(np.isnan(values))


# —— 密度图（Fig. 3 的景观密度）——


def test_density_returns_per_condition_field_and_leaves_no_figures():
    """守口径：每个条件一个密度场（长度 = 该条件点数）+ (2, N) 掩码，且不留下 SPOT 开的 figure。"""
    rng = np.random.default_rng(6)
    emb = rng.normal(size=(30, 2))
    conditions = np.array(["a"] * 15 + ["b"] * 15)
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    densities, select = land.density(emb, conditions)
    assert len(densities) == 2 and select.shape == (2, 30)
    # 两个条件的掩码必须把 30 个点无重不漏地分完（加起来正好 30）
    assert select.sum() == 30
    assert all(d.shape == (15,) for d in densities)
    assert all(np.all(np.isfinite(d)) for d in densities)
    # SPOT 每个条件开一个 figure 且从不关闭：包装层必须收拾干净
    assert set(plt.get_fignums()) == before


def test_density_without_conditions_treats_all_points_as_one_condition():
    """`all_conditions=None` 时 SPOT 把全部点当一个条件：返回 1 个密度场 + (1, N) 全 True 掩码。"""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(9)
    emb = rng.normal(size=(30, 2))
    before = set(plt.get_fignums())
    densities, select = land.density(emb)
    assert len(densities) == 1
    assert select.shape == (1, 30) and bool(select.all())
    assert densities[0].shape == (30,)
    assert np.all(np.isfinite(densities[0]))
    assert set(plt.get_fignums()) == before


# —— run_stage5 的产物与输入自检 ——


def test_run_stage5_writes_uid_tagged_artifacts(monkeypatch, tmp_path):
    """守落盘：四份产物都带 instance_uid，且簇编号 1..K 在原生离心率均值上递增。"""
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    uids, X = _build_inputs(tmp_path)

    land.run_stage5()

    out = tmp_path / "analysis"
    umap_df = pd.read_csv(out / "umap_2d.csv")
    # 行身份必须落盘：下游不许再假设行序
    assert list(umap_df.columns) == ["instance_uid", "umap_1", "umap_2"]
    assert umap_df["instance_uid"].tolist() == uids
    assert umap_df.shape[0] == X.shape[0]

    labels_df = pd.read_csv(out / "cluster_labels.csv")
    assert labels_df["instance_uid"].tolist() == uids
    # 1-based 编号（论文口径：1..K 按平均 eccentricity 递增）
    assert labels_df["cluster"].min() == 1
    assert sorted(labels_df["cluster"].unique()) == list(range(1, labels_df["cluster"].max() + 1))
    # 0-based 的原始 k-means 标签也留着，方便核对重排
    # cluster_raw 必须从 0 起（与 cluster 的 1-based 形成对照，一眼看出重排有没有生效）
    assert labels_df["cluster_raw"].min() == 0
    # day/fov/time_hours 从 all_instances.csv 按 uid 连接进来（Stage 6/7 画分面图要用）
    assert labels_df["day"].iloc[0] == "day0" and labels_df["fov"].iloc[0] == "A1"
    assert labels_df["fov"].tolist() == [u.split("_")[0] for u in uids]
    assert labels_df["time_hours"].iloc[-1] == cfg.TIME_HOURS["day5"]

    per_inst = pd.read_csv(out / "per_instance_features.csv")
    assert per_inst["instance_uid"].tolist() == uids
    assert {_AREA, "eccentricity", "intensity", "speed"} <= set(per_inst.columns)
    assert np.allclose(per_inst[_AREA].to_numpy(), X[:, 0])  # area 是 Stage 2 的原始列
    # 解释特征表里的 eccentricity 也必须是原始量纲（而不是 Stage 3 标准化之后的 score）
    assert np.allclose(per_inst["eccentricity"].to_numpy(), X[:, 1])
    # motion 被有意跳过（Spec §6-Stage2）：speed 恒为 NaN 占位
    assert per_inst["speed"].isna().all()

    metrics = _read_json(out / "kmeans_metrics.json")
    assert metrics["chosen_k"] == metrics["k"]
    assert len(metrics["inertia"]) == len(metrics["silhouette"]) == len(metrics["k_scanned"])
    assert metrics["umap_params"] == cfg.UMAP_PARAMS
    # 簇排序用的是 Stage2 的**原生** eccentricity 列（原生单位 = 论文编号口径）
    assert metrics["eccentricity_score"]["source"] == _ECC
    assert metrics["eccentricity_score"]["feature"] == _ECC
    assert "all_features.csv" in metrics["eccentricity_score"]["description"]
    assert metrics["eccentricity_score"]["monotone_native"] is True
    assert metrics["eccentricity_score"]["monotone_score"] is True
    # per-instance 的 eccentricity 列走原生列（与排序 score 是两个不同的决策）
    assert metrics["per_instance_features"]["eccentricity_path"] == _ECC
    assert metrics["per_instance_features"]["source_columns"][_ECC] is True
    assert metrics["n_objects"] == X.shape[0]
    assert metrics["row_identity"]["source"] == "features/all_features.csv 的 instance_uid 索引"
    # 论文口径：1..K 的簇均值在**原生**偏心率上递增
    native = pd.read_csv(tmp_path / "features" / "all_features.csv")[_ECC].to_numpy(dtype=np.float64)
    means = labels_df.assign(ecc=native).groupby("cluster")["ecc"].mean().sort_index()
    assert means.is_monotonic_increasing


def test_run_stage5_requires_preprocessed_matrix(monkeypatch, tmp_path):
    """守前置：缺 Stage3 的 `X.npy` 时给带 Stage3 上下文的中文错误。"""
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="Stage3"):
        land.run_stage5()


def test_run_stage5_rejects_row_count_mismatch(monkeypatch, tmp_path):
    """守自检：特征表比 `X.npy` 少 7 行（Stage2 重跑而 Stage3 是旧的）时报带"行"上下文的错误。"""
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    uids, _X = _build_inputs(tmp_path)
    # 特征表少 7 行（例如 Stage 2 被重跑过、而 Stage 3 的 X.npy 还是旧的）
    _write_stage2(tmp_path, uids, n_rows=len(uids) - 7)
    with pytest.raises(ValueError, match="行"):
        land.run_stage5()


def test_run_stage5_falls_back_to_bbox_eccentricity(monkeypatch, tmp_path):
    """原生 `moment_eccentricity` 不可用（Stage2 没这列）时，排序 score 退回 bbox 推导的偏心率。"""
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    raw_columns = (_AREA, "f1", "f2", "f3")
    kept = [_AREA, "f1", "f2"]
    _build_inputs(tmp_path, raw_columns=raw_columns, kept_names=kept, wide_first=60)

    land.run_stage5()

    out = tmp_path / "analysis"
    metrics = _read_json(out / "kmeans_metrics.json")
    assert metrics["eccentricity_score"]["source"] == "bbox"
    assert metrics["eccentricity_score"]["feature"] is None
    assert "bbox" in metrics["eccentricity_score"]["description"]
    # per-instance 列也只能用 bbox（原生列确实不存在）
    assert metrics["per_instance_features"]["eccentricity_path"] == "bbox"
    assert metrics["per_instance_features"]["source_columns"][_ECC] is False
    assert metrics["eccentricity_score"]["monotone_native"] is True

    # 从 bbox 独立复算偏心率，验证簇编号确实按它升序（而不是按面积）
    meta = pd.read_csv(tmp_path / "instances" / "all_instances.csv")
    w = (meta["c1"] - meta["c0"] + 1).to_numpy(dtype=np.float64)
    h = (meta["r1"] - meta["r0"] + 1).to_numpy(dtype=np.float64)
    ecc = np.sqrt(1.0 - (np.minimum(w, h) / np.maximum(w, h)) ** 2)
    labels_df = pd.read_csv(out / "cluster_labels.csv")
    means = labels_df.assign(ecc=ecc).groupby("cluster")["ecc"].mean().sort_index()
    assert means.is_monotonic_increasing
    # 扁长 bbox 的实例（ecc≈0.98）集中在编号更大的簇里
    assert labels_df.assign(ecc=ecc).loc[labels_df["cluster"] == labels_df["cluster"].max(), "ecc"].mean() > 0.5


def test_run_stage5_uses_native_eccentricity_when_dropped_by_variance_filter(monkeypatch, tmp_path):
    """Stage3 高方差过滤拿掉了 `moment_eccentricity`，但 Stage2 的原生列还在：
    排序 score 与 per-instance 列都走原生列，而且**完全不需要 bbox**（这里直接把 bbox 列删掉）。"""
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    _build_inputs(tmp_path, kept_names=[_AREA, _INTENSITY, "f1", "f2"])
    inst_path = tmp_path / "instances" / "all_instances.csv"
    meta = pd.read_csv(inst_path).drop(columns=["r0", "c0", "r1", "c1"])
    meta.to_csv(inst_path, index=False)

    land.run_stage5()

    out = tmp_path / "analysis"
    metrics = _read_json(out / "kmeans_metrics.json")
    assert metrics["eccentricity_score"]["source"] == _ECC
    # 关键（review finding 1）：per-instance 路径是自有决策，不能跟着 score 的 fallback 状态写成 bbox
    assert metrics["per_instance_features"]["eccentricity_path"] == _ECC
    assert metrics["per_instance_features"]["source_columns"][_ECC] is True
    assert metrics["eccentricity_score"]["monotone_native"] is True
    labels_df = pd.read_csv(out / "cluster_labels.csv")
    native = pd.read_csv(tmp_path / "features" / "all_features.csv")[_ECC].to_numpy(dtype=np.float64)
    means = labels_df.assign(ecc=native).groupby("cluster")["ecc"].mean().sort_index()
    assert means.is_monotonic_increasing


def test_cluster_numbering_follows_native_eccentricity_not_preprocessed(monkeypatch, tmp_path):
    """原生量纲 vs 预处理量纲可能给出相反的簇均值次序（两簇分布宽度不同）。

    构造：簇 A 的原生偏心率 {0, 10}（均值 5）、簇 B 恒为 5.5；两者平方（逐行单调）后均值变成
    50 vs 30.25，次序翻转。编号必须跟着**原生**（A = 1、B = 2），同时把 `monotone_preprocessed`
    记成 false（只 warn + flag，不改编号）。
    """
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    uids = _uids()
    native = np.array([0.0] * 30 + [10.0] * 30 + [5.5] * 60)
    raw = _write_stage2(tmp_path, uids)
    raw[_ECC] = native  # Stage2 落盘的是原生量纲
    raw.to_csv(tmp_path / "features" / "all_features.csv")
    _write_instances(tmp_path, uids)

    names = list(_RAW_COLUMNS)
    X = raw[names].to_numpy(dtype=np.float64)
    # Stage3 的预处理列：平方（非负区间上逐行单调递增，但会放大散布 → 翻转簇均值次序）
    X[:, names.index(_ECC)] = native**2
    _write_preprocessed(tmp_path, names, X)

    rng = np.random.default_rng(10)
    emb = np.vstack([rng.normal(0, 0.1, (60, 2)), rng.normal(6, 0.1, (60, 2))])
    monkeypatch.setattr(land, "embed", lambda _X: emb)

    land.run_stage5()

    out = tmp_path / "analysis"
    metrics = _read_json(out / "kmeans_metrics.json")
    numbering = metrics["eccentricity_score"]
    assert metrics["n_clusters"] == 2
    assert numbering["source"] == _ECC
    assert numbering["monotone_native"] is True
    assert numbering["monotone_preprocessed"] is False
    assert np.isclose(numbering["mean_per_cluster_native"]["1"], 5.0)
    assert np.isclose(numbering["mean_per_cluster_native"]["2"], 5.5)
    labels_df = pd.read_csv(out / "cluster_labels.csv")
    # 编号 1 = 原生均值更小的簇 A（前 60 行）
    assert (labels_df["cluster"].to_numpy()[:60] == 1).all()
    assert (labels_df["cluster"].to_numpy()[60:] == 2).all()
