"""Stage 4（SAM 模块发现，legacy 选择器）的测试：全部合成数据，不依赖真实 Stage 1–3 产物。

重点盯住四件容易悄悄错的事：
- 必须显式走 `module_selection='legacy'` + 论文参数（v2.0.1 默认是 multiscale，静默换掉会改模块数）；
- 模块输出必须与输入列序（= `keep_features.json` 的顺序）对齐；
- 行身份必须来自 Stage 2 的 `instance_uid`，**不是** `all_instances.csv` 的行序：Stage 1 给每个标签都留
  meta 行，Stage 2 只给有可用轮廓的实例算特征，两者行数一般不同；
- 输入自检：特征重名会让 `df_corr` 的 index 静默折叠，必须先拒绝。
"""

import json

import matplotlib

# SPOT 的 legacy 路径每调一次就 `sns.clustermap` 一次，而交互式后端（本机默认 TkAgg）会真的开窗口：
# 窗口 + sklearn 的线程池在 Windows 上会把进程卡住（实测单个用例从 1s 涨到 150s+，且抛 Tk 的
# "main thread is not in main loop"）。本模块只关心 figure 有没有被关掉、会不会累积，用无界面的 Agg。
# （必须在任何 pyplot / seaborn 导入之前设置）
matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

from biopheno import config as cfg
from biopheno import modules as mod

# —— 合成数据小工具 ——


def _covarying_matrix(n_objects: int, n_features: int, n_covarying: int, seed: int = 0):
    """前 n_covarying 列共享同一个潜变量（近似完全共线），其余列独立噪声。"""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_objects, 1))
    X = np.hstack(
        [base + 0.01 * rng.normal(size=(n_objects, 1)) for _ in range(n_covarying)]
        + [rng.normal(size=(n_objects, 1)) for _ in range(n_features - n_covarying)]
    )
    return X, [f"f{i}" for i in range(n_features)]


def _uid(fov: str, label: int, day: str) -> str:
    """Stage 1 的 uid 格式（Spec D13）：`{fov}_{label}_{frame_idx}`，frame_idx = cfg.DAYS[day]。"""
    return f"{fov}_{int(label)}_{int(cfg.DAYS[day])}"


def _write_preprocessed(root, names, X):
    outdir = root / "preprocessed"
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "X.npy", np.asarray(X, dtype=np.float64))
    (outdir / "keep_features.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")


def _write_stage2(root, uids, names, *, with_uid_index=True, shape_uids=None):
    """按 Stage 2 的落盘格式写行身份产物：`all_features.csv`（instance_uid 索引）+ `sam_shape.npz`。

    值本身无所谓（Stage 4 不读它们），但行数/uid 顺序就是 Stage 4 认定的实例顺序。
    `with_uid_index=False` 模拟"表被重写成纯数值索引"：此时行身份必须退回 `sam_shape.npz` 的 uid。
    """
    outdir = root / "features"
    outdir.mkdir(parents=True, exist_ok=True)
    data = np.zeros((len(uids), len(names)))
    if with_uid_index:
        df = pd.DataFrame(data, index=pd.Index(uids, name="instance_uid"), columns=names)
    else:
        df = pd.DataFrame(data, columns=names)
    df.to_csv(outdir / "all_features.csv")
    np.savez_compressed(outdir / "sam_shape.npz", uid=np.asarray(uids if shape_uids is None else shape_uids, dtype=str))


def _write_instances(root, groups, missing_patch_labels=()):
    """写 Stage1 的汇总表与每 (day, fov) 的 patches.npz；patch 形状编码 label（便于反查对齐）。

    `missing_patch_labels` 里的 `(day, fov, label)` 模拟"轮廓被丢弃、没有代表图"的实例：Stage 1 仍给它们
    留 meta 行，但 patch 文件里没有条目（Stage 2 随后也不会给它们算特征）。
    """
    missing = set(missing_patch_labels)
    inst_root = root / "instances"
    rows = []
    for day, fov, labels in groups:
        outdir = inst_root / day / fov
        outdir.mkdir(parents=True, exist_ok=True)
        flat = {}
        for label in labels:
            if (day, fov, label) in missing:
                continue
            flat[f"label{label}_native"] = np.full((6 + label, 6), label, dtype=np.uint8)
        np.savez_compressed(outdir / "patches.npz", **flat)
        for label in labels:
            rows.append({"instance_uid": _uid(fov, label, day), "day": day, "fov": fov, "label": label})
    inst_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(inst_root / "all_instances.csv", index=False)
    return [r["instance_uid"] for r in rows]


def _assert_panels_follow_uids(patches, uids, n_modules):
    """逐模块核对 uid ↔ 行号 ↔ 代表图同源。

    合成 patch 的高度编码了 label（`6 + label`，宽度恒为 6），而 SPOT 的马赛克格子高 = 最大 patch 高度，
    所以"面板高度 = 3 × 选中实例的 patch 高度"只有在 patch 是按 uid（而不是按位置）装配时才成立。
    """
    label_of_uid = {uid: int(uid.split("_")[1]) for uid in uids}
    for module_id in range(1, n_modules + 1):
        index = patches[f"module{module_id}_object_index"].tolist()
        panel_uids = patches[f"module{module_id}_instance_uid"].tolist()
        assert panel_uids == [uids[i] for i in index]
        panel = patches[f"module{module_id}_panel"]
        expected_cell_h = max(6 + label_of_uid[uid] for uid in panel_uids)
        assert panel.shape == (3 * expected_cell_h, 3 * 6), (
            f"模块 {module_id} 的代表图尺寸 {panel.shape} 与选中实例 {panel_uids} 的 patch 尺寸不符："
            "patch 与 uid 没有按 uid 对齐"
        )


def _run_stage4_into(monkeypatch, root, groups, names, X, *, perm=None, with_uid_index=True):
    """在一个干净的 OUTPUT_ROOT 下布线 Stage1/2/3 输入并跑 `run_stage4`，返回 `(meta_uids, stage2_uids)`。

    `perm` 给定时：Stage 2 的行序（= X 的行序）是 Stage 1 meta 行序的一个置换，用来证明对齐靠 uid 而非位置。
    """
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", root)
    meta_uids = _write_instances(root, groups)
    order = np.arange(len(meta_uids)) if perm is None else np.asarray(perm)
    stage2_uids = [meta_uids[i] for i in order]
    _write_preprocessed(root, names, np.asarray(X)[order])
    _write_stage2(root, stage2_uids, names, with_uid_index=with_uid_index)
    mod.run_stage4()
    return meta_uids, stage2_uids


# —— 模块发现（legacy 选择器 + 论文参数）的语义 ——


def test_covarying_features_form_one_module_with_legacy_paper_params():
    # 必须用 D=60：legacy 扫描是 `arange(2, len(linkage))[:max_clust_len+buffer]`，D=12 时只评得到
    # 3..11 个模块的切割（几乎全是退化切割），ALS(λ=1e0) 平滑后的 DB 峰落在紧致块内部，共变列会被拆开；
    # D≥40 后树上部才有可评估的结构、峰落在树高处，20 个强共变特征才能作为一个模块被发现。
    rng = np.random.default_rng(0)
    n_features, n_covarying = 60, 20
    base = rng.normal(size=(60, 1))
    X = np.hstack(
        [base + 0.01 * rng.normal(size=(60, 1)) for _ in range(n_covarying)]
        + [rng.normal(size=(60, 1)) for _ in range(n_features - n_covarying)]
    )
    names = [f"f{i}" for i in range(n_features)]
    mods = mod.discover_modules(X, names)
    labels = mods["module_labels"]
    assert labels.shape == (n_features,)
    assert 1 < len(np.unique(labels)) <= n_features
    # 反空转：含 f0 的那个模块必须**正好**是那 20 个共变特征——既没被拆散，也没把噪声列并进来。
    # （只看 `len(set(labels[:20])) == 1` 是空的：噪声列的标签与它们不同也能满足，但把噪声并进来同样会过）
    assert int((labels == labels[0]).sum()) == n_covarying
    assert np.unique(labels[:n_covarying]).tolist() == [labels[0]]


def test_module_expression_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 9))
    names = [f"f{i}" for i in range(9)]
    mods = mod.discover_modules(X, names)
    expr = mod.module_expression(X, mods)
    assert expr.shape[0] == 40


# —— 选择器与参数必须显式走 legacy ——


def test_discover_modules_uses_legacy_selector_and_paper_params(monkeypatch):
    # v2.0.1 的默认是 'multiscale'（null 模型 + bootstrap 共识）：整条调用链上任何一处漏传
    # module_selection，模块数都会静默变掉。这里把真实函数的 kwargs 抓下来核对。
    captured: dict = {}
    real = mod.sam.hierarchical_cluster_features_into_SAM_modules

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(mod.sam, "hierarchical_cluster_features_into_SAM_modules", _spy)
    X, names = _covarying_matrix(n_objects=30, n_features=8, n_covarying=3, seed=3)
    mod.discover_modules(X, names)

    assert captured["module_selection"] == "legacy"
    # MODULE_PARAMS 的三个键都必须原样传到（legacy 路径都真的用到了，见 report）
    for key, value in cfg.MODULE_PARAMS.items():
        assert key in captured, f"legacy 参数 {key} 没有传给 SPOT"
        assert captured[key] == value
    assert captured["random_state"] == cfg.RANDOM_STATE
    # legacy 路径不重算特征类型，必须由我们先算好传进去，否则 SPOT 会按默认名规则猜
    assert captured["feature_type"] is not None
    assert captured["feature_scope"] is not None
    assert np.asarray(captured["feature_type"]).shape == (8,)


def test_module_params_rejects_reserved_names(monkeypatch):
    # 这些 kwarg 由 discover_modules 显式传：cfg.MODULE_PARAMS 里再出现一次就是 "got multiple values"，
    # 必须在调用前用中文错误拦住，而不是让 Python 抛一个看不出原因的 TypeError
    monkeypatch.setattr(cfg, "MODULE_PARAMS", {**cfg.MODULE_PARAMS, "module_selection": "multiscale"})
    with pytest.raises(ValueError, match="重复赋值") as excinfo:
        mod._module_params()
    assert "module_selection" in str(excinfo.value)


def test_discover_modules_closes_clustermap_figure_on_failure(monkeypatch):
    # SPOT 的 legacy 路径内部建了 clustermap figure；自检失败（这里把模块标签改坏）时也必须关掉，
    # 否则异常路径上每失败一次就漏一个 figure（长跑/测试进程里会一直累积）
    import matplotlib.pyplot as plt

    real = mod.sam.hierarchical_cluster_features_into_SAM_modules

    def _broken(*args, **kwargs):
        grid, labels, colorbar, names_clusters, indices_clusters = real(*args, **kwargs)
        return grid, np.zeros_like(np.asarray(labels)), colorbar, names_clusters, indices_clusters

    monkeypatch.setattr(mod.sam, "hierarchical_cluster_features_into_SAM_modules", _broken)
    before = plt.get_fignums()
    X, names = _covarying_matrix(n_objects=20, n_features=8, n_covarying=3, seed=16)
    with pytest.raises(ValueError, match="不自洽"):
        mod.discover_modules(X, names)
    assert plt.get_fignums() == before


def test_discover_modules_keeps_what_a_replot_needs():
    # cluster_map 必须是"不用重算就能重画"的完整状态：相关系数矩阵 + 树 + 重排索引 + 类型颜色。
    X, names = _covarying_matrix(n_objects=40, n_features=10, n_covarying=3, seed=4)
    mods = mod.discover_modules(X, names)
    cmap = mods["cluster_map_data"]
    D = len(names)
    assert cmap["corr"].shape == (D, D)
    assert np.allclose(cmap["corr"], cmap["corr"].T)
    assert np.allclose(np.diag(cmap["corr"]), 1.0)
    assert cmap["linkage"].shape == (D - 1, 4)
    assert sorted(cmap["reordered_ind"].tolist()) == list(range(D))
    assert list(cmap["feature_names"]) == names
    assert np.asarray(cmap["module_labels"]).shape == (D,)
    assert np.asarray(cmap["feature_type"]).shape == (D,)
    assert np.asarray(cmap["feature_scope"]).shape == (D,)
    assert np.asarray(cmap["colorbar"]).shape == (D, 3)


# —— 输入自检 ——


def test_discover_modules_rejects_duplicate_names():
    # 重名会让 SPOT 里的 pd.DataFrame(columns=...)/df_corr index 静默折叠（按名取列全部错位）
    X, names = _covarying_matrix(n_objects=20, n_features=6, n_covarying=2, seed=5)
    names[3] = names[2]
    with pytest.raises(ValueError, match="重名"):
        mod.discover_modules(X, names)


def test_discover_modules_rejects_name_column_mismatch():
    X, names = _covarying_matrix(n_objects=20, n_features=6, n_covarying=2, seed=6)
    with pytest.raises(ValueError, match="列数"):
        mod.discover_modules(X, names[:-1])


def test_discover_modules_needs_enough_features():
    # legacy 扫描从第 2 个合并点起，特征太少时没有任何可评估的切割
    X, _ = _covarying_matrix(n_objects=20, n_features=3, n_covarying=1, seed=7)
    with pytest.raises(ValueError, match="特征"):
        mod.discover_modules(X, [f"f{i}" for i in range(3)])


# —— 模块表达与代表图 ——


def test_module_expression_columns_follow_modules():
    X, names = _covarying_matrix(n_objects=50, n_features=12, n_covarying=4, seed=8)
    mods = mod.discover_modules(X, names)
    expr = mod.module_expression(X, mods)
    assert expr.shape == (50, len(mods["feature_names"]))
    assert np.isfinite(expr).all()
    # 同一输入再算一次必须完全一致（固定 random_state，下游图表才能离线复现）
    assert np.allclose(expr, mod.module_expression(X, mods))


def test_representative_patches_returns_panels_and_purity_scores():
    rng = np.random.default_rng(9)
    X, names = _covarying_matrix(n_objects=40, n_features=9, n_covarying=3, seed=9)
    mods = mod.discover_modules(X, names)
    expr = mod.module_expression(X, mods)
    # patch 形状不一致（Stage1 的 native 裁剪图就是随 bbox 变的），必须能拼成统一的马赛克
    patches = [rng.integers(0, 255, size=(8 + (i % 5), 7 + (i % 3)), dtype=np.uint8) for i in range(40)]
    purity, panels = mod.representative_patches(expr, patches, n_rows=2)

    n_modules = len(mods["feature_names"])
    assert purity.shape == (40, n_modules)
    assert len(panels) == n_modules
    assert all(np.asarray(p).dtype == np.uint8 for p in panels)
    assert all(np.asarray(p).ndim == 2 for p in panels)


def test_representative_patches_rejects_misaligned_patches():
    X, names = _covarying_matrix(n_objects=30, n_features=8, n_covarying=3, seed=10)
    mods = mod.discover_modules(X, names)
    expr = mod.module_expression(X, mods)
    patches = [np.zeros((6, 6), dtype=np.uint8) for _ in range(29)]
    with pytest.raises(ValueError, match="patch"):
        mod.representative_patches(expr, patches, n_rows=2)


# —— run_stage4 落盘 ——


def test_run_stage4_requires_stage3(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="Stage3") as excinfo:
        mod.run_stage4()
    assert "先跑" in str(excinfo.value)


def test_run_stage4_requires_stage2(monkeypatch, tmp_path):
    # 行身份来自 Stage 2：缺了它就没有可信的实例顺序，必须报错指明先跑 Stage2
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    X, names = _covarying_matrix(n_objects=10, n_features=8, n_covarying=3, seed=17)
    _write_preprocessed(tmp_path, names, X)
    with pytest.raises(ValueError, match="Stage2") as excinfo:
        mod.run_stage4()
    assert "先跑" in str(excinfo.value)


def test_run_stage4_writes_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    names = [f"f{i}" for i in range(12)]
    # 两个 (day, fov) 分组；最后一个实例故意没有 patch（Stage1 丢弃轮廓时会这样 → 它也不进 Stage2）
    groups = [("day0", "A1", list(range(1, 16))), ("day3", "A2", list(range(1, 16)))]
    meta_uids = _write_instances(tmp_path, groups, missing_patch_labels={("day3", "A2", 15)})
    dropped_uid = _uid("A2", 15, "day3")
    stage2_uids = [uid for uid in meta_uids if uid != dropped_uid]
    X, _ = _covarying_matrix(n_objects=len(stage2_uids), n_features=len(names), n_covarying=4, seed=11)
    _write_preprocessed(tmp_path, names, X)
    _write_stage2(tmp_path, stage2_uids, names)

    mod.run_stage4()

    outdir = tmp_path / "modules"
    for fname in ("modules_labels.csv", "module_expr.npy", "cluster_map.npz", "module_patches.npz", "stage4_qc.json"):
        assert (outdir / fname).exists(), f"缺少产物 {fname}"

    labels_df = pd.read_csv(outdir / "modules_labels.csv")
    # 每个特征一行，顺序与 keep_features.json（= X 的列序）一致
    assert list(labels_df["feature"]) == names
    module_labels = labels_df["module"].to_numpy()
    assert module_labels.shape == (len(names),)

    expr = np.load(outdir / "module_expr.npy")
    n_modules = int(module_labels.max())
    assert expr.shape == (len(stage2_uids), n_modules)  # 行数 = Stage2 的实例数
    assert np.isfinite(expr).all()

    qc = json.loads((outdir / "stage4_qc.json").read_text(encoding="utf-8"))
    assert qc["module_selection"] == cfg.MODULE_SELECTION == "legacy"
    assert qc["module_params"] == cfg.MODULE_PARAMS
    assert qc["n_modules"] == n_modules
    assert qc["n_objects"] == len(stage2_uids)
    assert qc["n_instances_in_X"] == len(stage2_uids)
    assert qc["n_meta_rows"] == len(meta_uids)
    assert qc["n_meta_not_in_X"] == 1
    assert qc["meta_not_in_x_uids"] == [dropped_uid]
    assert "instance_uid" in qc["module_expr_row_source"]
    assert qc["n_features"] == len(names)
    assert sum(qc["module_sizes"].values()) == len(names)
    assert qc["module_sizes"][str(int(module_labels[0]))] == int((module_labels == module_labels[0]).sum())
    # feature → module 的对应关系也要能从 QC 复核
    first_module = str(int(module_labels[0]))
    assert names[0] in qc["module_features"][first_module]
    # Stage2 的实例全都应该有 patch（缺失会直接报错），所以这个计数为 0
    assert qc["n_rows_without_patch"] == 0
    assert qc["rows_without_patch_uids"] == []
    # total 必须把落盘算进去，否则"几分钟上限"里的写盘开销是隐形的
    timing = qc["timing_seconds"]
    assert timing["total"] >= timing["discover_modules"] + timing["module_expression"] + timing["artifacts"] > 0

    cmap = np.load(outdir / "cluster_map.npz", allow_pickle=False)
    assert cmap["corr"].shape == (len(names), len(names))
    assert cmap["linkage"].shape == (len(names) - 1, 4)
    assert sorted(cmap["reordered_ind"].tolist()) == list(range(len(names)))
    assert list(cmap["feature_names"]) == names

    with np.load(outdir / "module_patches.npz", allow_pickle=False) as patches:
        assert patches["purity"].shape == (len(stage2_uids), n_modules)  # 每个 Stage2 实例都是候选
        assert patches["patched_row_index"].tolist() == list(range(len(stage2_uids)))
        for module_id in range(1, n_modules + 1):
            assert patches[f"module{module_id}_panel"].ndim == 2
            assert len(patches[f"module{module_id}_object_index"]) == min(3 * 3, len(stage2_uids))
            assert dropped_uid not in patches[f"module{module_id}_instance_uid"].tolist()
        # patch 是按 uid 装配的（面板尺寸跟着选中实例的 patch 尺寸走）
        _assert_panels_follow_uids(patches, stage2_uids, n_modules)


def test_run_stage4_excludes_meta_instances_that_stage2_dropped(monkeypatch, tmp_path):
    # Stage1 给每个标签都留 meta 行，但轮廓被丢弃的标签没有 patch、也不会进 Stage2 的特征表：
    # all_instances.csv 因此是 Stage2 uid 的**严格超集**。Stage4 必须照常跑完并把排除数记进 QC，
    # 而不是要求两者行数相等（真实数据上那种要求会让 Stage4 永远跑不起来）。
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    names = [f"f{i}" for i in range(8)]
    groups = [("day0", "A1", list(range(1, 13))), ("day3", "A2", list(range(1, 11)))]
    dropped = {("day0", "A1", 12), ("day0", "A1", 11), ("day3", "A2", 10)}
    meta_uids = _write_instances(tmp_path, groups, missing_patch_labels=dropped)
    dropped_uids = sorted([_uid("A1", 11, "day0"), _uid("A1", 12, "day0"), _uid("A2", 10, "day3")])
    stage2_uids = [uid for uid in meta_uids if uid not in set(dropped_uids)]
    assert len(stage2_uids) == 19  # 22 个 meta 行 - 3 个被 Stage2 排除
    X, _ = _covarying_matrix(n_objects=len(stage2_uids), n_features=len(names), n_covarying=3, seed=13)
    _write_preprocessed(tmp_path, names, X)
    _write_stage2(tmp_path, stage2_uids, names)

    mod.run_stage4()  # 不抛错，且产物完整

    qc = json.loads((tmp_path / "modules" / "stage4_qc.json").read_text(encoding="utf-8"))
    assert qc["n_instances_in_X"] == 19
    assert qc["n_meta_rows"] == 22
    assert qc["n_meta_not_in_X"] == 3
    assert list(qc["meta_not_in_x_uids"]) == sorted(qc["meta_not_in_x_uids"])
    assert qc["meta_not_in_x_uids"] == dropped_uids
    expr = np.load(tmp_path / "modules" / "module_expr.npy")
    assert expr.shape[0] == 19
    # 被排除的实例不能以任何形式出现在代表图里
    with np.load(tmp_path / "modules" / "module_patches.npz", allow_pickle=False) as patches:
        for module_id in range(1, qc["n_modules"] + 1):
            assert not set(patches[f"module{module_id}_instance_uid"].tolist()) & set(dropped_uids)


def test_run_stage4_alignment_ignores_stage2_row_order(monkeypatch, tmp_path):
    # 行身份来自 instance_uid：把 Stage2 的行序（连同 X 的行）打乱后，模块划分、模块表达与代表图
    # 都应与原行序等价——若实现按位置装配 patch/表达，这里就会静默串行。
    names = [f"f{i}" for i in range(10)]
    groups = [("day0", "A1", list(range(1, 21)))]
    X, _ = _covarying_matrix(n_objects=20, n_features=len(names), n_covarying=4, seed=14)
    _base_meta, base_stage2 = _run_stage4_into(monkeypatch, tmp_path / "base", groups, names, X)

    perm = np.random.default_rng(15).permutation(len(base_stage2))
    _perm_meta, perm_stage2 = _run_stage4_into(monkeypatch, tmp_path / "perm", groups, names, X, perm=perm)
    assert perm_stage2 == [base_stage2[i] for i in perm]

    base_dir, perm_dir = tmp_path / "base" / "modules", tmp_path / "perm" / "modules"
    # 模块划分（特征 → 模块）与行序无关
    pd.testing.assert_frame_equal(
        pd.read_csv(base_dir / "modules_labels.csv"), pd.read_csv(perm_dir / "modules_labels.csv")
    )
    expr_base = np.load(base_dir / "module_expr.npy")
    expr_perm = np.load(perm_dir / "module_expr.npy")
    # perm_stage2[j] = base_stage2[perm[j]]，所以 perm_run 的第 j 行属于 base 的第 perm[j] 行：
    # 用逆置换把两边对齐到同一套 uid 上再逐元素比
    inv = np.argsort(perm)
    assert np.allclose(expr_perm[inv], expr_base)
    with (
        np.load(base_dir / "module_patches.npz", allow_pickle=False) as pb,
        np.load(perm_dir / "module_patches.npz", allow_pickle=False) as pp,
    ):
        # 纯度只跟着实例走（行序只是置换）
        assert np.allclose(pp["purity"][inv], pb["purity"])
        module_keys = [k for k in pb.files if k.endswith("_instance_uid")]
        assert module_keys
        for key in module_keys:
            # 每个模块选中的代表实例（按 uid）完全相同——证明选择靠 uid/表达值，而不是"第几行"
            assert sorted(pp[key].tolist()) == sorted(pb[key].tolist())
        # 代表图本身也必须跟着 uid 走（面板尺寸随选中实例的 patch 尺寸变）
        n_modules = len(module_keys)
        _assert_panels_follow_uids(pb, base_stage2, n_modules)
        _assert_panels_follow_uids(pp, perm_stage2, n_modules)


def test_run_stage4_falls_back_to_sam_shape_uids(monkeypatch, tmp_path):
    # all_features.csv 没有 instance_uid 索引（表被重写成纯数值索引）时，行身份退回 sam_shape.npz 的 uid
    names = [f"f{i}" for i in range(10)]
    groups = [("day0", "A1", list(range(1, 21)))]
    X, _ = _covarying_matrix(n_objects=20, n_features=len(names), n_covarying=4, seed=14)
    _base_meta, base_stage2 = _run_stage4_into(monkeypatch, tmp_path / "base", groups, names, X)

    perm = np.random.default_rng(16).permutation(len(base_stage2))
    _perm_meta, perm_stage2 = _run_stage4_into(
        monkeypatch, tmp_path / "fallback", groups, names, X, perm=perm, with_uid_index=False
    )

    fallback_dir = tmp_path / "fallback" / "modules"
    qc = json.loads((fallback_dir / "stage4_qc.json").read_text(encoding="utf-8"))
    assert "sam_shape.npz" in qc["module_expr_row_source"]
    assert qc["n_instances_in_X"] == len(perm_stage2)
    # 行序跟着 sam_shape.npz 的 uid 走（与 CSV 的行位置无关）
    expr_perm = np.load(fallback_dir / "module_expr.npy")
    expr_base = np.load(tmp_path / "base" / "modules" / "module_expr.npy")
    # 同上：行序被 perm 打乱过，用逆置换对齐回同一套 uid
    assert np.allclose(expr_perm[np.argsort(perm)], expr_base)
    with np.load(fallback_dir / "module_patches.npz", allow_pickle=False) as patches:
        _assert_panels_follow_uids(patches, perm_stage2, qc["n_modules"])


def test_run_stage4_rejects_instance_row_mismatch(monkeypatch, tmp_path):
    # 真正的对齐保护：X 的行数必须等于 Stage2 的 uid 数（all_instances.csv 的行数不是标准）
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    names = [f"f{i}" for i in range(8)]
    meta_uids = _write_instances(tmp_path, [("day0", "A1", list(range(1, 16)))])  # 15 个实例
    X, _ = _covarying_matrix(n_objects=30, n_features=len(names), n_covarying=3, seed=12)  # 但 X 有 30 行
    _write_preprocessed(tmp_path, names, X)
    _write_stage2(tmp_path, meta_uids, names)

    with pytest.raises(ValueError, match="实例数") as excinfo:
        mod.run_stage4()
    assert "Stage2" in str(excinfo.value)


def test_run_stage4_rejects_instances_without_patch(monkeypatch, tmp_path):
    # Stage2 有特征行、Stage1 却没给它的 patch：这不能用"轮廓被丢弃"解释，必须报错而不是静默少图
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    names = [f"f{i}" for i in range(8)]
    groups = [("day0", "A1", list(range(1, 13)))]
    meta_uids = _write_instances(tmp_path, groups, missing_patch_labels={("day0", "A1", 5)})
    X, _ = _covarying_matrix(n_objects=len(meta_uids), n_features=len(names), n_covarying=3, seed=18)
    _write_preprocessed(tmp_path, names, X)
    _write_stage2(tmp_path, meta_uids, names)

    with pytest.raises(ValueError, match="patch") as excinfo:
        mod.run_stage4()
    assert _uid("A1", 5, "day0") in str(excinfo.value)
