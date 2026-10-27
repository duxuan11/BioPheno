"""Stage 7 静态图（F1–F6b）的测试：全部合成产物，不依赖真实 Stage 1–5 数据。

`fake_analysis_dir` 故意与真实 `output/` **同构**（`analysis/` + `modules/` + `instances/` +
`preprocessed/`），因为这份 fixture 是唯一钉住上游产物 schema 的东西：只要哪一阶段改了列名或键名，
这里就会先炸，而不是等出图时静默画错。

测试盯住的点：
- `render_static` 必须产出任务说明里那 7 个**确切**文件名，且每张图不是空壳（> 1 KB）；
- 只有一个簇时不能 IndexError（Review Focus #5）：`force_single_cluster=True` 走退化路径仍然出图；
- 缺上游产物时给**带阶段名**的中文错误（不许把问题藏起来，也不许默默少画一张）；
- 每张图都要 `plt.close`：出完图进程里不留 figure（长跑/多次调用会吃内存）；
- 每簇代表图固定 4 张（2×2 马赛克）、且只能来自**本簇**实例，且只把选中的那几张图读进内存；
- F4 的 contribution score 优先取 Stage 4 落盘的 `stage4_qc.json`，缺键时用 `X.npy` +
  `keep_features.json` 重算（不重跑模块发现）。

出图**内容**（不只是"文件非空"）也被钉住：F1 的面板标题承认强度已归一化、F3 画的是共用网格的真实
二维密度场、F5 的显示矩阵就是落盘的 dendrogram 列序矩阵（对称 + 块结构 + 色条连续块）、F6b 每簇的
day 占比之和为 1。为此 `_spy_figures` 包一层 `figs._open_figure`，把 `plt.close` 之后的 figure 收起来看。

`fixture` 里的 `cluster_map.npz` 按**真实 Stage4 的口径**造：`corr` 是 seaborn 画出来的 `data2d`
（已经是 dendrogram 列序的对称块状距离矩阵），`colorbar` 是按同一列序逐特征的连续色块，`module_labels`
仍是 original 列序、模块号按 dendrogram 里出现的先后重编号。
"""

import contextlib
import json
import math
import warnings
from pathlib import Path

import matplotlib

# SPOT 的密度函数无条件 `plt.show()`；交互式后端会真的开窗口。必须在 pyplot 相关导入之前设置。
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from biopheno import figures as figs

# `render_static` 的出图顺序（= 图文件名清单的顺序），`_spy_figures` 收集到的 figure 也按这个序。
_RENDER_ORDER: tuple[str, ...] = (
    "F1_umap_clusters_patches.png",
    "F2_umap_by_day_fov.png",
    "F3_umap_density.png",
    "F4_contribution_scores.png",
    "F5_sam_modules.png",
    "F6_cluster_feature_bars.png",
    "F6b_cluster_day_composition.png",
)

# 任务说明里点名的 7 张图（集合相等，多一张少一张都算违约）。
_EXPECTED_FIGURES: set[str] = {
    "F1_umap_clusters_patches.png",
    "F2_umap_by_day_fov.png",
    "F3_umap_density.png",
    "F4_contribution_scores.png",
    "F5_sam_modules.png",
    "F6_cluster_feature_bars.png",
    "F6b_cluster_day_composition.png",
}

# 三个 (day, fov) 组合，每个 4 个实例 = 12 行；簇号在每个视野里都出现，F6b 的 day 组成才有信息量。
_GROUPS: tuple[tuple[str, str], ...] = (("day0", "A1"), ("day3", "A2"), ("day5", "A3"))
_PER_GROUP: int = 4
_CLUSTER_PATTERN: tuple[int, ...] = (1, 2, 3, 1)
# Stage 2 的特征列（含 SAM 分类需要的 intensity 关键词，F4 重算路径要用）。
_FEATURE_NAMES: tuple[str, ...] = ("area", "moment_eccentricity", "mean_intensity", "f1", "f2", "f3")
# 特征的"真实"模块归属（original 列序）：M1 = 前两个、M2 = 中间两个、M3 = 最后两个。
_SEED_MODULES: tuple[int, ...] = (1, 1, 2, 2, 3, 3)
# 模块色条用的显式 RGB（互不相同，图上配色与图例才能互相核对）。
_MODULE_COLOURS: dict[int, tuple[float, float, float]] = {
    1: (0.85, 0.20, 0.20),
    2: (0.20, 0.55, 0.85),
    3: (0.25, 0.75, 0.35),
}


def _contiguous_blocks(labels: np.ndarray) -> list[list[int]]:
    """把一维标签切成连续块（`[M1, M1, M2, M2]` → `[[M1, M1], [M2, M2]]`）。"""
    blocks: list[list[int]] = []
    for label in labels.tolist():
        if blocks and blocks[-1][0] == label:
            blocks[-1].append(label)
        else:
            blocks.append([label])
    return blocks


def _block_distance_matrix(seed_modules: np.ndarray, rng) -> np.ndarray:
    """块状距离矩阵（口径同 SPOT：`1 - 相关系数`）：同模块近、跨模块远，对称、对角为 0。"""
    n = int(seed_modules.size)
    distances = np.zeros((n, n), dtype=np.float64)
    for row in range(n):
        for column in range(row + 1, n):
            same = seed_modules[row] == seed_modules[column]
            value = rng.uniform(0.10, 0.30) if same else rng.uniform(1.30, 1.70)
            distances[row, column] = distances[column, row] = value
    return distances


def _uid(fov: str, label: int, frame: int) -> str:
    """Stage 1 的 uid 格式（Spec D13）：`{fov}_{label}_{frame_idx}`。"""
    return f"{fov}_{int(label)}_{int(frame)}"


def _instances() -> list[dict]:
    """逐实例的 meta 行（与 Stage 1 的 `all_instances.csv` 列名、uid 口径一致）。"""
    rows = []
    for day_idx, (day, fov) in enumerate(_GROUPS):
        for label in range(1, _PER_GROUP + 1):
            cluster = _CLUSTER_PATTERN[(label - 1) % len(_CLUSTER_PATTERN)]
            rows.append(
                {
                    "instance_uid": _uid(fov, label, day_idx),
                    "day": day,
                    "fov": fov,
                    "frame_idx": day_idx,
                    "time_hours": float(day_idx) * 60.0,
                    "label": label,
                    "centroid_row": 60.0 + 10.0 * label,
                    "centroid_col": 40.0 + 10.0 * day_idx,
                    "r0": 40,
                    "c0": 30,
                    "r1": 80 + label,
                    "c1": 70 + label,
                    "area_px": 500 + 50 * label,
                    "is_small": False,
                    "touches_border": False,
                    "cluster": cluster,
                }
            )
    return rows


def _write_patches(root, rows) -> None:
    """按真实布局写 `instances/<day>/<fov>/patches.npz`（键 `label{n}_native` / `label{n}_sift`）。

    native 尺寸逐实例不同（Stage 1 的裁剪图随 bbox 变），sift 固定 64×64。
    """
    rng = np.random.default_rng(3)
    for day, fov in _GROUPS:
        outdir = root / "instances" / day / fov
        outdir.mkdir(parents=True, exist_ok=True)
        flat = {}
        for row in rows:
            if row["day"] != day or row["fov"] != fov:
                continue
            label = int(row["label"])
            native = rng.integers(60, 200, size=(24 + label, 20 + label), dtype=np.uint8)
            sift = rng.integers(60, 200, size=(64, 64), dtype=np.uint8)
            flat[f"label{label}_native"] = native
            flat[f"label{label}_sift"] = sift
        np.savez_compressed(outdir / "patches.npz", **flat)


def _write_analysis(root, rows) -> None:
    """写 Stage 5 的 `analysis/`（umap_2d / cluster_labels / per_instance_features / kmeans_metrics）。"""
    outdir = root / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    centres = {1: np.array([0.0, 0.0]), 2: np.array([5.0, 0.5]), 3: np.array([0.5, 5.0])}
    coords = np.vstack([centres[row["cluster"]] + rng.normal(0, 0.25, 2) for row in rows])
    uids = [row["instance_uid"] for row in rows]
    pd.DataFrame({"instance_uid": uids, "umap_1": coords[:, 0], "umap_2": coords[:, 1]}).to_csv(
        outdir / "umap_2d.csv", index=False
    )
    pd.DataFrame(
        {
            "instance_uid": uids,
            "day": [row["day"] for row in rows],
            "fov": [row["fov"] for row in rows],
            "frame_idx": [row["frame_idx"] for row in rows],
            "time_hours": [row["time_hours"] for row in rows],
            "cluster": [row["cluster"] for row in rows],
            "cluster_raw": [row["cluster"] - 1 for row in rows],
        }
    ).to_csv(outdir / "cluster_labels.csv", index=False)
    # area / eccentricity / intensity 的原生量纲；speed 恒 NaN（motion 被有意跳过，Spec §6-Stage2）
    pd.DataFrame(
        {
            "instance_uid": uids,
            "area": [row["area_px"] * 1.0 for row in rows],
            "eccentricity": [0.2 + 0.15 * row["cluster"] for row in rows],
            "intensity": [180.0 - 10.0 * row["cluster"] for row in rows],
            "speed": np.full(len(rows), np.nan),
        }
    ).to_csv(outdir / "per_instance_features.csv", index=False)
    metrics = {
        "k": 3,
        "chosen_k": 3,
        "n_clusters": 3,
        "k_range": [2, 12],
        "k_scanned": [2, 3, 4],
        "inertia": {"2": 3.0, "3": 1.0, "4": 0.7},
        "silhouette": {"2": 0.5, "3": 0.6, "4": 0.55},
        "scaled_inertia": {"2": 0.10, "3": 0.05, "4": 0.06},
        "mean_per_cluster_score": {"1": 0.35, "2": 0.50, "3": 0.65},
        "eccentricity_score": {"source": "moment_eccentricity", "order": "ascending", "fallback_used": False},
        "cluster_mapping": {"0": 1, "1": 2, "2": 3},
    }
    (outdir / "kmeans_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")


def _write_modules(root) -> None:
    """写 Stage 4 的 `modules/`：模块表达、clustermap 的完整重画状态、QC 里的 contribution score。

    `cluster_map.npz` 严格照真实 Stage4（`modules.discover_modules`）的口径：

    - `corr` = `grid.data2d`，而 seaborn 在 `plot_matrix` 里已经自己做过 `data2d.iloc[yind, xind]`，
      所以它**已经是 dendrogram 列序**，对称且成块；
    - `colorbar` = SPOT 的 `features_clusters_colorbar`，**按 dendrogram 列序**逐特征给色（同一模块是
      一段连续色块），不是 original 列序；
    - `module_labels` 仍是 original 列序，但模块号按 dendrogram 里出现的先后重编号（1..m）。
    """
    outdir = root / "modules"
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    seed_modules = np.asarray(_SEED_MODULES, dtype=np.int64)
    original_distances = _block_distance_matrix(seed_modules, rng)
    tree = linkage(squareform(original_distances, checks=False), method="average")
    order = np.asarray(leaves_list(tree), dtype=np.int64)
    ordered_seed_modules = seed_modules[order]
    # fixture 自查：块内距离 <= 0.30、跨块 >= 1.30，所以每个模块在 dendrogram 里必然是一段连续叶子
    assert len(_contiguous_blocks(ordered_seed_modules)) == 3
    # SPOT 口径：模块号 = dendrogram 里出现的先后
    renumber = {int(seed): index + 1 for index, seed in enumerate(dict.fromkeys(ordered_seed_modules.tolist()))}
    module_labels = np.asarray([renumber[int(seed)] for seed in seed_modules], dtype=np.int64)
    module_of_position = np.asarray([renumber[int(seed)] for seed in ordered_seed_modules], dtype=np.int64)
    colorbar = np.vstack([np.asarray(_MODULE_COLOURS[int(module)], dtype=np.float64) for module in module_of_position])
    pd.DataFrame(
        {
            "feature": list(_FEATURE_NAMES),
            "module": module_labels,
            "feature_type": np.array([0, 0, 1, 0, 0, 0], dtype=np.int64),
            "feature_scope": np.array([0, 0, 1, 2, 0, 0], dtype=np.int64),
        }
    ).to_csv(outdir / "modules_labels.csv", index=False)
    np.save(outdir / "module_expr.npy", rng.normal(size=(len(_instances()), 3)))
    np.savez_compressed(
        outdir / "cluster_map.npz",
        # 已按 dendrogram 列序（= seaborn 画出来的 data2d），不是 original 列序
        corr=original_distances[np.ix_(order, order)],
        linkage=tree,
        reordered_ind=order,
        colorbar=colorbar,
        module_labels=module_labels,
        feature_type=np.array([0, 0, 1, 0, 0, 0], dtype=np.int64),
        feature_scope=np.array([0, 0, 1, 2, 0, 0], dtype=np.int64),
        feature_names=np.array(_FEATURE_NAMES, dtype=str),
    )
    qc = {
        "module_selection": "legacy",
        "n_modules": 3,
        "type_contribution": [0.62, 0.38, 0.0],
        "scope_contribution": [0.51, 0.30, 0.19],
        "module_sizes": {"1": 2, "2": 2, "3": 2},
    }
    (outdir / "stage4_qc.json").write_text(json.dumps(qc, ensure_ascii=False), encoding="utf-8")


def _write_preprocessed(root) -> None:
    """写 Stage 3 的 `preprocessed/`（F4 的贡献分重算路径要读它，不是要重跑预处理）。"""
    outdir = root / "preprocessed"
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9)
    X = rng.normal(size=(len(_instances()), len(_FEATURE_NAMES)))
    # 让 appearance 块与 shape 块的方差不同，F4 的条形才有可读的差异
    X[:, :3] *= 3.0
    np.save(outdir / "X.npy", X)
    (outdir / "keep_features.json").write_text(json.dumps(list(_FEATURE_NAMES)), encoding="utf-8")


def _build_fake_tree(root) -> None:
    """造出与真实 `output/` 同构的最小目录树。"""
    rows = _instances()
    root.mkdir(parents=True, exist_ok=True)
    meta_columns = [
        "instance_uid",
        "day",
        "fov",
        "frame_idx",
        "time_hours",
        "label",
        "centroid_row",
        "centroid_col",
        "r0",
        "c0",
        "r1",
        "c1",
        "area_px",
        "is_small",
        "touches_border",
    ]
    (root / "instances").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[meta_columns].to_csv(root / "instances" / "all_instances.csv", index=False)
    _write_patches(root, rows)
    _write_analysis(root, rows)
    _write_modules(root)
    _write_preprocessed(root)


@pytest.fixture
def fake_analysis_dir(tmp_path):
    """与真实 `output/` 同构的合成产物树（3 个簇 × 3 个视野、12 个实例）。"""
    root = tmp_path / "output"
    _build_fake_tree(root)
    return root


@contextlib.contextmanager
def _spy_figures(monkeypatch):
    """把 `figs._open_figure` 包一层，收集出图期间的 figure（`plt.close` 之后 artist 仍可读）。

    出图函数自己负责关闭 figure（这是被测行为之一），所以要看"画进去的是什么"就不能靠
    `plt.gcf()`，只能在这层把 figure 抄下来。收集顺序 = `_RENDER_ORDER`。
    """
    original = figs._open_figure
    figures: list = []

    @contextlib.contextmanager
    def spy(**kwargs):
        """透传 `_open_figure` 的上下文管理器，顺手把 figure 记进 `figures`。"""
        with original(**kwargs) as figure:
            figures.append(figure)
            yield figure

    monkeypatch.setattr(figs, "_open_figure", spy)
    yield figures


def _images(figure) -> list[np.ndarray]:
    """收集一张 figure 里所有 `imshow` 的数组（色条的 QuadMesh 不是 image，不会混进来）。"""
    return [np.asarray(image.get_array()) for axis in figure.axes for image in axis.images]


def test_render_static_writes_seven_figures(fake_analysis_dir, tmp_path):
    """守口径：必须正好产出 F1–F6b 这 7 个文件名，且每张都不是空壳。"""
    paths = figs.render_static(fake_analysis_dir, outdir=tmp_path)
    names = {p.name for p in paths}
    assert names == _EXPECTED_FIGURES
    # 1 KB 是"真画了东西"的下限：空白 PNG 只有几百字节，能顶住"建了 figure 但没画"的退化。
    assert all(p.stat().st_size > 1000 for p in paths)


def test_render_static_tolerates_single_cluster(fake_analysis_dir, tmp_path):
    """守边界：退化路径（所有实例归 1 簇）也必须出满 7 张图，不许 IndexError。"""
    # 只有一个簇时不应 IndexError（Review Focus #5）
    paths = figs.render_static(fake_analysis_dir, outdir=tmp_path, force_single_cluster=True)
    assert {p.name for p in paths} == _EXPECTED_FIGURES
    assert all(p.stat().st_size > 1000 for p in paths)


def test_render_static_closes_every_figure(fake_analysis_dir, tmp_path):
    """守口径：出图函数自己 plt.close，跑完进程里不留 figure（长跑/多次调用不吃内存）。"""
    figs.render_static(fake_analysis_dir, outdir=tmp_path)
    assert plt.get_fignums() == []


def test_render_static_names_the_stage_of_a_missing_artifact(fake_analysis_dir, tmp_path):
    """守口径：缺 Stage5 产物时报错且消息点名 Stage5，不许默默少画一张。"""
    (fake_analysis_dir / "analysis" / "cluster_labels.csv").unlink()
    with pytest.raises(ValueError, match="Stage5"):
        figs.render_static(fake_analysis_dir, outdir=tmp_path)


def test_render_static_missing_modules_points_at_stage4(fake_analysis_dir, tmp_path):
    """守口径：缺 Stage4 的 cluster_map.npz 时消息点名 Stage4（定位到阶段而非文件名）。"""
    (fake_analysis_dir / "modules" / "cluster_map.npz").unlink()
    with pytest.raises(ValueError, match="Stage4"):
        figs.render_static(fake_analysis_dir, outdir=tmp_path)


def test_render_static_rejects_a_label_uid_mismatch(fake_analysis_dir, tmp_path):
    """uid 与 (fov, label, frame_idx) 对不上时报错：否则代表图会静静拼到别的实例上。"""
    meta_path = fake_analysis_dir / "instances" / "all_instances.csv"
    meta = pd.read_csv(meta_path)
    meta.loc[0, "label"] = 99  # uid 仍是 A1_1_0，但 label 已对不上
    meta.to_csv(meta_path, index=False)

    with pytest.raises(ValueError, match="重建"):
        figs.render_static(fake_analysis_dir, outdir=tmp_path)


def test_cluster_panels_pick_four_images_from_own_cluster():
    """每簇代表图必须是 2×2 = 4 张、且只来自本簇（不能串到别的簇去）。"""
    clusters = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])
    coords = np.vstack([np.zeros((5, 2)), np.full((4, 2), 9.0)]) + np.arange(9)[:, None] * 0.01
    uids = [f"A1_{i + 1}_0" for i in range(9)]
    patches = {uid: np.full((10, 10), 100, dtype=np.uint8) for uid in uids}
    # 上面的 0.01 递增偏移让"离簇中心最近"的排序唯一，选中哪 4 张可复现、不靠随机数。

    panels = figs._cluster_panels(coords, clusters, uids, patches, patches, n_per_cluster=4)

    assert set(panels) == {1, 2}
    for cluster, panel in panels.items():
        assert panel.native.shape == (20, 20)  # 2 行 × 2 列，每格 10×10
        assert panel.sift.shape == (20, 20)
        assert len(panel.uids) == 4
        assert all(uids.index(uid) in np.flatnonzero(clusters == cluster) for uid in panel.uids)


def test_cluster_panels_survive_a_cluster_without_patches():
    """守边界：簇内可用 patch 少于 n_per_cluster 时照样出面板（其余格留白），不越界取图。"""
    clusters = np.array([1, 1, 2])
    coords = np.zeros((3, 2))
    uids = ["A1_1_0", "A1_2_0", "A1_3_0"]
    patches = {"A1_1_0": np.zeros((8, 8), dtype=np.uint8), "A1_3_0": np.zeros((8, 8), dtype=np.uint8)}

    panels = figs._cluster_panels(coords, clusters, uids, patches, patches, n_per_cluster=4)

    # 簇 1 只有 1 张图：照样给面板（另一格留白）；一个候选都没有时给 None，而不是崩
    assert panels[1].native.shape == (16, 16)
    assert len(panels[1].uids) == 1


def test_contribution_scores_fall_back_to_recompute(fake_analysis_dir):
    """Stage 4 没有落盘 contribution 时，用 X.npy + keep_features.json 重算（不重跑模块发现）。"""
    qc_path = fake_analysis_dir / "modules" / "stage4_qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    qc.pop("type_contribution")
    qc.pop("scope_contribution")
    qc_path.write_text(json.dumps(qc, ensure_ascii=False), encoding="utf-8")

    type_scores, scope_scores, source = figs._contribution_scores(fake_analysis_dir)

    assert source == "recomputed"
    assert type_scores.shape == (3,)
    assert scope_scores.shape == (3,)
    assert np.all(np.isfinite(type_scores))
    assert np.all(np.isfinite(scope_scores))


def test_contribution_scores_prefer_persisted_stage4_qc(fake_analysis_dir):
    """守口径：已落盘的 type/scope contribution 优先直接采用，不重算、不改写数值。"""
    type_scores, scope_scores, source = figs._contribution_scores(fake_analysis_dir)
    assert source == "stage4_qc"
    assert np.allclose(type_scores, [0.62, 0.38, 0.0])
    assert np.allclose(scope_scores, [0.51, 0.30, 0.19])


def test_cjk_font_missing_warns_instead_of_silently_degrading(monkeypatch):
    """没有候选中文字体时必须 warnings.warn（中文，点名影响的后果），不许静默出 7 张方框图。"""
    monkeypatch.setattr(figs, "_CJK_FONT_CANDIDATES", ("绝对不存在的测试字体",))
    with pytest.warns(UserWarning, match="方框"):
        assert figs._configure_fonts() is None


def test_cjk_font_configuration_never_degrades_silently():
    """要么字体名进了 rcParams，要么发过警告——不允许第三种结局。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        name = figs._configure_fonts()
    if name is None:
        assert any("中文" in str(item.message) for item in caught)
    else:
        assert name in plt.rcParams["font.sans-serif"]


def test_render_static_logs_the_cjk_font_degradation(fake_analysis_dir, tmp_path, monkeypatch, capsys):
    """字体降级要落到 render_static 的汇总日志行里（跑完才知道图上的中文是方框）。"""
    monkeypatch.setattr(figs, "_CJK_FONT", None)
    figs.render_static(fake_analysis_dir, outdir=tmp_path)
    summary = capsys.readouterr().out
    assert "字体" in summary
    assert "方框" in summary


# —— 出图内容 ——


def test_f1_native_panel_title_admits_the_intensity_is_rescaled(fake_analysis_dir, tmp_path, monkeypatch):
    """F1 的"原始尺寸"只指几何：强度经 SPOT 的 rescale_intensity 归一化过，标题必须说清楚。"""
    with _spy_figures(monkeypatch) as figures:
        figs.render_static(fake_analysis_dir, outdir=tmp_path)
    titles = [axis.get_title() for axis in figures[_RENDER_ORDER.index("F1_umap_clusters_patches.png")].axes]
    native_titles = [title for title in titles if "原始尺寸" in title]
    assert native_titles
    assert all("强度" in title and "归一化" in title for title in native_titles)


def test_f3_draws_a_shared_grid_density_field(fake_analysis_dir, tmp_path, monkeypatch):
    """F3 必须是真正的二维密度场（共用网格 + 共用色标），而不是按点着色的散点。"""
    with _spy_figures(monkeypatch) as figures:
        figs.render_static(fake_analysis_dir, outdir=tmp_path)
    figure = figures[_RENDER_ORDER.index("F3_umap_density.png")]
    field_axes = [axis for axis in figure.axes if axis.images]
    fields = [np.asarray(axis.images[0].get_array(), dtype=np.float64) for axis in field_axes]
    assert len(fields) == 3  # 三个簇各一层
    for field in fields:
        assert field.ndim == 2 and min(field.shape) > 8
        assert np.all(np.isfinite(field))
        assert np.unique(field).size > 10  # 不是一块平色
        assert field.max() > 0.0
    # 所有面板共用同一个网格范围与色标，密度才可比
    assert len({tuple(np.round(axis.images[0].get_extent(), 9)) for axis in field_axes}) == 1
    assert len({tuple(np.round(axis.images[0].get_clim(), 12)) for axis in field_axes}) == 1
    assert field_axes[0].get_title() and "密度场" in field_axes[0].get_title()


def test_f3_raises_when_a_cluster_has_no_density_field(tmp_path):
    """密度场缺一层就报错（说明布线坏了）：不许悄悄画成一块平色。"""
    coords = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [5.0, 5.0], [5.1, 5.0]])
    clusters = np.array([1, 1, 1, 2, 2])
    field = figs._density_fields(coords, clusters)
    broken = figs._DensityField(fields={1: field.fields[1]}, extent=field.extent, vmax=field.vmax)

    with pytest.raises(ValueError, match="密度场"):
        figs._plot_f3(tmp_path / "f3.png", coords, clusters, broken)
    # 先校验后落盘：报错时不能留下一张画了一半的 F3。
    assert not (tmp_path / "f3.png").exists()


def test_force_single_cluster_degrades_via_a_warning(fake_analysis_dir, tmp_path):
    """退化路径（所有实例归 1 簇）要会warnings.warn 告知，而不是悄悄换一套口径。"""
    with pytest.warns(UserWarning, match="force_single_cluster"):
        figs.render_static(fake_analysis_dir, outdir=tmp_path, force_single_cluster=True)


def test_f5_display_matrix_is_the_stored_dendrogram_order(fake_analysis_dir, tmp_path, monkeypatch):
    """F5 不许二次置换：画出来的矩阵就是落盘的（已重排）矩阵，对称 + 块结构与色条一致。"""
    cluster_map = figs._read_npz(fake_analysis_dir / "modules" / "cluster_map.npz", "Stage4")
    state = figs._clustermap_display(cluster_map)
    stored = np.asarray(cluster_map["corr"], dtype=np.float64)
    order = np.asarray(cluster_map["reordered_ind"], dtype=np.int64)
    module_labels = np.asarray(cluster_map["module_labels"], dtype=np.int64)

    assert np.array_equal(state.order, order)
    assert np.array_equal(state.distances, stored)  # 不再 corr[np.ix_(order, order)]
    assert np.allclose(state.distances, state.distances.T)
    assert np.allclose(np.diag(state.distances), 0.0)

    # 块结构：同模块内距离 < 跨模块距离（fixture 里同块 0.1–0.3、跨块 1.3–1.7）
    modules = state.module_of_position
    assert modules.tolist() == module_labels[order].tolist()
    same = modules[:, None] == modules[None, :]
    off_diagonal = ~np.eye(modules.size, dtype=bool)
    assert state.distances[same & off_diagonal].mean() < state.distances[~same].mean()

    # 色条按显示序连续成块，块长与模块特征数一致，块内颜色唯一
    assert state.module_counts == {1: 2, 2: 2, 3: 2}
    boundaries = np.cumsum(list(state.module_counts.values()))[:-1]
    for block, module in zip(np.split(state.strip, boundaries), state.module_counts):
        assert len({tuple(np.round(colour, 12)) for colour in block}) == 1
        assert tuple(np.round(block[0], 12)) == tuple(np.round(_MODULE_COLOURS[module], 12))


def test_f5_rejects_a_reordered_ind_that_disagrees_with_the_tree(fake_analysis_dir):
    """落盘的列序与 linkage 的 dendrogram 叶序不一致时报中文错误（说明 npz 是旧产物）。"""
    cluster_map = figs._read_npz(fake_analysis_dir / "modules" / "cluster_map.npz", "Stage4")
    tampered = dict(cluster_map)
    tampered["reordered_ind"] = np.roll(np.asarray(cluster_map["reordered_ind"], dtype=np.int64), 1)
    with pytest.raises(ValueError, match="dendrogram"):
        figs._clustermap_display(tampered)


def test_f5_figure_draws_the_stored_matrix(fake_analysis_dir, tmp_path, monkeypatch):
    """真正落到 F5 画面上的热图、色条与图例也必须是落盘口径（树/矩阵/图例一一对应）。"""
    cluster_map = figs._read_npz(fake_analysis_dir / "modules" / "cluster_map.npz", "Stage4")
    stored = np.asarray(cluster_map["corr"], dtype=np.float64)
    state = figs._clustermap_display(cluster_map)
    with _spy_figures(monkeypatch) as figures:
        figs.render_static(fake_analysis_dir, outdir=tmp_path)
    figure = figures[_RENDER_ORDER.index("F5_sam_modules.png")]
    images = _images(figure)
    matrices = [image for image in images if image.shape == stored.shape]
    assert len(matrices) == 1
    assert np.allclose(matrices[0], stored)
    strips = [image for image in images if image.ndim == 3 and image.shape[0] == 1]
    assert len(strips) == 1
    assert np.allclose(strips[0][0], np.asarray(cluster_map["colorbar"], dtype=np.float64))

    # 图例颜色必须是该模块自己的色块颜色（不能是拿 original 列序的 mask 去索引 dendrogram 列序的色条）
    strip_axis = next(axis for axis in figure.axes if axis.get_title().startswith("模块色条"))
    legend = strip_axis.get_legend()
    assert legend is not None
    # matplotlib 3.7+ 叫 legend_handles，旧版叫 legendHandles（图例是直接用 handles 建的，
    # 所以不能靠 get_legend_handles_labels）
    legend_handles = getattr(legend, "legend_handles", None) or legend.legendHandles
    labels = [text.get_text() for text in legend.get_texts()]
    starts = np.cumsum([0, *list(state.module_counts.values())[:-1]])
    expected = [tuple(np.round(state.strip[start], 12)) for start in starts]
    drawn = [tuple(np.round(handle.get_facecolor()[:3], 12)) for handle in legend_handles]
    assert drawn == expected
    assert [label.split(" ")[0] for label in labels] == [f"M{module}" for module in state.module_counts]


def test_f6b_day_fractions_sum_to_one(fake_analysis_dir, tmp_path, monkeypatch):
    """F6b 的堆叠条是"簇内占比"：每个簇的各 day 高度之和必须正好是 1。"""
    with _spy_figures(monkeypatch) as figures:
        figs.render_static(fake_analysis_dir, outdir=tmp_path)
    figure = figures[_RENDER_ORDER.index("F6b_cluster_day_composition.png")]
    axis = figure.axes[0]
    assert len(axis.patches) == 3 * 3  # 3 个簇 × 3 个 day
    totals: dict[float, float] = {}
    for patch in axis.patches:
        centre = round(float(patch.get_x() + patch.get_width() / 2.0), 6)
        totals[centre] = totals.get(centre, 0.0) + float(patch.get_height())
    assert len(totals) == 3
    assert all(abs(total - 1.0) < 1e-9 for total in totals.values())


def test_motion_note_distinguishes_absent_column_from_all_nan():
    """F6 的 motion 说明不能对"列不存在"声称已核对 NaN。"""
    uids = [row["instance_uid"] for row in _instances()]
    absent = pd.DataFrame({"instance_uid": uids, "area": np.ones(len(uids))})
    all_nan = pd.DataFrame({"instance_uid": uids, "area": np.ones(len(uids)), "speed": np.full(len(uids), np.nan)})
    with_values = pd.DataFrame({"instance_uid": uids, "area": np.ones(len(uids)), "speed": np.arange(len(uids)) * 1.0})

    absent_note = figs._motion_note(absent)
    assert "未提供" in absent_note
    assert "NaN" not in absent_note
    assert "NaN" in figs._motion_note(all_nan)
    assert "非 NaN" in figs._motion_note(with_values)


def test_representative_selection_skips_small_instances():
    """`is_small` 的实例不进代表图候选；整簇都是小实例时才退回包含它们（并记在 fallback 里）。"""
    clusters = np.array([1] * 6 + [2] * 2)
    coords = np.zeros((8, 2))
    # 沿 x 轴排一排：到簇中心（簇内均值）的距离严格递增，选中谁只由距离决定。
    coords[:, 0] = np.arange(8)
    uids = [f"A1_{index + 1}_0" for index in range(8)]
    index = {uid: (Path("patches.npz"), "native", "sift") for uid in uids}

    selection = figs._representative_uid_selection(
        coords, clusters, uids, index, n_per_cluster=2, small_uids={"A1_1_0", "A1_7_0", "A1_8_0"}
    )

    assert selection.uids_by_cluster[1] == ["A1_4_0", "A1_3_0"]  # 最近的 A1_1_0 是小实例 → 被跳过
    assert "A1_1_0" not in selection.uids_by_cluster[1]
    assert selection.n_excluded_small == 1
    # 簇 2 里两个全是小实例：不能出空面板，退回包含（图注会写明）
    assert set(selection.uids_by_cluster[2]) == {"A1_7_0", "A1_8_0"}
    assert selection.fallback_clusters == (2,)


def test_cluster_panels_apply_the_same_small_instance_filter():
    """`_cluster_panels` 必须和预选用同一套过滤，否则实际拼进图里的仍可能是小实例。"""
    clusters = np.array([1, 1, 1, 2])
    coords = np.zeros((4, 2))
    coords[:, 0] = np.arange(4)
    uids = ["A1_1_0", "A1_2_0", "A1_3_0", "A1_4_0"]
    patches = {uid: np.full((10, 10), 120, dtype=np.uint8) for uid in uids}

    panels = figs._cluster_panels(coords, clusters, uids, patches, patches, n_per_cluster=2, small_uids={"A1_1_0"})

    assert "A1_1_0" not in panels[1].uids
    assert len(panels[1].uids) == 2
    assert panels[2].uids == ["A1_4_0"]


def test_f1_caption_states_the_small_instance_filter(fake_analysis_dir, tmp_path, monkeypatch):
    """F1 图注要说明候选跳过了 `is_small`，并点名“整簇都是小实例”的簇。"""
    # 用 min() 决定保留哪一个，让结果可复现（集合序不确定）。
    meta_path = fake_analysis_dir / "instances" / "all_instances.csv"
    labels = pd.read_csv(fake_analysis_dir / "analysis" / "cluster_labels.csv")
    cluster_one = set(labels.loc[labels["cluster"] == 1, "instance_uid"])
    # 簇 1 里只留一个非小实例：其余的都不该进候选
    keep = min(cluster_one)
    meta = pd.read_csv(meta_path)
    meta.loc[meta["instance_uid"].isin(cluster_one) & (meta["instance_uid"] != keep), "is_small"] = True
    # 簇 2 全部标成小实例（包括另外两个视野的实例），验证退化分支
    cluster_two = set(labels.loc[labels["cluster"] == 2, "instance_uid"])
    meta.loc[meta["instance_uid"].isin(cluster_two), "is_small"] = True
    meta.to_csv(meta_path, index=False)

    seen: dict[str, list[str]] = {}
    original = figs._read_patch_arrays

    def spy(index, uids):
        """包一层 `_read_patch_arrays`，记下这次实际被读进内存的 uid 列表。"""
        seen["uids"] = list(uids)
        return original(index, uids)

    monkeypatch.setattr(figs, "_read_patch_arrays", spy)
    with _spy_figures(monkeypatch) as figures:
        figs.render_static(fake_analysis_dir, outdir=tmp_path)

    assert keep in seen["uids"]
    assert not (cluster_one - {keep}) & set(seen["uids"])
    # 簇 2 整簇都是小实例 → 仍会包含（否则出空图），但图注要说清楚
    assert cluster_two & set(seen["uids"])
    caption = figures[_RENDER_ORDER.index("F1_umap_clusters_patches.png")]._suptitle.get_text()
    assert "is_small" in caption
    assert "簇 2 内没有非小实例" in caption


def test_render_static_materialises_only_the_selected_patches(fake_analysis_dir, tmp_path, monkeypatch):
    """只读选中的 4 张/簇（12 个实例里簇 1 有 6 个候选 → 总共只读 10 张，不是全部）。"""
    seen: dict[str, list[str]] = {}
    original = figs._read_patch_arrays

    def spy(index, uids):
        """包一层 `_read_patch_arrays`，记下这次实际被读进内存的 uid 列表。"""
        seen["uids"] = list(uids)
        return original(index, uids)

    monkeypatch.setattr(figs, "_read_patch_arrays", spy)
    figs.render_static(fake_analysis_dir, outdir=tmp_path)

    # 4（簇 1 的 12 个候选里只取 4）+ 3 + 3 = 10；把整簇都读进来就会是 12。
    assert len(seen["uids"]) == 10
    assert len(set(seen["uids"])) == 10


def test_render_static_keeps_the_previous_figures_when_rendering_fails(fake_analysis_dir, tmp_path):
    """渲染失败时不能先把旧图删掉：第一遍成功，第二遍坏产物 → 旧图仍在，且不留半成品。"""
    paths = figs.render_static(fake_analysis_dir, outdir=tmp_path)
    sizes = {path.name: path.stat().st_size for path in paths}

    # 让 F5 在读产物时就炸：corr 变成非方阵
    cluster_map_path = fake_analysis_dir / "modules" / "cluster_map.npz"
    cluster_map = figs._read_npz(cluster_map_path, "Stage4")
    broken = dict(cluster_map)
    broken["corr"] = np.zeros((3, 3))
    np.savez_compressed(cluster_map_path, **broken)

    with pytest.raises(ValueError, match="Stage4"):
        figs.render_static(fake_analysis_dir, outdir=tmp_path)

    for name, size in sizes.items():
        kept = tmp_path / name
        assert kept.exists()
        assert kept.stat().st_size == size
    # 原子写入的临时文件必须清干净：失败时不许留 .tmp-* 半成品。
    assert not list(tmp_path.glob(".tmp-*"))


# —— F1 的版式：景观 1:1、簇号够大、代表图铺进 ~4 行 ——


@pytest.mark.parametrize(
    ("n_clusters", "expected"),
    # expected = (per_row, rows, panel_cols)：panel_cols 是 2 张/簇拼图的列数，不是行数。
    [
        (1, (1, 1, 2)),
        (3, (1, 3, 2)),
        (4, (1, 4, 2)),
        (5, (2, 3, 4)),
        (7, (2, 4, 4)),
        (8, (2, 4, 4)),
        (9, (3, 3, 6)),
        (11, (3, 4, 6)),
    ],
)
def test_f1_patch_grid_stays_within_four_rows(n_clusters, expected):
    """右侧代表图铺进 ~4 行（K=7 → 4 行 4 列），不再"每簇一行"把景观拉长。"""
    per_row, rows, panel_cols = figs._f1_patch_grid(n_clusters)
    assert (per_row, rows, panel_cols) == expected
    assert panel_cols == 2 * per_row
    assert per_row * rows >= n_clusters  # 每个簇都有格子
    assert rows <= figs._F1_TARGET_ROWS  # 行数不随簇数爆炸
    if n_clusters > 1:
        assert per_row == math.ceil(n_clusters / figs._F1_TARGET_ROWS)


def test_f1_patch_grid_rejects_empty_cluster_list():
    """守边界：簇数为 0 时报 ValueError，不许返回一个看似合法的空网格。"""
    with pytest.raises(ValueError, match="至少要有 1 个簇"):
        figs._f1_patch_grid(0)


def test_f1_umap_panel_is_square_with_readable_cluster_numbers(fake_analysis_dir, tmp_path, monkeypatch):
    """景观必须 1:1（x/y 每单位等长且跨度相同），簇号字号够大——这是"别把景观拉长"的可测形式。"""
    with _spy_figures(monkeypatch) as figures:
        figs.render_static(fake_analysis_dir, outdir=tmp_path)
    figure = figures[_RENDER_ORDER.index("F1_umap_clusters_patches.png")]
    map_ax = figure.axes[0]  # 第一个建的就是景观面板

    assert map_ax.get_aspect() == 1.0
    x_span = map_ax.get_xlim()[1] - map_ax.get_xlim()[0]
    y_span = map_ax.get_ylim()[1] - map_ax.get_ylim()[0]
    assert x_span == pytest.approx(y_span)  # 两个方向共用同一个跨度 → 方框是正方形

    labels = [text for text in map_ax.texts if text.get_text().isdigit()]
    assert {text.get_text() for text in labels} == {"1", "2", "3"}  # fixture 的 3 个簇都标了号
    assert all(text.get_fontsize() >= figs._F1_CLUSTER_LABEL_FONTSIZE for text in labels)


def test_square_view_rejects_non_two_dimensional_coordinates():
    """守边界：坐标不是 (n, 2) 时直接报错，不许静默只取前两列画成正方形。"""
    figure = plt.figure()
    try:
        with pytest.raises(ValueError, match="UMAP 坐标"):
            figs._square_view(figure.add_subplot(1, 1, 1), np.zeros((4, 3)))
    finally:
        plt.close(figure)


# —— 图说明 README.md ——


def test_render_static_writes_the_figure_readme(fake_analysis_dir, tmp_path):
    """出图顺手写 `README.md`：13 张图全列出来，并标出这棵目录里哪些还没画。"""
    figs.render_static(fake_analysis_dir, outdir=tmp_path)
    readme = tmp_path / "README.md"
    assert readme.exists()
    text = readme.read_text(encoding="utf-8")

    for name in _EXPECTED_FIGURES:  # 7 张静态图：已生成
        assert name in text
    for name in ("F7_stacked_cluster_proportions.png", "F11_match_events.png"):  # 时间图：还没画也要列
        assert name in text
    assert "已生成 **7 / 13**" in text
    assert "缺失 6 张" in text
    # 三条统一口径（motion 置空 / 离散采样点 / 簇号含义）必须在说明里
    assert "motion 是空的，不是 0" in text
    assert "离散采样点" in text
    assert "平均偏心率" in text
    assert not list(tmp_path.glob(".tmp-*"))


def test_readme_marks_every_figure_generated_when_all_pngs_exist(tmp_path):
    """静态图与时间图都画完之后，说明里不该再出现"缺失"。"""
    for name in _RENDER_ORDER:
        (tmp_path / name).write_bytes(b"png")
    for name in (
        "F7_stacked_cluster_proportions.png",
        "F8_transition_graph.png",
        "F9_umap_trajectories.png",
        "F10a_trajectory_affinity_fov.png",
        "F10b_trajectory_affinity_cluster.png",
        "F11_match_events.png",
    ):
        (tmp_path / name).write_bytes(b"png")

    lines = figs._readme_lines(tmp_path)

    assert any("已生成 **13 / 13**" in line for line in lines)
    assert not any("缺失 6 张" in line for line in lines)


def test_readme_documents_every_rendered_figure():
    """`_FIGURE_DOCS` 必须覆盖两张图名清单的全部图（少一条就是"图有、说明没有"）。"""
    documented = [doc.name for doc in figs._FIGURE_DOCS]
    assert documented == list(figs._FIGURE_NAMES) + list(figs._TEMPORAL_FIGURE_NAMES)
