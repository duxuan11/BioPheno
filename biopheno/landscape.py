"""Stage 5：静态表型景观（UMAP + k-means 表型簇 + 按 eccentricity 重排；Spec §6-Stage5）。

上游（`SPOT/Analysis/sam_analysis.py`）的真实返回契约，逐条对照源码记下：

- `KMeans_cluster_low_dimensional_embedding_with_scan(low_dimensional_feats, k_range=(2,20), ploton=True)`
  → **四元组** `(cluster_labels, clusterer, best_k, scan_results)`。`cluster_labels` 是 (N,) 的 **0-based**
  整数标签（= `clusterer.predict`），`clusterer` 是最终拟合的 `KMeans(n_clusters=best_k, random_state=0,
  n_init=10)`，`best_k` 是 int，`scan_results` 是 pandas DataFrame（索引 = k、只有一列 `'Scaled Inertia'`）。
  两个容易踩的点：**扫描范围是 `range(k_range[0], k_range[1])`，不含上界**（所以 `cfg.KMEANS_K_RANGE=(2,12)`
  实际只扫 2–11），且选 K 的依据不是 inertia 本身，而是"adjusted inertia"
  `inertia_ / 总平方和 + 0.02*k`（`kMeansRes`）。`ploton=True` 会开 figure 并 `plt.show()`，本项目一律传 False。
  **但它在当前环境（pandas 3.0.6）下直接抛 `KeyError: 0`**：内部调用的 `chooseBestKforKMeansParallel`
  末行是 `results.idxmin()[0]`，而 pandas>=3 不再把整数键退化为按位置取样（`Series.__getitem__` 一律按
  标签），于是取 `'Scaled Inertia'` 那列的标量时崩掉。本模块**不改 SPOT 源码**，只在调用期间把这个
  内部函数换成逐行等价的实现（见 `_pandas3_elbow_compat` / `_choose_best_k_pandas3`）。
- `reorder_and_relabel_clusters(clusters, score, order='ascending')` → `(relabeled_clusters,
  cluster_mapping_dict)`。前者把簇 id 换成**排序号**，而排序号从 **0** 开始（源码里 `enumerate(ordering)`）；
  后者是 `{旧标签: 新标签}` 的 dict。论文的编号口径是 1..K（按平均 eccentricity 从低到高），
  所以本模块统一 +1（`order_clusters_by_score`）。
- `compute_heatmap_density_image_Embedding(lower_dimension_pts_2D, all_conditions=None, ...)` →
  `(all_pts_density, all_pts_density_select)`：前者是长度 = 条件数的 list，第 i 项是该条件下**每个点处**的
  局部密度（不是整张网格图）；后者是 (n_conditions, N) 的布尔矩阵。它对每个条件**无条件**
  `plt.subplots()` + `plt.show()` 且从不关闭 figure——所以本模块的 `density` 包装层负责把 figure 关掉。
  它不落盘任何东西（`saveplotsfolder=None` 时不写 svg），Stage 5 只把它作为可复用的计算入口，
  密度图（论文 Fig. 3 的 F3）留给绘图阶段按簇调用。
- `umap.UMAP(**cfg.UMAP_PARAMS)`：umap-learn 0.5.12 在 `random_state` 非 None 时会把 `n_jobs` 强制改成 1
  （源码 `umap_.py:1952` 的 UserWarning "n_jobs value ... overridden to 1 by setting random_state"），
  所以论文参数可以逐字传、不需要额外补 `n_jobs=1` 也已经确定（实测 3 次坐标完全一致）。

簇编号的排序 score 是**平均 eccentricity**（论文口径：1..K 按簇内平均偏心率递增），而且必须用**原生量纲**：
论文的编号、图与报告都按原生单位（0 = 类圆、越接近 1 越扁长），而 Yeo-Johnson + 标准化虽然逐行单调，
却可能翻转**簇均值**的先后（各簇分布宽度不同时）。所以选择顺序是 `all_features.csv` 的原生
`moment_eccentricity` → bbox 推导值 →（最后手段）Stage 3 预处理列；实际走了哪条路写在 `kmeans_metrics.json`
的 `eccentricity_score.source` 里，绝不静默换口径，并把每簇均值（原生单位 / score 单位）与三套单调性检查
（`monotone_native` / `monotone_score` / `monotone_preprocessed`）一并落盘。
`per_instance_features.csv` 的 `eccentricity` 列是**另一件事**：它只需要原生量纲，取原生列、缺列时用 bbox，
与"score 走了哪条路"无关，所以单独记 `per_instance_features.eccentricity_path`。

产物（`output/analysis/`）：`umap_2d.csv`（`instance_uid` + 两维坐标）、`cluster_labels.csv`
（`instance_uid` / `day` / `fov` / `frame_idx` / `time_hours` / 1-based `cluster` / 0-based `cluster_raw`）、
`per_instance_features.csv`（area / eccentricity / intensity / speed，其中 speed 恒 NaN——motion 被有意跳过）、
`kmeans_metrics.json`（K 扫描的 inertia / silhouette / adjusted inertia、选中的 K、簇映射、排序口径与 QC）。
每份产物都带 `instance_uid`：行身份不再靠"假设 Stage 2 的行序"传递。
"""

import contextlib
import json
import time
import warnings
from pathlib import Path
from typing import NoReturn

import matplotlib

# SPOT 的密度函数无条件 `plt.show()`：交互式后端（本机默认 TkAgg）会真的开窗口、在测试/长跑里挂住。
# 必须在任何 pyplot 导入之前设置（本模块自己不导入 pyplot，`density` 里按需导入）。
matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from SPOT.Analysis import sam_analysis as sam

from biopheno import config as cfg

# —— 输入产物（与 preprocess.py / features.py / instances.py 的落盘位置一致）——
_PREPROCESSED_SUBDIR: str = "preprocessed"
_FEATURES_SUBDIR: str = "features"
_INSTANCES_SUBDIR: str = "instances"
_X_FILE: str = "X.npy"
_KEEP_FILE: str = "keep_features.json"
_ALL_FEATURES_FILE: str = "all_features.csv"
_ALL_INSTANCES_FILE: str = "all_instances.csv"

# —— 本阶段产物（Spec §6-Stage5）——
_ANALYSIS_SUBDIR: str = "analysis"
_UMAP_FILE: str = "umap_2d.csv"
_LABELS_FILE: str = "cluster_labels.csv"
_PER_INSTANCE_FILE: str = "per_instance_features.csv"
_METRICS_FILE: str = "kmeans_metrics.json"

# 簇排序 score 的特征名（Stage 2 的 binary morphology 家族，见 SPOT preprocessing 的 SAM_FEATURE_GROUPS）。
_ECC_FEATURE: str = "moment_eccentricity"
# 排序 score 的来源标签（写进 QC 的 `eccentricity_score.source`，机器可读、便于下游核对）。
_NATIVE_SOURCE: str = "moment_eccentricity"
_BBOX_SOURCE: str = "bbox"
_PREPROCESSED_SOURCE: str = "preprocessed"
# bbox 偏心率兜底需要的列（Stage 1 的 all_instances.csv）。
_BBOX_COLUMNS: tuple[str, ...] = ("r0", "c0", "r1", "c1")
# 每实例解释特征（Spec §6-Stage5：area / eccentricity / intensity，speed 置空）。
_AREA_FEATURE: str = "area"
_INTENSITY_FEATURE: str = "mean_intensity"
_SPEED_FEATURE: str = "speed"

# 行身份的来源描述（写进 QC，便于下游核对）。
_ROW_SOURCE: str = "features/all_features.csv 的 instance_uid 索引"

# 逐实例特征表的列序（下游 Stage 6/7 直接按列名取）。
_PER_INSTANCE_COLUMNS: tuple[str, ...] = ("instance_uid", _AREA_FEATURE, "eccentricity", "intensity", _SPEED_FEATURE)

# QC 里记录的 pandas-3 兼容说明（见 `_pandas3_elbow_compat`）。
_ELBOW_SHIM_NOTE: str = (
    "SPOT 2.0.1 的 chooseBestKforKMeansParallel 末行是 `results.idxmin()[0]`，pandas>=3 把整数键当标签、"
    "抛 KeyError: 0。本模块不改 SPOT 源码，只在调用扫描函数期间把这个内部函数换成逐行等价的实现"
    "（复用它的 kMeansRes 打分与 Parallel 并行，只把 `[0]` 换成 `.iloc[0]`），调用结束立即还原。"
)


def _fail(message: str) -> NoReturn:
    """抛出一句带上下文的中文 ValueError（薄封装只为统一文案格式）。"""
    raise ValueError(message)


def _validate_embedding(embedding) -> np.ndarray:
    """核对嵌入坐标：二维、(N, >=2)、至少 2 个点、全有限。

    非有限坐标会让 k-means 直接崩在 sklearn 里（报错看不出是哪个上游产物坏了），密度图也会给出
    全 NaN，所以先在这里拦下。
    """
    emb = np.asarray(embedding, dtype=np.float64)
    if emb.ndim != 2:
        _fail(f"嵌入坐标必须是二维 (n_objects, n_dim)，实际 {emb.ndim} 维、形状 {emb.shape}")
    if emb.shape[1] < 2:
        _fail(f"嵌入坐标至少要有 2 维（景观用前两维），实际只有 {emb.shape[1]} 维")
    if emb.shape[0] < 2:
        _fail(f"聚类/密度估计至少需要 2 个实例，实际只有 {emb.shape[0]} 个")
    if not np.all(np.isfinite(emb)):
        n_bad = int((~np.isfinite(emb)).any(axis=1).sum())
        _fail(f"嵌入坐标里有 {n_bad} 个非有限值（NaN/Inf）：聚类与密度图都算不了，请检查上游特征矩阵")
    return emb


def embed(X: np.ndarray) -> np.ndarray:
    """把特征矩阵铺成 2D 表型景观：`umap.UMAP(**cfg.UMAP_PARAMS)`，参数原样传、一个都不改。

    为什么不加 `n_jobs=1` 也不调 `n_neighbors`：论文写死了这几个参数（`cfg.UMAP_PARAMS`），而 umap-learn
    0.5.12 在 `random_state` 非 None 时自己就会把 `n_jobs` 覆盖成 1，所以确定性与"参数原样"可以同时成立。

    Returns
    -------
    np.ndarray
        (N, 2) 的 float64 坐标，行序与输入 X 的行序一致（= Stage 2 的实例序）。
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        _fail(f"特征矩阵必须是二维 (n_objects, n_feats)，实际 {X.ndim} 维、形状 {X.shape}")
    if X.shape[0] < 2:
        _fail(f"UMAP 至少需要 2 个实例（要构造近邻图），实际只有 {X.shape[0]} 个")
    reducer = umap.UMAP(**cfg.UMAP_PARAMS)
    embedding = np.asarray(reducer.fit_transform(X), dtype=np.float64)
    if embedding.ndim != 2 or embedding.shape[1] != 2:
        _fail(f"UMAP 返回的坐标形状是 {embedding.shape}，不是预期的 (N, 2)：无法作为 2D 景观")
    return embedding


def _choose_best_k_pandas3(low_dimensional_feats: np.ndarray, k_range) -> tuple[int, pd.DataFrame]:
    """`chooseBestKforKMeansParallel` 的逐行等价替代（唯一差别：pandas-3 兼容的 `.iloc[0]`）。

    elbow 规则原样来自 SPOT：直接复用它的 `kMeansRes`（`inertia_ / 总平方和 + 0.02*k`）打分、同样的
    `Parallel` 并行与 `DataFrame` 组装，选最小 adjusted inertia 的 k，所以结果与上游一致。
    """
    from joblib import Parallel

    ks = list(k_range)
    if not ks:
        _fail(f"k 范围 {k_range} 是空的：没有可扫描的 K（SPOT 的扫描不含上界）")
    scaled = Parallel(n_jobs=cfg.N_JOBS, verbose=0)(sam.kMeansRes(low_dimensional_feats, k) for k in ks)
    results = pd.DataFrame(list(zip(ks, scaled)), columns=["k", "Scaled Inertia"]).set_index("k")
    return int(results.idxmin().iloc[0]), results


@contextlib.contextmanager
def _pandas3_elbow_compat():
    """在调用 SPOT 的扫描期间临时换上 pandas-3 可用的 elbow 选择（见模块 docstring 与 `_ELBOW_SHIM_NOTE`）。

    为什么不用 try/except 兜底：那个 KeyError 是 SPOT 在**算完**所有 KMeans 之后才抛的，retry 会把扫描
    白跑一遍（真实数据上 10 个 K × n_init=10 可能几十秒）。这里只换末行的取标量方式，既不重复计算，
    也不改 SPOT 源码；无论成败都还原原函数，不会把补丁泄漏到进程里其他调用点。
    """
    original = sam.chooseBestKforKMeansParallel
    sam.chooseBestKforKMeansParallel = _choose_best_k_pandas3
    try:
        yield
    finally:
        sam.chooseBestKforKMeansParallel = original


def _inertia_and_silhouette(embedding: np.ndarray, ks: list[int]) -> tuple[dict[str, float], dict[str, float]]:
    """按 k 重算 inertia 与 silhouette（SPOT 的扫描只返回 adjusted inertia，没有这两列）。

    重算用的 `KMeans(n_clusters=k, random_state=cfg.RANDOM_STATE, n_init=10)` 与 SPOT 内部
    （`kMeansRes` 与最终重聚类都用 `random_state=0, n_init=10`）完全同参，所以同一 k 的分区一致。
    轮廓系数只在簇数 > 1 时有定义，退化情形记 NaN（不编造 0）。

    返回 `{str(k): 数值}` 映射（不是 list）：list 一旦与 `k_scanned` 错位，指标就会静默挂到别的 k 上，
    而映射自带 k 标签，读的人与下游都不用数位置。
    """
    inertia: dict[str, float] = {}
    silhouette: dict[str, float] = {}
    for k in ks:
        clusterer = KMeans(n_clusters=int(k), random_state=cfg.RANDOM_STATE, n_init=10).fit(embedding)
        inertia[str(int(k))] = float(clusterer.inertia_)
        n_labels = len(np.unique(clusterer.labels_))
        silhouette[str(int(k))] = (
            float(silhouette_score(embedding, clusterer.labels_)) if n_labels > 1 else float("nan")
        )
    return inertia, silhouette


def cluster(embedding: np.ndarray, k_range=cfg.KMEANS_K_RANGE) -> tuple[np.ndarray, dict]:
    """在 2D 景观上用 SPOT 的 elbow 扫描切表型簇，并补齐 inertia / silhouette 两列指标。

    K 由 SPOT 的 `KMeans_cluster_low_dimensional_embedding_with_scan(..., ploton=False)` 选（最小 adjusted
    inertia）。注意它的扫描是 `range(k_range[0], k_range[1])`——**上界取不到**：`cfg.KMEANS_K_RANGE=(2,12)`
    实际扫 2–11，`metrics["k_scanned"]` 把这个事实落盘，免得下游误以为试过 K=12。

    Returns
    -------
    cluster_labels : (N,) int array
        **0-based** 的 k-means 标签（与 SPOT 一致；1-based 的论文编号由 `order_clusters_by_score` 给）。
    metrics : dict
        `k` / `chosen_k`（SPOT 选中的 K，两者相等，前者是任务说明要求的键）、`k_range`、`k_scanned`、
        `inertia` / `silhouette` / `scaled_inertia`（**都是 `{str(k): 数值}` 映射**：簇内平方和、平均轮廓系数、
        SPOT 自己用的 adjusted inertia）、`scan_rule`（口径说明）。映射而不是 list，是为了让"哪个数对应哪个 k"
        不靠数位置——list 与 `k_scanned` 一旦错位，指标就会静默挂到错的 k 上。
    """
    emb = _validate_embedding(embedding)
    k_min, k_max = int(k_range[0]), int(k_range[1])
    ks = list(range(k_min, k_max))
    if not ks:
        _fail(
            f"k 范围 {k_min}–{k_max} 没有可扫描的 K：SPOT 的扫描是 range(k_range[0], k_range[1])（不含上界），"
            "上界必须严格大于下界"
        )
    if k_min < 2:
        _fail(f"k 范围下界必须 >= 2（1 个簇没有轮廓系数、也不是表型簇），实际 {k_min}")
    if emb.shape[0] < max(ks):
        _fail(f"实例数 {emb.shape[0]} 少于扫描的最大 K {max(ks)}：k-means 无法切出这么多簇，请缩小 k_range")

    with _pandas3_elbow_compat():
        cluster_labels, _clusterer, best_k, scan_results = sam.KMeans_cluster_low_dimensional_embedding_with_scan(
            emb, k_range=(k_min, k_max), ploton=False
        )
    cluster_labels = np.asarray(cluster_labels, dtype=np.int64)
    best_k = int(best_k)
    inertia, silhouette = _inertia_and_silhouette(emb, ks)
    scaled = np.asarray(scan_results["Scaled Inertia"], dtype=np.float64).ravel()
    if scaled.size != len(ks):
        _fail(
            f"SPOT 的 elbow 扫描返回 {scaled.size} 个 adjusted inertia，而 k_scanned 有 {len(ks)} 个"
            f"（{ks}）：数值与 k 对应不上，不能落盘"
        )

    metrics = {
        "k": best_k,
        "chosen_k": best_k,
        "k_range": [k_min, k_max],
        "k_scanned": ks,
        "scan_rule": (
            "SPOT 的 chooseBestKforKMeansParallel 扫 range(k_range[0], k_range[1])（不含上界），"
            "按 adjusted inertia = inertia_ / 总平方和 + 0.02 * k 的最小值选 K"
        ),
        "inertia": inertia,
        "silhouette": silhouette,
        "scaled_inertia": {str(int(k)): float(v) for k, v in zip(ks, scaled)},
        "elbow_shim": {
            "spot_function": "sam_analysis.KMeans_cluster_low_dimensional_embedding_with_scan(..., ploton=False)",
            "pandas3_compat": _ELBOW_SHIM_NOTE,
        },
    }
    return cluster_labels, metrics


def order_clusters_by_score(clusters: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按簇内平均 score（论文用 eccentricity）升序重排簇编号，返回 1-based 编号与映射表。

    SPOT 的 `reorder_and_relabel_clusters(clusters, score, order="ascending")` 给的是 0-based 排行号与
    `{旧标签: 新标签}` dict；论文口径是 1..K，所以这里统一 +1。

    Returns
    -------
    ordered : (N,) int array
        1-based 新簇标签（1 = 平均 score 最小）。
    mapping : (K, 2) int array
        映射表，每行 `(旧标签, 新标签)`，按旧标签升序。用二维数组而不是 dict 是为了支持任意整数标签
        （k-means 给的是 0..K-1，但外部传入的标签不保证连续）。
    """
    clusters = np.asarray(clusters)
    score = np.asarray(score, dtype=np.float64)
    if clusters.ndim != 1 or score.ndim != 1:
        _fail(f"簇标签与 score 都必须是一维，实际 {clusters.ndim} 维与 {score.ndim} 维")
    if clusters.shape != score.shape:
        _fail(f"簇标签 {clusters.size} 个、score {score.size} 个，长度不一致：无法按簇取平均")
    n_clusters = int(np.unique(clusters).size)
    if n_clusters < 2:
        _fail(f"只有 {n_clusters} 个簇，没有可排序的对象（k-means 至少应给出 2 个簇）")
    if not np.all(np.isfinite(score)):
        n_bad = int((~np.isfinite(score)).sum())
        _fail(f"簇排序 score 里有 {n_bad} 个非有限值：算不出簇内平均 score，无法给簇编号")

    relabeled, cluster_mapping_dict = sam.reorder_and_relabel_clusters(
        clusters.astype(np.int64), score, order="ascending"
    )
    relabeled = np.asarray(relabeled, dtype=np.int64)
    # SPOT 的排序号从 0 开始；若上游改成 1-based，我们的 +1 会静默错一格，所以先断言再偏移
    if set(np.unique(relabeled).tolist()) != set(range(n_clusters)):
        _fail(
            f"SPOT 的 reorder_and_relabel_clusters 返回了非 0..{n_clusters - 1} 的编号"
            f"（实际 {sorted(set(np.unique(relabeled).tolist()))}）：上游的编号口径变了，请检查实现"
        )
    ordered = relabeled + 1
    pairs = sorted((int(old), int(new) + 1) for old, new in cluster_mapping_dict.items())
    if len(pairs) != n_clusters:
        _fail(f"SPOT 的簇映射表有 {len(pairs)} 项，而实际有 {n_clusters} 个簇：映射表与簇数对不上，无法落盘")
    try:
        mapping = np.asarray(pairs, dtype=np.int64).reshape(n_clusters, 2)
    except ValueError as exc:
        _fail(
            f"SPOT 的簇映射表组装成 ({n_clusters}, 2) 矩阵失败（{len(pairs)} 项、{type(exc).__name__}: {exc}）："
            "上游返回的映射表口径变了，请检查实现"
        )
    return ordered, mapping


def density(embedding: np.ndarray, all_conditions=None) -> tuple[list[np.ndarray], np.ndarray]:
    """局部点密度（景观热图 F3 / 轨迹计算共用）：包一层 `compute_heatmap_density_image_Embedding`。

    SPOT 那个函数对每个条件都无条件建 figure 并 `plt.show()`、且从不关闭；在 Agg 后端下 `show()` 只是
    一条 "FigureCanvasAgg is non-interactive" 警告，figure 却会一直攒在进程里（长跑会吃内存）。所以这里
    - 静音那条预期的警告；
    - 无论成功与否都把本次新建的 figure 关掉。

    Returns
    -------
    densities : list[np.ndarray]
        长度 = 条件数；第 i 项是该条件下**每个点处**的局部密度（长度 = 该条件的实例数）。
    select : (n_conditions, N) bool array
        SPOT 给出的"哪些行属于该条件"的掩码。
    """
    emb = _validate_embedding(embedding)
    conditions = None if all_conditions is None else np.asarray(all_conditions)
    if conditions is not None and conditions.shape != (emb.shape[0],):
        _fail(f"条件数组长度 {conditions.shape} 与实例数 {emb.shape[0]} 不一致：密度按条件分组，长度必须一一对应")
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive", category=UserWarning)
            densities, select = sam.compute_heatmap_density_image_Embedding(emb, all_conditions=conditions)
    finally:
        for num in set(plt.get_fignums()) - before:
            plt.close(num)
    return [np.asarray(d, dtype=np.float64).ravel() for d in densities], np.asarray(select, dtype=bool)


def _read_preprocessed() -> tuple[np.ndarray, list[str]]:
    """读 Stage 3 的 `X.npy` + `keep_features.json`，并核对列数一致。"""
    prep_root = Path(cfg.OUTPUT_ROOT) / _PREPROCESSED_SUBDIR
    x_path = prep_root / _X_FILE
    if not x_path.exists():
        _fail(
            f"找不到 Stage3 的预处理矩阵 {x_path}：Stage5 只读预处理产物、不重算特征，"
            "请先跑 Stage3（biopheno.preprocess.run_stage3）"
        )
    try:
        X = np.load(x_path)
    except (OSError, ValueError) as exc:
        _fail(f"Stage3 的预处理矩阵 {x_path} 读不出来（{type(exc).__name__}: {exc}）：请重跑 Stage3")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        _fail(f"Stage3 的预处理矩阵 {x_path} 不是二维（形状 {X.shape}）：请重跑 Stage3")

    keep_path = prep_root / _KEEP_FILE
    if not keep_path.exists():
        _fail(f"找不到 Stage3 的特征名清单 {keep_path}：列的语义无法确定，请先跑 Stage3")
    try:
        recorded = json.loads(keep_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"Stage3 的特征名清单 {keep_path} 不是合法 JSON（{exc}）：请重跑 Stage3")
    if not isinstance(recorded, list):
        _fail(f"Stage3 的特征名清单 {keep_path} 不是一个名字列表（实际是 {type(recorded).__name__}）：请重跑 Stage3")
    names = [str(n) for n in recorded]
    if len(names) != X.shape[1]:
        _fail(
            f"Stage3 的特征名清单 {keep_path}（{len(names)} 个）与预处理矩阵列数 {X.shape[1]} 不一致："
            "列的语义无法确定，请重跑 Stage3"
        )
    return X, names


def _read_stage2_table() -> pd.DataFrame:
    """读 Stage 2 的 `all_features.csv`：**行身份**（instance_uid 索引）与原始解释特征的来源。"""
    feat_root = Path(cfg.OUTPUT_ROOT) / _FEATURES_SUBDIR
    csv_path = feat_root / _ALL_FEATURES_FILE
    if not csv_path.exists():
        _fail(
            f"找不到 Stage2 的特征表 {csv_path}：Stage5 用它确定 X.npy 的行身份（instance_uid）与每实例解释特征，"
            "请先跑 Stage2（biopheno.features.run_stage2）"
        )
    try:
        table = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        _fail(f"Stage2 的特征表 {csv_path} 读不出来（空文件、被截断或不是 UTF-8 文本：{type(exc).__name__}: {exc}）")
    if len(table) == 0 or table.shape[1] == 0:
        _fail(f"Stage2 的特征表 {csv_path} 是空的（{len(table)} 行 × {table.shape[1]} 列）：请重跑 Stage2")
    if "instance_uid" not in table.columns:
        _fail(
            f"Stage2 的特征表 {csv_path} 没有 instance_uid 列（现有列如 {list(table.columns[:5])}）："
            "行身份无法确定，请重跑 Stage2"
        )
    uids = [str(u) for u in table["instance_uid"].tolist()]
    if len(set(uids)) != len(uids):
        duplicated = sorted({u for u in uids if uids.count(u) > 1})
        _fail(
            f"Stage2 的特征表 {csv_path} 里有重复的 instance_uid（如 {duplicated[:5]}）："
            "同一 uid 对应多行，无法确定哪一行是实例的特征，请重跑 Stage2"
        )
    return table


def _read_instance_meta() -> pd.DataFrame:
    """读 Stage 1 的 `all_instances.csv`（只强制要求 day/fov/时间；bbox 列按需检查）。

    bbox 只在"原生 `moment_eccentricity` 不可用、要用 bbox 兜底"时才需要，所以这里**不**硬要求它：
    原生列在手时缺 bbox 列也照样能跑（真正的需求由 `_has_bbox_columns` / `_bbox_eccentricity` 把）。
    """
    inst_root = Path(cfg.OUTPUT_ROOT) / _INSTANCES_SUBDIR
    all_path = inst_root / _ALL_INSTANCES_FILE
    if not all_path.exists():
        _fail(
            f"找不到 Stage1 的实例汇总表 {all_path}：Stage5 要从它取每个实例的 day/fov、frame_idx 与 time_hours，"
            "请先跑 Stage1（biopheno.instances.run_stage1）"
        )
    try:
        meta = pd.read_csv(all_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        _fail(f"Stage1 的实例汇总表 {all_path} 读不出来（{type(exc).__name__}: {exc}）：请重跑 Stage1")
    needed = ("instance_uid", "day", "fov", "frame_idx", "time_hours")
    missing = [c for c in needed if c not in meta.columns]
    if missing:
        _fail(f"Stage1 的实例汇总表 {all_path} 缺少列 {missing}：取不到 day/fov/时间，请重跑 Stage1")
    if len(meta) == 0:
        _fail(f"Stage1 的实例汇总表 {all_path} 是空的：请重跑 Stage1")
    return meta.reset_index(drop=True)


def _has_bbox_columns(meta_rows: pd.DataFrame) -> bool:
    """`all_instances.csv` 是否带齐了推导 bbox 偏心率所需的列。"""
    return all(column in meta_rows.columns for column in _BBOX_COLUMNS)


def _bbox_eccentricity(meta_rows: pd.DataFrame) -> np.ndarray:
    """用 bbox 的宽/高推导偏心率：`sqrt(1 - (短边/长边)^2)`（正方形 = 0，越扁越接近 1）。

    bbox 边长按像素计数（含两端）：`w = c1 - c0 + 1`、`h = r1 - r0 + 1`，与 Stage 1 的 `area_px` 口径一致。
    缺列时给带上下文的中文错误（而不是让 pandas 的 KeyError 冒上去）。
    """
    missing = [column for column in _BBOX_COLUMNS if column not in meta_rows.columns]
    if missing:
        _fail(
            f"all_instances.csv 缺少 bbox 列 {missing}：原生 {_ECC_FEATURE} 也不可用时无法用它推导偏心率，请重跑 Stage1"
        )
    width = (meta_rows["c1"] - meta_rows["c0"] + 1).to_numpy(dtype=np.float64)
    height = (meta_rows["r1"] - meta_rows["r0"] + 1).to_numpy(dtype=np.float64)
    long_side = np.maximum(width, height)
    short_side = np.minimum(width, height)
    if np.any(long_side <= 0):
        n_bad = int((long_side <= 0).sum())
        _fail(f"{n_bad} 个实例的 bbox 长边为 0（表被截断或列写错）：推导不出偏心率，请重跑 Stage1")
    return np.sqrt(1.0 - (short_side / long_side) ** 2)


def _raw_column(table: pd.DataFrame, name: str) -> tuple[np.ndarray, bool]:
    """按名取 Stage 2 的原始特征列；不存在时返回全 NaN 并标记为缺失（缺失清单会落进 QC）。"""
    if name in table.columns:
        return table[name].to_numpy(dtype=np.float64), True
    return np.full(len(table), np.nan, dtype=np.float64), False


def _eccentricity_score(
    X: np.ndarray, names: list[str], table: pd.DataFrame, bbox_ecc: np.ndarray | None
) -> tuple[np.ndarray, dict]:
    """给出簇排序 score（平均 eccentricity，原生量纲优先）及其口径说明。

    选择顺序（控制者裁定）：Stage 2 的**原生** `moment_eccentricity` → bbox 推导值 → Stage 3 预处理列。
    为什么不优先用 UMAP 所在的那份预处理列：Yeo-Johnson + 标准化是逐行单调变换，但各簇分布宽度不同时
    **簇均值**的先后会翻转，而论文按原生量纲的平均偏心率给簇编号，所有图与报告也都是原生单位。
    走了哪条路写进 `info["source"]`（`moment_eccentricity` / `bbox` / `preprocessed`），绝不静默换口径。
    """
    if _ECC_FEATURE in table.columns:
        return np.asarray(table[_ECC_FEATURE], dtype=np.float64), {
            "source": _NATIVE_SOURCE,
            "feature": _ECC_FEATURE,
            "units": "原生量纲（0 = 类圆，越接近 1 越扁长）",
            "description": (f"Stage2 的 all_features.csv 里 {_ECC_FEATURE} 原始列：与论文编号口径一致的原生单位"),
            "fallback_used": False,
        }
    if bbox_ecc is not None:
        return np.asarray(bbox_ecc, dtype=np.float64), {
            "source": _BBOX_SOURCE,
            "feature": None,
            "units": "原生量纲（bbox 的 sqrt(1 - (短边/长边)^2)，0 = 正方形）",
            "description": (
                f"{_ECC_FEATURE} 不在 Stage2 的特征表里，改用 all_instances.csv 的 bbox："
                "sqrt(1 - (短边/长边)^2)，短边 = min(c1-c0+1, r1-r0+1)"
            ),
            "fallback_used": True,
        }
    if _ECC_FEATURE in names:
        column = names.index(_ECC_FEATURE)
        return np.asarray(X[:, column], dtype=np.float64), {
            "source": _PREPROCESSED_SOURCE,
            "feature": _ECC_FEATURE,
            "units": "Stage3 预处理后的无量纲取值（最后手段：原生列与 bbox 都不可用）",
            "description": (
                f"原生 {_ECC_FEATURE} 与 bbox 都不可用，只能用 preprocessed/X.npy 的第 {column} 列"
                "（Stage3 全局 power transform + 标准化之后）；各簇分布宽度不同时簇均值次序可能与原生量纲不同"
            ),
            "fallback_used": True,
        }
    _fail(
        f"拿不到任何 eccentricity：Stage2 的 all_features.csv 里没有 {_ECC_FEATURE} 列，"
        f"all_instances.csv 里没有 bbox 列 {list(_BBOX_COLUMNS)}，Stage3 保留的特征里也没有 {_ECC_FEATURE}。"
        "簇编号需要按平均偏心率排序，请重跑 Stage1/Stage2/Stage3"
    )


def _per_instance_eccentricity(table: pd.DataFrame, bbox_ecc: np.ndarray | None) -> tuple[np.ndarray, bool, str | None]:
    """每实例解释特征里的 eccentricity：原生列优先、缺列时 bbox。

    这是一个**自有决策**，与"簇排序 score 走了哪条路"无关（review finding：曾经用 score 的
    `fallback_used` 推导这一列，结果 Stage3 过滤掉该特征时明明有原生列、JSON 却说 bbox）。
    返回 `(取值, 原生列是否存在, 路径标签)`；两者都不可用时给 NaN + 无路径（QC 的 `missing` 会记名）。
    """
    raw_ecc, has_raw_ecc = _raw_column(table, _ECC_FEATURE)
    if has_raw_ecc:
        return raw_ecc, True, _NATIVE_SOURCE
    if bbox_ecc is not None:
        return np.asarray(bbox_ecc, dtype=np.float64), False, _BBOX_SOURCE
    return np.full(len(table), np.nan, dtype=np.float64), False, None


def _cluster_means(labels: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """每个 1..K 簇的均值（键是簇号字符串，插入顺序 = 簇号数值升序，方便直接看单调性）。"""
    return {str(int(k)): float(np.mean(values[labels == k])) for k in sorted(np.unique(labels).tolist())}


def _is_non_decreasing(means: dict[str, float]) -> bool:
    """簇号升序时均值是否非递减（浮点误差按 1e-12 的相对容差吸收）。"""
    values = np.asarray(list(means.values()), dtype=np.float64)
    if values.size < 2:
        return True
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(values))))
    return bool(np.all(np.diff(values) >= -tolerance))


def _has_ties(means: dict[str, float]) -> bool:
    """簇均值里是否有并列（并列时编号在并列簇之间是任意的，QC 要标出来）。"""
    values = np.asarray(list(means.values()), dtype=np.float64)
    return bool(np.unique(np.round(values, 12)).size < values.size)


def _numbering_qc(labels: np.ndarray, *, score: np.ndarray, native, preprocessed, source: str) -> dict:
    """核对 1-based 簇编号的单调性，并把"原生 / score / 预处理"三套簇均值一起落盘。

    编号必须按**实际使用的 score** 升序，否则说明 `reorder_and_relabel_clusters` 的升序语义被破坏
    （那是实现 bug，直接报中文错误）。原生量纲与预处理量纲的次序**可能不同**（各簇分布宽度不同时，
    逐行单调的 Yeo-Johnson + 标准化会翻转簇均值次序）：论文口径以原生量纲为准，所以这种分歧只 warn + flag。
    """
    score_means = _cluster_means(labels, np.asarray(score, dtype=np.float64))
    if not _is_non_decreasing(score_means):
        _fail(
            f"簇编号不是按排序 score（来源 {source}）升序：每个 1..K 簇的均值是 {score_means}。"
            "这是重排实现的问题（reorder_and_relabel_clusters 应按簇均值升序给号），"
            "请检查 landscape.order_clusters_by_score"
        )
    qc = {
        "mean_per_cluster_score": score_means,
        "monotone_score": True,
        "mean_per_cluster_native": None,
        "monotone_native": None,
        "monotone_preprocessed": None,
        "tied_score_means": _has_ties(score_means),
    }
    if native is not None:
        native_means = _cluster_means(labels, np.asarray(native, dtype=np.float64))
        qc["mean_per_cluster_native"] = native_means
        qc["monotone_native"] = _is_non_decreasing(native_means)
    if preprocessed is not None:
        preprocessed_means = _cluster_means(labels, np.asarray(preprocessed, dtype=np.float64))
        qc["monotone_preprocessed"] = _is_non_decreasing(preprocessed_means)
    return qc


def run_stage5() -> None:
    """跑完 Stage 5 并落盘（Spec §6-Stage5）：读 `output/preprocessed/` + `output/features/` +
    `output/instances/`，写 `output/analysis/`。

    流程：`embed`（UMAP，参数全来自 `cfg.UMAP_PARAMS`）→ `cluster`（SPOT 的 elbow 扫描，ploton=False）
    → `order_clusters_by_score`（按簇内平均 eccentricity 升序编号 1..K，score 用**原生量纲**：
    `all_features.csv` 的 `moment_eccentricity` → bbox 推导值 → Stage3 预处理列）。

    产物（全部带 `instance_uid`，行序 = Stage 2 的实例序，下游不用再猜行序）：
    - `umap_2d.csv`：`instance_uid` / `umap_1` / `umap_2`；
    - `cluster_labels.csv`：`instance_uid` / `day` / `fov` / `frame_idx` / `time_hours` / `cluster`（1-based）/
      `cluster_raw`（k-means 的 0-based 标签）；
    - `per_instance_features.csv`：`instance_uid` / `area` / `eccentricity` / `intensity` / `speed`
      —— area/eccentricity/intensity 取 Stage 2 的**原始量纲**（柱状图要用真实单位）；不可用的列写 NaN 并
      在 QC 里记名；`speed` 恒为 NaN（motion 被有意跳过，Spec §6-Stage2）；
    - `kmeans_metrics.json`：K 扫描的 inertia / silhouette / adjusted inertia、选中的 K、簇映射、
      排序口径与单调性 QC（`eccentricity_score`：`source` / 原生与 score 单位的每簇均值 /
      `monotone_native` / `monotone_score` / `monotone_preprocessed`）、行身份来源、每实例解释特征的取列来源、耗时。

    失败时的磁盘状态与 Stage 1–4 一致：落盘前失败的路径不动上一轮产物；确定要写时才先删旧的四份。
    """
    t_start = time.perf_counter()
    X, names = _read_preprocessed()
    table = _read_stage2_table()
    if len(table) != X.shape[0]:
        _fail(
            f"行数对不上：Stage3 的预处理矩阵 X.npy 有 {X.shape[0]} 行，而 Stage2 的特征表 "
            f"all_features.csv 有 {len(table)} 行；两者必须一一对应，否则行身份与解释特征都会错位。"
            "这通常意味着 X.npy 与 all_features.csv 不是同一次 Stage2/3 的产物，请按顺序重跑 Stage2 → Stage3"
        )
    missing_features = [name for name in names if name not in table.columns]
    if missing_features:
        _fail(
            f"Stage3 保留的特征名有 {len(missing_features)} 个不在 Stage2 的特征表里（如 {missing_features[:5]}）："
            "X.npy 与 all_features.csv 不是同一次 Stage2/3 的产物，请重跑 Stage2/3"
        )
    uids = [str(u) for u in table["instance_uid"].tolist()]

    meta = _read_instance_meta()
    if not meta["instance_uid"].is_unique:
        _fail("Stage1 的实例汇总表里 instance_uid 有重复：按 uid 连接 day/fov 会一对多，请重跑 Stage1")
    meta_by_uid = meta.set_index(meta["instance_uid"].astype(str))
    missing_uids = [uid for uid in uids if uid not in meta_by_uid.index]
    if missing_uids:
        head = "、".join(missing_uids[:5])
        _fail(
            f"{len(missing_uids)} 个 Stage2 实例在 Stage1 的 all_instances.csv 里找不到（如 {head}"
            f"{' …' if len(missing_uids) > 5 else ''}）：取不到 day/fov 与 bbox，请重跑 Stage1/Stage2"
        )
    # 按 X 的行序取 meta（Stage 2 只给有可用轮廓的实例算特征，行序与 all_instances.csv 天然不同）
    meta_rows = meta_by_uid.loc[uids]

    # 原生量纲的偏心率只算一次：簇排序 score 与 per_instance_features 都从这里取（以前两处各算一遍）。
    # bbox 只在原生列不可用时才算，所以缺 bbox 列不会拖累"原生列在手"的正常路径。
    raw_ecc, has_raw_ecc = _raw_column(table, _ECC_FEATURE)
    bbox_ecc = _bbox_eccentricity(meta_rows) if (not has_raw_ecc and _has_bbox_columns(meta_rows)) else None
    score, ecc_info = _eccentricity_score(X, names, table, bbox_ecc)
    # 预处理列不参与编号，只在 QC 里用来对比"原生 vs 预处理"的簇均值次序
    preprocessed_ecc = np.asarray(X[:, names.index(_ECC_FEATURE)], dtype=np.float64) if _ECC_FEATURE in names else None

    t_embed = time.perf_counter()
    embedding = embed(X)
    t_cluster = time.perf_counter()
    raw_labels, metrics = cluster(embedding)
    t_order = time.perf_counter()
    ordered, mapping = order_clusters_by_score(raw_labels, score)
    numbering = _numbering_qc(
        ordered,
        score=score,
        native=raw_ecc if has_raw_ecc else bbox_ecc,
        preprocessed=preprocessed_ecc,
        source=ecc_info["source"],
    )
    t_accounting = time.perf_counter()

    # 每实例解释特征取 Stage 2 的**原始量纲**（柱状图要用真实单位）。这里的 eccentricity 列与
    # "排序 score 走了哪条路"是两件不同的事：只要原生量纲，所以取原生列、缺列时才用 bbox 推导值。
    area, has_area = _raw_column(table, _AREA_FEATURE)
    intensity, has_intensity = _raw_column(table, _INTENSITY_FEATURE)
    per_instance_ecc, _has_per_instance_ecc, per_instance_ecc_path = _per_instance_eccentricity(table, bbox_ecc)
    per_instance = pd.DataFrame(
        {
            "instance_uid": uids,
            _AREA_FEATURE: area,
            "eccentricity": per_instance_ecc,
            "intensity": intensity,
            _SPEED_FEATURE: np.full(len(uids), np.nan),
        }
    )[list(_PER_INSTANCE_COLUMNS)]

    labels_df = pd.DataFrame(
        {
            "instance_uid": uids,
            "day": meta_rows["day"].astype(str).to_numpy(),
            "fov": meta_rows["fov"].astype(str).to_numpy(),
            "frame_idx": meta_rows["frame_idx"].to_numpy(dtype=np.int64),
            "time_hours": meta_rows["time_hours"].to_numpy(dtype=np.float64),
            "cluster": ordered.astype(np.int64),
            "cluster_raw": raw_labels.astype(np.int64),
        }
    )
    umap_df = pd.DataFrame({"instance_uid": uids, "umap_1": embedding[:, 0], "umap_2": embedding[:, 1]})

    out_root = Path(cfg.OUTPUT_ROOT) / _ANALYSIS_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)
    for stale in (_UMAP_FILE, _LABELS_FILE, _PER_INSTANCE_FILE, _METRICS_FILE):
        (out_root / stale).unlink(missing_ok=True)

    umap_df.to_csv(out_root / _UMAP_FILE, index=False, float_format="%.10g")
    labels_df.to_csv(out_root / _LABELS_FILE, index=False)
    per_instance.to_csv(out_root / _PER_INSTANCE_FILE, index=False, float_format="%.10g")
    t_written = time.perf_counter()

    ecc_info = {**ecc_info, "order": "ascending", **numbering}
    metrics = {
        **metrics,
        "umap_params": dict(cfg.UMAP_PARAMS),
        "random_state": cfg.RANDOM_STATE,
        "n_objects": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_clusters": int(np.unique(ordered).size),
        "row_identity": {
            "source": _ROW_SOURCE,
            "n_instances": len(uids),
            "uid_column": "instance_uid（umap_2d.csv / cluster_labels.csv / per_instance_features.csv 都有）",
        },
        "eccentricity_score": ecc_info,
        "cluster_mapping": {str(old): int(new) for old, new in mapping},
        "cluster_sizes": {str(int(k)): int(v) for k, v in zip(*np.unique(ordered, return_counts=True))},
        "per_instance_features": {
            "columns": list(_PER_INSTANCE_COLUMNS),
            "source_columns": {_AREA_FEATURE: has_area, _ECC_FEATURE: has_raw_ecc, _INTENSITY_FEATURE: has_intensity},
            "missing": [
                name
                for name, present in (
                    (_AREA_FEATURE, has_area),
                    (_ECC_FEATURE, has_raw_ecc),
                    (_INTENSITY_FEATURE, has_intensity),
                )
                if not present
            ],
            "eccentricity_path": per_instance_ecc_path,
            "eccentricity_path_note": (
                "这一列只要能用的原生量纲；与排序 score 的选择（eccentricity_score.source）相互独立"
            ),
            "speed": "NaN 占位（motion 特征被有意跳过，Spec §6-Stage2）",
        },
        "density": {
            "persisted": False,
            "reason": "密度图（Fig.3 F3）由绘图阶段按簇调用 biopheno.landscape.density；Stage5 只负责坐标与簇",
        },
        "timing_seconds": {
            "umap": round(t_cluster - t_embed, 3),
            "kmeans_scan_and_metrics": round(t_order - t_cluster, 3),
            "cluster_ordering": round(t_accounting - t_order, 3),
            "artifacts": round(t_written - t_accounting, 3),
            "total": round(t_written - t_start, 3),
        },
    }
    (out_root / _METRICS_FILE).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Stage5 完成：{X.shape[0]} 个实例 × {X.shape[1]} 维 → UMAP({cfg.UMAP_PARAMS}) → "
        f"{metrics['n_clusters']} 个表型簇（K={metrics['k']}，扫描 {metrics['k_scanned']}），"
        f"按平均 eccentricity（{ecc_info['source']}）升序编号，耗时 {metrics['timing_seconds']['total']}s → {out_root}"
    )
    if ecc_info["fallback_used"]:
        print(
            f"警告：簇排序 score 退到 {ecc_info['source']}（{ecc_info['description']}），"
            f"详见 {out_root / _METRICS_FILE}"
        )
    if numbering["monotone_preprocessed"] is False and ecc_info["source"] != _PREPROCESSED_SOURCE:
        print(
            f"警告：簇编号在原生量纲（{ecc_info['source']}）下单调，但在 Stage3 预处理列下不单调"
            f"（各簇分布宽度不同）；按论文口径以原生量纲为准，{out_root / _METRICS_FILE} 里 monotone_preprocessed=false"
        )
    if numbering["tied_score_means"]:
        print(
            f"警告：排序 score（{ecc_info['source']}）的簇均值有并列：{numbering['mean_per_cluster_score']}，"
            "并列簇之间的编号是任意的"
        )
    missing_per_instance = metrics["per_instance_features"]["missing"]
    if missing_per_instance:
        print(f"警告：Stage2 特征表里缺 {missing_per_instance}，per_instance_features.csv 对应列写成 NaN")
