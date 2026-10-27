"""Stage 4：SAM 模块发现（legacy 选择器，Spec §6-Stage4）。

本阶段把 Stage 3 预处理后的特征矩阵按"共变"层次聚类成 SAM 模块，并给出每个模块的代表实例图。
"为什么要用 legacy 选择器"写在 `discover_modules` 的 docstring 里（v2.0.1 的默认换成了另一套方法）。

上游（`SPOT/Analysis/sam_analysis.py`）的真实返回契约，逐条对照源码记下：

- `compute_SAM_features_and_scope_contribution(all_features, feature_names, random_state=0)` →
  `((feature_type, feature_scope), (SAM_expr, Scope_expr), (SAM_contribution, scope_contribution))`。
  `feature_type` / `feature_scope` 是长度 D 的整数数组（0/1/2，见 `get_SAM_feature_type` 的名字规则），
  两个贡献是 3 维向量（对三个子集各取 PC1 再对 3 列做 PCA 的第一主成分绝对值）。特征名不含
  `intensity`/`appear`/`flow`/`speed`/`motion`/`sift_*` 等关键词时全部落成 type=0、scope=0。
- `hierarchical_cluster_features_into_SAM_modules(all_feats, feature_names, feature_type=None,
  feature_scope=None, ..., module_selection='multiscale', random_state=0)` →
  `(g, final_feature_clusters, features_clusters_colorbar, final_feature_names_clusters,
  final_feature_names_ids)`。`feature_type`/`feature_scope` 为 None 时它会自己按名字猜；我们显式传它
  返回过的那一份，保证配色与 Stage 3 的名字完全同源。`module_selection='legacy'` 路径里真正被用到的
  只有 `max_clust_len`（None = 按 dendrogram 里"单例区间"的长度自动推，推不出时退回 25）、`buffer`
  （扫描切割数上限加这个缓冲）、`lam_smooth`（Asymmetric Least Squares 平滑的 λ）以及 `p_smooth` /
  `niter_smooth` / `min_peak_distance`（这三个用默认值）；`scale`、`k_max`、`n_scales`、`n_null`、
  `n_boot`、`min_stability`、`parsimony`、`min_modules`、`random_state` 全部只在 multiscale 路径
  生效（legacy 路径没有任何随机性）。
- `compute_individual_feature_contributions_in_SAM_modules(all_feats, SAM_module_indexes, random_state=0)`
  → `(feature_module_expr, feature_module_expr_contrib)`：前者 (n_objects, n_modules) 的每模块 PC1，
  后者是 list，逐个模块给出特征载荷（长度 = 该模块特征数）。
- `compute_most_representative_image_patches(object_feature_module_expr, all_patches, n_rows=3, n_cols=3,
  rescale_intensity=True)` → `(object_SAM_module_purity_scores, sample_images_modules,
  sample_object_index_modules)`。`all_patches` 只要能被整数下标取用即可（list 或数组都行），每项形状
  可以是 (h, w) 或 (h, w, ch) 且**允许各不相同**（它按最大值做格子、居中贴图）；返回的
  `sample_images_modules` 是 dtype=object 的数组（每模块一张 n_rows×n_cols 马赛克，uint8），
  `sample_object_index_modules` 是 list of `argsort(purity)[::-1][:n_rows*n_cols]`，即**对 all_patches
  的下标**（不是实例 uid）。

产物（`output/modules/`）：`modules_labels.csv`（特征 → 模块）、`module_expr.npy`（实例 × 模块）、
`cluster_map.npz`（重画 clustermap 所需的相关系数矩阵 / 树 / 重排索引 / 颜色条）、
`module_patches.npz`（每模块的代表图马赛克 + 被选中的实例）、`stage4_qc.json`（模块数、各模块特征数、
选择方式与参数、被排除的实例、耗时）。
"""

import json
import time
from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd

from biopheno import config as cfg
from SPOT.Analysis import sam_analysis as sam

# —— 输入产物（Stage 3 / Stage 2 / Stage 1 的落盘位置，与 preprocess.py / features.py / instances.py 一致）——
# 行身份（实例顺序）来自 Stage 2 自己的产物，不是 Stage 1 的 all_instances.csv：
# Stage 1 给每个标签都留 meta 行（含轮廓全被丢弃的），Stage 2 只给有可用轮廓的实例算特征，
# 所以 X 的行集合是 all_instances.csv 的子集，按后者装配行序会整体错位。
_PREPROCESSED_SUBDIR: str = "preprocessed"
_FEATURES_SUBDIR: str = "features"
_ALL_FEATURES_FILE: str = "all_features.csv"
_SHAPE_FILE: str = "sam_shape.npz"
_X_FILE: str = "X.npy"
_KEEP_FILE: str = "keep_features.json"
_INSTANCES_SUBDIR: str = "instances"
_ALL_INSTANCES_FILE: str = "all_instances.csv"
_PATCH_FILE: str = "patches.npz"

# —— 本阶段产物（Spec §6-Stage4）——
_MODULES_SUBDIR: str = "modules"
_LABELS_FILE: str = "modules_labels.csv"
_EXPR_FILE: str = "module_expr.npy"
_CLUSTER_MAP_FILE: str = "cluster_map.npz"
_PATCHES_FILE: str = "module_patches.npz"
_QC_FILE: str = "stage4_qc.json"

# 每个模块代表图马赛克的边长（论文 Fig. 3e 用 3×3）：取该模块"纯度"最高的 n×n 个实例。
_PANEL_ROWS: int = 3

# legacy 选择器至少要能扫到一个切割：SPOT 的扫描是 `arange(2, len(linkage))`，
# 特征数 D 时 len(linkage)=D-1，故 D>=4 才有点可评。
_MIN_FEATURES: int = 4
# SAM 贡献那一步要 PCA(n_components=3)（shape/appearance/motion 三列），实例数少于 3 会直接抛 sklearn 错。
_MIN_OBJECTS: int = 3

# 这些 kwarg 由本模块显式传，若出现在 cfg.MODULE_PARAMS 里会造成 "got multiple values" 的 TypeError。
_RESERVED_PARAMS: frozenset[str] = frozenset({"feature_type", "feature_scope", "module_selection", "random_state"})


def _fail(message: str) -> NoReturn:
    """抛出一句带上下文的中文 ValueError（薄封装只为统一文案格式）。"""
    raise ValueError(message)


def _validate_matrix(X, names) -> tuple[np.ndarray, np.ndarray]:
    """核对特征矩阵与特征名：二维、个数一致、无重名、特征数与实例数够用。

    重名必须拒绝：SPOT 里 `pd.DataFrame(all_feats, columns=feature_names)` 与 `df_corr` 的 index
    都会把重名列折成一列（`feature_names[mask]` 也会静默变花式索引），模块标签随后整体错位且不报错。
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        _fail(f"特征矩阵必须是二维 (n_objects, n_feats)，实际 {X.ndim} 维、形状 {X.shape}")
    names_arr = np.asarray([str(n) for n in names], dtype=object)
    if names_arr.shape != (X.shape[1],):
        _fail(f"特征名个数 {names_arr.size} 与特征矩阵列数 {X.shape[1]} 不一致：列的语义无法确定，拒绝做模块发现")
    if len(set(names_arr.tolist())) != names_arr.size:
        duplicated = sorted({n for n in names_arr.tolist() if names_arr.tolist().count(n) > 1})
        _fail(
            f"特征名有重名（{duplicated[:5]}）：SPOT 用名字做 DataFrame 列与 df_corr 的 index，"
            "重名会静默折叠、按名取列全部错位，请先去重或改名"
        )
    if X.shape[1] < _MIN_FEATURES:
        _fail(
            f"模块发现至少需要 {_MIN_FEATURES} 个特征（legacy 选择器从 dendrogram 的第 2 个合并点开始扫描，"
            f"更少时没有任何可评估的切割），实际只有 {X.shape[1]} 个特征"
        )
    if X.shape[0] < _MIN_OBJECTS:
        _fail(f"模块发现至少需要 {_MIN_OBJECTS} 个实例（要估计 3 个 SAM 主成分），实际只有 {X.shape[0]} 个")
    return X, names_arr


def _module_params() -> dict:
    """把 cfg.MODULE_PARAMS 摊平成一串 kwarg，并挡住与显式 kwarg 撞名的情况。"""
    params = dict(cfg.MODULE_PARAMS)
    conflicts = _RESERVED_PARAMS & set(params)
    if conflicts:
        _fail(f"cfg.MODULE_PARAMS 里有本模块显式传的参数 {sorted(conflicts)}：会造成重复赋值，请在 config 里改名或删掉")
    return params


def _close_grid(grid) -> None:
    """关掉 SPOT 内部簇热图留下的 matplotlib figure（长跑进程/测试里不关会一直累积）。"""
    fig = getattr(grid, "fig", None)
    if fig is not None:
        import matplotlib.pyplot as plt

        plt.close(fig)


def discover_modules(X: np.ndarray, names: list[str]) -> dict:
    """在特征矩阵上跑 SAM 模块发现（**legacy** 选择器 + 论文参数）。

    为什么不用 v2.0.1 默认的 `module_selection='multiscale'`：v2.0.1 把默认改成了"多尺度选择"——
    对特征做 null 模型（每列独立置换）× bootstrap 重采样共识评分，再在稳定候选里按 parsimony 取最粗的
    一个；它是另一套方法，模块数与本项目要复现的论文（Fig. 3d）不同，而且明显更慢（null 树 + bootstrap
    都要重算 dendrogram）。论文脚本走的是 legacy 规则：在 dendrogram 的切割序列上算
    `davies_bouldin_score`，用 ALS 平滑后取**第一个峰**作为模块数（DB 越低越好，所以这个峰其实是
    "局部最差的划分"，保留它是为了复现早期结果）。因此这里显式传 `module_selection=cfg.MODULE_SELECTION`
    （= "legacy"）与 `**cfg.MODULE_PARAMS`（= 论文的 max_clust_len=None / lam_smooth=1e0 / buffer=30），
    绝不静默回落到 multiscale。

    输入必须是 Stage 3 **预处理后**的矩阵（模块靠共变结构，量纲必须可比）。返回的模块标签与
    `names`/`X` 的列序一一对应。

    Returns
    -------
    dict
        `module_labels`：长度 D 的整数数组（1..m）；
        `feature_indices`：list[array]，每个模块的特征**列下标**；
        `feature_names`：list[list[str]]，与 `feature_indices` 同序；
        `contribution`：list[array]，每个模块 PC1 的特征载荷；
        `type_contribution` / `scope_contribution`：长度 3，shape/appearance/motion 与
        global/regional/distribution 对变异的贡献；
        `cluster_map_data`：`corr`（SPOT 实际绘制的相关系数矩阵）、`linkage`（dendrogram 用的树）、
        `reordered_ind`（dendrogram 列序）、`colorbar`（模块颜色条，按 dendrogram 列序）、
        `module_labels`、`feature_type`、`feature_scope`、`feature_names`——重画 clustermap 不再重算。
    """
    X, names_arr = _validate_matrix(X, names)

    (feature_type, feature_scope), _sam_exprs, (type_contribution, scope_contribution) = (
        sam.compute_SAM_features_and_scope_contribution(X, names_arr, random_state=cfg.RANDOM_STATE)
    )
    # feature_type / feature_scope 显式传：SPOT 只在两者任一为 None 时才自己按名字猜，
    # 传进去保证聚类图的类型/scope 配色与我们这份名字完全同源。
    grid = None
    try:
        grid, module_labels, colorbar, names_clusters, indices_clusters = (
            sam.hierarchical_cluster_features_into_SAM_modules(
                X,
                names_arr,
                feature_type=feature_type,
                feature_scope=feature_scope,
                module_selection=cfg.MODULE_SELECTION,
                **_module_params(),
                random_state=cfg.RANDOM_STATE,
            )
        )
        module_labels = np.asarray(module_labels, dtype=int)
        indices_clusters = [np.asarray(ix, dtype=int) for ix in indices_clusters]

        # SPOT 会把模块按 dendrogram 列序重新编号（1..m），标签与索引必须自洽；不自洽说明上游改了重排逻辑，
        # 那种情况下 feature_indices 与 module_labels 会指向不同模块，下游模块表达会静默错位。
        if len(indices_clusters) != len(names_clusters) or set(np.unique(module_labels)) != set(
            range(1, len(indices_clusters) + 1)
        ):
            _fail(
                f"SPOT 返回的模块标签（{sorted(set(np.unique(module_labels).tolist()))}）与模块数 "
                f"{len(indices_clusters)} 不自洽：无法确定特征属于哪个模块，请检查上游重排逻辑"
            )
        for module_id, index in enumerate(indices_clusters, start=1):
            if not np.all(module_labels[index] == module_id):
                _fail(f"SPOT 返回的第 {module_id} 个模块的特征下标与 module_labels 不一致：模块定义已错位")

        cluster_map_data = {
            # 直接取 clustermap 实际绘制的矩阵（= SPOT 内部那个 1 - correlation 距离的 df_corr，已 fillna(0)）
            "corr": np.asarray(grid.data2d, dtype=np.float64),
            "linkage": np.asarray(grid.dendrogram_col.calculated_linkage, dtype=np.float64),
            "reordered_ind": np.asarray(grid.dendrogram_col.reordered_ind, dtype=int),
            "colorbar": np.asarray(colorbar, dtype=np.float64),
            "module_labels": module_labels,
            "feature_type": np.asarray(feature_type, dtype=int),
            "feature_scope": np.asarray(feature_scope, dtype=int),
            "feature_names": np.asarray(names_arr, dtype=str),
        }
    finally:
        # 任何出口（含上面那些 _fail 的异常路径）都要关掉 SPOT 建的 clustermap figure
        _close_grid(grid)

    expr_contrib = sam.compute_individual_feature_contributions_in_SAM_modules(
        X, indices_clusters, random_state=cfg.RANDOM_STATE
    )
    _module_expr, contribution = expr_contrib

    return {
        "module_labels": module_labels,
        "feature_indices": indices_clusters,
        "feature_names": [[str(n) for n in np.asarray(arr).tolist()] for arr in names_clusters],
        "contribution": [np.asarray(c, dtype=np.float64) for c in contribution],
        "type_contribution": np.asarray(type_contribution, dtype=np.float64),
        "scope_contribution": np.asarray(scope_contribution, dtype=np.float64),
        "cluster_map_data": cluster_map_data,
    }


def module_expression(X: np.ndarray, modules: dict) -> np.ndarray:
    """实例 × 模块的模块表达矩阵：每个模块独立做 PCA 后取 PC1（SPOT 的模块表达口径）。

    `modules["feature_indices"]` 是相对 `X` 列的下标，所以 `X` 必须与 `discover_modules` 用的是同一份
    矩阵、同一列序（Stage 3 的 `X.npy` + `keep_features.json` 同序）。
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        _fail(f"特征矩阵必须是二维 (n_objects, n_feats)，实际 {X.ndim} 维、形状 {X.shape}")
    indices = [np.asarray(ix, dtype=int) for ix in modules.get("feature_indices", [])]
    if not indices:
        _fail("modules 里没有特征索引（feature_indices 为空）：无法计算模块表达")
    max_index = max(int(ix.max()) for ix in indices if ix.size)
    if max_index >= X.shape[1]:
        _fail(
            f"模块引用的特征下标 {max_index} 超出特征矩阵列数 {X.shape[1]}："
            "传进来的 X 与发现模块时用的不是同一份（列序或列数变了）"
        )
    expr, _contrib = sam.compute_individual_feature_contributions_in_SAM_modules(
        X, indices, random_state=cfg.RANDOM_STATE
    )
    return np.asarray(expr, dtype=np.float64)


def _representative_patches(
    expr: np.ndarray, patches: list[np.ndarray], n_rows: int
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """`compute_most_representative_image_patches` 的薄封装，额外把选中实例的下标也返回。

    "纯度" = 该模块的模块表达减去其他模块表达的最大值（越大越代表这个模块而非别的模块），
    每个模块取纯度最高的 n_rows×n_rows 个实例拼成一张马赛克。
    """
    expr = np.asarray(expr, dtype=np.float64)
    if expr.ndim != 2:
        _fail(f"模块表达必须是二维 (n_objects, n_modules)，实际 {expr.ndim} 维、形状 {expr.shape}")
    patches = list(patches)
    if len(patches) != expr.shape[0]:
        _fail(
            f"代表图与模块表达的行数不一致：{len(patches)} 个 patch vs {expr.shape[0]} 个实例；"
            "patch 必须与模块表达的行一一对应，否则图上的人物与实例对不上"
        )
    if expr.shape[0] == 0:
        _fail("没有任何实例可用于挑选代表图（模块表达 0 行）")
    if int(n_rows) < 1:
        _fail(f"代表图每行至少 1 张，实际 {n_rows}")
    purity, panels, indices = sam.compute_most_representative_image_patches(
        expr, patches, n_rows=int(n_rows), n_cols=int(n_rows)
    )
    return (
        np.asarray(purity, dtype=np.float64),
        [np.asarray(p) for p in panels],
        [np.asarray(i, dtype=int) for i in indices],
    )


def representative_patches(
    expr: np.ndarray, patches: list[np.ndarray], n_rows: int = _PANEL_ROWS
) -> tuple[np.ndarray, list[np.ndarray]]:
    """每模块挑最"纯"的实例拼成 n_rows×n_rows 马赛克，返回 `(纯度矩阵, 每模块一张拼图)`。

    `patches` 与 `expr` 的行必须一一对应（同一实例），且各 patch 尺寸可以不同（Stage 1 的 native 裁剪
    图就是随 bbox 变的），上游按最大格子居中贴图。
    """
    purity, panels, _indices = _representative_patches(expr, patches, n_rows)
    return purity, panels


def _read_preprocessed() -> tuple[np.ndarray, list[str]]:
    """读 Stage 3 的 `X.npy` + `keep_features.json`，并核对列数一致。"""
    prep_root = Path(cfg.OUTPUT_ROOT) / _PREPROCESSED_SUBDIR
    x_path = prep_root / _X_FILE
    if not x_path.exists():
        _fail(
            f"找不到 Stage3 的预处理矩阵 {x_path}：Stage4 只读预处理产物、不重算特征，"
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


def _read_stage2_order(names: list[str]) -> tuple[list[str], str]:
    """读 Stage 2 落盘的行身份（实例顺序），返回 `(uids, 来源描述)`。

    行身份必须取 Stage 2 自己的产物：`features/all_features.csv` 的 `instance_uid` 索引（Stage 2 就是这么
    写的、Stage 3 也按它校验过），没有索引时退回 `features/sam_shape.npz` 的 `uid` 数组。
    **不能用 Stage 1 的 `all_instances.csv`**：Stage 1 给每个标签都留 meta 行（包括轮廓全被丢弃的），
    Stage 2 只给有可用轮廓的实例算特征，所以 `all_instances.csv` 的行序与 `X` 的行序天然不同。
    """
    feat_root = Path(cfg.OUTPUT_ROOT) / _FEATURES_SUBDIR
    csv_path = feat_root / _ALL_FEATURES_FILE
    if not csv_path.exists():
        _fail(
            f"找不到 Stage2 的特征表 {csv_path}：Stage4 用它的 instance_uid 索引确定特征矩阵的行身份，"
            "请先跑 Stage2（biopheno.features.run_stage2）"
        )
    try:
        # 不用 index_col：Stage2 写的是 uid 索引（读回来就是首列 instance_uid），
        # 而没有 uid 索引时 index_col=0 会把第一个**特征**列当成索引，列名核对会误判
        table = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        _fail(f"Stage2 的特征表 {csv_path} 读不出来（空文件、被截断或不是 UTF-8 文本：{type(exc).__name__}: {exc}）")
    if len(table) == 0 or table.shape[1] == 0:
        _fail(f"Stage2 的特征表 {csv_path} 是空的（{len(table)} 行 × {table.shape[1]} 列）：请重跑 Stage2")
    missing = [name for name in names if name not in table.columns]
    if missing:
        _fail(
            f"Stage3 保留的特征名有 {len(missing)} 个不在 Stage2 的特征表 {csv_path} 里（如 {missing[:5]}）："
            "X.npy 与这张表不是同一次 Stage2/3 的产物，请重跑 Stage2/3"
        )

    if "instance_uid" in table.columns:
        uids = [str(u) for u in table["instance_uid"].tolist()]
        source = "features/all_features.csv 的 instance_uid 索引"
    else:
        uids, source = _read_stage2_shape_uids(feat_root, csv_path)
    if len(set(uids)) != len(uids):
        duplicated = sorted({u for u in uids if uids.count(u) > 1})
        _fail(
            f"Stage2 的行身份（{source}）里有重复的 instance_uid（如 {duplicated[:5]}）："
            "同一 uid 对应多行、无法确定哪一行是实例的特征，请重跑 Stage2"
        )
    return uids, source


def _read_stage2_shape_uids(feat_root: Path, csv_path: Path) -> tuple[list[str], str]:
    """`all_features.csv` 没有 uid 索引时的退路：读 `sam_shape.npz` 的 `uid` 数组（Stage 2 的逐行 uid）。"""
    shape_path = feat_root / _SHAPE_FILE
    if not shape_path.exists():
        _fail(
            f"Stage2 的特征表 {csv_path} 没有 instance_uid 索引，而形状特征块 {shape_path} 也不存在："
            "行身份无法确定，请重跑 Stage2"
        )
    try:
        with np.load(shape_path, allow_pickle=False) as npz:
            files = sorted(npz.files)
            # 先把数据取出来再校验：`_fail` 也会抛 ValueError，不能让它被下面的 except 吞掉
            uids = [str(u) for u in npz["uid"].tolist()] if "uid" in npz.files else None
    except (OSError, ValueError) as exc:
        _fail(f"Stage2 的形状特征块 {shape_path} 读不出来（{type(exc).__name__}: {exc}）：请重跑 Stage2")
    if uids is None:
        _fail(f"Stage2 的形状特征块 {shape_path} 里没有 uid 数组（现有键 {files}）：行身份无法确定，请重跑 Stage2")
    if not uids:
        _fail(f"Stage2 的形状特征块 {shape_path} 里的 uid 数组是空的：请重跑 Stage2")
    return uids, "features/sam_shape.npz 的 uid 数组"


def _read_instance_order() -> tuple[pd.DataFrame, list[str]]:
    """读 Stage 1 的实例汇总表，返回 `(meta, uids)`。

    **这里的行序不是特征矩阵的行序**：Stage 1 给每个标签都留 meta 行，Stage 2 只算有轮廓的那些。
    meta 只用来重建 `instance_uid → patch` 的映射（patch 按 (day, fov) 分文件、以 label 为键）。
    """
    inst_root = Path(cfg.OUTPUT_ROOT) / _INSTANCES_SUBDIR
    all_path = inst_root / _ALL_INSTANCES_FILE
    if not all_path.exists():
        _fail(
            f"找不到 Stage1 的实例汇总表 {all_path}：Stage4 要按实例行序取代表图，"
            "请先跑 Stage1（biopheno.instances.run_stage1）"
        )
    try:
        meta = pd.read_csv(all_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        _fail(f"Stage1 的实例汇总表 {all_path} 读不出来（{type(exc).__name__}: {exc}）：请重跑 Stage1")
    needed = ("instance_uid", "day", "fov", "label")
    missing = [c for c in needed if c not in meta.columns]
    if missing:
        _fail(f"Stage1 的实例汇总表 {all_path} 缺少列 {missing}：取不到代表图，请重跑 Stage1")
    if len(meta) == 0:
        _fail(f"Stage1 的实例汇总表 {all_path} 是空的：请重跑 Stage1")
    # 丢掉 csv 里可能多出来的索引列带来的行标签：下面按 0..n-1 的位置号取 patch，
    # 行标签不是 RangeIndex 时（表被手工重写过）会静默错位或索引越界。
    meta = meta.reset_index(drop=True)
    return meta, [str(u) for u in meta["instance_uid"].tolist()]


def _load_patches_by_uid(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    """把 Stage 1 的 patch 装配成 `instance_uid → patch` 的映射（**按 uid 对齐，不依赖行序**）。

    patch 是按 (day, fov) 分文件、以 `label<label>_native` 为键存的，所以从 meta 的 (day, fov, label)
    重建 `instance_uid = f"{fov}_{label}_{frame_idx}"`（Spec D13 的格式，Stage 1 就是这么做 uid 的），
    而且要和 meta 自己那列 `instance_uid` 对上。这样即使 Stage 2 的行序与 Stage 1 完全不同，patch 也不会串。
    轮廓被丢弃的实例在 patch 文件里没有条目，它们本来就不在 Stage 2 的行里，不参与装配。
    """
    inst_root = Path(cfg.OUTPUT_ROOT) / _INSTANCES_SUBDIR
    has_frame = "frame_idx" in meta.columns
    patches: dict[str, np.ndarray] = {}
    seen: set[str] = set()
    for (day, fov), group in meta.groupby(["day", "fov"], sort=False):
        path = inst_root / str(day) / str(fov) / _PATCH_FILE
        if not path.exists():
            _fail(f"找不到 {day}/{fov} 的代表图 {path}：请先跑 Stage1（biopheno.instances.run_stage1）")
        if has_frame:
            frames = group["frame_idx"].to_numpy()
        elif str(day) in cfg.DAYS:
            frames = np.full(len(group), int(cfg.DAYS[str(day)]), dtype=np.int64)
        else:
            _fail(f"实例汇总表里有未知日期 {day!r}（cfg.DAYS 里没有）：重建不出 instance_uid，请重跑 Stage1")
        with np.load(path) as npz:
            for label, frame, recorded_uid in zip(group["label"].to_numpy(), frames, group["instance_uid"].to_numpy()):
                uid = f"{fov}_{int(label)}_{int(frame)}"
                if str(recorded_uid) != uid:
                    _fail(
                        f"实例汇总表里的 uid {str(recorded_uid)!r} 与按 (fov, label, frame_idx) 重建出的 "
                        f"{uid!r} 不一致：行身份与代表图的对齐不可信，请重跑 Stage1"
                    )
                if uid in seen:
                    _fail(f"重建出的 instance_uid {uid!r} 重复（同一 (fov, label, frame) 出现多次）：请重跑 Stage1")
                seen.add(uid)
                key = f"label{int(label)}_native"
                if key in npz.files:
                    patches[uid] = np.asarray(npz[key])
    if not patches:
        _fail(f"{len(meta)} 个实例全部没有代表图（instances/<day>/<fov>/patches.npz 里一条都没匹配上）：请重跑 Stage1")
    return patches


def _json_ready(values) -> list:
    """把 numpy 向量转成 JSON 能写的 float 列表（QC 里记贡献用）。"""
    return [float(v) for v in np.asarray(values, dtype=np.float64).ravel()]


def run_stage4() -> None:
    """跑完 Stage 4 并落盘（Spec §6-Stage4）：读 `output/preprocessed/` + `output/features/` + `output/instances/`，
    写 `output/modules/`。

    行身份全部按 `instance_uid` 对齐，不依赖任何表的行序：
    - `X.npy` 的行序 = Stage 2 的实例序（`features/all_features.csv` 的 `instance_uid` 索引，缺失则退回
      `features/sam_shape.npz` 的 `uid`），并断言 `X.shape[0] == len(Stage2 uid)`；
    - patch 按 meta 的 `(day, fov, label, frame_idx)` 重建 uid 后索引，所以不论 Stage 2 的行序与
      `all_instances.csv` 是否一致（一般不一致）都不会串行；
    - Stage 1 有 meta 行、但 Stage 2 没有特征行的实例（轮廓被丢弃）不是错误：它们不在 `module_expr` 里，
      排除数记进 QC（`n_meta_not_in_X`）。

    产物：
    - `modules_labels.csv`：每个特征一行（`feature` / `module` / `feature_type` / `feature_scope`），
      行序 = `keep_features.json` 的列序 = `X.npy` 的列序；
    - `module_expr.npy`：实例 × 模块的 PC1 表达，float64，行序 = **Stage 2 的实例序**（QC 里记了来源）；
    - `cluster_map.npz`：`corr` / `linkage` / `reordered_ind` / `colorbar` / `module_labels` /
      `feature_type` / `feature_scope` / `feature_names`，重画 clustermap 不用重算；
    - `module_patches.npz`：`purity`、`patched_row_index`（现恒为 0..n-1，保留字段是为了下游不必分两条路），
      每模块的 `module<i>_panel` / `module<i>_object_index` / `module<i>_instance_uid`；
    - `stage4_qc.json`：模块数、各模块的特征数/特征名、选择方式与参数、行身份校对的三个计数、耗时。

    失败时的磁盘状态与 Stage 1–3 一致：任何在落盘前失败的路径都不动上一轮的产物；只有确定要写时才先删
    旧的四份 + 旧 QC，避免"新标签 + 旧模块表达"的混合态。
    """
    t_start = time.perf_counter()
    X, names = _read_preprocessed()
    stage2_uids, stage2_source = _read_stage2_order(names)
    if len(stage2_uids) != X.shape[0]:
        _fail(
            f"实例数对不上：Stage3 的预处理矩阵 {X.shape[0]} 行，而 Stage2 的行身份（{stage2_source}）"
            f"有 {len(stage2_uids)} 个；行身份无法确定（这不是 Stage1 轮廓被丢弃那种情形，那种只影响 "
            "all_instances.csv、不影响 X），请重跑 Stage2/3"
        )
    meta, meta_uids = _read_instance_order()
    patches_by_uid = _load_patches_by_uid(meta)

    # Stage 2 只给有可用轮廓的实例算特征；它的每个实例都必然有 patch，缺了就是真不对劲（不是"轮廓被丢弃"）
    without_patch = [uid for uid in stage2_uids if uid not in patches_by_uid]
    if without_patch:
        head = "、".join(without_patch[:5])
        _fail(
            f"{len(without_patch)} 个 Stage2 实例在 Stage1 的 patch 里找不到对应项（如 {head}"
            f"{' …' if len(without_patch) > 5 else ''}）：Stage2 只给有可用轮廓的实例算特征、Stage1 也会给它们存 "
            "patch，这种缺失不是【轮廓被丢弃】能解释的，请重跑 Stage1/Stage2"
        )
    patches = [patches_by_uid[uid] for uid in stage2_uids]
    stage2_set = set(stage2_uids)
    meta_not_in_x = [uid for uid in meta_uids if uid not in stage2_set]
    if meta_not_in_x:
        print(
            f"{len(meta_not_in_x)} 个实例有 Stage1 的 meta 行但 Stage2 没有算特征（轮廓被丢弃），"
            f"不进入模块表达与代表图候选（如 {'、'.join(meta_not_in_x[:5])}）；完整清单见 stage4_qc.json"
        )
    # 行身份已经校对上，所以每个 Stage 2 实例都参与代表图挑选（patched_row_index 保留为全量行号）
    patched_rows = np.arange(len(stage2_uids), dtype=np.int64)

    t_discover = time.perf_counter()
    modules = discover_modules(X, names)
    t_expr = time.perf_counter()
    expr = module_expression(X, modules)
    t_patches = time.perf_counter()
    purity, panels, indices = _representative_patches(expr, patches, _PANEL_ROWS)
    t_accounting = time.perf_counter()

    out_root = Path(cfg.OUTPUT_ROOT) / _MODULES_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)
    for stale in (_LABELS_FILE, _EXPR_FILE, _CLUSTER_MAP_FILE, _PATCHES_FILE, _QC_FILE):
        (out_root / stale).unlink(missing_ok=True)

    labels_df = pd.DataFrame(
        {
            "feature": names,
            "module": np.asarray(modules["module_labels"], dtype=int),
            "feature_type": np.asarray(modules["cluster_map_data"]["feature_type"], dtype=int),
            "feature_scope": np.asarray(modules["cluster_map_data"]["feature_scope"], dtype=int),
        }
    )
    labels_df.to_csv(out_root / _LABELS_FILE, index=False)
    np.save(out_root / _EXPR_FILE, np.asarray(expr, dtype=np.float64))
    np.savez_compressed(out_root / _CLUSTER_MAP_FILE, **modules["cluster_map_data"])

    patch_payload: dict[str, np.ndarray] = {
        "n_rows": np.asarray(_PANEL_ROWS, dtype=np.int64),
        "patched_row_index": patched_rows.astype(np.int64),
        "purity": purity,
    }
    for module_id, panel in enumerate(panels, start=1):
        patch_payload[f"module{module_id}_panel"] = np.asarray(panel)
        patch_payload[f"module{module_id}_object_index"] = patched_rows[indices[module_id - 1]].astype(np.int64)
        patch_payload[f"module{module_id}_instance_uid"] = np.asarray(
            [stage2_uids[i] for i in patched_rows[indices[module_id - 1]]], dtype=str
        )
    np.savez_compressed(out_root / _PATCHES_FILE, **patch_payload)

    # total 必须含落盘：四份大产物（标签/表达/clustermap/patch）都已写完才计时，
    # 只有 QC 自身的写盘（几 ms）不计入
    t_written = time.perf_counter()

    module_sizes = {str(i): len(ix) for i, ix in enumerate(modules["feature_indices"], start=1)}
    module_features = {str(i): list(names_arr) for i, names_arr in enumerate(modules["feature_names"], start=1)}
    qc = {
        "module_selection": cfg.MODULE_SELECTION,
        "module_params": dict(cfg.MODULE_PARAMS),
        "random_state": cfg.RANDOM_STATE,
        "n_objects": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        # 行身份校对：X 的行 = Stage2 的实例序；Stage1 的 meta 一般比它多（轮廓被丢弃的实例）
        "n_instances_in_X": int(X.shape[0]),
        "n_meta_rows": len(meta),
        "n_meta_not_in_X": len(meta_not_in_x),
        "meta_not_in_x_uids": meta_not_in_x,
        "module_expr_row_source": stage2_source,
        "n_modules": len(modules["feature_indices"]),
        "module_sizes": module_sizes,
        "module_features": module_features,
        "type_contribution": _json_ready(modules["type_contribution"]),
        "scope_contribution": _json_ready(modules["scope_contribution"]),
        "cluster_map": {
            "corr_shape": list(np.asarray(modules["cluster_map_data"]["corr"]).shape),
            "linkage_shape": list(np.asarray(modules["cluster_map_data"]["linkage"]).shape),
            "fields": sorted(modules["cluster_map_data"].keys()),
        },
        "representative_patches": {
            "n_rows": int(_PANEL_ROWS),
            "n_candidates": int(patched_rows.size),
        },
        # Stage2 的实例缺 patch 现在直接报错（不是"轮廓被丢弃"能解释的）,所以这里恒为 0；
        # 保留字段是为了下游 QC 读法不变
        "n_rows_without_patch": len(without_patch),
        "rows_without_patch_uids": without_patch,
        "timing_seconds": {
            "discover_modules": round(t_expr - t_discover, 3),
            "module_expression": round(t_patches - t_expr, 3),
            "representative_patches": round(t_accounting - t_patches, 3),
            "artifacts": round(t_written - t_accounting, 3),
            "total": round(t_written - t_start, 3),
        },
    }
    (out_root / _QC_FILE).write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")

    sizes = "、".join(f"M{i}:{n}" for i, n in module_sizes.items())
    print(
        f"Stage4 完成（{cfg.MODULE_SELECTION} 选择器，参数 {dict(cfg.MODULE_PARAMS)}）："
        f"{X.shape[0]} 个实例 × {X.shape[1]} 个特征 → {qc['n_modules']} 个模块（{sizes}），"
        f"Stage1 另有 {len(meta_not_in_x)} 个实例被 Stage2 排除，"
        f"耗时 {qc['timing_seconds']['total']}s → {out_root}"
    )
