"""Stage 6b：时间统计（簇占比、簇间转移、表型轨迹与轨迹亲和；Spec §6-Stage6）。

**时间轴是"3 个离散采样点"，不是连续视频。** 数据是 day0 / day3 / day5 三天的静态切片，
motion 特征被有意置空，所以这里的一切"时间"结论都只是"第 1/2/3 次采样"之间的差异：
簇占比随采样点变化、簇间转移按"下一个采样点"统计、表型轨迹每一步跨 3 天或 2 天。
**任何速度类结论（迁移速率、扩散系数、单位时间变化率）都不能从这些产物里得出。**

**时间轴口径**：分析时间用 `frame_idx`（0/1/2；真实天数差 3 与 2 无法用单一 `d_Frame` 表达，
见 `config.DAYS` 的注释），`time_intervals = [0, 1, 2, 3]` 让每个采样点落进自己的箱；真实小时数
（0/72/120 h）只用来给箱贴标签（`bin_time_hours` / `bin_day`），**不参与分箱**。
注意 SPOT 的时间选择是**两端闭区间**（`t >= lo and t <= hi`），`frame_idx` 恰好等于内部边界（1、2）
的样本会被相邻两个箱各算一次，所以本模块把分析时间再平移半个箱宽（`frame_idx + 0.5`，箱中心），
并在 `cluster_proportions` / `phenotype_trajectories` 里拒绝任何落在内部边界上的时间值。

**上游返回契约**（逐条对照 `SPOT/Analysis/sam_analysis.py` 源码，元素顺序极易搞反）：

- `compute_temporal_cluster_proportions(...)` → `(uniq_clust_labels, proportion_time_condition)`，
  **标签在前、占比在后**；占比形状 `(n_conditions, n_time_bins, n_unique_labels)`（条件 → 时间箱 → 簇），
  空箱整行 nan。
- `construct_obj_traj_from_uniq_obj_ids(...)` → dict，键 `all_obj_trajectories`（每条轨迹是**输入数组
  的下标**列表）、`all_obj_trajectories_times`、`all_obj_trajectories_uniq_ids`、
  `all_obj_uniq_row_ids_next`、`all_obj_uniq_ids`。
- `get_labels_for_trajectory(trajs, labels, all_conditions, all_unique_conditions)` → list（每条件一项，
  每项是该条件下各轨迹的标签序列）。
- `compute_phenotype_cluster_transitions_from_label_trajectories(...)` → list（每条件一项），每项是
  `[transition_probs, transition_counts]`。矩阵朝向是 **(来源, 目标)**、行 = 来源、行和 = 1
  （源码里 `label_seq_tt[::-1]` 把 t 放回第 0 行，再 `counts[t, t+1] += 1`）。
  **标签值被直接当矩阵下标**（`transition_counts[label, label]`），所以必须是 0-based 连续整数；
  Stage5 的簇号是 1-based，本模块用 `np.searchsorted` 重映射后再喂给 SPOT。
  `transition_counts` 初始化为 `pseudocounts`（默认 0，会让"从未作为起点"的簇出现 nan 行），
  本模块固定传 `pseudocounts=1`（拉普拉斯平滑）并同时返回**未平滑的原始计数**供审计。
- `compute_phenotypic_trajectory(...)` → `(all_trajectories, all_density_contours)`，前者形状
  `(n_conditions, n_time_bins, 2)`、空箱为 nan；`all_conditions` 传 None 时它内部把条件当成
  `['na']`（长度 1 的 list），所以**必须**显式传条件数组与条件列表。
- `construct_affinity_matrix_phenotypic_trajectory(...)` → 只返回亲和矩阵 `exp(-距离 / std)`
  （`use_std_denom=True`）；两两距离全等时 std = 0，矩阵会全 nan。
- `hcluster_phenotypic_trajectory_affinity_matrix(...)` → `seaborn.clustermap` 的 Grid 对象，内部
  **会开一个 figure 且从不关闭**；本模块调用后按 fignum 差集关掉（长跑/测试里不堆 figure）。

**轨迹退化与兜底**：只有 3 个采样点时，密度阈值法（`mean + 3σ`）可能选空（→ 全 nan）或塌缩成
一个点。退化判据、实际走了哪条路（`spot_density` / `cluster_centroid_weighted` / `bin_centroid`）
全部落进 QC，绝不静默产出一条空轨迹。退化是**逐条件**记录的：一个条件塌缩、或某一步是 nan，都会
在 QC 的 `trajectories.degeneracy` 里单独列出，不会被 `nanmax` / `nansum` 抹平。

**转移矩阵的先验与观测**：`pseudocounts=1` 让零观测的行也保持行随机（不出 nan 行），代价是
“从未作为来源被观测到”的簇会拿到**均匀先验** `1/(K-1)`，看起来像“流向了所有其他簇”。所以
`inter_graph_data` 的每条边都带 `count`（未平滑原始计数）与 `prior_only`（count == 0），每个节点带
`n_obs_out`（观测到的出边数）；逐条件的原始计数落进 `transitions.npz` 的 `counts_by_condition`
（= SPOT 的平滑计数 − `PSEUDOCOUNTS`），画图的人据此把观测与先验分开。

**全局副作用**：本模块在导入时把 matplotlib 后端设为 `Agg`（`matplotlib.use("Agg")`，不是只对
本模块生效的局部设置）。SPOT 的 hcluster 用 `seaborn.clustermap` 建 figure 且从不关闭，交互式
后端会真的开窗口、在测试/长跑里挂住；同进程里随后导入 `pyplot` 的代码也会跟着用 `Agg`。

产物（`output/analysis/`）：`proportions.npz`、`transitions.npz`、`trajectories.npz`、
`affinity.npz`、`stage6b_qc.json`。每份 npz 都自带它用的标签顺序（簇 id、fov 名、bin→day/小时映射、
分析时间 `bin_analysis_time = frame_idx + 0.5`），绘图阶段不需要再猜；`stage6b_qc.json` 额外记录每份
产物用的是哪个实例子集（占比用**全部**有簇标签的实例，含长度为 1 的 track；转移只统计相邻采样点的对，
长度为 1 的 track 不产生转移）。
"""

import contextlib
import io
import json
import time
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import NoReturn

import matplotlib

# SPOT 的 hcluster 用 seaborn.clustermap 建 figure 且从不关闭；交互式后端（本机默认 TkAgg）
# 会真的开窗口、在测试/长跑里挂住。必须在任何 pyplot 导入之前设置。
# 注意这是**全局**副作用：同进程里随后导入 pyplot 的代码也会用 Agg（见模块 docstring）。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SPOT.Analysis import sam_analysis as sam

from biopheno import config as cfg

# —— 上游产物（Stage 5 / Stage 6 的落盘位置）——
_ANALYSIS_SUBDIR: str = "analysis"
_TRACKS_FILE: str = "tracks.csv"
_LABELS_FILE: str = "cluster_labels.csv"
_UMAP_FILE: str = "umap_2d.csv"

# —— 本阶段产物（Spec §6-Stage6b）——
_PROPORTIONS_FILE: str = "proportions.npz"
_TRANSITIONS_FILE: str = "transitions.npz"
_TRAJECTORIES_FILE: str = "trajectories.npz"
_AFFINITY_FILE: str = "affinity.npz"
_QC_FILE: str = "stage6b_qc.json"

# 一个采样点一个箱：内部边界是 1、2（frame_idx 的整数值），所以分析时间用"箱中心" frame_idx + 0.5。
def time_intervals() -> tuple[float, ...]:
    """时间箱边界：每个采样点一箱，即 (0, 1, …, len(cfg.DAYS))。

    必须在**调用时**按当前的 `cfg.DAYS` 算：命令行 `--images/--masks` 会让采样点数变成任意值
    （这份数据是 3 天，换一份数据可能是 2 天或 5 天）。写死成模块级常量会在 2 天数据上多出
    一个空箱、在 5 天数据上少一个箱，而这两种错误都不会抛异常，只会让图上的时间轴悄悄变形。
    """
    return tuple(float(i) for i in range(len(cfg.DAYS) + 1))


BIN_OFFSET: float = 0.5

# 转移矩阵的拉普拉斯平滑强度：0 会让"从未作为起点"的簇出现 nan 行（SPOT 默认行为）。
PSEUDOCOUNTS: float = 1.0
# 表型轨迹的密度阈值（SPOT 默认 mean + 3σ，论文 Stage3 脚本里试到 2–2.5）。
DENSITY_THRESHOLD_SIGMA: float = 3.0
# 轨迹亲和：dtw + 用 std 归一（SPOT 参考脚本的默认调用）。
AFFINITY: str = "dtw"
# 轨迹层次聚类（SPOT 参考脚本画图时用 average/euclidean）。
LINKAGE: str = "average"
METRIC: str = "euclidean"
# 轨迹"塌缩"的判据：整条轨迹的总位移小于这个值就认为密度选择退化成一个点。
_COLLAPSE_EPS: float = 1e-9


def _fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def _as_float_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} 必须是 1 维数组，实际是 {array.shape}")
    return array


def _check_bin_edges(times: np.ndarray, edges: np.ndarray) -> None:
    """拒绝落在内部箱边界上的时间值。

    SPOT 的时间选择是**两端闭区间**，内部边界上的样本会被相邻两个箱各算一次。用箱中心
    （`frame_idx + 0.5`）或更细的箱就能避开；撞上了要早失败，而不是产出一条"每天都重复计入"
    的时间序列。
    """
    interior = edges[1:-1]
    if interior.size == 0:
        return
    on_edge = np.isin(times, interior)
    if on_edge.any():
        offending = np.unique(times[on_edge])
        raise ValueError(
            f"有 {int(on_edge.sum())} 个样本的时间值恰好落在时间箱的**内部边界**上（{offending.tolist()}）："
            "SPOT 的 compute_temporal_cluster_proportions / compute_phenotypic_trajectory 用的是两端闭区间"
            f"（t >= lo 且 t <= hi），这些样本会被相邻两个箱各算一次。时间箱边界是 {edges.tolist()}，"
            "请改用箱中心（frame_idx + 0.5）或更细的箱，不要在边界上放样本"
        )


def _validate_time_inputs(
    times: np.ndarray, conditions: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """校验时间/条件/箱边界三者的形状与取值，返回 `(排序后的唯一条件, 箱边界)`。"""
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(f"time_intervals 至少要有 2 个边界，实际是 {edges.tolist()}")
    if np.any(np.diff(edges) <= 0):
        raise ValueError(f"time_intervals 必须严格递增，实际是 {edges.tolist()}")
    if times.size != conditions.size:
        raise ValueError(f"时间与条件的长度不一致：{times.size} vs {conditions.size}")
    if times.size == 0:
        _fail("没有任何实例：时间统计需要至少一个带簇标签的实例")
    _check_bin_edges(times, edges)
    outside = (times < edges[0]) | (times > edges[-1])
    if outside.any():
        _fail(
            f"有 {int(outside.sum())} 个样本的时间值落在时间箱之外（{edges[0]}–{edges[-1]}）："
            "请确认 frame_idx 与分析时间口径一致，否则这些实例会从全部时间统计里消失"
        )
    return np.unique(conditions), edges


def _label_positions(clusters) -> tuple[np.ndarray, np.ndarray]:
    """把簇标签映射成 SPOT 要求的 0-based 连续下标，返回 `(原始标签, 下标)`。

    SPOT 的转移统计直接拿标签值当矩阵下标（`transition_counts[label, label]`），所以标签必须是
    0..K-1。Stage5 的簇号是 1-based（按平均 eccentricity 升序），这里按**升序**重映射，
    返回的 `labels` 就是矩阵轴上的簇 id 顺序。
    """
    labels = np.unique(np.asarray(clusters))
    if labels.size and not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"簇标签必须是整数，实际 dtype 是 {labels.dtype}")
    positions = np.searchsorted(labels, np.asarray(clusters)).astype(np.int64)
    return labels, positions


def cluster_proportions(
    clusters: np.ndarray,
    time_hours: np.ndarray,
    conditions: np.ndarray,
    time_intervals: list[float] | tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每个 (条件, 时间箱) 里各表型簇的占比（行和为 1，空箱为 nan）。

    时间轴是"3 个离散采样点"，不是连续视频：这里的占比序列只说明"第 1/2/3 次采样时各簇的
    相对丰度"，不能换算成任何单位时间的增殖/迁移速率（motion 已置空）。

    参数
    ----
    clusters : (N,) 整数数组
        每个实例的表型簇 id（Stage5 的 1-based 簇号即可）。
    time_hours : (N,) 数组
        **分析时间**，本模块用箱中心 `frame_idx + 0.5`（真实小时数只用于给箱贴标签）。
    conditions : (N,) 数组
        每个实例的条件，本模块用 `fov`。
    time_intervals : 长度 n_bins+1 的边界
        本模块用 `[0, 1, 2, 3]`（一个采样点一个箱）。

    返回
    ----
    (proportions, uniq_labels, uniq_conditions)
        `proportions` 形状 `(n_conditions, n_bins, n_labels)`（条件 → 时间箱 → 簇），
        `uniq_labels` 是矩阵第 3 轴上的簇 id 顺序（升序），`uniq_conditions` 是第 1 轴的条件顺序。
        注意 SPOT 的返回顺序是 `(标签, 占比)`，本函数把它转成更自然的 `(占比, 标签, 条件)`。
    """
    labels_arr = np.asarray(clusters)
    times = _as_float_array(time_hours, "time_hours")
    conds = np.asarray(conditions)
    edges = np.asarray(time_intervals, dtype=np.float64)
    uniq_conditions, edges = _validate_time_inputs(times, conds, edges)
    if labels_arr.size != times.size:
        raise ValueError(f"簇标签与时间的长度不一致：{labels_arr.size} vs {times.size}")

    # SPOT 返回 (uniq_clust_labels, proportion_time_condition)，顺序容易搞反。
    uniq_labels, proportions = sam.compute_temporal_cluster_proportions(
        labels_arr,
        objects_time=times,
        time_intervals=edges,
        all_conditions=conds,
        unique_conditions=uniq_conditions,
    )
    proportions = np.asarray(proportions, dtype=np.float64)
    expected = (uniq_conditions.size, edges.size - 1, np.asarray(uniq_labels).size)
    if proportions.shape != expected:
        _fail(f"SPOT 的占比矩阵形状是 {proportions.shape}，与预期的 (条件, 时间箱, 簇) {expected} 不符")
    return proportions, np.asarray(uniq_labels), uniq_conditions


def _inter_from_probs(probs: np.ndarray) -> np.ndarray:
    """由完整的 (来源, 目标) 转移矩阵得到"簇间"分布：去掉自转移后按行重归一。"""
    n_labels = probs.shape[0]
    if n_labels < 2:
        # 只有一个簇时没有"簇间"转移可言，用 nan 明说，而不是给一行 0。
        return np.full((n_labels, n_labels), np.nan)
    off = np.asarray(probs, dtype=np.float64).copy()
    np.fill_diagonal(off, 0.0)
    row_sums = off.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return off / row_sums


def _raw_counts(model) -> np.ndarray:
    """SPOT 的计数矩阵初值是 `pseudocounts`；减掉先验得到**未平滑的原始观测计数**。

    浮点误差（`k - 1.0`）抹成 0，这样“边缘到底有没有被观测到”可以严格判断。
    """
    counts = np.asarray(model, dtype=np.float64) - PSEUDOCOUNTS
    counts[np.abs(counts) < 1e-9] = 0.0
    return counts


def _graph_data(
    labels: np.ndarray, inter: np.ndarray, self_probs: np.ndarray, counts: np.ndarray, condition: str
) -> dict:
    """给绘图阶段（F8 转移图）用的、可 JSON 序列化的边表：(来源, 目标) 概率 + **原始计数**。

    `probability` 来自平滑后的 `inter`（`pseudocounts=1`，保证行随机、不出 nan 行），所以零观测的
    边也会拿到非零概率：一整行都没有观测时，非对角概率就是均匀的 `1/(K-1)`。为了不让画图的人把先验
    当观测，每条边额外带 `count`（未平滑原始计数）与 `prior_only`（count == 0），每个节点带
    `n_obs_out`（观测到的出边数）。
    """
    edges = []
    n_obs_out = []
    for source_i, label_i in enumerate(labels):
        row_counts = np.asarray(counts[source_i], dtype=np.float64)
        n_obs_out.append(round(float(row_counts.sum())))
        for target_i, label_j in enumerate(labels):
            probability = inter[source_i, target_i]
            if target_i == source_i or not np.isfinite(probability) or probability <= 0:
                continue
            count = float(row_counts[target_i])
            edges.append(
                {
                    "source": int(label_i),
                    "target": int(label_j),
                    "probability": float(probability),
                    "count": count,
                    "prior_only": bool(count == 0.0),
                }
            )
    return {
        "condition": condition,
        "labels": [int(label) for label in labels],
        "nodes": [str(int(label)) for label in labels],
        "n_obs_out": n_obs_out,
        "self": [None if not np.isfinite(value) else float(value) for value in self_probs],
        "edges": edges,
        "pseudocounts": float(PSEUDOCOUNTS),
        "prior_only_note": (
            "probability 来自 pseudocounts=1 的平滑矩阵；count == 0 的边（prior_only=True）只是先验，"
            "不是观测到的转移，画图时应当过滤或另行标注"
        ),
        "orientation": "source->target（行 = 来源，行和 = 1；对角单列为 self）",
    }


def transitions(tracks: pd.DataFrame, clusters: np.ndarray) -> dict:
    """按"下一个采样点"统计表型簇之间的转移矩阵（行 = 来源，列 = 目标，行和 = 1）。

    时间轴是"3 个离散采样点"（day0 → day3 → day5），不是连续视频：矩阵里的每个元素是
    "同一颗类器官在相邻两次采样间换到另一个表型簇"的经验频率，一步跨 3 天或 2 天，
    所以**不能**读成单位时间的转移速率（motion 已置空，也没有真实视频帧）。

    参数
    ----
    tracks : DataFrame
        至少要有 `spot_uid` / `fov` / `track_id` / `frame_idx`，行序必须与 `clusters` 逐行对齐。
        `spot_uid` 形如 `f"{fov}_{track_id}_{frame_idx}"`（Stage6 的硬要求）。
    clusters : (N,) 整数数组
        每行的表型簇 id（1-based 也可以，内部会重映射成 SPOT 需要的 0-based 下标）。

    返回
    ----
    dict，键：
        - `inter`：(K, K) 簇间转移（对角为 0，按行重归一，行和 1）；
        - `self`：(K,) 各簇的自转移概率；
        - `inter_graph_data`：给 F8 的 JSON 友好边表（含**未平滑计数** `count` 与 `prior_only`）；
        - `inter_graph_data_by_condition`：同上，但按 fov 各一份；
        - `probs` / `counts`：(K, K) 完整 (来源, 目标) 概率（含自转移）与**未平滑**的原始计数；
        - `counts_by_condition`：(n_conditions, K, K) 逐条件的**未平滑**原始计数（
          全 0 就意味着该条件的矩阵完全来自先验）；
        - `labels` / `conditions` / `by_condition` / `inter_by_condition` / `self_by_condition`；
        - `qc`：实例数、**SPOT 段数**与真实 track 数（两者不是一回事，见 `segment_note`）、
          长度 1 的段数、相邻采样点对数（这些数决定哪些行被用到了）。
    """
    required = ("spot_uid", "fov", "track_id", "frame_idx")
    missing = [name for name in required if name not in tracks.columns]
    if missing:
        raise ValueError(f"tracks 缺少必需的列 {missing}：转移统计要靠 spot_uid 重建轨迹、靠 fov 分条件")
    labels_arr = np.asarray(clusters)
    if len(labels_arr) != len(tracks):
        raise ValueError(f"tracks 有 {len(tracks)} 行，但簇标签有 {len(labels_arr)} 个：两者必须逐行对齐")

    uids = tracks["spot_uid"].to_numpy(dtype=str)
    if np.unique(uids).size != uids.size:
        raise ValueError(
            "spot_uid 有重复：SPOT 的 construct_obj_traj_from_uniq_obj_ids 会把同一帧的两个实例当成两段轨迹，"
            "请先回 Stage6 检查 track 分配（同一 (fov, track_id, frame_idx) 只能有一行）"
        )
    fovs = tracks["fov"].to_numpy(dtype=object)
    labels, positions = _label_positions(labels_arr)
    uniq_conditions = np.unique(fovs)

    # tqdm 进度条走 stderr、SPOT 在转移统计里无条件 print 矩阵尺寸——都不属于本模块的输出。
    with contextlib.redirect_stderr(io.StringIO()):
        traj_dict = sam.construct_obj_traj_from_uniq_obj_ids(uids, separator="_", wanted_entries=[0, 1], d_Frame=1)
    all_trajectories = traj_dict["all_obj_trajectories"]
    label_trajectories = sam.get_labels_for_trajectory(
        all_trajectories,
        positions,
        all_conditions=fovs,
        all_unique_conditions=uniq_conditions,
    )
    pooled = [[np.asarray(seq) for per_condition in label_trajectories for seq in per_condition]]
    with contextlib.redirect_stdout(io.StringIO()):
        pooled_models = sam.compute_phenotype_cluster_transitions_from_label_trajectories(
            pooled, positions, pseudocounts=PSEUDOCOUNTS, time_interval=1
        )
        per_condition_models = sam.compute_phenotype_cluster_transitions_from_label_trajectories(
            label_trajectories, positions, pseudocounts=PSEUDOCOUNTS, time_interval=1
        )

    probs = np.asarray(pooled_models[0][0], dtype=np.float64)
    counts = _raw_counts(pooled_models[0][1])  # 去掉平滑的先验
    self_probs = np.diag(probs).astype(np.float64)
    inter = _inter_from_probs(probs)

    by_condition = {}
    graph_by_condition = {}
    inter_by_condition = np.full((uniq_conditions.size,) + inter.shape, np.nan)
    self_by_condition = np.full((uniq_conditions.size, labels.size), np.nan)
    counts_by_condition = np.zeros((uniq_conditions.size,) + counts.shape, dtype=np.float64)
    for condition_i, condition in enumerate(uniq_conditions):
        cond_probs = np.asarray(per_condition_models[condition_i][0], dtype=np.float64)
        cond_counts = _raw_counts(per_condition_models[condition_i][1])
        cond_inter = _inter_from_probs(cond_probs)
        cond_self = np.diag(cond_probs).astype(np.float64)
        inter_by_condition[condition_i] = cond_inter
        self_by_condition[condition_i] = cond_self
        counts_by_condition[condition_i] = cond_counts
        by_condition[str(condition)] = {"inter": cond_inter, "self": cond_self}
        graph_by_condition[str(condition)] = _graph_data(
            labels, cond_inter, cond_self, cond_counts, condition=str(condition)
        )

    # SPOT 把 (fov, track_id) 按帧号断档切成多段，所以“段数”不等于 track 数，两个数都记。
    lengths = np.array([len(seq) for seq in all_trajectories], dtype=np.int64)
    qc = {
        "n_instances": len(uids),
        "n_spot_segments": len(all_trajectories),
        "n_single_instance_segments": int((lengths < 2).sum()),
        "n_tracks": int(tracks[["fov", "track_id"]].drop_duplicates().shape[0]),
        "n_transition_pairs": int(np.clip(lengths - 1, 0, None).sum()),
        "n_clusters": int(labels.size),
        "pseudocounts": float(PSEUDOCOUNTS),
        "track_length_filter": ">=2（SPOT 只统计相邻采样点的对，长度 1 的段不产生任何转移）",
        "orientation": "inter/probs 都是 (来源, 目标)：行 = 来源、行和 = 1",
        "segment_note": (
            "`n_spot_segments` 是 SPOT 的 construct_obj_traj_from_uniq_obj_ids 切出的**段数**"
            "（同一 track 的帧号断档会裂成多段），`n_tracks` 才是 (fov, track_id) 去重后的真实 track 数；"
            "`n_single_instance_segments` 是只含 1 个实例的段数（它们不产生转移对）"
        ),
        "prior_only_note": (
            "pseudocounts=1 让零观测的行也是均匀分布；原始计数在 `counts` / `counts_by_condition`，"
            "边表里用 `prior_only` 标出 count == 0 的纯先验边"
        ),
    }
    return {
        "inter": inter,
        "self": self_probs,
        "inter_graph_data": _graph_data(labels, inter, self_probs, counts, condition="pooled"),
        "inter_graph_data_by_condition": graph_by_condition,
        "probs": probs,
        "counts": counts,
        "counts_by_condition": counts_by_condition,
        "labels": labels,
        "conditions": uniq_conditions,
        "by_condition": by_condition,
        "inter_by_condition": inter_by_condition,
        "self_by_condition": self_by_condition,
        "qc": qc,
    }


def _condition_names(conditions, n_conditions: int) -> list[str]:
    """把条件数组变成可读的名字；没给条件时用下标。"""
    if conditions is None:
        return [str(index) for index in range(n_conditions)]
    return [str(condition) for condition in np.asarray(conditions)]


def _degeneracy(trajectories, error: str | None, conditions=None) -> dict:
    """判断密度阈值法产出的轨迹是否退化（空选择 / 空箱 / 单条件塌缩成一个点）。

    退化必须**逐条件**记录：`nanmax` 会把“只有一个条件塌缩”藏成“整体正常”，`nansum` 会把
    一个 nan 步当作 0 位移。所以这里同时给出 `finite_bins`（逐条件逐箱）、`step_lengths`、
    `n_nan_steps`、`collapsed` 与 `collapsed_conditions`，让 QC 里能看到是哪个条件、哪个箱出的问题。
    """
    if trajectories is None:
        return {
            "degenerate": True,
            "reasons": [f"compute_phenotypic_trajectory 抛异常：{error}"],
            "conditions": [] if conditions is None else _condition_names(conditions, len(conditions)),
            "n_finite_bins": [],
            "finite_bins": [],
            "step_lengths": [],
            "n_nan_steps": [],
            "path_length": [],
            "collapsed": [],
            "collapsed_conditions": [],
        }
    traj = np.asarray(trajectories, dtype=np.float64)
    n_conditions, n_bins = traj.shape[0], traj.shape[1]
    names = _condition_names(conditions, n_conditions)
    finite = np.isfinite(traj).all(axis=-1)  # (n_conditions, n_bins)
    n_finite = finite.sum(axis=1)
    if n_bins > 1:
        step_lengths = np.linalg.norm(np.diff(traj, axis=1), axis=-1)  # nan 步保持 nan
    else:
        step_lengths = np.zeros((n_conditions, 0), dtype=np.float64)
    n_nan_steps = np.isnan(step_lengths).sum(axis=1)
    # 等价于 nansum，但全 nan 的条件算 0 而不是报“All-NaN slice”警告；nan 步数另记在 n_nan_steps。
    path_length = np.where(np.isfinite(step_lengths), step_lengths, 0.0).sum(axis=1)
    collapsed = path_length < _COLLAPSE_EPS
    reasons = []
    if not finite.all():
        empty = [
            f"{names[index]}：第 {np.flatnonzero(~finite[index]).tolist()} 个时间箱"
            for index in range(n_conditions)
            if (~finite[index]).any()
        ]
        reasons.append("存在空选择的时间箱（密度阈值下没有任何点 → 坐标是 nan）：" + "；".join(empty))
    if (n_finite < 2).any():
        reasons.append(f"条件 {[names[index] for index in np.flatnonzero(n_finite < 2)]} 的有效时间箱 < 2，构不成轨迹")
    if collapsed.any():
        reasons.append(
            f"条件 {[names[index] for index in np.flatnonzero(collapsed)]} 的轨迹总位移 < {_COLLAPSE_EPS}"
            "（密度选择塌缩成一个点；path_length 不含 nan 步，逐条步长见 step_lengths / n_nan_steps）"
        )
    if n_bins < 2:
        reasons.append("时间箱只有 1 个，无法构成轨迹")
    return {
        "degenerate": bool(reasons),
        "reasons": reasons,
        "conditions": names,
        "n_finite_bins": [int(value) for value in n_finite],
        "finite_bins": [[bool(value) for value in row] for row in finite],
        "step_lengths": [[None if not np.isfinite(value) else float(value) for value in row] for row in step_lengths],
        "n_nan_steps": [int(value) for value in n_nan_steps],
        "path_length": [float(value) for value in path_length],
        "collapsed": [bool(value) for value in collapsed],
        "collapsed_conditions": [names[index] for index in np.flatnonzero(collapsed)],
    }


def _finiteness(trajectories, conditions) -> dict:
    """统计**实际落盘**的那条轨迹在每个 (条件, 时间箱) 是否有限。

    与被拒绝的 SPOT 密度输出分开记：`degeneracy` 描述被拒的那一份，`shipped_finiteness` 描述
    真正写进 `trajectories.npz` 的这一份，QC 里不会把两者搞混。
    """
    traj = np.asarray(trajectories, dtype=np.float64)
    finite = np.isfinite(traj).all(axis=-1)
    return {
        "all_finite": bool(finite.all()),
        "conditions": _condition_names(conditions, traj.shape[0]),
        "finite_bins": [[bool(value) for value in row] for row in finite],
        "n_finite_bins": [int(value) for value in finite.sum(axis=1)],
        "note": "这里统计的是实际落盘的轨迹（path 见同级字段）；被拒的 SPOT 密度输出在 degeneracy 里",
    }


def _bin_assignments(times: np.ndarray, conditions: np.ndarray, edges: np.ndarray, uniq_conditions, bin_i: int):
    """取第 bin_i 个时间箱里、属于某条件的实例（与 SPOT 同样的两端闭区间选择）。"""
    return np.logical_and(times >= edges[bin_i], times <= edges[bin_i + 1]) & np.isin(conditions, uniq_conditions)


def _fallback_trajectories(
    embedding: np.ndarray,
    times: np.ndarray,
    conditions: np.ndarray,
    edges: np.ndarray,
    uniq_conditions: np.ndarray,
    clusters: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    """密度阈值法退化时的兜底：**箱内**簇质心按该 (条件, 时间箱) 的簇占比加权。

    每个 (条件, 时间箱) 里先取各簇在该箱内的质心，再按该箱的簇占比加权；因为簇是箱内实例的
    划分，这个加权和恒等于"该箱所有实例的嵌入均值"（测试用这条恒等式验算）。
    这里用**箱内**质心而不是全局质心：同一个簇在不同天的形状会漂移，全局质心会把这部分漂移摸平。
    没有簇标签时退回纯粹的逐箱质心（`bin_centroid`），并在 QC 里区分。
    """
    n_bins = edges.size - 1
    n_conditions = uniq_conditions.size
    trajectories = np.full((n_conditions, n_bins, embedding.shape[1]), np.nan)
    if clusters is not None:
        cluster_array = np.asarray(clusters)
        labels, _ = _label_positions(cluster_array)
        proportions, uniq_labels, _ = cluster_proportions(cluster_array, times, conditions, edges)
        # 占比的簇轴与这里的 labels 必须同序（两者都来自 np.unique 升序），否则加权会把簇的权重串位。
        if not np.array_equal(np.asarray(uniq_labels), labels):
            _fail(
                f"簇占比的簇顺序 {np.asarray(uniq_labels).tolist()} 与兜底用的簇顺序 {labels.tolist()} 不一致："
                "两者都应当来自 np.unique 的升序，出现差异说明聚类标签在两次调用之间被改动"
            )
        for condition_i, condition in enumerate(uniq_conditions):
            for bin_i in range(n_bins):
                weights = proportions[condition_i, bin_i]
                if np.all(np.isnan(weights)):
                    continue
                select = _bin_assignments(times, conditions, edges, np.array([condition]), bin_i)
                weighted = np.zeros(embedding.shape[1], dtype=np.float64)
                for label_i, label in enumerate(labels):
                    in_cluster = select & (cluster_array == label)
                    if not in_cluster.any():
                        continue
                    weighted += weights[label_i] * embedding[in_cluster].mean(axis=0)
                trajectories[condition_i, bin_i] = weighted
        return trajectories, "cluster_centroid_weighted"

    for condition_i, condition in enumerate(uniq_conditions):
        for bin_i in range(n_bins):
            select = _bin_assignments(times, conditions, edges, np.array([condition]), bin_i)
            if select.any():
                trajectories[condition_i, bin_i] = embedding[select].mean(axis=0)
    return trajectories, "bin_centroid"


def phenotype_trajectories(
    embedding: np.ndarray,
    time_hours: np.ndarray,
    conditions: np.ndarray,
    time_intervals: list[float] | tuple[float, ...],
    clusters: np.ndarray | None = None,
    thresh_density_sigma: float = DENSITY_THRESHOLD_SIGMA,
) -> dict:
    """每个条件在表型景观（前两维 UMAP 坐标）里的"主群体坐标"随采样点的移动。

    时间轴是"3 个离散采样点"，不是连续视频：轨迹的每一段跨 3 天或 2 天，所以只能读出
    "第 1/2/3 次采样时主群体落在景观的哪个区域"，不能读成移动速度（motion 已置空）。

    做法：先走 SPOT 的 `compute_phenotypic_trajectory`（每个时间箱取密度 `mean + 3σ` 以上的
    区域的质心）。只有 3 个采样点时这一步可能退化（选空 → nan，或塌缩成一个点），
    退化时退到"簇质心按该箱簇占比加权"的轨迹，并把实际路径写进 `path`。

    参数
    ----
    embedding : (N, >=2) 数组
        联合 UMAP 坐标，只用前两维（SPOT 内部也只用前两维）。
    time_hours : (N,) 数组
        分析时间（箱中心 `frame_idx + 0.5`）。
    conditions : (N,) 数组
        每个实例的条件（`fov`）。
    time_intervals : 长度 n_bins+1 的边界（`[0, 1, 2, 3]`）。
    clusters : (N,) 整数数组，可选
        退化兜底时用来做"簇质心 × 占比"加权；不给则退到逐箱质心。
    thresh_density_sigma : float
        密度阈值（以 σ 计）。

    返回
    ----
    dict：`trajectories`（选定的轨迹 `(n_conditions, n_bins, 2)`）、`spot_trajectories`（SPOT 原始
    输出，便于审计退化）、`conditions`、`bin_centers`、`bin_edges`、`path`、`degenerate`、
    `degeneracy`（逐条件有效箱数、逐条件/逐箱步长与 nan 步数、哪些条件塌缩）、
    `shipped_finiteness`（**实际落盘**轨迹的逐条件/逐箱有限性）、`thresh_density_sigma`、`warnings`。
    """
    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.ndim != 2 or embedding.shape[1] < 2:
        raise ValueError(f"embedding 必须是 (N, >=2) 的数组，实际是 {embedding.shape}")
    times = _as_float_array(time_hours, "time_hours")
    conds = np.asarray(conditions)
    edges = np.asarray(time_intervals, dtype=np.float64)
    uniq_conditions, edges = _validate_time_inputs(times, conds, edges)
    if embedding.shape[0] != times.size:
        raise ValueError(f"embedding 有 {embedding.shape[0]} 行，但时间有 {times.size} 个：两者必须逐行对齐")
    if clusters is not None and len(np.asarray(clusters)) != times.size:
        raise ValueError("簇标签与时间的长度不一致：退化兜底要靠簇质心加权，必须逐行对齐")
    points = embedding[:, :2]

    warning_messages: list[str] = []
    spot_trajectories = None
    contours = None
    error = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            spot_trajectories, contours = sam.compute_phenotypic_trajectory(
                points,
                objects_time=times,
                time_intervals=edges,
                all_conditions=conds,
                unique_conditions=uniq_conditions,
                thresh_density_sigma=thresh_density_sigma,
                debugviz=False,
            )
        except Exception as exc:  # noqa: BLE001 —— 任何失败都退到兜底路径，并把原因写进 QC
            error = f"{type(exc).__name__}: {exc}"
        warning_messages.extend(str(item.message) for item in caught)

    degeneracy = _degeneracy(spot_trajectories, error, conditions=uniq_conditions)
    if degeneracy["degenerate"]:
        trajectories, path = _fallback_trajectories(points, times, conds, edges, uniq_conditions, clusters)
    else:
        trajectories, path = np.asarray(spot_trajectories, dtype=np.float64), "spot_density"

    bin_centers = edges[:-1] + 0.5 * np.diff(edges)
    return {
        "trajectories": trajectories,
        "spot_trajectories": None if spot_trajectories is None else np.asarray(spot_trajectories, dtype=np.float64),
        "density_contours": contours,
        "conditions": uniq_conditions,
        "bin_centers": bin_centers,
        "bin_edges": edges,
        "path": path,
        "degenerate": bool(degeneracy["degenerate"]),
        "degeneracy": degeneracy,
        "shipped_finiteness": _finiteness(trajectories, uniq_conditions),
        "thresh_density_sigma": float(thresh_density_sigma),
        "warnings": warning_messages,
    }


def _trajectory_matrix(trajectories) -> tuple[np.ndarray, list[str]]:
    """从 `phenotype_trajectories` 的 dict（或裸数组）里取出 (条件, 时间箱, 2) 轨迹与条件名。"""
    if isinstance(trajectories, dict):
        matrix = np.asarray(trajectories["trajectories"], dtype=np.float64)
        labels = [str(condition) for condition in trajectories.get("conditions", range(len(matrix)))]
    else:
        matrix = np.asarray(trajectories, dtype=np.float64)
        labels = [str(index) for index in range(len(matrix))]
    if matrix.ndim != 3 or matrix.shape[-1] != 2:
        raise ValueError(f"轨迹必须是 (n_conditions, n_bins, 2) 的数组，实际是 {matrix.shape}")
    if len(labels) != matrix.shape[0]:
        raise ValueError(f"条件名有 {len(labels)} 个，但轨迹有 {matrix.shape[0]} 条")
    return matrix, labels


def _affinity_with_info(matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    """DTW 轨迹距离 → 亲和矩阵，并在距离全等（std = 0）时退到"全 1"。"""
    n_conditions = matrix.shape[0]
    info = {"affinity": AFFINITY, "use_std_denom": True, "fallback": None, "warnings": []}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            affinity = sam.construct_affinity_matrix_phenotypic_trajectory(
                matrix, affinity=AFFINITY, use_std_denom=True
            )
        except Exception as exc:  # noqa: BLE001 —— 例如所有时间箱都含 nan
            affinity = None
            info["fallback"] = f"construct_affinity_matrix_phenotypic_trajectory 抛异常：{type(exc).__name__}: {exc}"
        info["warnings"] = [str(item.message) for item in caught]
    if affinity is None or not np.all(np.isfinite(affinity)):
        if info["fallback"] is None:
            info["fallback"] = "轨迹两两距离全等（std = 0）导致 exp(-d / std) 无定义：亲和矩阵整体置 1"
        affinity = np.ones((n_conditions, n_conditions), dtype=np.float64)
    return np.asarray(affinity, dtype=np.float64), info


def trajectory_affinity(trajectories: dict, return_meta: bool = False):
    """条件之间的轨迹亲和矩阵（DTW 距离的指数核，对称、对角为 1）。

    时间轴是"3 个离散采样点"，不是连续视频：DTW 在这里只是"两条 3 点折线形状有多像"的度量，
    与"移动快慢"无关（motion 已置空）。

    返回 `(affinity, labels)`：`affinity[i, j] = exp(-dtw_i_j / std(所有距离))`，
    `labels` 是矩阵两个轴的顺序（= `phenotype_trajectories` 的 `conditions`）。

    两两距离全等（std = 0）时 `exp(-d / std)` 无定义，本函数退到"全 1"——这是一个**必须被调用方
    知道**的退化，否则看起来只是"各条件轨迹都很像"。所以 `return_meta=True` 时额外返回 info
    （`fallback` / `warnings`），把这层退化显式交给调用方。
    """
    matrix, labels = _trajectory_matrix(trajectories)
    affinity, info = _affinity_with_info(matrix)
    if return_meta:
        return affinity, labels, info
    return affinity, labels


def trajectory_clusters(
    affinity: np.ndarray,
    labels: Iterable[str],
    linkage: str = LINKAGE,
    metric: str = METRIC,
) -> dict:
    """用 SPOT 的 `hcluster_phenotypic_trajectory_affinity_matrix` 给轨迹做层次聚类。

    时间轴是"3 个离散采样点"，不是连续视频：聚类只说明"哪几个条件的 3 点轨迹形状相近"，
    不涉及速度。SPOT 的这个函数内部用 `seaborn.clustermap` 建 figure 且从不关闭，
    所以这里在 `finally` 里把本次调用新开的 figure 全部关掉（按 fignum 差集，不误伤别人的图）。

    返回 dict：`order`（重排后的条件下标）、`conditions`（重排后的条件名）、`linkage`、`metric`、
    `skipped`（条件数 < 2 时写明跳过原因）、`fallback`（seaborn/SPOT 报错时的补救说明）。
    聚类图属于"额外一张图"，失败了不该把整条 Stage6b 拖挂：报错时退回原始条件顺序并记进 `fallback`。
    """
    names = [str(label) for label in labels]
    affinity = np.asarray(affinity, dtype=np.float64)
    if affinity.shape != (len(names), len(names)):
        raise ValueError(f"亲和矩阵形状 {affinity.shape} 与条件数 {len(names)} 不匹配")
    if len(names) < 2:
        return {
            "order": [0] if names else [],
            "conditions": names,
            "linkage": linkage,
            "metric": metric,
            "skipped": "条件数 < 2，层次聚类无意义",
            "fallback": None,
        }

    before = set(plt.get_fignums())
    fallback = None
    try:
        try:
            grid = sam.hcluster_phenotypic_trajectory_affinity_matrix(
                affinity,
                uniq_conditions=np.asarray(names),
                linkage=linkage,
                metric=metric,
            )
            order = [int(index) for index in grid.dendrogram_col.reordered_ind]
        except Exception as exc:  # noqa: BLE001 —— 一张聚类图失败不该终止整条 Stage6b
            fallback = f"hcluster_phenotypic_trajectory_affinity_matrix 抛异常：{type(exc).__name__}: {exc}"
            order = list(range(len(names)))
    finally:
        for fignum in set(plt.get_fignums()) - before:
            plt.close(fignum)
    return {
        "order": order,
        "conditions": [names[index] for index in order],
        "linkage": linkage,
        "metric": metric,
        "skipped": None,
        "fallback": fallback,
    }


def _bin_labels(n_bins: int) -> tuple[np.ndarray, list[str], list[float]]:
    """把时间箱编号映射到 (frame_idx, day 名, 真实小时数)；绘图阶段靠它给箱贴标签。"""
    frame_to_day = {int(frame): day for day, frame in cfg.DAYS.items()}
    bin_frame_idx = np.arange(n_bins, dtype=np.int64)
    bin_day = [frame_to_day.get(int(frame), f"frame{int(frame)}") for frame in bin_frame_idx]
    bin_time_hours = [float(cfg.TIME_HOURS.get(day, np.nan)) for day in bin_day]
    return bin_frame_idx, bin_day, bin_time_hours


def _read_csv(path: Path, missing_message: str) -> pd.DataFrame:
    if not path.exists():
        _fail(missing_message)
    return pd.read_csv(path)


def _stage6b_inputs() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict]:
    """读 Stage5/6 的产物，按 `instance_uid` 对齐成 (tracks 子集, 簇标签, 条件, 嵌入, 对齐 QC)。"""
    analysis_dir = Path(cfg.OUTPUT_ROOT) / _ANALYSIS_SUBDIR
    tracks_path = analysis_dir / _TRACKS_FILE
    labels_path = analysis_dir / _LABELS_FILE
    umap_path = analysis_dir / _UMAP_FILE
    tracks = _read_csv(
        tracks_path,
        f"找不到 Stage6 的 {_TRACKS_FILE}（{tracks_path}）：时间统计要先有跨天 track，请先跑 run_stage6（Stage 6）",
    )
    labels_df = _read_csv(
        labels_path,
        f"找不到 Stage5 的 {_LABELS_FILE}（{labels_path}）：表型簇是时间统计的前提，请先跑 run_stage5（Stage 5）",
    )
    umap_df = _read_csv(
        umap_path,
        f"找不到 Stage5 的 {_UMAP_FILE}（{umap_path}）：表型轨迹需要联合 UMAP 坐标，请先跑 run_stage5（Stage 5）",
    )
    for table, columns, name in (
        (tracks, ("instance_uid", "spot_uid", "track_id", "fov", "frame_idx"), _TRACKS_FILE),
        (labels_df, ("instance_uid", "cluster"), _LABELS_FILE),
        (umap_df, ("instance_uid", "umap_1", "umap_2"), _UMAP_FILE),
    ):
        missing = [column for column in columns if column not in table.columns]
        if missing:
            _fail(f"{name} 缺少列 {missing}：不是当前 Stage6b 期望的产物，请按顺序重跑 Stage5/Stage6")

    for table, name in ((tracks, _TRACKS_FILE), (labels_df, _LABELS_FILE), (umap_df, _UMAP_FILE)):
        if not table["instance_uid"].is_unique:
            duplicated = table.loc[table["instance_uid"].duplicated(), "instance_uid"].iloc[0]
            _fail(
                f"{name} 里 instance_uid 有重复（如 {duplicated}）：按 uid 对齐会把一行复制成多行"
                "（占比会被重复计入、实际参与统计的实例数与 n_without_cluster 都可能变成负数），"
                "时间统计的行身份必须唯一，请重跑对应的 Stage（tracks 来自 Stage6，cluster_labels / umap_2d 来自 Stage5）"
            )
    merged = tracks.merge(labels_df[["instance_uid", "cluster"]], on="instance_uid", how="inner")
    if merged.empty:
        _fail(
            f"{_TRACKS_FILE} 与 {_LABELS_FILE} 的 instance_uid 没有任何交集：两者不是同一批实例的产物，"
            "请按顺序重跑 Stage1 → Stage5 → Stage6"
        )
    n_without_cluster = len(tracks) - len(merged)
    merged = merged.merge(umap_df[["instance_uid", "umap_1", "umap_2"]], on="instance_uid", how="left")
    without_embedding = merged[["umap_1", "umap_2"]].isna().any(axis=1)
    if without_embedding.any():
        _fail(
            f"有 {int(without_embedding.sum())} 个实例在 {_UMAP_FILE} 里没有坐标（如 "
            f"{merged.loc[without_embedding, 'instance_uid'].iloc[0]}）：表型轨迹无法计算，请重跑 Stage5"
        )
    alignment_qc = {
        "n_tracks_rows": len(tracks),
        "n_instances_without_cluster": int(n_without_cluster),
        "note": ("Stage5 只给有可用轮廓的实例聚类；没有簇标签的实例无法参与任何时间统计，数量记在这里，不当成静默丢弃"),
    }
    clusters = merged["cluster"].to_numpy(dtype=np.int64)
    conditions = merged["fov"].astype(str).to_numpy(dtype=object)
    embedding = merged[["umap_1", "umap_2"]].to_numpy(dtype=np.float64)
    return merged, clusters, conditions, embedding, alignment_qc


def run_stage6b() -> None:
    """跑完 Stage 6b 并落盘（Spec §6-Stage6b）：读 `output/analysis/` 的 Stage5/6 产物 → 写四份 npz + QC。

    时间轴是"3 个离散采样点"（day0/day3/day5），不是连续视频：所有产物都只描述这三次采样之间的
    差异，**不支持任何速度类结论**（motion 特征被有意置空）。

    分析时间用箱中心 `frame_idx + 0.5`，时间箱边界是 `[0, 1, 2, 3]`：SPOT 的时间选择是两端闭区间，
    直接用 `frame_idx` 会让采样点 1、2 同时落进相邻两个箱。真实小时数（0/72/120 h）只作为箱标签
    落进每份产物与 QC，不参与分箱；分析时间本身（`bin_analysis_time`）也写进每份 npz，不只是一句 QC 散文。
    两步的真实宽度并不相等（day0→day3 = 72 h、day3→day5 = 48 h），本模块把两步都当作“一步”。

    产物（都自带标签顺序：`cluster_labels` / `conditions` / `bin_day` / `bin_frame_idx` /
    `bin_time_hours`）：
    - `proportions.npz`：`proportions`（条件 × 时间箱 × 簇，行和 1，空箱 nan）；
    - `transitions.npz`：`inter`（来源 × 目标，对角 0、行和 1）、`self_transitions`（`self` 是 Python
      关键字，落盘时改名）、`probs`、`counts`、`counts_by_condition`（逐条件的**未平滑**原始计数，
      全 0 即该条件的矩阵完全来自先验）、`inter_by_condition` / `self_by_condition`（按 fov 分开）
      与 `pseudocounts`；
    - `trajectories.npz`：`trajectories`（条件 × 时间箱 × 2）、`path`（实际走的路径）、
      `spot_trajectories`（SPOT 原始输出，便于审计退化）；
    - `affinity.npz`：`affinity`（条件 × 条件，DTW 指数核）、`order`（层次聚类重排后的下标）、
      `linkage` / `metric`；
    - `stage6b_qc.json`：时间轴口径、标签顺序、每份产物用到的实例子集（占比用全部有簇标签的实例，
      含长度 1 的 track；转移只统计相邻采样点的对）、轨迹退化判据与实际路径、耗时。
    """
    t_start = time.perf_counter()
    merged, clusters, conditions, embedding, alignment_qc = _stage6b_inputs()
    t_loaded = time.perf_counter()

    intervals = time_intervals()
    edges = np.asarray(intervals, dtype=np.float64)
    n_bins = edges.size - 1
    analysis_time = merged["frame_idx"].to_numpy(dtype=np.float64) + BIN_OFFSET
    frames_present = np.unique(merged["frame_idx"].to_numpy(dtype=np.int64))
    if frames_present.min() < 0 or frames_present.max() >= n_bins:
        _fail(
            f"frame_idx 的取值是 {frames_present.tolist()}，但时间箱只有 {n_bins} 个（{edges.tolist()}）："
            "帧轴必须是 0..n_bins-1 的连续整数（Stage6 的 spot_uid 硬要求），请重跑 Stage6"
        )

    proportions, proportion_labels, proportion_conditions = cluster_proportions(
        clusters, analysis_time, conditions, intervals
    )
    trans = transitions(merged, clusters)
    t_transitions = time.perf_counter()
    trajectories = phenotype_trajectories(embedding, analysis_time, conditions, intervals, clusters=clusters)
    t_trajectories = time.perf_counter()
    affinity, affinity_info = _affinity_with_info(trajectories["trajectories"])
    clustered = trajectory_clusters(affinity, trajectories["conditions"])
    t_affinity = time.perf_counter()

    # 三条路径（占比 / 转移 / 轨迹）必须用同一套轴顺序，否则图会把簇或 fov 对错轴。
    condition_order = list(map(str, proportion_conditions))
    for name, other in (
        ("transitions", list(map(str, trans["conditions"]))),
        ("trajectories", list(map(str, trajectories["conditions"]))),
    ):
        if other != condition_order:
            _fail(f"{name} 的条件顺序 {other} 与占比的条件顺序 {condition_order} 不一致：标签顺序必须统一")
    if list(map(int, trans["labels"])) != list(map(int, proportion_labels)):
        _fail(
            f"转移矩阵的簇顺序 {trans['labels'].tolist()} 与占比的簇顺序 {proportion_labels.tolist()} 不一致："
            "两者必须都来自同一份 cluster_labels.csv"
        )

    bin_frame_idx, bin_day, bin_time_hours = _bin_labels(n_bins)
    bin_analysis_time = bin_frame_idx.astype(np.float64) + BIN_OFFSET
    common = {
        "cluster_labels": np.asarray(proportion_labels, dtype=np.int64),
        "conditions": np.asarray(condition_order),
        "bin_frame_idx": bin_frame_idx,
        "bin_analysis_time": bin_analysis_time,
        "bin_time_hours": np.asarray(bin_time_hours, dtype=np.float64),
        "bin_day": np.asarray(bin_day),
        "time_intervals": edges,
    }

    outdir = Path(cfg.OUTPUT_ROOT) / _ANALYSIS_SUBDIR
    outdir.mkdir(parents=True, exist_ok=True)
    for stale in (_PROPORTIONS_FILE, _TRANSITIONS_FILE, _TRAJECTORIES_FILE, _AFFINITY_FILE, _QC_FILE):
        (outdir / stale).unlink(missing_ok=True)

    np.savez(outdir / _PROPORTIONS_FILE, proportions=proportions, **common)
    np.savez(
        outdir / _TRANSITIONS_FILE,
        inter=trans["inter"],
        self_transitions=trans["self"],
        probs=trans["probs"],
        counts=trans["counts"],
        counts_by_condition=trans["counts_by_condition"],
        inter_by_condition=trans["inter_by_condition"],
        self_by_condition=trans["self_by_condition"],
        pseudocounts=np.asarray(PSEUDOCOUNTS, dtype=np.float64),
        **common,
    )
    np.savez(
        outdir / _TRAJECTORIES_FILE,
        trajectories=trajectories["trajectories"],
        spot_trajectories=(
            np.full_like(trajectories["trajectories"], np.nan)
            if trajectories["spot_trajectories"] is None
            else trajectories["spot_trajectories"]
        ),
        path=np.asarray(trajectories["path"]),
        density_threshold_sigma=np.asarray(trajectories["thresh_density_sigma"], dtype=np.float64),
        **common,
    )
    np.savez(
        outdir / _AFFINITY_FILE,
        affinity=affinity,
        order=np.asarray(clustered["order"], dtype=np.int64),
        linkage=np.asarray(clustered["linkage"]),
        metric=np.asarray(clustered["metric"]),
        **common,
    )
    t_written = time.perf_counter()

    warnings_all = list(trajectories["warnings"])
    if trajectories["degenerate"]:
        warnings_all.append(
            f"密度阈值法退化（{'; '.join(trajectories['degeneracy']['reasons'])}），"
            f"轨迹改用 {trajectories['path']}，详见 QC 的 trajectories.degeneracy"
        )
    if affinity_info["fallback"]:
        warnings_all.append(f"轨迹亲和退化：{affinity_info['fallback']}")
    if clustered["skipped"]:
        warnings_all.append(f"轨迹层次聚类跳过：{clustered['skipped']}")
    if clustered["fallback"]:
        warnings_all.append(f"轨迹层次聚类退化：{clustered['fallback']}（已退回原始条件顺序）")

    qc = {
        "time_axis": {
            "mode": "frame_idx 箱中心（frame_idx + 0.5）",
            "why": (
                "分析时间必须用 frame_idx（真实天数差 3 与 2 无法用单一 d_Frame 表达）；"
                "而 SPOT 的时间选择是两端闭区间，直接用 frame_idx 会让采样点 1、2 同时落进相邻两个箱，"
                "所以平移到箱中心"
            ),
            "bin_frame_idx": bin_frame_idx.tolist(),
            "bin_analysis_time": bin_analysis_time.tolist(),
            "intervals": edges.tolist(),
            "bin_day": bin_day,
            "bin_time_hours": bin_time_hours,
            "n_bins": int(n_bins),
            "discrete_days_note": (
                "时间轴是 3 个离散采样点（day0/day3/day5），不是连续视频；且两步宽度**不等**："
                "day0→day3 是 72 h、day3→day5 是 48 h，而本模块把两步都当作**一步**转移"
                "（SPOT 的 time_interval=1、轨迹的一步），所以转移矩阵与轨迹里混着两种真实时长，"
                "不能换算成单位时间的速率；motion 特征被有意置空，因此不支持任何速度类结论"
            ),
        },
        "label_ordering": {
            "cluster_labels": [int(label) for label in proportion_labels],
            "cluster_source": "cluster_labels.csv 的 cluster 列（Stage5 按簇内平均 eccentricity 升序编号，1-based）",
            "conditions": condition_order,
            "condition_source": "fov（Stage5/6 取实例所属视野）",
            "bin_day": bin_day,
            "bin_frame_idx": bin_frame_idx.tolist(),
            "bin_analysis_time": bin_analysis_time.tolist(),
            "bin_time_hours": bin_time_hours,
        },
        "inputs": {
            "tracks": str(Path(cfg.OUTPUT_ROOT) / _ANALYSIS_SUBDIR / _TRACKS_FILE),
            "cluster_labels": str(Path(cfg.OUTPUT_ROOT) / _ANALYSIS_SUBDIR / _LABELS_FILE),
            "umap": str(Path(cfg.OUTPUT_ROOT) / _ANALYSIS_SUBDIR / _UMAP_FILE),
            **alignment_qc,
        },
        "subsets": {
            "proportions": {
                "n_instances": len(merged),
                "track_length_filter": "none（每个实例都是某一帧里的真实观测，长度 1 的 track 也计入占比）",
            },
            "transitions": {
                **trans["qc"],
                "n_instances": len(merged),
            },
            "trajectories": {
                "n_instances": len(merged),
                "track_length_filter": "none（轨迹用全部有簇标签的实例的嵌入坐标）",
            },
            "affinity": {
                "n_conditions": len(condition_order),
                "note": "亲和矩阵来自 trajectories，因此同样用全部有簇标签的实例",
            },
        },
        "trajectories": {
            "path": trajectories["path"],
            "degenerate": trajectories["degenerate"],
            "degeneracy": trajectories["degeneracy"],
            "shipped_finiteness": trajectories["shipped_finiteness"],
            "density_threshold_sigma": trajectories["thresh_density_sigma"],
            "fallback_note": (
                "退化时用**箱内**簇质心按该 (条件, 时间箱) 的簇占比加权（恒等于该箱全部实例的嵌入均值）；"
                "没有簇标签时退到 bin_centroid"
            ),
        },
        "transitions": {
            "pseudocounts": float(PSEUDOCOUNTS),
            "orientation": "inter / probs 都是 (来源, 目标)，行 = 来源、行和 = 1",
            "inter_graph_data": trans["inter_graph_data"],
            "inter_graph_data_by_condition": trans["inter_graph_data_by_condition"],
        },
        "affinity": {
            "affinity": AFFINITY,
            "use_std_denom": True,
            "fallback": affinity_info["fallback"],
            "hcluster": {
                "order": clustered["order"],
                "conditions": clustered["conditions"],
                "linkage": clustered["linkage"],
                "metric": clustered["metric"],
                "skipped": clustered["skipped"],
                "fallback": clustered["fallback"],
            },
        },
        "warnings": warnings_all,
        "timing_seconds": {
            "load_inputs": round(t_loaded - t_start, 3),
            "proportions_and_transitions": round(t_transitions - t_loaded, 3),
            "trajectories": round(t_trajectories - t_transitions, 3),
            "affinity_and_hcluster": round(t_affinity - t_trajectories, 3),
            "artifacts": round(t_written - t_affinity, 3),
            "total": round(t_written - t_start, 3),
        },
    }
    (outdir / _QC_FILE).write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Stage6b 完成：{len(merged)} 个实例 → {len(proportion_labels)} 个簇 × {n_bins} 个采样点 × "
        f"{len(condition_order)} 个视野；轨迹走 {trajectories['path']}"
        f"（{'退化' if trajectories['degenerate'] else '正常'}），"
        f"耗时 {qc['timing_seconds']['total']}s → {outdir}"
    )
    for message in warnings_all:
        print(f"警告：{message}")
