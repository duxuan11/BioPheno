"""Stage 6b（时间统计：簇占比、转移矩阵、表型轨迹、轨迹亲和）的测试：全部合成数据。

两个契约测试（占比逐 (条件, 时间箱) 归一、转移矩阵行随机）取自任务说明。其余测试盯住
实现里最容易悄悄错的地方——这些地方上游 SPOT 的源码本身就把语义写得含糊：

- **矩阵朝向**：`compute_phenotype_cluster_transitions_from_label_trajectories` 的返回矩阵
  是 (来源, 目标)，行=来源、行和为 1（源码里 `label_seq_tt[::-1]` + `counts[0, xxx+1]` 才让它
  变回正向，注释里那句 "this is the bug" 说的是反过来的那次翻转）。用**非对称**轨迹钉死朝向：
  只有 1→1、1→2、2→2 时，(来源, 目标) 与它的转置完全不同。
- **簇 id 必须 0-based 连续**：SPOT 直接拿标签值当矩阵下标（`transition_counts[label, label]`），
  所以 Stage5 的 1-based 簇号必须在这里重映射，否则不是静默错，而是 IndexError。
- **时间箱边界**：SPOT 的时间选择是**两端闭区间** `t >= lo and t <= hi`，所以落在箱边界上的
  样本会被相邻两个箱各算一次。占比必须拒绝这种输入（而不是让人误以为时间序列是真的）。
- **长度 1 的 track**：它在某一帧里是真实存在的实例，必须计入占比；但 SPOT 的转移统计只
  统计相邻帧的对，长度 1 的 track 不产生任何转移。
- **表型轨迹退化**：只有 3 个时间点时 `compute_phenotypic_trajectory` 的密度阈值可能选空
  （空选择 → 全 nan）或选成一个点（轨迹塌缩）。必须退到"簇质心按占比加权"的轨迹并记录
  实际走了哪条路，不能静默落一份空轨迹。
- **hcluster 会开图**：`hcluster_phenotypic_trajectory_affinity_matrix` 内部用
  `sns.clustermap` 建 figure 且从不关闭，必须由本模块关掉（否则长跑/测试里堆 figure）。
  seaborn 自己报错时也不允许把整条 Stage6b 拖挂，必须退回原始顺序并记录原因。
- **先验 vs 观测**：`pseudocounts=1` 让零观测的行也变成均匀分布（`1/(K-1)`），所以
  `inter_graph_data` 的每条边都带原始计数 `count` 与 `prior_only`，每个节点带 `n_obs_out`；
  逐条件的原始计数落进 `counts_by_condition`，否则“这条边是观测到的还是平滑出来的”无法分辨。
- **段数 vs track 数**：SPOT 按 `(fov, track_id)` 并允许帧号断档，段数可能大于 track 数，
  所以 QC 里 `n_spot_segments` / `n_single_instance_segments` / `n_tracks` 三个数分开记。
- **退化逐条件可见**：一个条件塌缩、或某一步是 nan，不允许被 `nanmax` / `nansum` 抹平。
- `run_stage6b` 的四份产物都必须自带标签顺序（簇 id、fov 名、bin→day 映射、分析时间偏移），
  QC 里必须同时记“落盘轨迹的有限性”与被拒的 SPOT 密度输出的退化数字；绘图阶段不许再猜任何顺序。
"""

import json

import matplotlib

# 测试进程自己的后端设置（与 temporal_stats 模块一致：不强制覆盖已加载的后端）。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biopheno import config as cfg
from biopheno import temporal_stats as tstat

# 合成数据规模：3 个视野 × 3 天 × 30 个实例。每箱里的主簇随天推进（24/3/3 → 3/24/3 → 3/3/24），
# 这样表型轨迹一定是"从一个密集团移向下一个"，密度阈值法不会选空。
_FOVS: tuple[str, ...] = ("A1", "A2", "A3")
_PER_BIN: int = 30
_N_DOMINANT: int = 24
_DAYS: tuple[str, ...] = ("day0", "day3", "day5")
# 三个簇的嵌入质心（离得足够远，密度阈值只会选中主簇）
_CENTROIDS: dict[int, tuple[float, float]] = {1: (0.0, 0.0), 2: (8.0, 0.0), 3: (16.0, 0.0)}


# ---------------------------------------------------------------- 合成数据工具


def _cluster_for(frame: int, k: int) -> int:
    """返回第 frame 天第 k 个实例的表型簇（1-based）：主簇 = frame+1，其余两个簇各 3 个。"""
    dominant = frame + 1
    if k < _N_DOMINANT:
        return dominant
    return 1 + (frame + 1 + (k - _N_DOMINANT)) % 3


def _fake_analysis(root, *, singleton_track: bool = False) -> dict:
    """在 root/analysis/ 下造 Stage5 + Stage6 的三份输入，返回它们的 DataFrame。

    - `tracks.csv`：每个视野 30 条 track，每条贯穿 0/1/2 三帧（`spot_uid = fov_track_frame`）；
    - `cluster_labels.csv`：`cluster` 为 1-based（Stage5 按 eccentricity 升序编号）；
    - `umap_2d.csv`：三个簇各一个紧凑团，位置见 `_CENTROIDS`（+ 确定性噪声）。

    `singleton_track=True` 时额外加一条**只出现在 day0** 的实例（长度 1 的 track），
    用来验证它仍被算进占比、但不产生任何转移。
    """
    rng = np.random.default_rng(0)
    tracks, labels, umap = [], [], []
    for fov in _FOVS:
        for frame, day in enumerate(_DAYS):
            for k in range(_PER_BIN):
                uid = f"{fov}_{k}_{frame}"
                cluster = _cluster_for(frame, k)
                cx, cy = _CENTROIDS[cluster]
                tracks.append(
                    {
                        "instance_uid": uid,
                        "spot_uid": f"{fov}_{k}_{frame}",
                        "track_id": k,
                        "fov": fov,
                        "frame_idx": frame,
                        "time_hours": cfg.TIME_HOURS[day],
                        "day": day,
                    }
                )
                labels.append(
                    {
                        "instance_uid": uid,
                        "day": day,
                        "fov": fov,
                        "frame_idx": frame,
                        "time_hours": cfg.TIME_HOURS[day],
                        "cluster": cluster,
                        "cluster_raw": cluster - 1,
                    }
                )
                umap.append(
                    {
                        "instance_uid": uid,
                        "umap_1": cx + rng.normal(0.0, 0.4),
                        "umap_2": cy + rng.normal(0.0, 0.4),
                    }
                )
        if singleton_track:
            uid = f"{fov}_{_PER_BIN}_0"
            tracks.append(
                {
                    "instance_uid": uid,
                    "spot_uid": uid,
                    "track_id": _PER_BIN,
                    "fov": fov,
                    "frame_idx": 0,
                    "time_hours": cfg.TIME_HOURS["day0"],
                    "day": "day0",
                }
            )
            labels.append(
                {
                    "instance_uid": uid,
                    "day": "day0",
                    "fov": fov,
                    "frame_idx": 0,
                    "time_hours": cfg.TIME_HOURS["day0"],
                    "cluster": 2,
                    "cluster_raw": 1,
                }
            )
            umap.append({"instance_uid": uid, "umap_1": 8.0, "umap_2": 0.0})

    outdir = root / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    frames = {
        "tracks": pd.DataFrame(tracks),
        "labels": pd.DataFrame(labels),
        "umap": pd.DataFrame(umap),
    }
    frames["tracks"].to_csv(outdir / "tracks.csv", index=False)
    frames["labels"].to_csv(outdir / "cluster_labels.csv", index=False)
    frames["umap"].to_csv(outdir / "umap_2d.csv", index=False)
    return frames


def _two_condition_bins() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2 个条件 × 2 个时间箱 × 3 个簇的占比输入（每个 (条件, 箱) 都非空、构成各不相同）。"""
    clusters = np.array([1, 1, 2, 3, 3, 3, 2, 2, 2, 1])
    times = np.array([0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5, 1.5])
    conditions = np.array(["A1"] * 5 + ["A2"] * 5)
    return clusters, times, conditions


def _tracks_with_sequences(sequences: dict[str, list[int]]) -> tuple[pd.DataFrame, np.ndarray]:
    """把 {fov: [标签序列]} 变成 (tracks 表, 与之逐行对齐的簇标签数组)。"""
    rows, clusters = [], []
    for fov, seq in sequences.items():
        for frame, cluster in enumerate(seq):
            cluster = int(cluster)
            rows.append(
                {
                    "instance_uid": f"{fov}_0_{frame}",
                    "spot_uid": f"{fov}_0_{frame}",
                    "track_id": 0,
                    "fov": fov,
                    "frame_idx": frame,
                    "time_hours": float(frame * 72),
                    "day": _DAYS[frame],
                }
            )
            clusters.append(cluster)
    return pd.DataFrame(rows), np.array(clusters)


# ---------------------------------------------------------------- 占比


def test_proportions_sum_to_one_per_time_and_condition():
    clusters = np.array([1, 1, 2, 2, 1, 2])
    times = np.array([0.0, 0.0, 0.0, 72.0, 72.0, 72.0])
    conds = np.array(["A1"] * 6)
    props, uniq_labels, uniq_cond = tstat.cluster_proportions(clusters, times, conds, time_intervals=[0.0, 72.0])
    assert props.shape[0] == 1
    assert np.allclose(props.sum(axis=-1), 1.0, atol=1e-6)
    assert list(uniq_labels) == [1, 2] and list(uniq_cond) == ["A1"]


def test_proportions_axes_are_condition_bin_cluster():
    clusters, times, conditions = _two_condition_bins()
    props, uniq_labels, uniq_cond = tstat.cluster_proportions(
        clusters, times, conditions, time_intervals=[0.0, 1.0, 2.0]
    )
    # 轴顺序 (条件, 时间箱, 簇)：搞反会让堆叠柱状图把簇画成时间箱
    assert props.shape == (2, 2, 3)
    assert list(uniq_cond) == ["A1", "A2"]
    assert list(uniq_labels) == [1, 2, 3]
    # A1 的 day0 箱 = 簇 [1, 1, 2]；A1 的 day3 箱 = 簇 [3, 3]
    assert np.allclose(props[0, 0], [2 / 3, 1 / 3, 0.0])
    assert np.allclose(props[0, 1], [0.0, 0.0, 1.0])
    # A2 的 day0 箱 = 簇 [3, 2]；A2 的 day3 箱 = 簇 [2, 2, 1]
    assert np.allclose(props[1, 0], [0.0, 0.5, 0.5])
    assert np.allclose(props[1, 1], [1 / 3, 2 / 3, 0.0])
    assert np.allclose(props.sum(axis=-1), 1.0, atol=1e-6)


def test_proportions_reject_objects_exactly_on_interior_bin_edge():
    # SPOT 的选择是两端闭区间（t >= lo 且 t <= hi），落在内部边界上的样本会被两个箱各算一次
    clusters = np.array([1, 2, 1, 2])
    times = np.array([0.0, 72.0, 72.0, 120.0])
    conditions = np.array(["A1"] * 4)
    with pytest.raises(ValueError, match="边界"):
        tstat.cluster_proportions(clusters, times, conditions, time_intervals=[0.0, 72.0, 120.0])


def test_proportions_empty_bin_is_nan_and_still_row_stochastic_elsewhere():
    # A2 只出现在第二个时间箱：第一个箱的占比必须是 nan（缺数据），不能悄悄填 0
    clusters = np.array([1, 2, 2, 1])
    times = np.array([0.5, 0.5, 1.5, 1.5])
    conditions = np.array(["A1", "A1", "A2", "A2"])
    props, _, uniq_cond = tstat.cluster_proportions(clusters, times, conditions, time_intervals=[0.0, 1.0, 2.0])
    assert list(uniq_cond) == ["A1", "A2"]
    assert np.all(np.isnan(props[0, 1]))
    assert np.allclose(props[1, 1], [0.5, 0.5])


# ---------------------------------------------------------------- 转移矩阵


def test_transitions_are_row_stochastic():
    tracks = pd.DataFrame(
        {
            "spot_uid": ["A1_0_0", "A1_0_1", "A2_0_0", "A2_0_1"],
            "fov": ["A1", "A1", "A2", "A2"],
            "track_id": [0, 0, 0, 0],
            "frame_idx": [0, 1, 0, 1],
        }
    )
    clusters = np.array([1, 2, 1, 1])
    out = tstat.transitions(tracks, clusters)
    assert np.allclose(out["inter"].sum(axis=1), 1.0, atol=1e-6)


def test_transitions_matrix_is_source_by_target():
    # 只观测到 1→1、1→2、2→2（A1/A2 各一条两帧 track，A3 一条 2→2）：非对称，能钉死朝向
    tracks, clusters = _tracks_with_sequences({"A1": [1, 1], "A2": [1, 2], "A3": [2, 2]})
    out = tstat.transitions(tracks, clusters)
    assert list(out["labels"]) == [1, 2]
    assert list(out["conditions"]) == ["A1", "A2", "A3"]
    # 平滑（拉普拉斯 +1）后的 (来源, 目标) 计数矩阵：[[2, 2], [1, 2]] → 行归一
    assert np.allclose(out["counts"], [[1.0, 1.0], [0.0, 1.0]])  # 未平滑的原始计数
    assert np.allclose(out["probs"], [[0.5, 0.5], [1 / 3, 2 / 3]])
    # self = 行内自转移概率；inter = 去掉对角后按行重归一（= "既然离开了这个簇，去哪"）
    assert np.allclose(out["self"], [0.5, 2 / 3])
    assert np.allclose(out["inter"], [[0.0, 1.0], [1.0, 0.0]])
    # 转置就完全不同（probs 转置后行和不再是 1），所以这组断言能钉死朝向
    assert not np.allclose(out["probs"], out["probs"].T)


def test_transitions_graph_data_is_json_serializable_and_keeps_asymmetric_edges():
    tracks, clusters = _tracks_with_sequences({"A1": [1, 1], "A2": [1, 2], "A3": [2, 2]})
    out = tstat.transitions(tracks, clusters)
    graph = out["inter_graph_data"]
    assert graph["nodes"] == ["1", "2"]
    json.dumps(graph, ensure_ascii=False)  # 落 QC 前必须能序列化
    edges = {(e["source"], e["target"]): e["probability"] for e in graph["edges"]}
    # 边表必须与 inter 一致（对角不进边表，自转移单列在 graph["self"]）
    assert edges[(1, 2)] == pytest.approx(out["inter"][0, 1])
    assert edges[(2, 1)] == pytest.approx(out["inter"][1, 0])
    assert (1, 1) not in edges
    # 原始计数与先验标记：簇 2 只有 2→2 一次观测，所以 2→1 只是平滑出来的先验
    by_pair = {(e["source"], e["target"]): e for e in graph["edges"]}
    assert by_pair[(1, 2)]["count"] == 1.0
    assert by_pair[(1, 2)]["prior_only"] is False
    assert by_pair[(2, 1)]["count"] == 0.0
    assert by_pair[(2, 1)]["prior_only"] is True
    assert graph["n_obs_out"] == [2, 1]


def test_transitions_flags_prior_only_edges_and_zero_observation_rows():
    # 簇 3 只在最后一次采样出现：它作为来源从未被观测到，平滑后整行变成均匀的 1/(K-1)
    tracks, clusters = _tracks_with_sequences({"A1": [1, 2, 2], "A2": [1, 2, 3]})
    out = tstat.transitions(tracks, clusters)
    assert list(out["labels"]) == [1, 2, 3]
    # 未平滑的原始计数 (来源, 目标)：行 3 全是 0
    assert np.allclose(out["counts"], [[0.0, 2.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    graph = out["inter_graph_data"]
    json.dumps(graph, ensure_ascii=False)
    assert graph["n_obs_out"] == [2, 2, 0]
    by_pair = {(e["source"], e["target"]): e for e in graph["edges"]}
    # 观测充足的行：计数精确、不是先验；概率仍是平滑后的 3/4（另一个非对角格也拿到 1 个先验）
    assert by_pair[(1, 2)]["count"] == 2.0
    assert by_pair[(1, 2)]["prior_only"] is False
    assert by_pair[(1, 2)]["probability"] == pytest.approx(3 / 4)
    # 零观测行：概率是均匀的 1/(K-1)，但每条边 count == 0 且带 prior_only 标记
    from_three = [by_pair[(3, target)] for target in (1, 2)]
    assert [edge["probability"] for edge in from_three] == pytest.approx([1 / 2, 1 / 2])
    assert all(edge["count"] == 0.0 for edge in from_three)
    assert all(edge["prior_only"] is True for edge in from_three)


def test_transitions_per_condition_counts_mark_all_prior_condition():
    # A2 只有一条长度 1 的 track：该条件没有任何相邻采样点对，矩阵完全来自先验
    tracks, clusters = _tracks_with_sequences({"A1": [1, 2], "A2": [2]})
    out = tstat.transitions(tracks, clusters)
    assert list(out["conditions"]) == ["A1", "A2"]
    counts_by_condition = out["counts_by_condition"]
    assert counts_by_condition.shape == (2, 2, 2)
    assert np.allclose(counts_by_condition[0], [[0.0, 1.0], [0.0, 0.0]])
    assert np.allclose(counts_by_condition[1], 0.0)
    # A2 的矩阵可以**从产物本身**识别为"全部来自先验"：计数全 0、无观测出边、每条边 prior_only
    graph_a2 = out["inter_graph_data_by_condition"]["A2"]
    assert graph_a2["n_obs_out"] == [0, 0]
    assert graph_a2["edges"]
    assert all(edge["prior_only"] is True and edge["count"] == 0.0 for edge in graph_a2["edges"])
    # 对照：A1 有一条真实观测到的 1→2
    graph_a1 = out["inter_graph_data_by_condition"]["A1"]
    by_pair = {(e["source"], e["target"]): e for e in graph_a1["edges"]}
    assert by_pair[(1, 2)]["count"] == 1.0
    assert by_pair[(1, 2)]["prior_only"] is False


def test_transitions_count_singleton_tracks_but_they_produce_no_pairs():
    tracks, clusters = _tracks_with_sequences({"A1": [1, 2]})
    singleton = tracks.iloc[[1]].copy()
    singleton["instance_uid"] = "A2_0_0"
    singleton["spot_uid"] = "A2_0_0"
    singleton["fov"] = "A2"
    singleton["frame_idx"] = 0
    tracks = pd.concat([tracks, singleton], ignore_index=True)
    clusters = np.append(clusters, 2)

    out = tstat.transitions(tracks, clusters)
    assert out["qc"]["n_instances"] == 3
    # 长度 1 的段不进转移统计，但被计数（名字里写明是 SPOT 段，不是 track）
    assert out["qc"]["n_single_instance_segments"] == 1
    assert out["qc"]["n_spot_segments"] == 2
    assert out["qc"]["n_tracks"] == 2
    assert out["qc"]["n_transition_pairs"] == 1
    assert np.allclose(out["inter"].sum(axis=1), 1.0, atol=1e-6)


def test_transitions_qc_counts_segments_and_tracks_separately():
    # 同一 track 的帧号断档（0 → 2）会被 SPOT 切成两段：段数 2，但 track 数仍是 1
    tracks = pd.DataFrame(
        {
            "spot_uid": ["A1_0_0", "A1_0_2"],
            "fov": ["A1", "A1"],
            "track_id": [0, 0],
            "frame_idx": [0, 2],
        }
    )
    out = tstat.transitions(tracks, np.array([1, 2]))
    assert out["qc"]["n_tracks"] == 1
    assert out["qc"]["n_spot_segments"] == 2
    assert out["qc"]["n_single_instance_segments"] == 2
    assert out["qc"]["n_transition_pairs"] == 0
    assert "段" in out["qc"]["segment_note"]


def test_transitions_with_single_cluster_does_not_crash():
    tracks, clusters = _tracks_with_sequences({"A1": [1, 1]})
    out = tstat.transitions(tracks, clusters)
    assert out["inter"].shape == (1, 1)
    assert np.all(np.isnan(out["inter"]))  # 只有一个簇时没有"簇间"转移可言
    assert np.allclose(out["self"], [1.0])


# ---------------------------------------------------------------- 表型轨迹


def _fake_temporal_inputs(root):
    frames = _fake_analysis(root)
    merged = frames["labels"].merge(frames["umap"], on="instance_uid", how="inner")
    embedding = merged[["umap_1", "umap_2"]].to_numpy(dtype=float)
    analysis_time = merged["frame_idx"].to_numpy(dtype=float) + 0.5
    conditions = merged["fov"].to_numpy(dtype=object)
    clusters = merged["cluster"].to_numpy(dtype=np.int64)
    return frames, embedding, analysis_time, conditions, clusters


def test_phenotype_trajectories_use_density_path_on_separated_blobs(tmp_path):
    _, embedding, times, conditions, _ = _fake_temporal_inputs(tmp_path)
    out = tstat.phenotype_trajectories(embedding, times, conditions, time_intervals=[0.0, 1.0, 2.0, 3.0])
    assert out["path"] == "spot_density"
    assert out["trajectories"].shape == (3, 3, 2)
    assert np.all(np.isfinite(out["trajectories"]))
    assert list(out["conditions"]) == ["A1", "A2", "A3"]
    # 主簇每天右移 8，所以轨迹必须单调右移（密度法选的是主簇的位置）
    assert np.all(np.diff(out["trajectories"][:, :, 0], axis=1) > 1.0)


def test_phenotype_trajectories_fall_back_to_cluster_centroid_weighting(tmp_path, monkeypatch):
    _, embedding, times, conditions, clusters = _fake_temporal_inputs(tmp_path)

    def _empty_selection(*args, **kwargs):
        return np.full((3, 3, 2), np.nan), [[], [], []]

    monkeypatch.setattr(tstat.sam, "compute_phenotypic_trajectory", _empty_selection)
    out = tstat.phenotype_trajectories(
        embedding, times, conditions, time_intervals=[0.0, 1.0, 2.0, 3.0], clusters=clusters
    )
    assert out["path"] == "cluster_centroid_weighted"
    assert out["degenerate"] is True
    traj = out["trajectories"]
    assert np.all(np.isfinite(traj))
    # 被拒绝的 SPOT 输出（全 nan）在 degeneracy 里，落盘轨迹的有限性另记一份
    assert out["degeneracy"]["finite_bins"] == [[False, False, False]] * 3
    assert out["shipped_finiteness"]["all_finite"] is True
    assert out["shipped_finiteness"]["finite_bins"] == [[True, True, True]] * 3
    assert out["shipped_finiteness"]["n_finite_bins"] == [3, 3, 3]
    # 簇质心按占比加权 == 该 (条件, 箱) 的实例均值（代数恒等），所以直接和逐箱均值比
    for cond_i, cond in enumerate(out["conditions"]):
        for bin_i, analysis_time in enumerate(out["bin_centers"]):
            select = (conditions == cond) & np.isclose(times, analysis_time)
            assert np.allclose(traj[cond_i, bin_i], embedding[select].mean(axis=0), atol=1e-9)


def test_fallback_trajectories_rejects_mismatched_cluster_order(monkeypatch):
    times = np.array([0.5, 0.5, 1.5, 1.5])
    conditions = np.array(["A1"] * 4)
    clusters = np.array([1, 2, 1, 2])
    embedding = np.zeros((4, 2))

    def _reordered(clusters, time_hours, conditions, time_intervals):
        return np.ones((1, 2, 2)), np.array([2, 1]), np.array(["A1"])

    monkeypatch.setattr(tstat, "cluster_proportions", _reordered)
    with pytest.raises(RuntimeError, match="簇顺序"):
        tstat._fallback_trajectories(
            embedding, times, conditions, np.array([0.0, 1.0, 2.0]), np.array(["A1"]), clusters
        )


def test_degeneracy_makes_one_collapsed_condition_visible():
    trajectories = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # A1 塌缩成一个点
            [[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]],  # A2 正常移动
        ]
    )
    info = tstat._degeneracy(trajectories, None, conditions=np.array(["A1", "A2"]))
    assert info["degenerate"] is True
    # 逐个条件记录，不再用 nanmax 把"只有一个条件塌缩"藏成"全部正常"
    assert info["collapsed"] == [True, False]
    assert info["collapsed_conditions"] == ["A1"]
    assert any("A1" in reason for reason in info["reasons"])
    assert info["path_length"] == pytest.approx([0.0, 6.0])
    assert info["n_nan_steps"] == [0, 0]


def test_degeneracy_makes_nan_step_visible():
    trajectories = np.array(
        [
            [[0.0, 0.0], [np.nan, np.nan], [6.0, 0.0]],  # A1 中间箱选空
            [[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]],
        ]
    )
    info = tstat._degeneracy(trajectories, None, conditions=np.array(["A1", "A2"]))
    assert info["degenerate"] is True
    # nansum 会把 nan 步当作 0，所以必须同时给出 nan 步数与逐箱有限性
    assert info["n_nan_steps"] == [2, 0]
    assert info["step_lengths"][0] == [None, None]
    assert info["finite_bins"][0] == [True, False, True]
    assert any("nan" in reason for reason in info["reasons"])


def test_phenotype_trajectories_falls_back_to_bin_centroid_without_clusters(tmp_path, monkeypatch):
    _, embedding, times, conditions, _ = _fake_temporal_inputs(tmp_path)

    def _empty_selection(*args, **kwargs):
        return np.full((3, 3, 2), np.nan), [[], [], []]

    monkeypatch.setattr(tstat.sam, "compute_phenotypic_trajectory", _empty_selection)
    out = tstat.phenotype_trajectories(embedding, times, conditions, time_intervals=[0.0, 1.0, 2.0, 3.0])
    assert out["path"] == "bin_centroid"
    assert np.all(np.isfinite(out["trajectories"]))


def test_phenotype_trajectories_falls_back_when_trajectory_collapses(tmp_path, monkeypatch):
    _, embedding, times, conditions, _ = _fake_temporal_inputs(tmp_path)

    def _collapsed(*args, **kwargs):
        return np.zeros((3, 3, 2)), [[], [], []]

    monkeypatch.setattr(tstat.sam, "compute_phenotypic_trajectory", _collapsed)
    out = tstat.phenotype_trajectories(embedding, times, conditions, time_intervals=[0.0, 1.0, 2.0, 3.0])
    assert out["degenerate"] is True
    assert out["path"] == "bin_centroid"
    assert np.ptp(out["trajectories"][:, :, 0]) > 1.0  # 退化的塌缩轨迹被替换掉了


# ---------------------------------------------------------------- 轨迹亲和与层次聚类


def test_trajectory_affinity_is_symmetric_with_unit_diagonal(tmp_path):
    _, embedding, times, conditions, _ = _fake_temporal_inputs(tmp_path)
    out = tstat.phenotype_trajectories(embedding, times, conditions, time_intervals=[0.0, 1.0, 2.0, 3.0])
    affinity, labels = tstat.trajectory_affinity(out)
    assert list(labels) == ["A1", "A2", "A3"]
    assert affinity.shape == (3, 3)
    assert np.allclose(affinity, affinity.T)
    assert np.allclose(np.diag(affinity), 1.0)
    assert np.all((affinity >= 0.0) & (affinity <= 1.0))


def test_trajectory_affinity_surfaces_fallback_to_callers():
    # 三条轨迹完全相同 → 两两 DTW 距离全 0 → std = 0 → exp(-d / std) 无定义，只能退到全 1
    identical = {"trajectories": np.zeros((3, 3, 2)), "conditions": ["A1", "A2", "A3"]}
    affinity, labels, info = tstat.trajectory_affinity(identical, return_meta=True)
    assert info["fallback"] is not None
    assert np.allclose(affinity, np.ones((3, 3)))
    assert list(labels) == ["A1", "A2", "A3"]
    # 默认调用仍是两元组（不破坏既有调用方），信息通过 return_meta 显式索取
    affinity_only, labels_only = tstat.trajectory_affinity(identical)
    assert np.allclose(affinity_only, affinity)
    assert labels_only == labels


def test_trajectory_clusters_falls_back_when_hcluster_raises(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("seaborn 挂了")

    monkeypatch.setattr(tstat.sam, "hcluster_phenotypic_trajectory_affinity_matrix", _boom)
    affinity = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])
    plt.close("all")
    clustered = tstat.trajectory_clusters(affinity, ["A1", "A2", "A3"])
    # 一张聚类图失败不许把整条 Stage6b 拖挂：退到原始条件顺序并记录原因
    assert clustered["order"] == [0, 1, 2]
    assert clustered["conditions"] == ["A1", "A2", "A3"]
    assert clustered["skipped"] is None
    assert "seaborn 挂了" in clustered["fallback"]
    assert plt.get_fignums() == []


def test_trajectory_clusters_returns_permutation_and_closes_figures(tmp_path):
    _, embedding, times, conditions, _ = _fake_temporal_inputs(tmp_path)
    out = tstat.phenotype_trajectories(embedding, times, conditions, time_intervals=[0.0, 1.0, 2.0, 3.0])
    affinity, labels = tstat.trajectory_affinity(out)
    plt.close("all")
    clustered = tstat.trajectory_clusters(affinity, labels)
    assert sorted(clustered["order"]) == [0, 1, 2]
    assert sorted(clustered["conditions"]) == ["A1", "A2", "A3"]
    # SPOT 的 hcluster 内部 sns.clustermap 会留下 figure，必须由本模块关掉
    assert plt.get_fignums() == []


# ---------------------------------------------------------------- Stage 6b 落盘


def test_run_stage6b_writes_four_artifacts_with_their_label_ordering(tmp_path, monkeypatch):
    _fake_analysis(tmp_path)
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)

    tstat.run_stage6b()

    outdir = tmp_path / "analysis"
    for name in ("proportions.npz", "transitions.npz", "trajectories.npz", "affinity.npz"):
        assert (outdir / name).exists(), name
    qc = json.loads((outdir / "stage6b_qc.json").read_text(encoding="utf-8"))
    assert qc["time_axis"]["bin_day"] == ["day0", "day3", "day5"]
    assert qc["time_axis"]["bin_time_hours"] == [0.0, 72.0, 120.0]
    assert qc["label_ordering"]["cluster_labels"] == [1, 2, 3]
    assert qc["label_ordering"]["conditions"] == ["A1", "A2", "A3"]

    with np.load(outdir / "proportions.npz", allow_pickle=False) as props:
        assert props["proportions"].shape == (3, 3, 3)
        assert np.allclose(props["proportions"].sum(axis=-1), 1.0, atol=1e-6)
        assert props["bin_day"].tolist() == ["day0", "day3", "day5"]
        assert props["cluster_labels"].tolist() == [1, 2, 3]
        assert props["conditions"].tolist() == ["A1", "A2", "A3"]
    with np.load(outdir / "transitions.npz", allow_pickle=False) as trans:
        assert np.allclose(trans["inter"].sum(axis=1), 1.0, atol=1e-6)
        assert trans["cluster_labels"].tolist() == [1, 2, 3]
        assert trans["bin_day"].tolist() == ["day0", "day3", "day5"]
        assert trans["inter_by_condition"].shape == (3, 3, 3)
        assert trans["counts_by_condition"].shape == (3, 3, 3)
        # 主簇每天换一个，所以对角（自转移）应当很低：轨迹是"往后走"而不是"停在原地"
        assert np.all(np.diag(trans["probs"]) < 0.5)
    with np.load(outdir / "trajectories.npz", allow_pickle=False) as traj:
        assert traj["trajectories"].shape == (3, 3, 2)
        assert np.all(np.isfinite(traj["trajectories"]))
        assert str(traj["path"]) in {"spot_density", "cluster_centroid_weighted", "bin_centroid"}
        assert traj["conditions"].tolist() == ["A1", "A2", "A3"]
    with np.load(outdir / "affinity.npz", allow_pickle=False) as aff:
        assert aff["affinity"].shape == (3, 3)
        assert np.allclose(aff["affinity"], aff["affinity"].T)
        assert sorted(aff["order"].tolist()) == [0, 1, 2]


def test_run_stage6b_keeps_singleton_tracks_in_proportions(tmp_path, monkeypatch):
    frames = _fake_analysis(tmp_path, singleton_track=True)
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)

    tstat.run_stage6b()

    qc = json.loads((tmp_path / "analysis" / "stage6b_qc.json").read_text(encoding="utf-8"))
    # 每个实例都是某个时间箱里的真实观测：长度 1 的 track 不许从占比里消失
    assert qc["subsets"]["proportions"]["n_instances"] == len(frames["tracks"])
    assert qc["subsets"]["transitions"]["n_single_instance_segments"] == len(_FOVS)
    # 每个 fov 有 _PER_BIN 条三段 track + 1 条长度 1 的 track：段数 = track 数 = 3 × 31
    assert qc["subsets"]["transitions"]["n_spot_segments"] == len(_FOVS) * (_PER_BIN + 1)
    assert qc["subsets"]["transitions"]["n_tracks"] == len(_FOVS) * (_PER_BIN + 1)
    with np.load(tmp_path / "analysis" / "proportions.npz", allow_pickle=False) as props:
        assert np.allclose(props["proportions"].sum(axis=-1), 1.0, atol=1e-6)


def test_run_stage6b_persists_counts_offset_and_shipped_finiteness(tmp_path, monkeypatch):
    _fake_analysis(tmp_path)
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)

    tstat.run_stage6b()

    outdir = tmp_path / "analysis"
    # 分析时间的偏移（frame_idx + 0.5）必须进 npz，而不是只写在 QC 散文里
    for name in ("proportions.npz", "transitions.npz", "trajectories.npz", "affinity.npz"):
        with np.load(outdir / name, allow_pickle=False) as data:
            assert data["bin_analysis_time"].tolist() == pytest.approx([0.5, 1.5, 2.5]), name

    qc = json.loads((outdir / "stage6b_qc.json").read_text(encoding="utf-8"))
    assert qc["time_axis"]["bin_analysis_time"] == pytest.approx([0.5, 1.5, 2.5])
    # 落盘轨迹的有限性：逐条件、逐时间箱
    shipped = qc["trajectories"]["shipped_finiteness"]
    assert shipped["all_finite"] is True
    assert shipped["finite_bins"] == [[True, True, True]] * 3
    assert shipped["n_finite_bins"] == [3, 3, 3]
    assert shipped["conditions"] == ["A1", "A2", "A3"]
    # 两步的宽度不同（72 h 与 48 h），必须写进 QC 而不是当成等距时间轴
    note = qc["time_axis"]["discrete_days_note"]
    assert "72" in note and "48" in note and "一步" in note
    # 转移的边带原始计数与先验标记（pooled 与逐条件各一份）
    pooled_graph = qc["transitions"]["inter_graph_data"]
    assert "n_obs_out" in pooled_graph
    assert all({"count", "prior_only"} <= set(edge) for edge in pooled_graph["edges"])
    per_condition = qc["transitions"]["inter_graph_data_by_condition"]
    assert sorted(per_condition) == ["A1", "A2", "A3"]


def test_run_stage6b_rejects_duplicate_instance_uid(tmp_path, monkeypatch):
    _fake_analysis(tmp_path)
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    for filename in ("cluster_labels.csv", "umap_2d.csv"):
        path = tmp_path / "analysis" / filename
        table = pd.read_csv(path)
        pd.concat([table, table.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
        with pytest.raises(RuntimeError, match=f"{filename} 里 instance_uid 有重复"):
            tstat.run_stage6b()
        # 复原，下一轮只污染另一份产物
        table.to_csv(path, index=False)


def test_run_stage6b_requires_upstream_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="tracks.csv"):
        tstat.run_stage6b()

    frames = _fake_analysis(tmp_path)
    (tmp_path / "analysis" / "cluster_labels.csv").unlink()
    with pytest.raises(RuntimeError, match="cluster_labels.csv"):
        tstat.run_stage6b()
    assert (tmp_path / "analysis" / "tracks.csv").exists()
    assert len(frames["labels"]) > 0
