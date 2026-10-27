"""Stage 7 时间图（F7–F11）的测试：全部合成产物，不依赖真实 Stage 5/6/6b 数据。

`fake_temporal_dir` 造出与真实 `output/` **同构**的树（`output/analysis/` 里放 Stage5/6/6b 的产物），
因为这份 fixture 是钉住上游 schema 的东西：只要哪一阶段改了文件名、键名或列名，这里会先炸，
而不是等出图时静默画错。四份 npz 的公共键（`cluster_labels` / `conditions` / `bin_frame_idx` /
`bin_day` / `bin_time_hours` / `bin_analysis_time` / `time_intervals`）与真实 Stage6b 落盘逐键对齐。

测试盯住的点（都是**图的内容**，不只是"文件存在"）：

- `render_temporal` 产出任务说明里那 6 个**确切**文件名，且 `analysis_dir` 既能指 `output/` 根，
  也能直接指 `output/analysis/`；
- F7 的每根堆叠条 = 该 (视野, 采样点) 的簇占比，和恰为 1；合并面板 = 按实例数**加权**平均
  （不是等权：fixture 里各视野实例数不同，两种口径能区分开）；
- F8 画出来的边只是**观测到**的转移（`count >= 1`），`count == 0` 的纯先验边一条都不画；
  自转移热图就是平滑后的 (来源, 目标) 矩阵，行和为 1；图上的边概率是去掉自转移后按行重归一
  的 `inter`，与热图的 `probs` 分母不同（图注必须写明，否则同一个 (来源, 目标) 会读出两个数）；
- F9 只画长度 ≥ 2 的 track（图上写明子集），全长度 1 的 fixture 也要出图并把退化写在图上；
- F10a（视野级）与 F10b（簇级）画的是**不同形状**的亲和矩阵，且都是读盘得到的产物/质心；
  F10b 的数值要等于 `tstats.trajectory_affinity(簇质心轨迹)`；两张图的图注要如实写 DTW
  （对时间伸缩不敏感、但包含位置/偏移差异）并声明两者各自用自己的 std 归一、不可横比；
- F11 的每帧对事件数与 `match_qc.json` 的 `n_one_to_one / n_merge / n_split / unmatched` 一致；
- 出完图进程里不留 figure；缺产物/坏产物给带阶段名的中文错误（不静默少画）；
- 落盘 helper `figs._stage_and_replace`（静态图与时间图共用）：全部渲染成功才换名，失败时旧图
  不动且不留 `.tmp-*`（包含 `savefig` 写到一半、来不及登记的那一份）。

`fake_singleton_dir` 是同一棵树、但**每条 track 长度都是 1**（跨天匹配全失败）的退化现场：
它是任务说明里"F9 出图但不崩"的测试前提。
"""

import contextlib
import itertools
import json

import matplotlib

# 无头环境：SPOT 的 hcluster 会建 figure 且从不关闭，交互式后端会真的开窗口。
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from biopheno import figures as figs
from biopheno import temporal_stats as tstats

# `render_temporal` 的出图顺序（= 文件名清单的顺序），`_spy_figures` 收集到的 figure 也按这个序。
_RENDER_ORDER: tuple[str, ...] = (
    "F7_stacked_cluster_proportions.png",
    "F8_transition_graph.png",
    "F9_umap_trajectories.png",
    "F10a_trajectory_affinity_fov.png",
    "F10b_trajectory_affinity_cluster.png",
    "F11_match_events.png",
)
_EXPECTED_FIGURES: set[str] = set(_RENDER_ORDER)

# 时间轴：3 个离散采样点（真实数据是 day0/day3/day5，差 3 天与 2 天不等宽）。
_FOVS: tuple[str, ...] = ("A1", "A2", "A3")
_FRAMES: tuple[int, ...] = (0, 1, 2)
_DAYS: tuple[str, ...] = ("day0", "day3", "day5")
_HOURS: tuple[float, ...] = (0.0, 72.0, 120.0)
_CLUSTERS: tuple[int, ...] = (1, 2, 3, 4)

# 每个 (采样点, 簇) 的基础实例数；簇 1 再按视野与采样点加量，所以各视野实例数不同，
# "按实例数加权"的合并面板与"等权平均"能算出不同的数（测试据此区分两种口径）。
_BASE_COUNTS: tuple[tuple[int, ...], ...] = ((2, 3, 2, 2), (4, 2, 2, 2), (2, 2, 3, 4))

# 转移矩阵的**未平滑原始计数**（行 = 来源、列 = 目标）：刻意留出 count == 0 的纯先验格
# （例如簇 1 → 簇 3），F8 必须把这些边过滤掉。
_RAW_COUNTS: np.ndarray = np.array(
    [
        [4.0, 1.0, 0.0, 0.0],
        [0.0, 3.0, 1.0, 0.0],
        [0.0, 0.0, 2.0, 1.0],
        [1.0, 0.0, 0.0, 3.0],
    ]
)
_PSEUDOCOUNTS: float = 1.0

# 视野级的轨迹亲和矩阵（对称、对角 1）；`order` 由 1 - 亲和 的 average 层次聚类算出来，
# 与真实 Stage6b 落盘口径一致（亲和矩阵来自 DTW 指数核）。
_AFFINITY: np.ndarray = np.array(
    [
        [1.00, 0.85, 0.30],
        [0.85, 1.00, 0.35],
        [0.30, 0.35, 1.00],
    ]
)

# 合成 UMAP 坐标：每个簇一个中心 + 随采样点漂移，轨迹才不是静止的点。
_CENTRES: dict[int, tuple[float, float]] = {1: (0.0, 0.0), 2: (6.0, 0.5), 3: (0.5, 6.0), 4: (6.0, 6.0)}
_DRIFT: dict[int, tuple[float, float]] = {1: (0.6, 0.1), 2: (-0.2, 0.5), 3: (0.3, -0.4), 4: (-0.5, -0.3)}


def _cluster_counts(fov_index: int, frame: int) -> tuple[int, ...]:
    """某个 (视野, 采样点) 里每簇的实例数（簇 1 随视野与采样点加量，制造视野间权重差异）。"""
    counts = list(_BASE_COUNTS[frame])
    counts[0] += fov_index + frame
    return tuple(counts)


def _instances(chained: bool) -> list[dict]:
    """逐实例的元数据行（列名与 Stage5 的 `cluster_labels.csv` / Stage6 的 `tracks.csv` 一致）。

    `chained=True`：每簇的**第一个**实例在 3 个采样点之间连成一条长度 3 的 track（跨天匹配成功），
    其余实例各自是一条长度 1 的 track——这正是真实数据里 merge/分裂之外的主干。
    `chained=False`：每个实例自成一个 track（跨天匹配全失败，用来测 F9 的退化标注）。
    """
    rows: list[dict] = []
    next_track = 0
    for fov_index, fov in enumerate(_FOVS):
        chain: dict[int, int] = {}
        if chained:
            for cluster in _CLUSTERS:
                chain[cluster] = next_track
                next_track += 1
        label = 0
        for frame in _FRAMES:
            for cluster in _CLUSTERS:
                for variant in range(_cluster_counts(fov_index, frame)[cluster - 1]):
                    label += 1
                    if chained and variant == 0:
                        track_id = chain[cluster]
                    else:
                        track_id = next_track
                        next_track += 1
                    rows.append(
                        {
                            "instance_uid": f"{fov}_{label}_{frame}",
                            "spot_uid": f"{fov}_{track_id}_{frame}",
                            "track_id": track_id,
                            "fov": fov,
                            "frame_idx": frame,
                            "time_hours": _HOURS[frame],
                            "day": _DAYS[frame],
                            "label": label,
                            "cluster": cluster,
                            "variant": variant,
                        }
                    )
    return rows


def _proportions(rows: list[dict]) -> np.ndarray:
    """簇占比矩阵 (n_conditions, n_bins, n_labels)：与 Stage6b 的落盘口径一致（行和 1）。"""
    out = np.full((len(_FOVS), len(_FRAMES), len(_CLUSTERS)), np.nan)
    for fov_index, fov in enumerate(_FOVS):
        for bin_index, frame in enumerate(_FRAMES):
            labels = np.array([row["cluster"] for row in rows if row["fov"] == fov and row["frame_idx"] == frame])
            for cluster_index, cluster in enumerate(_CLUSTERS):
                out[fov_index, bin_index, cluster_index] = float(np.mean(labels == cluster))
    return out


def _bin_counts(rows: list[dict]) -> np.ndarray:
    """每 (视野, 采样点) 的实例数（= 占比的分母，F7 合并面板的加权权重）。"""
    out = np.zeros((len(_FOVS), len(_FRAMES)), dtype=np.int64)
    for fov_index, fov in enumerate(_FOVS):
        for bin_index, frame in enumerate(_FRAMES):
            out[fov_index, bin_index] = sum(1 for row in rows if row["fov"] == fov and row["frame_idx"] == frame)
    return out


def _coords(rows: list[dict]) -> np.ndarray:
    """按 Stage5 的口径给每个实例一个 UMAP 坐标（簇中心 + 采样点漂移 + 固定种子的小抖动）。"""
    rng = np.random.default_rng(11)
    coords = np.vstack(
        [
            np.asarray(_CENTRES[row["cluster"]], dtype=np.float64)
            + np.asarray(_DRIFT[row["cluster"]], dtype=np.float64) * float(row["frame_idx"])
            + rng.normal(0.0, 0.2, 2)
            for row in rows
        ]
    )
    return coords


def _common_arrays() -> dict[str, np.ndarray]:
    """四份 npz 共有的标签顺序（真实 Stage6b 每份都写一遍，绘图阶段不必猜）。"""
    return {
        "cluster_labels": np.asarray(_CLUSTERS, dtype=np.int64),
        "conditions": np.asarray(_FOVS),
        "bin_frame_idx": np.asarray(_FRAMES, dtype=np.int64),
        "bin_analysis_time": np.asarray(_FRAMES, dtype=np.float64) + 0.5,
        "bin_time_hours": np.asarray(_HOURS, dtype=np.float64),
        "bin_day": np.asarray(_DAYS),
        "time_intervals": np.asarray([0.0, 1.0, 2.0, 3.0]),
    }


def _transition_probs() -> np.ndarray:
    """平滑后的 (来源, 目标) 转移概率（pseudocounts=1 → 行和恰好 1，不出 NaN 行）。"""
    smoothed = _RAW_COUNTS + _PSEUDOCOUNTS
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def _inter_from_probs(probs: np.ndarray) -> np.ndarray:
    """对角置 0 后按行重归一（= Stage6b 的 `_inter_from_probs` 口径）。"""
    off = np.asarray(probs, dtype=np.float64).copy()
    np.fill_diagonal(off, 0.0)
    return off / off.sum(axis=1, keepdims=True)


def _write_analysis(root, rows: list[dict]) -> None:
    """写 `analysis/` 下的 Stage5 + Stage6 + Stage6b 产物（列名/键名与真实落盘逐一对齐）。"""
    outdir = root / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    uids = [row["instance_uid"] for row in rows]
    coords = _coords(rows)
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
    pd.DataFrame(
        {
            "instance_uid": uids,
            "spot_uid": [row["spot_uid"] for row in rows],
            "track_id": [row["track_id"] for row in rows],
            "fov": [row["fov"] for row in rows],
            "frame_idx": [row["frame_idx"] for row in rows],
            "time_hours": [row["time_hours"] for row in rows],
            "day": [row["day"] for row in rows],
        }
    ).to_csv(outdir / "tracks.csv", index=False)

    common = _common_arrays()
    np.savez(outdir / "proportions.npz", proportions=_proportions(rows), **common)

    probs = _transition_probs()
    inter = _inter_from_probs(probs)
    counts_by_condition = np.stack([_RAW_COUNTS * weight for weight in (1.0, 0.5, 0.0)])
    np.savez(
        outdir / "transitions.npz",
        inter=inter,
        self_transitions=np.diag(probs).astype(np.float64),
        probs=probs,
        counts=_RAW_COUNTS.copy(),
        counts_by_condition=counts_by_condition,
        inter_by_condition=np.repeat(inter[None, :, :], len(_FOVS), axis=0),
        self_by_condition=np.repeat(np.diag(probs)[None, :], len(_FOVS), axis=0),
        pseudocounts=np.asarray(_PSEUDOCOUNTS, dtype=np.float64),
        **common,
    )

    trajectories = np.zeros((len(_FOVS), len(_FRAMES), 2), dtype=np.float64)
    for fov_index, fov in enumerate(_FOVS):
        for bin_index, frame in enumerate(_FRAMES):
            members = [index for index, row in enumerate(rows) if row["fov"] == fov and row["frame_idx"] == frame]
            trajectories[fov_index, bin_index] = coords[members].mean(axis=0)
    np.savez(
        outdir / "trajectories.npz",
        trajectories=trajectories,
        spot_trajectories=trajectories.copy(),
        path=np.asarray("spot_density"),
        density_threshold_sigma=np.asarray(3.0, dtype=np.float64),
        **common,
    )

    order = np.asarray(
        leaves_list(linkage(squareform(1.0 - _AFFINITY, checks=False), method="average")), dtype=np.int64
    )
    np.savez(
        outdir / "affinity.npz",
        affinity=_AFFINITY.copy(),
        order=order,
        linkage=np.asarray("average"),
        metric=np.asarray("euclidean"),
        **common,
    )
    (outdir / "stage6b_qc.json").write_text(json.dumps(_stage6b_qc(), ensure_ascii=False), encoding="utf-8")


def _stage6b_qc() -> dict:
    """`stage6b_qc.json` 里绘图阶段真正会读的那几项（时间轴口径、子集、轨迹路径/退化）。"""
    return {
        "time_axis": {
            "mode": "frame_idx 箱中心（frame_idx + 0.5）",
            "bin_frame_idx": list(_FRAMES),
            "bin_day": list(_DAYS),
            "bin_time_hours": list(_HOURS),
            "n_bins": len(_FRAMES),
            "discrete_days_note": "时间轴是 3 个离散采样点（day0/day3/day5），不是连续视频；motion 已置空",
        },
        "subsets": {
            "proportions": {"n_instances": 0, "track_length_filter": "none"},
            "transitions": {"n_instances": 0, "track_length_filter": ">=2"},
            "trajectories": {
                "n_instances": 0,
                "track_length_filter": "none（轨迹用全部有簇标签的实例的嵌入坐标）",
            },
        },
        "trajectories": {
            "path": "spot_density",
            "degenerate": False,
            "degeneracy": {"degenerate": False, "reasons": []},
            "shipped_finiteness": {"all_finite": True},
        },
        "transitions": {"pseudocounts": _PSEUDOCOUNTS, "orientation": "inter / probs 都是 (来源, 目标)，行 = 来源"},
        "affinity": {"affinity": "dtw", "use_std_denom": True, "fallback": None},
        "warnings": [],
    }


def _match_rows(rows: list[dict], chained: bool) -> tuple[pd.DataFrame, dict]:
    """造逐帧对匹配表 + `match_qc.json`（两边的数**手工一致**：统计都由同一批行算出来）。

    `chained=True` 时每个帧对有 4 条 one_to_one、1 条 merge（上一帧的小实例被吸收进父实例）、
    1 条 split（父实例已有继承人后新出现的子实例）；`chained=False` 时一条匹配都没有（全部未匹配）。
    """
    by_key = {(row["fov"], row["frame_idx"], row["cluster"], row["variant"]): row for row in rows}
    records: list[dict] = []
    fovs_qc: dict[str, dict] = {}
    for fov in _FOVS:
        frame_totals = {
            frame: sum(1 for row in rows if row["fov"] == fov and row["frame_idx"] == frame) for frame in _FRAMES
        }
        pairs: dict[str, dict] = {}
        for prev_frame, cur_frame in itertools.pairwise(_FRAMES):
            pair = f"{prev_frame}->{cur_frame}"
            pair_rows: list[dict] = []
            if chained:
                for cluster in _CLUSTERS:
                    pair_rows.append(
                        {
                            "prev_uid": by_key[(fov, prev_frame, cluster, 0)]["instance_uid"],
                            "cur_uid": by_key[(fov, cur_frame, cluster, 0)]["instance_uid"],
                            "iou": 0.5 + 0.05 * cluster,
                            "cent_dist": 2.0 * cluster,
                            "event": "one_to_one",
                        }
                    )
                pair_rows.append(
                    {
                        "prev_uid": by_key[(fov, prev_frame, 2, 1)]["instance_uid"],
                        "cur_uid": by_key[(fov, cur_frame, 2, 0)]["instance_uid"],
                        "iou": 0.2,
                        "cent_dist": 12.0,
                        "event": "merge",
                    }
                )
                pair_rows.append(
                    {
                        "prev_uid": by_key[(fov, prev_frame, 4, 0)]["instance_uid"],
                        "cur_uid": by_key[(fov, cur_frame, 4, 1)]["instance_uid"],
                        "iou": 0.1,
                        "cent_dist": 20.0,
                        "event": "split",
                    }
                )
            for record in pair_rows:
                records.append({"fov": fov, "pair": pair, **record})
            n_prev, n_cur = frame_totals[prev_frame], frame_totals[cur_frame]
            n_prev_used = len({record["prev_uid"] for record in pair_rows})
            n_cur_used = len({record["cur_uid"] for record in pair_rows})
            n_one_to_one = sum(1 for record in pair_rows if record["event"] == "one_to_one")
            pairs[pair] = {
                "n_prev": n_prev,
                "n_cur": n_cur,
                "n_candidates": len(pair_rows),
                "n_one_to_one": n_one_to_one,
                "n_merge": sum(1 for record in pair_rows if record["event"] == "merge"),
                "n_merge_targets": 1 if any(record["event"] == "merge" for record in pair_rows) else 0,
                "n_split": sum(1 for record in pair_rows if record["event"] == "split"),
                "n_split_parents": 1 if any(record["event"] == "split" for record in pair_rows) else 0,
                "n_prev_unmatched": n_prev - n_prev_used,
                "n_cur_unmatched": n_cur - n_cur_used,
                "prev_link_rate": n_one_to_one / n_prev,
                "cur_link_rate": n_one_to_one / n_cur,
                "offset": [0, 0],
                "n_match_rows": len(pair_rows),
                # 视野 A2 的第二个帧对故意标一条"复合偏移超限"的告警，F11 要在图上标出来
                "offset_exceeds_limit": fov == "A2" and pair == "1->2",
            }
        sizes = pd.Series([row["track_id"] for row in rows if row["fov"] == fov]).value_counts()
        fovs_qc[fov] = {
            "registration": {
                "reference_frame": 0,
                "offset_to_reference": {"0": [0, 0], "1": [1, 0], "2": [1, 2]},
                "align_offsets": {"0": [0, 0], "1": [0, 0], "2": [0, 0]},
                "pair_offsets": {"0->1": [1, 0], "1->2": [0, 2]},
                "warnings": [],
            },
            "pairs": pairs,
            "n_instances": sum(frame_totals.values()),
            "n_tracks": int(sizes.size),
            "n_tracks_len_ge2": int((sizes >= 2).sum()),
            "track_length_histogram": {str(int(k)): int(v) for k, v in sizes.value_counts().sort_index().items()},
        }
    table = pd.DataFrame(records, columns=["fov", "pair", "prev_uid", "cur_uid", "iou", "cent_dist", "event"])
    if len(table):
        table = table.sort_values(["fov", "pair", "cur_uid", "event", "prev_uid"], kind="mergesort").reset_index(
            drop=True
        )
    totals = {
        "n_instances": len(rows),
        "n_tracks": len({(row["fov"], row["track_id"]) for row in rows}),
        "n_tracks_len_ge2": 0,
        "n_one_to_one": int((table["event"] == "one_to_one").sum()),
        "n_merge": int((table["event"] == "merge").sum()),
        "n_split": int((table["event"] == "split").sum()),
        "n_match_rows": len(table),
    }
    qc = {
        "config": {"frame_axis": {day: frame for day, frame in zip(_DAYS, _FRAMES)}},
        "fovs": fovs_qc,
        "totals": totals,
    }
    return table, qc


def _build_tree(root, chained: bool) -> None:
    """造出与真实 `output/` 同构的最小目录树（`analysis/` 里是 8 份时间产物）。"""
    rows = _instances(chained)
    root.mkdir(parents=True, exist_ok=True)
    _write_analysis(root, rows)
    table, qc = _match_rows(rows, chained)
    outdir = root / "analysis"
    table.to_csv(outdir / "matches.csv", index=False)
    (outdir / "match_qc.json").write_text(json.dumps(qc, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def fake_temporal_dir(tmp_path):
    """与真实 `output/` 同构的合成产物树（3 个视野 × 3 个采样点 × 4 个簇，含长度 3 的 track）。"""
    root = tmp_path / "output"
    _build_tree(root, chained=True)
    return root


@pytest.fixture
def fake_singleton_dir(tmp_path):
    """同构的树，但**每条 track 长度都是 1**（跨天匹配全失败）：F9 的退化现场。"""
    root = tmp_path / "output"
    _build_tree(root, chained=False)
    return root


@contextlib.contextmanager
def _spy_figures(monkeypatch):
    """把 `figs._open_figure` 包一层，收集出图期间的 figure（`plt.close` 之后 artist 仍可读）。"""
    original = figs._open_figure
    figures: list = []

    @contextlib.contextmanager
    def spy(**kwargs):
        """替身 savefig：转调真正的 savefig，并把产出的 figure 记进 figures，供“是否全部关闭”的断言用。"""
        with original(**kwargs) as figure:
            figures.append(figure)
            yield figure

    monkeypatch.setattr(figs, "_open_figure", spy)
    yield figures


def _all_texts(figure) -> str:
    """一张 figure 上的全部文字（轴内文字 + 轴标题 + suptitle + `fig.text` 图注）。

    `fig.text` 不在 `figure.axes[*].texts` 里：F10a/F10b 的长图注是 figure 级文字，漏掉它就等于
    没测"图注到底写了什么"。
    """
    parts = [text.get_text() for axis in figure.axes for text in axis.texts]
    parts.extend(axis.get_title() for axis in figure.axes)
    parts.extend(text.get_text() for text in figure.texts)
    if figure._suptitle is not None:
        parts.append(figure._suptitle.get_text())
    return "\n".join(parts)


def _panel_totals(axis) -> dict[float, float]:
    """把柱状图的柱子按 x 轴位置聚起来（键 = 最近的整数横坐标，值 = 该位置的高度之和）。

    分组柱（未匹配面板）的中心在 x ± 0.18，取整到最近的整数才能与横坐轴刻度对齐。
    """
    totals: dict[float, float] = {}
    for patch in axis.patches:
        centre = round(float(patch.get_x() + patch.get_width() / 2.0))
        totals[centre] = totals.get(centre, 0.0) + float(patch.get_height())
    return totals


def _flow_gids(figure) -> list[str]:
    """F8 里每条转移边的 gid（`edge-<来源>-<目标>-n<观测计数>`），边是 FancyArrowPatch。"""
    return [
        str(artist.get_gid())
        for axis in figure.axes
        for artist in axis.patches
        if str(artist.get_gid() or "").startswith("edge-")
    ]


def _line_gids(figure) -> list[str]:
    """F9 里每条折线的 gid（`track-<fov>-<track_id>` / `trajectory-<fov>`）。"""
    return [str(artist.get_gid()) for axis in figure.axes for artist in axis.lines if artist.get_gid()]


def _images(figure) -> list[np.ndarray]:
    """一张 figure 里所有 `imshow` 的数值数组（色条的 QuadMesh 不是 image，不会混进来）。"""
    return [np.asarray(image.get_array(), dtype=np.float64) for axis in figure.axes for image in axis.images]


# —— 契约 ——


def test_render_temporal_writes_all_figures(fake_temporal_dir, tmp_path):
    """守口径：出图清单必须恰好是任务说明里那 6 个文件名，多一张少一张都算错。"""
    paths = figs.render_temporal(fake_temporal_dir, outdir=tmp_path)
    names = {p.name for p in paths}
    assert names == _EXPECTED_FIGURES
    # 1KB 下限只挡"0 字节 / 写到一半"的坏图；画面内容由下面各 F7–F11 用例逐一盯。
    assert all(p.stat().st_size > 1000 for p in paths)


def test_render_temporal_updates_the_figure_readme(fake_temporal_dir, tmp_path):
    """时间图出完也要刷新 `README.md`（与静态图共用同一份说明，标出本目录实际有哪些图）。"""
    figs.render_temporal(fake_temporal_dir, outdir=tmp_path)
    readme = tmp_path / "README.md"
    assert readme.exists()
    text = readme.read_text(encoding="utf-8")
    for name in _EXPECTED_FIGURES:  # 6 张时间图
        assert name in text
    assert "已生成 **6 / 13**" in text
    assert "缺失 7 张" in text
    assert not list(tmp_path.glob(".tmp-*"))


def test_render_temporal_accepts_the_analysis_dir_itself(fake_temporal_dir, tmp_path):
    """`analysis_dir` 既能指 `output/` 根，也能直接指 `output/analysis/`（两种调用都支持）。"""
    paths = figs.render_temporal(fake_temporal_dir / "analysis", outdir=tmp_path / "figures")
    assert {p.name for p in paths} == _EXPECTED_FIGURES


def test_render_temporal_closes_every_figure(fake_temporal_dir, tmp_path):
    """顶住退化：出完图后进程里一张 figure 都不留（否则无头环境会持续攒内存）。"""
    figs.render_temporal(fake_temporal_dir, outdir=tmp_path)
    assert plt.get_fignums() == []


def test_render_temporal_names_the_stage_of_a_missing_artifact(fake_temporal_dir, tmp_path):
    """缺产物不许静默少画：报错必须点名阶段（proportions.npz 属 Stage6b）。"""
    (fake_temporal_dir / "analysis" / "proportions.npz").unlink()
    with pytest.raises(ValueError, match="Stage6b"):
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)


def test_render_temporal_missing_tracks_points_at_stage6(fake_temporal_dir, tmp_path):
    """缺 tracks.csv 必须点名 Stage6（而不是 Stage6b：`Stage6` 单独一个词也是 `Stage6b` 的前缀）。"""
    (fake_temporal_dir / "analysis" / "tracks.csv").unlink()
    with pytest.raises(ValueError, match="请先跑 Stage6（"):
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)


def test_render_temporal_rejects_a_non_row_stochastic_matrix(fake_temporal_dir, tmp_path):
    """转移矩阵的契约是"行 = 来源、行和 = 1"：坏了就报错，绝不把比例画错。"""
    path = fake_temporal_dir / "analysis" / "transitions.npz"
    payload = figs._read_npz(path, "Stage6b")
    payload["probs"] = np.asarray(payload["probs"], dtype=np.float64) * 2.0
    # 乘 2 后行和变成 2：专打"行 = 来源、行和 = 1"这条契约。
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="Stage6b"):
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)


def test_transitions_reject_a_non_scalar_pseudocount(fake_temporal_dir, tmp_path):
    """`pseudocounts` 只允许一个值：多值时不许悄悄取第 0 个元素（图注会拿错数）。"""
    path = fake_temporal_dir / "analysis" / "transitions.npz"
    payload = figs._read_npz(path, "Stage6b")
    payload["pseudocounts"] = np.asarray([_PSEUDOCOUNTS, 5.0], dtype=np.float64)
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="pseudocounts 应只有 1 个值"):
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)


def test_trajectories_reject_a_non_scalar_path(fake_temporal_dir, tmp_path):
    """`path` 只允许一个值：多值时不许悄悄取第 0 个元素（图注会拿错来源）。"""
    path = fake_temporal_dir / "analysis" / "trajectories.npz"
    payload = figs._read_npz(path, "Stage6b")
    payload["path"] = np.asarray(["spot_density", "fallback_density"])
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="path 应只有 1 个值"):
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)


# —— 落盘 helper（render_static / render_temporal 共用）——


def test_staged_write_keeps_the_previous_figures_and_leaves_no_temporary(tmp_path):
    """渲染到一半崩掉（savefig 写了半个文件）：旧图不动、临时文件清干净（包括没登记的半成品）。"""
    out = tmp_path / "figures"
    out.mkdir()
    (out / "a.png").write_bytes(b"old-a")
    (out / "b.png").write_bytes(b"old-b")

    def render_a(path):
        """替身：只写出 a.png 就返回（模拟正常完成的一步）。"""
        path.write_bytes(b"new-a")

    def render_b(path):
        """替身：先建出 b.png 再抛错，模拟 savefig 中途失败（文件已存在但不能算 staged）。"""
        path.write_bytes(b"half-b")  # 模拟 savefig 中途抛错：文件已建出来，但没登记进 staged
        raise RuntimeError("savefig 挂了")

    with pytest.raises(RuntimeError, match="savefig"):
        figs._stage_and_replace(out, ("a.png", "b.png"), (render_a, render_b))
    assert (out / "a.png").read_bytes() == b"old-a"
    assert (out / "b.png").read_bytes() == b"old-b"
    assert not list(out.glob(".tmp-*"))


def test_staged_write_replaces_only_after_every_render_succeeds(tmp_path):
    """全部渲染成功之前不换名：渲染期间旧图仍在，成功后每一张都是新图。"""
    out = tmp_path / "figures"
    out.mkdir()
    (out / "a.png").write_bytes(b"old-a")
    seen: list[bytes] = []

    def render_a(path):
        """替身：先记录当时磁盘上的 a.png，再写新内容（用来看换名时机）。"""
        seen.append((out / "a.png").read_bytes())
        path.write_bytes(b"new-a")

    def render_b(path):
        """替身：同样先记录 a.png，再写新内容——验证两张图都渲染完才统一换名。"""
        seen.append((out / "a.png").read_bytes())  # 第一张也还没换名
        path.write_bytes(b"new-b")

    paths = figs._stage_and_replace(out, ("a.png", "b.png"), (render_a, render_b))
    assert [path.name for path in paths] == ["a.png", "b.png"]
    assert seen == [b"old-a", b"old-a"]
    assert (out / "a.png").read_bytes() == b"new-a"
    assert (out / "b.png").read_bytes() == b"new-b"
    assert not list(out.glob(".tmp-*"))


# —— F7 ——


def test_f7_stacked_bars_sum_to_one_per_bin(fake_temporal_dir, tmp_path, monkeypatch):
    """F7 每根堆叠条 = 该 (视野, 采样点) 的簇占比，和必须正好是 1；面板数 = 视野数 + 合并面板。"""
    with _spy_figures(monkeypatch) as figures:
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)
    figure = figures[_RENDER_ORDER.index("F7_stacked_cluster_proportions.png")]
    panels = [axis for axis in figure.axes if axis.patches]
    assert len(panels) == len(_FOVS) + 1
    for axis in panels:
        totals = _panel_totals(axis)
        assert len(totals) == len(_FRAMES)
        # 容差 1e-9 只吸收浮点累加误差：占比本身行和为 1，堆叠条高度之和必须是精确 1。
        assert all(abs(total - 1.0) < 1e-9 for total in totals.values())


def test_f7_pooled_panel_is_weighted_by_instances(fake_temporal_dir, tmp_path, monkeypatch):
    """合并面板 = 按每视野每采样点实例数加权平均（不是等权：fixture 里两种口径的数不同）。"""
    rows = _instances(chained=True)
    proportions = _proportions(rows)
    weights = _bin_counts(rows)
    weighted = (proportions * weights[:, :, None]).sum(axis=0) / weights.sum(axis=0)[:, None]
    equal_weight = proportions.mean(axis=0)
    assert not np.allclose(weighted, equal_weight)  # fixture 必须真的能区分两种口径

    with _spy_figures(monkeypatch) as figures:
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)
    figure = figures[_RENDER_ORDER.index("F7_stacked_cluster_proportions.png")]
    pooled_axis = [axis for axis in figure.axes if axis.patches][-1]
    for bin_index in range(len(_FRAMES)):
        drawn = sorted(
            float(patch.get_height())
            for patch in pooled_axis.patches
            if round(float(patch.get_x() + patch.get_width() / 2.0), 6) == 0.0 + bin_index
        )
        assert np.allclose(sorted(drawn), sorted(weighted[bin_index]), atol=1e-9)
    assert "加权" in _all_texts(figure)
    # 图上的图注是纯文本，不是 markdown：`**加权**` 会把星号原样画进图片
    assert "**" not in _all_texts(figure)


# —— F8 ——


def test_f8_draws_only_observed_flows(fake_temporal_dir, tmp_path, monkeypatch):
    """`count == 0` 的边只是 pseudocounts=1 的先验，不是观测：一条都不许画成流动。"""
    with _spy_figures(monkeypatch) as figures:
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)
    figure = figures[_RENDER_ORDER.index("F8_transition_graph.png")]
    gids = _flow_gids(figure)
    drawn = {(int(parts[1]), int(parts[2])) for parts in (gid.split("-") for gid in gids)}
    observed = {(i + 1, j + 1) for i in range(len(_CLUSTERS)) for j in range(len(_CLUSTERS)) if i != j}
    observed = {(source, target) for source, target in observed if _RAW_COUNTS[source - 1, target - 1] > 0}
    assert drawn == observed
    assert all(int(gid.split("-n")[1]) >= 1 for gid in gids)
    assert (1, 3) not in drawn  # 簇 1 → 簇 3 的原始计数是 0（纯先验）
    assert "先验" in _all_texts(figure)


def test_f8_edges_use_the_off_diagonal_denominator(fake_temporal_dir, tmp_path, monkeypatch):
    """图上的边概率 = 去掉自转移后按行重归一（`inter`）；右侧热图是含自转移的 `probs`。

    同一个 (来源, 目标) 在两处数字不同，所以两处标号必须写清楚分母：光写"转移概率"会让读者
    以为左图的 0.50 和右图的 0.22 是矛盾的数据。
    """
    with _spy_figures(monkeypatch) as figures:
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)
    figure = figures[_RENDER_ORDER.index("F8_transition_graph.png")]
    prob = _transition_probs()
    inter = _inter_from_probs(prob)

    drawn: dict[tuple[int, int], float] = {}
    for axis in figure.axes:
        for patch in axis.patches:
            gid = str(patch.get_gid() or "")
            if not gid.startswith("edge-"):
                continue
            parts = gid.split("-")
            drawn[(int(parts[1]), int(parts[2]))] = float(str(patch.get_label()).removeprefix("p="))
    assert drawn
    for (source, target), probability in drawn.items():
        assert np.isclose(probability, inter[source - 1, target - 1], atol=1e-6)

    heatmap = _images(figure)[0]
    assert np.allclose(heatmap, prob)
    # 簇 1 有自转移（probs[0, 0] > 0）：去掉自转移后分母变小，所以边概率必须**高于**热图同格
    source, target = 1, 2
    assert (source, target) in drawn
    assert prob[source - 1, source - 1] > 0.0
    assert drawn[(source, target)] > heatmap[source - 1, target - 1] + 0.1
    assert not np.isclose(drawn[(source, target)], heatmap[source - 1, target - 1])

    texts = _all_texts(figure)
    assert "去掉自转移" in texts
    assert "不同分母" in texts
    assert "含自转移" in texts


def test_f8_self_transition_heatmap_is_row_stochastic(fake_temporal_dir, tmp_path, monkeypatch):
    """自转移热图 = 平滑后的 (来源, 目标) 矩阵：行和 1，对角就是自转移概率。"""
    with _spy_figures(monkeypatch) as figures:
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)
    figure = figures[_RENDER_ORDER.index("F8_transition_graph.png")]
    images = _images(figure)
    assert len(images) == 1
    heatmap = images[0]
    assert heatmap.shape == (len(_CLUSTERS), len(_CLUSTERS))
    assert np.allclose(heatmap.sum(axis=1), 1.0)
    assert np.allclose(np.diag(heatmap), np.diag(_transition_probs()))


# —— F9 ——


def test_f9_plots_only_long_tracks(fake_temporal_dir, tmp_path, monkeypatch):
    """F9 只画长度 ≥ 2 的 track（长度 1 的 track 只有单点），并把子集写在图上。"""
    rows = _instances(chained=True)
    sizes = pd.Series([(row["fov"], row["track_id"]) for row in rows]).value_counts()
    long_tracks = {(fov, int(track)) for (fov, track), size in sizes.items() if size >= 2}
    short_tracks = {(fov, int(track)) for (fov, track), size in sizes.items() if size < 2}
    assert long_tracks

    with _spy_figures(monkeypatch) as figures:
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)
    figure = figures[_RENDER_ORDER.index("F9_umap_trajectories.png")]
    drawn = {(gid.split("-")[1], int(gid.split("-")[2])) for gid in _line_gids(figure) if gid.startswith("track-")}
    assert drawn == long_tracks
    assert not (drawn & short_tracks)
    assert any(gid.startswith("trajectory-") for gid in _line_gids(figure))
    texts = _all_texts(figure)
    assert "长度 ≥ 2" in texts
    # 背景上是**全部视野**的实例（共用参照系），面板标题却是单个视野：不写明会被读成"该视野的点"
    assert "全部视野的实例" in texts
    assert "参照系" in texts


def test_trajectory_plot_skips_singleton_tracks(fake_singleton_dir, tmp_path, monkeypatch):
    """fixture 里全是长度 1 的 track：F9 照样出图，但图上必须写明"没有长度 ≥ 2 的 track"。"""
    with pytest.warns(UserWarning, match="长度 ≥ 2"), _spy_figures(monkeypatch) as figures:
        paths = figs.render_temporal(fake_singleton_dir, outdir=tmp_path)
    assert any(p.name == "F9_umap_trajectories.png" for p in paths)
    figure = figures[_RENDER_ORDER.index("F9_umap_trajectories.png")]
    assert not [gid for gid in _line_gids(figure) if gid.startswith("track-")]
    texts = _all_texts(figure)
    assert "长度 ≥ 2" in texts
    assert "没有长度 ≥ 2" in texts


def test_f9_labels_a_degenerate_phenotype_trajectory(fake_temporal_dir, tmp_path, monkeypatch):
    """表型轨迹全为 NaN（Stage6b 退化）时，面板必须写在图上，而不是悄悄空白。"""
    path = fake_temporal_dir / "analysis" / "trajectories.npz"
    payload = figs._read_npz(path, "Stage6b")
    payload["trajectories"] = np.full_like(np.asarray(payload["trajectories"], dtype=np.float64), np.nan)
    # Stage6b 退化时轨迹会整体落成 NaN（不是 0）：用 NaN 而不是 0 才能区分"没有轨迹"和"轨迹在原点"。
    np.savez_compressed(path, **payload)

    with pytest.warns(UserWarning, match="退化"), _spy_figures(monkeypatch) as figures:
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)
    figure = figures[_RENDER_ORDER.index("F9_umap_trajectories.png")]
    assert "退化" in _all_texts(figure)
    assert not [gid for gid in _line_gids(figure) if gid.startswith("trajectory-")]


# —— F10a / F10b ——


def test_f10a_and_f10b_draw_different_matrix_shapes(fake_temporal_dir, tmp_path, monkeypatch):
    """F10a 是视野级亲和（3×3），F10b 是簇级亲和（4×4）：形状不同、都来自读盘产物。"""
    with _spy_figures(monkeypatch) as figures:
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)
    fov_images = _images(figures[_RENDER_ORDER.index("F10a_trajectory_affinity_fov.png")])
    cluster_images = _images(figures[_RENDER_ORDER.index("F10b_trajectory_affinity_cluster.png")])
    assert len(fov_images) == 1
    assert len(cluster_images) == 1
    fov_matrix, cluster_matrix = fov_images[0], cluster_images[0]
    assert fov_matrix.shape == (len(_FOVS), len(_FOVS))
    assert cluster_matrix.shape == (len(_CLUSTERS), len(_CLUSTERS))
    assert fov_matrix.shape != cluster_matrix.shape
    for matrix in (fov_matrix, cluster_matrix):
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(np.diag(matrix), 1.0)
    # F10a 画的就是 affinity.npz 的那份矩阵（重排行序不改变取值集合）
    assert np.allclose(np.sort(fov_matrix.ravel()), np.sort(_AFFINITY.ravel()))


def test_f10b_matches_tstats_affinity_of_the_cluster_centroids(fake_temporal_dir, tmp_path, monkeypatch):
    """F10b 的热图 = `tstats.trajectory_affinity(簇质心轨迹)`：bin 映射错一格就对不上。"""
    inputs = figs._load_temporal_inputs(fake_temporal_dir)
    centroids = figs._cluster_trajectories(inputs.frame, inputs.proportions_labels, inputs.bin_frame_idx)
    # 质心要按 (簇, 采样点) 逐格算：手算一遍，bin 映射错位会在这一步先失败
    manual = np.full_like(centroids, np.nan)
    for label_index, label in enumerate(_CLUSTERS):
        for bin_index, frame_idx in enumerate(_FRAMES):
            members = inputs.frame[(inputs.frame["cluster"] == label) & (inputs.frame["frame_idx"] == frame_idx)]
            manual[label_index, bin_index] = members[["umap_1", "umap_2"]].to_numpy(dtype=np.float64).mean(axis=0)
    assert np.allclose(centroids, manual, equal_nan=True)
    expected, _labels = tstats.trajectory_affinity(
        {"trajectories": centroids, "conditions": [f"簇 {label}" for label in _CLUSTERS]}
    )

    with _spy_figures(monkeypatch) as figures:
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)
    figure = figures[_RENDER_ORDER.index("F10b_trajectory_affinity_cluster.png")]
    axis = next(axis for axis in figure.axes if axis.images)
    drawn = np.asarray(axis.images[0].get_array(), dtype=np.float64)
    # 重排行序不改变数值：轴刻度标签就是显示序，按它把期望矩阵重排成同一序再逐格比
    order = [int(text.get_text().removeprefix("簇 ")) - 1 for text in axis.get_yticklabels() if text.get_text()]
    assert sorted(order) == list(range(len(_CLUSTERS)))
    assert drawn.shape == expected.shape
    assert np.allclose(drawn, expected[np.ix_(order, order)])


def test_f10_captions_say_what_the_dtw_kernel_measures(fake_temporal_dir, tmp_path, monkeypatch):
    """DTW 只是对时间伸缩不敏感，**不**抹掉位置/偏移差异；F10b 的路径来自质心，更要说清楚。"""
    with _spy_figures(monkeypatch) as figures:
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)
    captions = {
        name: _all_texts(figures[_RENDER_ORDER.index(name)])
        for name in ("F10a_trajectory_affinity_fov.png", "F10b_trajectory_affinity_cluster.png")
    }
    for name, texts in captions.items():
        assert "时间伸缩" in texts, name
        assert "位置/偏移" in texts, name
        assert "只比较 3 点折线的形状" not in texts, name  # 这句话是错的：DTW 不做平移归一
        assert "不可直接比较" in texts, name  # 两图各自用自己的 std 归一
        assert "**" not in texts, name  # 图注是纯文本，星号会被原样画进图片
    assert "质心间距" in captions["F10b_trajectory_affinity_cluster.png"]
    assert "spot_density" in captions["F10a_trajectory_affinity_fov.png"]
    assert "degenerate=False" in captions["F10a_trajectory_affinity_fov.png"]


def test_f10a_names_a_degenerate_trajectory_source(fake_temporal_dir, tmp_path, monkeypatch):
    """F10a 也要报轨迹来源与退化状态（和 F9 一样）：兜底路径的亲和不能静默当成真轨迹。"""
    analysis = fake_temporal_dir / "analysis"
    trajectory_path = analysis / "trajectories.npz"
    payload = figs._read_npz(trajectory_path, "Stage6b")
    payload["path"] = np.asarray("fallback_density")
    np.savez_compressed(trajectory_path, **payload)
    qc_path = analysis / "stage6b_qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    qc["trajectories"]["path"] = "fallback_density"
    qc["trajectories"]["degenerate"] = True
    qc_path.write_text(json.dumps(qc, ensure_ascii=False), encoding="utf-8")

    with pytest.warns(UserWarning, match="F10a"), _spy_figures(monkeypatch) as figures:
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)
    texts = _all_texts(figures[_RENDER_ORDER.index("F10a_trajectory_affinity_fov.png")])
    assert "fallback_density" in texts
    assert "degenerate=True" in texts


# —— F11 ——


def test_f11_match_events_follow_the_qc(fake_temporal_dir, tmp_path, monkeypatch):
    """每个帧对的堆叠条 = QC 的 one_to_one/merge/split；未匹配柱 = QC 的 prev/cur unmatched。"""
    rows = _instances(chained=True)
    table, qc = _match_rows(rows, chained=True)
    expected_events = [
        qc["fovs"][fov]["pairs"][pair]
        # 按数字元组排序（不是字符串序）：否则 "10->11" 会排到 "2->3" 前面，和柱子 x 位置错开。
        for fov in _FOVS
        for pair in sorted(qc["fovs"][fov]["pairs"], key=lambda text: tuple(int(part) for part in text.split("->")))
    ]

    with _spy_figures(monkeypatch) as figures:
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)
    figure = figures[_RENDER_ORDER.index("F11_match_events.png")]
    panels = [axis for axis in figure.axes if axis.patches]
    assert len(panels) == 2
    events_axis, unmatched_axis = panels
    event_totals = _panel_totals(events_axis)
    assert len(event_totals) == len(expected_events)
    for index, stats in enumerate(expected_events):
        assert abs(event_totals[float(index)] - (stats["n_one_to_one"] + stats["n_merge"] + stats["n_split"])) < 1e-9
    unmatched_totals = _panel_totals(unmatched_axis)
    for index, stats in enumerate(expected_events):
        assert abs(unmatched_totals[float(index)] - (stats["n_prev_unmatched"] + stats["n_cur_unmatched"])) < 1e-9
    assert len(table) == qc["totals"]["n_match_rows"]
    assert "merge" in _all_texts(figure) and "split" in _all_texts(figure)


def test_f11_rejects_match_qc_fovs_that_disagree_with_conditions(fake_temporal_dir, tmp_path):
    """`match_qc.json` 的 fovs 必须与 Stage6b 的 conditions 同一批：不一致就报错，不靠估算猜。"""
    path = fake_temporal_dir / "analysis" / "match_qc.json"
    qc = json.loads(path.read_text(encoding="utf-8"))
    qc["fovs"].pop("A3")
    path.write_text(json.dumps(qc, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="conditions"):
        figs.render_temporal(fake_temporal_dir, outdir=tmp_path)


def test_f11_survives_a_qc_without_any_match(fake_singleton_dir, tmp_path, monkeypatch):
    """全部实例未匹配时 F11 仍出图（右面板有柱子、左面板写明没有事件），不崩也不静默空白。"""
    with pytest.warns(UserWarning), _spy_figures(monkeypatch) as figures:
        figs.render_temporal(fake_singleton_dir, outdir=tmp_path)
    figure = figures[_RENDER_ORDER.index("F11_match_events.png")]
    events_axis = next(axis for axis in figure.axes if axis.patches)
    # 左面板仍然有柱子（高度全 0）而非空面板："没有事件"要画出来，不能悄悄空白。
    assert all(float(patch.get_height()) == 0.0 for patch in events_axis.patches)
    assert "没有" in _all_texts(figure)


def test_pooled_proportions_weight_by_bin_counts():
    """合并口径的单元测试：权重是每 (视野, 采样点) 的实例数，不是等权。"""
    proportions = np.array(
        [
            [[1.0, 0.0], [0.5, 0.5]],
            [[0.0, 1.0], [0.5, 0.5]],
        ]
    )
    counts = np.array([[3, 0], [1, 2]])
    pooled = figs._pooled_proportions(proportions, counts)
    assert np.allclose(pooled[0], [0.75, 0.25])  # (3×[1,0] + 1×[0,1]) / 4
    assert np.allclose(pooled[1], [0.5, 0.5])
