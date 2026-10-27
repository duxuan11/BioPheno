"""Stage 7 报告（`report.write_report`）与一键运行（`pipeline.run_all`）的测试。

全部用合成产物：`fake_full_output` 造出一棵与真实 `output/` **同构**的树
（`instances/` + `features/` + `preprocessed/` + `modules/` + `analysis/` + `figures/` + `logs/`），
所以测试不依赖 1.2 GB 真实数据，也不依赖任何阶段真的跑过。

测试盯住的点：

- 报告必须含任务说明要求的全部小节关键词（实例数 / 特征维度 / UMAP / K / 模块 / 匹配率 / 局限）；
- 实例数合计与数据事实 6741 的比较**必须显式写出来**（差额成因也要写，不允许沉默）；
- 每张**真实存在**的图都要在报告的图索引里，且写明对应论文哪张图；缺图时报告如实写"缺失"而不是崩掉；
- 有 `logs/run_all_timings.json` 时报告给出每阶段与总耗时；没有时也不崩（只写"未记录"）；
- 缺上游产物时抛**点名阶段**的中文错误，而不是静默生成一份数字残缺的报告；
- `run_all` 严格按 Stage1→2→3→4→5→6→6b→静态图→时间图→报告 的顺序调用，逐阶段写日志、
  写耗时 JSON、返回报告路径；任一阶段抛错就**停下并把错误再抛出**（脚本因此非零退出），
  且不继续跑后面的阶段、不写报告。
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from biopheno import config as cfg
from biopheno import pipeline, report

# (day, fov, 实例数)：合计 48 个实例，与数据事实 6741 差距很大——合成树本来就该差，
# 测试要看的是"报告把这个差额写出来了"，而不是差额恰好为 0。
_GROUPS: tuple[tuple[str, str, int], ...] = (
    ("day0", "A1", 10),
    ("day0", "A2", 8),
    ("day3", "A1", 9),
    ("day3", "A2", 7),
    ("day5", "A1", 8),
    ("day5", "A2", 6),
)
_REFERENCE_TOTAL: int = 6741
_FOVS: tuple[str, ...] = ("A1", "A2")
# Stage 2 在每个视野跳过的实例数（没有可用轮廓），用于报告里的"下游口径"解释。
_SKIPPED: dict[str, int] = {"A1": 1, "A2": 0}
# Stage 2 的特征列：给定一条 shape、一条 appearance、一条 ECT 家族列（曲率尺度归一会用到）与 motion 占位。
_FEATURE_NAMES: tuple[str, ...] = (
    "area",
    "equivalent_diameter",
    "curvature_mean",
    "mean_intensity",
    "motion_placeholder_mean_speed_global",
)
_MOTION_COLUMNS: int = 360
_FIGURES: tuple[str, ...] = (
    "F1_umap_clusters_patches.png",
    "F2_umap_by_day_fov.png",
    "F3_umap_density.png",
    "F4_contribution_scores.png",
    "F5_sam_modules.png",
    "F6_cluster_feature_bars.png",
    "F6b_cluster_day_composition.png",
    "F7_stacked_cluster_proportions.png",
    "F8_transition_graph.png",
    "F9_umap_trajectories.png",
    "F10a_trajectory_affinity_fov.png",
    "F10b_trajectory_affinity_cluster.png",
    "F11_match_events.png",
)


def _uid(fov: str, label: int, frame: int) -> str:
    """Stage 1 的 uid 口径：`{fov}_{label}_{frame_idx}`（与 `instances.build_catalog` 一致）。"""
    return f"{fov}_{int(label)}_{int(frame)}"


def _instances() -> list[dict]:
    """逐实例的 meta 行 + track 归属（同标签的前两个实例跨天连成 track）。"""
    rows: list[dict] = []
    next_track = 0
    for day, fov, count in _GROUPS:
        frame = int(cfg.DAYS[day])
        for label in range(1, count + 1):
            # 标签 1：三天都连（长度 3）；标签 2：只在 day0/day3 连（长度 2）；其余各自成一轨（长度 1）
            if label == 1 and fov not in _chain_seen:
                _chain_seen[fov] = next_track
                next_track += 1
            if label == 2 and fov not in _pair_seen:
                _pair_seen[fov] = next_track
                next_track += 1
            if label == 1:
                track_id = _chain_seen[fov]
            elif label == 2 and day != "day5":
                track_id = _pair_seen[fov]
            else:
                track_id = next_track
                next_track += 1
            rows.append(
                {
                    "instance_uid": _uid(fov, label, frame),
                    "spot_uid": _uid(fov, track_id, frame),
                    "track_id": track_id,
                    "day": day,
                    "fov": fov,
                    "frame_idx": frame,
                    "time_hours": float(cfg.TIME_HOURS[day]),
                    "label": label,
                    "centroid_row": 60.0 + 5.0 * label,
                    "centroid_col": 40.0 + 3.0 * frame,
                    "r0": 40,
                    "c0": 30,
                    "r1": 60 + label,
                    "c1": 50 + label,
                    "area_px": 300 + 20 * label,
                    "touches_border": False,
                    "is_small": label <= 2,
                }
            )
    return rows


# 跨天共享 track 的起点（每个视野一条"长度 3"与一条"长度 2"的 track）
_chain_seen: dict[str, int] = {}
_pair_seen: dict[str, int] = {}


def _write_stage1(root: Path, rows: list[dict]) -> None:
    """`instances/stage1_qc.json` + `instances/all_instances.csv`（列名与 Stage 1 落盘一致）。"""
    outdir = root / "instances"
    outdir.mkdir(parents=True, exist_ok=True)
    qc = {}
    for day, fov, count in _GROUPS:
        qc[f"{day}/{fov}"] = {
            "n_instances": count,
            "align_dy": 0,
            "align_dx": 1 if fov == "A2" else 0,
            "n_labels_discarded_contours": 0,
            "n_is_small": 2,
        }
    (outdir / "stage1_qc.json").write_text(json.dumps(qc, ensure_ascii=False), encoding="utf-8")
    columns = [
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
        "touches_border",
        "is_small",
    ]
    pd.DataFrame(rows)[columns].to_csv(outdir / "all_instances.csv", index=False)


def _write_stage2(root: Path, rows: list[dict]) -> None:
    """`features/stage2_qc.json` + `features/feature_names.json`（含 motion 占位列）。"""
    outdir = root / "features"
    outdir.mkdir(parents=True, exist_ok=True)
    qc = {}
    for day, fov, count in _GROUPS:
        skipped = _SKIPPED[fov]
        qc[f"{day}/{fov}"] = {
            "n_instances_meta": count,
            "n_instances_computed": count - skipped,
            "n_skipped_no_contour": skipped,
            "uid_cross_check": "ok",
            "skipped_reason": None,
            "n_shape_features_raw": 17,
            "n_shape_features_reduced": 6,
            "n_appearance_features": 8,
            "n_nonfinite_shape": 0,
            "n_nonfinite_appearance": 0,
            "n_motion_placeholder": _MOTION_COLUMNS,
            "n_features_total": 6 + 8 + _MOTION_COLUMNS,
            "n_nonfinite_ect_filled": 0,
            "shape_seconds": 1.0,
            "appearance_seconds": 2.0,
        }
    (outdir / "stage2_qc.json").write_text(json.dumps(qc, ensure_ascii=False), encoding="utf-8")
    (outdir / "feature_names.json").write_text(json.dumps(list(_FEATURE_NAMES)), encoding="utf-8")


def _write_stage3(root: Path) -> None:
    """`preprocessed/stage3_qc.json`（预处理前后的维度与剔除明细）。"""
    outdir = root / "preprocessed"
    outdir.mkdir(parents=True, exist_ok=True)
    qc = {
        "n_objects": 45,
        "n_features_in": len(_FEATURE_NAMES) - 1 + _MOTION_COLUMNS,
        "n_features_out": len(_FEATURE_NAMES) - 1,
        "n_dropped_features": _MOTION_COLUMNS,
        "variance_threshold_sigma": 2.0,
        "all_conditions": "single global condition (np.zeros(n))",
        "n_dropped_per_step": {
            "remove_zero_features": 0,
            "remove_nan_variance_features": _MOTION_COLUMNS,
            "remove_non_finite_features": 0,
            "remove_high_variance_features": 0,
        },
        "n_curvature_scaled": 1,
        "n_other_set_features": 0,
        "feature_sets": {"shape": 3, "appearance": 1, "motion": 0},
        "feature_groups": {"shape_binary": 1, "shape_ECC": 2, "appearance_intensity": 1},
        "degenerate_no_features": False,
        "dropped_zero": [],
        "dropped_all_nan": [list(_FEATURE_NAMES)[-1]],
        "dropped_partial_nan": [],
        "dropped_non_finite": [],
        "dropped_high_variance": [],
    }
    (outdir / "stage3_qc.json").write_text(json.dumps(qc, ensure_ascii=False), encoding="utf-8")


def _write_stage4(root: Path) -> None:
    """`modules/stage4_qc.json`（模块数、各模块特征数、contribution、Stage1 有 meta 但无特征的行数）。"""
    outdir = root / "modules"
    outdir.mkdir(parents=True, exist_ok=True)
    qc = {
        "module_selection": "legacy",
        "module_params": {"max_clust_len": None, "lam_smooth": 1.0, "buffer": 30},
        "random_state": 0,
        "n_objects": 45,
        "n_features": len(_FEATURE_NAMES) - 1,
        "n_instances_in_X": 45,
        "n_meta_rows": 46,
        "n_meta_not_in_X": 1,
        "module_expr_row_source": "features/all_features.csv 的 instance_uid 索引",
        "n_modules": 2,
        "module_sizes": {"1": 3, "2": 1},
        "module_features": {"1": ["area", "equivalent_diameter", "curvature_mean"], "2": ["mean_intensity"]},
        "type_contribution": [0.7, 0.3, 0.0],
        "scope_contribution": [0.5, 0.3, 0.2],
        "representative_patches": {"n_rows": 3, "n_candidates": 45},
        "n_rows_without_patch": 0,
        "timing_seconds": {"total": 18.0},
    }
    (outdir / "stage4_qc.json").write_text(json.dumps(qc, ensure_ascii=False), encoding="utf-8")


def _write_analysis(root: Path, rows: list[dict], *, zero_offsets: bool = False) -> None:
    """Stage 5/6/6b 的产物：UMAP 坐标、簇标签、K 扫描指标、tracks、match_qc、stage6b_qc。"""
    outdir = root / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    uids = [row["instance_uid"] for row in rows]
    clusters = [1 + (row["label"] - 1) % 2 for row in rows]
    pd.DataFrame(
        {
            "instance_uid": uids,
            "umap_1": [0.1 * i for i in range(len(rows))],
            "umap_2": [0.05 * i for i in range(len(rows))],
        }
    ).to_csv(outdir / "umap_2d.csv", index=False)
    pd.DataFrame(
        {
            "instance_uid": uids,
            "day": [row["day"] for row in rows],
            "fov": [row["fov"] for row in rows],
            "frame_idx": [row["frame_idx"] for row in rows],
            "time_hours": [row["time_hours"] for row in rows],
            "cluster": clusters,
            "cluster_raw": [value - 1 for value in clusters],
        }
    ).to_csv(outdir / "cluster_labels.csv", index=False)
    metrics = {
        "k": 2,
        "chosen_k": 2,
        "n_clusters": 2,
        "k_range": [2, 12],
        "k_scanned": [2, 3, 4],
        "inertia": {"2": 12.0, "3": 8.0, "4": 6.0},
        "silhouette": {"2": 0.61, "3": 0.55, "4": 0.48},
        "scaled_inertia": {"2": 0.10, "3": 0.12, "4": 0.15},
        "umap_params": dict(cfg.UMAP_PARAMS),
        "random_state": cfg.RANDOM_STATE,
        "n_objects": len(rows),
        "n_features": len(_FEATURE_NAMES) - 1,
        "cluster_sizes": {"1": 24, "2": 21},
        "eccentricity_score": {
            "source": "moment_eccentricity",
            "order": "ascending",
            "fallback_used": False,
            "mean_per_cluster_score": {"1": 0.42, "2": 0.66},
            "monotone_native": True,
            "monotone_score": True,
            "monotone_preprocessed": False,
        },
        "timing_seconds": {"total": 62.0},
    }
    (outdir / "kmeans_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")

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
    (outdir / "match_qc.json").write_text(
        json.dumps(_match_qc(rows, zero_offsets=zero_offsets), ensure_ascii=False), encoding="utf-8"
    )
    (outdir / "stage6b_qc.json").write_text(json.dumps(_stage6b_qc(rows), ensure_ascii=False), encoding="utf-8")


def _match_qc(rows: list[dict], *, zero_offsets: bool = False) -> dict:
    """按 tracks 的真实配对造 `match_qc.json`（与 tracks.csv 自洽，不用手写数字）。

    `zero_offsets=True` 造真实数据上实际发生的情形：相邻帧对的复合偏移全是 `(0, 0)`
    （相位相关估计被 ±8 px 上限拒掉后退回零偏移），用于覆盖“配准未生效”这条分支。
    """
    pair_offset = [0, 0] if zero_offsets else [0, 1]
    fovs: dict[str, dict] = {}
    totals = {"n_instances": len(rows), "n_tracks": len({(r["fov"], r["track_id"]) for r in rows})}
    agg = {"n_one_to_one": 0, "n_merge": 0, "n_split": 0, "n_prev_unmatched": 0, "n_cur_unmatched": 0}
    n_prev_total = n_cur_total = 0
    for fov in _FOVS:
        pairs: dict[str, dict] = {}
        for prev_frame, cur_frame in ((0, 1), (1, 2)):
            prev_tracks = {r["track_id"] for r in rows if r["fov"] == fov and r["frame_idx"] == prev_frame}
            cur_tracks = {r["track_id"] for r in rows if r["fov"] == fov and r["frame_idx"] == cur_frame}
            prev_rows = [r for r in rows if r["fov"] == fov and r["frame_idx"] == prev_frame]
            cur_rows = [r for r in rows if r["fov"] == fov and r["frame_idx"] == cur_frame]
            common = prev_tracks & cur_tracks
            pairs[f"{prev_frame}->{cur_frame}"] = {
                "n_prev": len(prev_rows),
                "n_cur": len(cur_rows),
                "n_candidates": len(common),
                "n_one_to_one": len(common),
                "n_merge": 0,
                "n_split": 0,
                "n_prev_unmatched": len(prev_rows) - len(common),
                "n_cur_unmatched": len(cur_rows) - len(common),
                "prev_link_rate": len(common) / len(prev_rows),
                "cur_link_rate": len(common) / len(cur_rows),
                "offset": [0, 0] if zero_offsets else [0, 1 if fov == "A2" else 0],
                "n_match_rows": len(common),
                "offset_exceeds_limit": False,
            }
            agg["n_one_to_one"] += len(common)
            agg["n_prev_unmatched"] += len(prev_rows) - len(common)
            agg["n_cur_unmatched"] += len(cur_rows) - len(common)
            n_prev_total += len(prev_rows)
            n_cur_total += len(cur_rows)
        fovs[fov] = {
            "registration": {
                "reference_frame": 0,
                "offset_to_reference": {"0": [0, 0], "1": pair_offset, "2": [0, 2]},
                "align_offsets": {"0": [0, 0], "1": [0, 0], "2": [0, 0]},
                "pair_offsets": {"0->1": pair_offset, "1->2": pair_offset},
                "warnings": [],
            },
            "pairs": pairs,
            "n_instances": sum(1 for r in rows if r["fov"] == fov),
            "n_tracks": len({r["track_id"] for r in rows if r["fov"] == fov}),
        }
    totals.update(agg)
    totals["n_pairs_offset_exceeding_limit"] = 0
    totals["n_tracks_len_ge2"] = len(
        {key for key in {(r["fov"], r["track_id"]) for r in rows} if _track_length(rows, key) >= 2}
    )
    return {
        "config": {
            "min_bbox_iou": 0.05,
            "max_cent_dist_px": 60.0,
            "align_max_shift_px": 8,
            "frame_axis": dict(cfg.DAYS),
        },
        "fovs": fovs,
        "totals": totals,
        "n_prev_total": n_prev_total,
        "n_cur_total": n_cur_total,
        "warnings": [],
        "match_table": {"file": "matches.csv", "n_rows": agg["n_one_to_one"], "columns": []},
    }


def _track_length(rows: list[dict], key: tuple[str, int]) -> int:
    return sum(1 for row in rows if (row["fov"], row["track_id"]) == key)


def _stage6b_qc(rows: list[dict]) -> dict:
    return {
        "time_axis": {
            "mode": "frame_idx 箱中心（frame_idx + 0.5）",
            "bin_frame_idx": [0, 1, 2],
            "bin_day": ["day0", "day3", "day5"],
            "bin_time_hours": [0.0, 72.0, 120.0],
            "n_bins": 3,
            "discrete_days_note": "时间轴是 3 个离散采样点，不是连续视频",
        },
        "subsets": {
            "proportions": {"n_instances": len(rows), "track_length_filter": "none"},
            "transitions": {"n_instances": len(rows), "track_length_filter": "相邻采样点的对"},
            "trajectories": {"n_instances": len(rows), "track_length_filter": "none"},
        },
        "trajectories": {
            "path": "spot_density",
            "degenerate": False,
            "degeneracy": {"reasons": []},
            "shipped_finiteness": {"all_finite": True},
        },
        "transitions": {"pseudocounts": 1.0, "orientation": "inter / probs 都是 (来源, 目标)"},
        "affinity": {"affinity": "dtw", "use_std_denom": True, "fallback": None},
        "warnings": [],
        "timing_seconds": {"total": 3.5},
    }


def _write_figures(root: Path, names=_FIGURES) -> None:
    """写占位 PNG（只有文件名与大小有意义：报告只看"图在不在"）。"""
    outdir = root / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (outdir / name).write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode("ascii"))


def _build_fake_output(root: Path, *, zero_offsets: bool = False) -> Path:
    """造一棵与真实 `output/` 同构的树（6 个 (day, fov)、48 个实例、2 个簇、2 个模块）。"""
    _chain_seen.clear()
    _pair_seen.clear()
    root.mkdir(parents=True, exist_ok=True)
    rows = _instances()
    _write_stage1(root, rows)
    _write_stage2(root, rows)
    _write_stage3(root)
    _write_stage4(root)
    _write_analysis(root, rows, zero_offsets=zero_offsets)
    _write_figures(root)
    return root


@pytest.fixture
def fake_full_output(tmp_path) -> Path:
    """完整的合成树（配准偏移非零：走“配准有偏移”那条分支）。"""
    return _build_fake_output(tmp_path / "output")


@pytest.fixture
def fake_zero_offset_output(tmp_path) -> Path:
    """配准复合偏移全为 (0, 0) 的合成树：真实数据上实际发生的那条分支。"""
    return _build_fake_output(tmp_path / "zero_offset_output", zero_offsets=True)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_report_contains_required_sections(fake_full_output):
    """任务说明 Step1：七个小节关键词与数据事实 6741 必须出现。"""
    path = report.write_report(fake_full_output)
    assert path == fake_full_output / "report.md"
    text = _text(path)
    for key in ["实例数", "特征维度", "UMAP", "K", "模块", "匹配率", "局限"]:
        assert key in text, f"报告缺少 {key} 小节"
    assert "6741" in text
    assert "PIXEL_RES" in text


def test_figure_index_uses_actual_fov_count_not_config(monkeypatch, fake_full_output):
    """图索引与局限里的“N 个视野”要按**数据里真实出现的视野数**写。

    原来写死“只有 3 个视野”：换成 1 个视野的数据（命令行 --images/--masks 只喂一个视野时就是这样）
    报告就会自相矛盾。这里把 cfg.FOVS 改成 5 个，证明数字来自数据而不是配置。
    """
    monkeypatch.setattr(cfg, "FOVS", ("A1", "A2", "A3", "A4", "A5"))
    text = _text(report.write_report(fake_full_output))
    assert "只有 2 个视野" in text  # fixture 里真实出现的是 A1/A2
    assert "只有 3 个视野" not in text
    assert "2 个视野是生物学重复" in text


def test_limitations_time_axis_follows_config(monkeypatch, fake_full_output):
    """“N 个离散采样点（day…，… h）”要按 cfg.DAYS / cfg.TIME_HOURS 写，不能照抄 0/72/120。"""
    monkeypatch.setattr(cfg, "TIME_HOURS", {"day0": 0.0, "day3": 48.0, "day5": 96.0})
    text = _text(report.write_report(fake_full_output))
    assert "**只有 3 个离散采样点**（day0 / day3 / day5，0 / 48 / 96 h）" in text


def test_instance_count_comparison_skipped_for_cli_input(monkeypatch, fake_full_output):
    """命令行指定输入时，不该拿"原 9 视野数据的 6741"当成本轮数据的期望值做差。

    6741 是 Spec §2 那份数据的标签总数；`--images/--masks` 换成 3 帧数据后它不属于本轮数据，
    继续写"数据事实是 6741、差额 …"就是报告自相矛盾。此时应如实说明"没有外部期望总数可比"。
    """
    for name in ("IMAGE_DIR", "MASK_DIR"):
        monkeypatch.setattr(cfg, name, getattr(cfg, name))
    monkeypatch.setattr(cfg, "IMAGE_DIR", Path("D:/somewhere/images"))
    monkeypatch.setattr(cfg, "MASK_DIR", Path("D:/somewhere/masks"))
    text = _text(report.write_report(fake_full_output))
    assert "不做 6741 那类比对" in text
    assert "差额 6693" not in text


def test_report_reports_instance_counts_and_explains_gap(fake_full_output):
    """实例数：逐 (day, fov) 的实测数、合计、与 6741 的差额及成因都要写出来。"""
    text = _text(report.write_report(fake_full_output))
    assert "day0/A1" in text and "10" in text  # 逐视野的实测数
    assert "合计" in text
    # 合成树合计 48，差额 6741 - 48 = 6693，必须显式写出
    assert "6693" in text
    assert "差额" in text
    # 下游口径的差额（Stage2 跳过 / Stage4 排除）也要有
    assert "跳过" in text and "排除" in text


def test_report_reports_cluster_day_composition(fake_full_output):
    """F6b 的读数：每簇的 day 占比与优势 day 份额要算在报告里（Spec §11-Q1 的证据）。"""
    text = _text(report.write_report(fake_full_output))
    assert "day 组成" in text
    assert "优势 day" in text
    assert "单日占比 ≥ 50%" in text


def test_report_names_dropped_features_and_the_area_consequence(fake_full_output):
    """§2 必须逐类列出被剔除的特征名，并写清 area 被 σ=2 规则剔除的后果（不能只给计数）。"""
    text = _text(report.write_report(fake_full_output))
    features = text.split("## 2. 特征维度")[1].split("## 3.")[0]
    # 默认 fixture 没有特征被高方差剔掉：报告要如实说“没有”，而不是编一句 area 被剔除
    assert "本轮没有特征被高方差规则" in features
    assert "`area` / `convex_area`" not in features
    assert "被剔除的特征名" in features
    assert f"{_MOTION_COLUMNS}" in features  # motion 占位列数

    qc_path = fake_full_output / "preprocessed" / "stage3_qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    qc["dropped_zero"] = ["Shape_Context_10", "Shape_Context_15"]
    qc["n_dropped_zero"] = 2
    qc["dropped_high_variance"] = ["area", "convex_area"]
    qc["n_dropped_high_variance"] = 2
    qc_path.write_text(json.dumps(qc, ensure_ascii=False), encoding="utf-8")

    features = _text(report.write_report(fake_full_output)).split("## 2. 特征维度")[1].split("## 3.")[0]
    assert "`Shape_Context_10`" in features and "`Shape_Context_15`" in features
    assert "**高方差（σ=2.0，2）：`area`、`convex_area`" in features
    assert "原始面积特征就在这一步被剔除" in features
    assert "进入 UMAP 之前" in features
    assert "没有原始面积特征" in features
    # 尺寸信息仍从保留下来的形状特征间接进入（fixture 的 equivalent_diameter 没被剔除）
    assert "`equivalent_diameter`" in features


def test_report_counts_only_rejected_registration_estimates(fake_zero_offset_output):
    """告警类型混合时，“被拒的相位相关估计”只数 offset_exceeds_limit，不是 len(warnings)。"""
    qc_path = fake_zero_offset_output / "analysis" / "match_qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    qc["fovs"]["A1"]["registration"]["warnings"] = [
        "配准偏移 (dy=12, dx=0) 超过 ±8 px 上限，偏移退回 (0, 0)",
        "空掩码（前景像素 prev=0、cur=0），偏移退回 (0, 0)",
    ]
    qc["fovs"]["A1"]["registration"]["warning_kinds"] = {
        report._REGISTRATION_REJECTION_KIND: 1,
        "empty_mask": 1,
    }
    qc["fovs"]["A2"]["registration"]["warnings"] = [
        "配准失败（RuntimeError: a），偏移退回 (0, 0)",
        "配准失败（RuntimeError: b），偏移退回 (0, 0)",
    ]
    qc["fovs"]["A2"]["registration"]["warning_kinds"] = {"upstream_error": 2}
    qc_path.write_text(json.dumps(qc, ensure_ascii=False), encoding="utf-8")

    text = _text(report.write_report(fake_zero_offset_output))
    assert "按告警类型计数，1 次相位相关估计" in text
    # 4 条字符串告警里只有 1 条是被拒的估计，报告不能说成 4 次
    assert "4 次相位相关估计" not in text

    # 旧产物（没有 warning_kinds）必须明说“数不出来”，而不是拿告警总数充当被拒次数
    without_kinds = json.loads(qc_path.read_text(encoding="utf-8"))
    for fov in ("A1", "A2"):
        without_kinds["fovs"][fov]["registration"].pop("warning_kinds")
    qc_path.write_text(json.dumps(without_kinds, ensure_ascii=False), encoding="utf-8")
    text = _text(report.write_report(fake_zero_offset_output))
    assert "无法只数“相位相关估计被上限拒收”这一类" in text

    # 只有部分视野带 warning_kinds 时同理：缺类型的视野会静默贡献 0 次被拒（等于低估），所以退回“数不出来”
    partial = json.loads(json.dumps(without_kinds))
    partial["fovs"]["A1"]["registration"]["warning_kinds"] = {
        report._REGISTRATION_REJECTION_KIND: 1,
        "empty_mask": 1,
    }
    qc_path.write_text(json.dumps(partial, ensure_ascii=False), encoding="utf-8")
    text = _text(report.write_report(fake_zero_offset_output))
    assert "无法只数“相位相关估计被上限拒收”这一类" in text
    assert "按告警类型计数" not in text
    assert "只有 1/2 个视野带告警类型标签" in text
    limitations = text.split("## 8. 局限与已知偏差")[1].split("## 9.")[0]
    assert "只有 1/2 个视野带告警类型标签" in limitations


def test_report_and_temporal_agree_on_warning_kind_names():
    """报告里的“被拒类型”常量必须与 Stage6 的告警类型常量同值（两边都写字符串，容易漂移）。"""
    from biopheno import temporal as temp

    assert report._REGISTRATION_REJECTION_KIND == temp.WARNING_KIND_OFFSET_EXCEEDS_LIMIT


def _geometry_payload(*, all_aligned: bool = True) -> dict:
    """造一份“由核验脚本产出”的几何核验 JSON（4 个帧对，对齐时零偏移 IoU ≈ 1）。"""

    def entry(zero: float, best: float, offset: list[int]) -> dict:
        return {
            "zero_offset_iou": zero,
            "best_iou": best,
            "best_offset": offset,
            "offset_unit": "px",
            "iou_gain": round(best - zero, 6),
            "aligned_at_zero": bool(best - zero <= 0.02),
        }

    pairs = {
        "A1": {
            "0->1": entry(0.984, 0.986, [0, 1]),
            "1->2": entry(0.991, 0.991, [0, 0]),
        },
        "A2": {
            "0->1": entry(0.972, 0.978, [1, 0]),
            "1->2": entry(0.963, 0.966, [0, -1] if all_aligned else [-6, 0]),
        },
    }
    if not all_aligned:
        pairs["A2"]["1->2"] = entry(0.610, 0.948, [-6, 0])
    n_aligned = sum(1 for fov in pairs for item in pairs[fov].values() if item["aligned_at_zero"])
    return {
        "source": report.GEOMETRY_CHECK_TOOL,
        "checked_at": "2026-10-22T21:10:00+08:00",
        "offset_unit": "px",
        "shift_step_px": 1,
        "max_shift_px": 8,
        "tolerance": 0.02,
        "pairs": pairs,
        "n_pairs": 4,
        "n_aligned_at_zero": n_aligned,
        "all_aligned_at_zero": n_aligned == 4,
    }


def _legacy_geometry_payload() -> dict:
    """旧产物：没有 `offset_unit`，`best_offset` / `max_shift_px` 存的是 4 px 网格步（曾被当成 px 引用）。"""
    payload = _geometry_payload()
    payload.pop("offset_unit")
    payload.pop("shift_step_px")
    payload["downsample_step"] = 4
    payload["max_shift_px"] = 8  # 旧口径：8 个网格步，真实窗口是 ±32 px
    for pairs in payload["pairs"].values():
        for item in pairs.values():
            item.pop("offset_unit")
            item["best_offset"] = [0, 1]  # 网格步 → 真实 4 px
    payload["pairs"]["A2"]["1->2"]["best_offset"] = [-3, 0]  # 网格步 → 真实 -12 px
    return payload


def test_report_verifies_zero_offset_geometry_when_the_check_says_aligned(fake_zero_offset_output):
    """有核验脚本产出的几何证据时：零偏移回退写“无害 + 数字”，不再写“必须去核对”。"""
    logdir = fake_zero_offset_output / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    check_path = logdir / report.GEOMETRY_CHECK_FILE
    # 手写的“自证”文件（没有 source/checked_at）不认
    check_path.write_text(
        json.dumps({"all_aligned_at_zero": True, "n_pairs": 4, "pairs": _geometry_payload()["pairs"]}),
        encoding="utf-8",
    )
    text = _text(report.write_report(fake_zero_offset_output))
    assert "已核验为无害" not in text
    assert "需人工核对" in text

    check_path.write_text(json.dumps(_geometry_payload(), ensure_ascii=False), encoding="utf-8")
    text = _text(report.write_report(fake_zero_offset_output))
    assert "零偏移回退已核验为无害" in text
    assert "掩码在平移意义上本来就对齐" in text
    # “对齐”只指没有整体平移，不能被读成“重叠度高”
    assert "不等于重叠度高" in text
    assert "0.963–0.991" in text  # 零偏移 IoU 的范围
    assert "4/4" in text  # 4 个帧对都判为已对齐
    assert "配准几何核验" in text and "零偏移已对齐" in text  # 逐帧对表
    # 偏移/窗口的单位必须写清是像素，且不能再出现“隔 N 个像素取点”的旧口径
    assert "逐像素扫 ±8 px" in text
    assert "隔 4 个像素取点" not in text
    limitations = text.split("## 8. 局限与已知偏差")[1].split("## 9.")[0]
    assert "已核验为无害" in limitations
    assert "需人工核对" not in limitations

    # 反过来：几何核验说“未对齐”时，报告必须仍要求人工核对
    check_path.write_text(json.dumps(_geometry_payload(all_aligned=False), ensure_ascii=False), encoding="utf-8")
    text = _text(report.write_report(fake_zero_offset_output))
    assert "已核验为无害" not in text
    assert "几何核验不支持“本来就对齐”" in text
    assert "未对齐的帧对必须人工核对" in text


def test_report_prefers_geometry_from_stage6_qc(fake_zero_offset_output):
    """Stage6 自己算过 `registration_geometry` 时优先用它（不必再依赖 logs 文件）。"""
    qc_path = fake_zero_offset_output / "analysis" / "match_qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    payload = _geometry_payload()
    payload.pop("source")
    payload.pop("checked_at")
    qc["registration_geometry"] = payload
    qc_path.write_text(json.dumps(qc, ensure_ascii=False), encoding="utf-8")

    text = _text(report.write_report(fake_zero_offset_output))
    assert "零偏移回退已核验为无害" in text
    assert "analysis/match_qc.json" in text and "registration_geometry" in text


def test_report_converts_legacy_grid_step_offsets_to_pixels(fake_zero_offset_output):
    """旧产物的偏移是“4 px 网格步”：必须换算成 px 并注明，不能把网格步当像素印出去。"""
    logdir = fake_zero_offset_output / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    check_path = logdir / report.GEOMETRY_CHECK_FILE
    check_path.write_text(json.dumps(_legacy_geometry_payload(), ensure_ascii=False), encoding="utf-8")

    text = _text(report.write_report(fake_zero_offset_output))
    # 8 个网格步 = ±32 px；-3 个网格步 = -12 px
    assert "±32 px" in text
    assert "(-12, 0)" in text
    assert "网格步" in text  # 明说换算过，不假装这是新产物的读数
    assert "±8 px 内就已经对齐" not in text

    # 新口径（px）来自 Stage6 QC、旧口径同时躺在 logs 里时，必须用新口径那份
    qc_path = fake_zero_offset_output / "analysis" / "match_qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    payload = _legacy_geometry_payload()
    payload.pop("source")
    payload.pop("checked_at")
    qc["registration_geometry"] = payload  # 旧口径落在 QC 里
    qc_path.write_text(json.dumps(qc, ensure_ascii=False), encoding="utf-8")
    check_path.write_text(json.dumps(_geometry_payload(), ensure_ascii=False), encoding="utf-8")

    text = _text(report.write_report(fake_zero_offset_output))
    assert f"{report.LOG_SUBDIR}/{report.GEOMETRY_CHECK_FILE}" in text
    assert "(-12, 0)" not in text
    assert "逐像素扫 ±8 px" in text


def test_report_notes_the_flat_adjusted_inertia_curve(fake_full_output):
    """相邻 k 的 adjusted inertia 几乎持平时提醒一句，别把 K=7 读成唯一解。"""
    text = _text(report.write_report(fake_full_output))
    assert "几乎持平" not in text  # fixture 的 k=2 与 k=3 差 0.02，不算持平
    metrics_path = fake_full_output / "analysis" / "kmeans_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["k"] = 7
    metrics["chosen_k"] = 7
    metrics["k_scanned"] = [6, 7, 8]
    metrics["inertia"] = {"6": 5852.611, "7": 4989.324, "8": 4439.978}
    metrics["silhouette"] = {"6": 0.367, "7": 0.357, "8": 0.344}
    metrics["scaled_inertia"] = {"6": 0.308, "7": 0.300, "8": 0.302}
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")

    text = _text(report.write_report(fake_full_output))
    assert "几乎持平" in text
    assert "k=7（0.300）" in text and "k=8（0.302）" in text
    assert "不宜过度解读" in text


def test_report_cross_references_day_dominance_in_limitations(fake_full_output):
    """§8（局限）必须交叉引用 §3.1 的天数主导读数，它不能只活在 §3.1。"""
    text = _text(report.write_report(fake_full_output))
    limitations = text.split("## 8. 局限与已知偏差")[1].split("## 9.")[0]
    assert "§3.1" in limitations
    assert "天数混杂" in limitations
    assert "单日占比 ≥ 50%" in limitations
    # 数字必须与 §3.1 一致（同一份读数，不是另算一遍）
    labels = pd.read_csv(fake_full_output / "analysis" / "cluster_labels.csv")
    dominance = report._day_dominance(labels)
    assert f"{dominance['n_dominated']} 个单日占比 ≥ 50%" in limitations
    assert f"{dominance['mean_share']:.1%}" in limitations


def test_report_states_f1_skips_small_instances(fake_full_output):
    """§7 要写清 F1 代表图跳过 is_small 的计数；整簇都是小实例时也要点名。"""
    text = _text(report.write_report(fake_full_output))
    assert "F1 代表图的口径" in text
    assert "跳过 `is_small`" in text
    assert "12 个小实例" in text  # 每组标签 1、2 是小实例（6 组 × 2）
    assert "每个簇都有非小实例" in text

    meta_path = fake_full_output / "instances" / "all_instances.csv"
    labels = pd.read_csv(fake_full_output / "analysis" / "cluster_labels.csv")
    cluster_one = set(labels.loc[labels["cluster"] == 1, "instance_uid"])
    meta = pd.read_csv(meta_path)
    meta.loc[meta["instance_uid"].isin(cluster_one), "is_small"] = True
    meta.to_csv(meta_path, index=False)

    text = _text(report.write_report(fake_full_output))
    assert "簇 1 里**没有非小实例**" in text


def test_report_lists_figures_with_paper_mapping(fake_full_output):
    """图索引：每张真实存在的图都要出现，且写明对应论文哪张图。"""
    text = _text(report.write_report(fake_full_output))
    for name in _FIGURES:
        assert name in text, f"报告图索引缺少 {name}"
    assert text.count("Fig.") >= len(_FIGURES)
    # F10 的两版必须都出现（Spec §6-Stage6-5 的双版本要求）
    assert "F10a" in text and "F10b" in text


def test_report_marks_missing_figures_without_crashing(fake_full_output):
    """只生成了部分图时：报告照样出，但要把缺的那几张写清楚（不许静默当没这回事）。"""
    (fake_full_output / "figures" / "F7_stacked_cluster_proportions.png").unlink()
    text = _text(report.write_report(fake_full_output))
    assert "缺失" in text
    assert "F7_stacked_cluster_proportions.png" in text


def test_report_records_stage_timings_when_present(fake_full_output):
    """`logs/run_all_timings.json` 存在时，报告给出每阶段与总耗时。"""
    logdir = fake_full_output / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / report.TIMING_FILE).write_text(
        json.dumps({"stages": {"stage1_instances": 101.5, "stage5_landscape": 62.25}, "total_seconds": 163.75}),
        encoding="utf-8",
    )
    text = _text(report.write_report(fake_full_output))
    assert "运行耗时" in text
    assert "101.5" in text and "163.75" in text


def test_report_without_timings_says_not_recorded(fake_full_output):
    """没有耗时文件时报告不崩，只写"未记录"。"""
    text = _text(report.write_report(fake_full_output))
    assert "未记录" in text


def test_report_declares_umap_determinism_check(fake_full_output):
    """可复现性：有核验脚本产出的 `logs/umap_repeat_check.json` 时写明坐标是否一致；没有时写明未验证。"""
    text = _text(report.write_report(fake_full_output))
    assert "未验证" in text
    logdir = fake_full_output / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / report.REPEAT_CHECK_FILE).write_text(
        json.dumps(
            {
                "source": report.REPEAT_CHECK_TOOL,
                "checked_at": "2026-10-22T20:30:00+08:00",
                "identical": True,
                "n_points": 48,
                "max_abs_diff": 0.0,
                "shipped_md5": "ddb8d9acbc608ab996554c90264ceed0",
                "rerun_md5": "ddb8d9acbc608ab996554c90264ceed0",
                "umap_params": dict(cfg.UMAP_PARAMS),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    text = _text(report.write_report(fake_full_output))
    assert "已核验" in text
    assert "一致" in text


def test_report_requires_check_file_from_the_verifier_script(fake_full_output):
    """可复现性不能自证：缺文件或文件不是本仓库脚本产出的，报告只能说"未核验"。"""
    logdir = fake_full_output / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    check_path = logdir / report.REPEAT_CHECK_FILE
    text = _text(report.write_report(fake_full_output))
    assert "未核验" in text
    assert "已核验" not in text
    # 手写的“自证”文件（有 verified/identical，但没有 source/checked_at）：报告不认
    check_path.write_text(
        json.dumps({"verified": True, "identical": True, "n_points": 48, "max_abs_diff": 0.0}),
        encoding="utf-8",
    )
    text = _text(report.write_report(fake_full_output))
    assert "未核验" in text
    assert "已核验" not in text
    # 核验脚本产出的文件：写“已核验”，并带上时间戳与 md5
    check_path.write_text(
        json.dumps(
            {
                "source": report.REPEAT_CHECK_TOOL,
                "checked_at": "2026-10-22T20:30:00+08:00",
                "identical": True,
                "n_points": 48,
                "max_abs_diff": 0.0,
                "shipped_md5": "ddb8d9acbc608ab996554c90264ceed0",
                "rerun_md5": "ddb8d9acbc608ab996554c90264ceed0",
                "umap_params": dict(cfg.UMAP_PARAMS),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    text = _text(report.write_report(fake_full_output))
    assert "已核验" in text
    assert "2026-10-22T20:30:00+08:00" in text
    assert "ddb8d9acbc608ab996554c90264ceed0" in text


def test_report_monotonicity_clause_follows_the_actual_value(fake_full_output):
    """“预处理量纲不单调是预期的”只在单调性真的是 false 时出现（否则报告自相矛盾）。"""
    text = _text(report.write_report(fake_full_output))
    assert "预处理量纲单调：否" in text
    assert "预处理量纲不单调是预期的" in text
    metrics_path = fake_full_output / "analysis" / "kmeans_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["eccentricity_score"]["monotone_preprocessed"] = True
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")
    text = _text(report.write_report(fake_full_output))
    assert "预处理量纲单调：是" in text
    assert "预处理量纲不单调是预期的" not in text


def test_report_treats_zero_offset_registration_as_suspicious(fake_zero_offset_output):
    """全零偏移是可疑模式：写事实 + 需人工核对，并把“配准未生效”列进局限。"""
    text = _text(report.write_report(fake_zero_offset_output))
    assert "可疑" in text
    assert "需人工核对" in text
    assert "假设" in text  # 因果解释必须明确标成假设
    assert "跨天匹配实际是在零偏移几何上跑的" in text
    assert "掩码本来就是对齐的" not in text  # 旧的、没有证据的断言
    limitations = text.split("## 8. 局限与已知偏差")[1].split("## 9.")[0]
    assert "跨天配准在本数据上未生效" in limitations
    assert "只依赖 bbox IoU 与质心距离" in limitations
    assert "配准失效的系统性原因需人工核对" in limitations


def test_report_states_matching_rate_denominator_and_merge_free_rate(fake_full_output):
    """分母是这些帧对里的全部实例（不是“参与匹配”的），并给出不含 merge 的口径。"""
    text = _text(report.write_report(fake_full_output))
    assert "分母是这些相邻帧对里**全部**实例" in text
    assert "不含 merge 的 `one_to_one / Σn_prev`" in text


def test_report_lists_warning_tail_instead_of_silently_truncating(fake_full_output):
    """告警 7 条时不能只列 5 条还说“共 7 条”：要写明只列前 5 条与其余部分的出处。"""
    qc_path = fake_full_output / "analysis" / "match_qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    qc["warnings"] = [f"告警 {i}" for i in range(1, 8)]
    qc_path.write_text(json.dumps(qc, ensure_ascii=False), encoding="utf-8")
    text = _text(report.write_report(fake_full_output))
    assert "配准告警共 7 条" in text
    assert "告警 5" in text and "告警 6" not in text
    assert "其余 2 条" in text
    assert "只列前 5 条" in text


def test_report_labels_offset_limit_row_as_final_offsets(fake_full_output):
    """“复合偏移超上限的帧对数”要写明是按最终使用的偏移统计（退回 0 后不再计入）。"""
    text = _text(report.write_report(fake_full_output))
    assert "按**最终使用**的偏移统计" in text


def test_report_header_states_the_actual_generating_call(fake_full_output):
    """直接调用报告函数时：写明只重写了报告、上游产物沿用上一次运行，不谎称是 run_all 的末阶段。"""
    text = _text(report.write_report(fake_full_output))
    assert "直接调用报告函数" in text
    assert "只重写了本文件" in text
    assert "的最后一个阶段生成" not in text


def test_report_rejects_tracks_without_long_tracks(fake_full_output):
    """全长度 1 的 track 说明时间编码失效：报告不许把它当成正常结果写出去。"""
    tracks = pd.read_csv(fake_full_output / "analysis" / "tracks.csv")
    tracks["track_id"] = range(len(tracks))
    tracks.to_csv(fake_full_output / "analysis" / "tracks.csv", index=False)
    with pytest.raises(RuntimeError, match="长度"):
        report.write_report(fake_full_output)


def test_report_missing_stage1_qc_names_the_stage(fake_full_output):
    """缺上游产物 → 点名阶段的中文错误（不是 KeyError，也不是一份数字残缺的报告）。"""
    (fake_full_output / "instances" / "stage1_qc.json").unlink()
    with pytest.raises(RuntimeError, match="Stage1"):
        report.write_report(fake_full_output)


# ---------------------------------------------------------------- run_all 的装配


def _stub(name: str, calls: list[str], *, boom: bool = False, touch: Path | None = None):
    """造一个记录调用顺序的假阶段；`boom=True` 时抛异常（模拟真实数据上的阶段失败）。"""

    def _run(*args, **kwargs):
        calls.append(name)
        if touch is not None:
            touch.parent.mkdir(parents=True, exist_ok=True)
            touch.write_text(name, encoding="utf-8")
        if boom:
            raise RuntimeError(f"{name} 炸了")

    return _run


def test_run_all_calls_stages_in_order_and_writes_logs(fake_full_output, monkeypatch, tmp_path):
    """`run_all` 的顺序、逐阶段日志、耗时 JSON、返回的报告路径。"""
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", fake_full_output)
    calls: list[str] = []
    for name in ("run_stage1", "run_stage2", "run_stage3", "run_stage4", "run_stage5", "run_stage6", "run_stage6b"):
        monkeypatch.setattr(pipeline, name, _stub(name, calls))

    static_dirs: list[tuple[Path, Path]] = []
    temporal_dirs: list[tuple[Path, Path]] = []

    def fake_static(analysis_dir, outdir, **kwargs):
        calls.append("render_static")
        static_dirs.append((Path(analysis_dir), Path(outdir)))
        return []

    def fake_temporal(analysis_dir, outdir=None):
        calls.append("render_temporal")
        temporal_dirs.append((Path(analysis_dir), Path(outdir)))
        return []

    monkeypatch.setattr(pipeline, "render_static", fake_static)
    monkeypatch.setattr(pipeline, "render_temporal", fake_temporal)

    path = pipeline.run_all()

    assert path == fake_full_output / "report.md"
    assert calls == [
        "run_stage1",
        "run_stage2",
        "run_stage3",
        "run_stage4",
        "run_stage5",
        "run_stage6",
        "run_stage6b",
        "render_static",
        "render_temporal",
    ]
    # 图都落在 <output>/figures：静态图吃 output/ 根，时间图吃 output/analysis/
    assert static_dirs == [(fake_full_output, fake_full_output / "figures")]
    assert temporal_dirs == [(fake_full_output / "analysis", fake_full_output / "figures")]
    # 每个阶段一份日志，且都写了内容
    logs = sorted(p.name for p in (fake_full_output / "logs").glob("*.log"))
    assert logs == [
        "stage1_instances.log",
        "stage2_features.log",
        "stage3_preprocess.log",
        "stage4_modules.log",
        "stage5_landscape.log",
        "stage6_tracks.log",
        "stage6b_temporal_stats.log",
        "stage7_static_figures.log",
        "stage7_temporal_figures.log",
    ]
    timing = json.loads((fake_full_output / "logs" / report.TIMING_FILE).read_text(encoding="utf-8"))
    assert set(timing["stages"]) == {
        "stage1_instances",
        "stage2_features",
        "stage3_preprocess",
        "stage4_modules",
        "stage5_landscape",
        "stage6_tracks",
        "stage6b_temporal_stats",
        "stage7_static_figures",
        "stage7_temporal_figures",
    }
    assert timing["total_seconds"] >= 0.0


def test_run_all_stops_on_stage_failure(fake_full_output, monkeypatch):
    """任一阶段抛错：立刻停下、把错误再抛出去（脚本因此非零退出）、不写报告。"""
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", fake_full_output)
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "run_stage1", _stub("run_stage1", calls))
    monkeypatch.setattr(pipeline, "run_stage2", _stub("run_stage2", calls))
    monkeypatch.setattr(pipeline, "run_stage3", _stub("run_stage3", calls, boom=True))
    monkeypatch.setattr(pipeline, "run_stage4", _stub("run_stage4", calls))
    report_path = fake_full_output / "report.md"
    report_path.unlink(missing_ok=True)

    with pytest.raises(RuntimeError, match="stage3_preprocess"):
        pipeline.run_all()
    assert calls == ["run_stage1", "run_stage2", "run_stage3"]
    assert not report_path.exists()
    # 失败也要留下耗时记录（在哪一步停的），否则事后无法判断跑了多久
    timing = json.loads((fake_full_output / "logs" / report.TIMING_FILE).read_text(encoding="utf-8"))
    assert timing["failed_stage"] == "stage3_preprocess"
    assert "stage3_preprocess" in timing["stages"]


def test_run_all_writes_report_from_real_artifacts(fake_full_output, monkeypatch):
    """报告那一步真的读盘：`run_all` 生成的 report.md 与直接调 `write_report` 的内容同源。

    这里不 stub `write_report`，只 stub 各阶段：断言的正是"最后一步确实去读产物并落 report.md"。
    """
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", fake_full_output)
    calls: list[str] = []
    for name in ("run_stage1", "run_stage2", "run_stage3", "run_stage4", "run_stage5", "run_stage6", "run_stage6b"):
        monkeypatch.setattr(pipeline, name, _stub(name, calls))
    monkeypatch.setattr(pipeline, "render_static", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "render_temporal", lambda *a, **k: [])

    path = pipeline.run_all()
    text = _text(path)
    assert "6741" in text and "匹配率" in text
    assert "运行耗时" in text
    # 由 run_all 生成时必须如实说是“最后一个阶段”，而不是“直接调用报告函数”
    assert "的最后一个阶段" in text
    assert "直接调用报告函数" not in text


def test_run_all_records_report_as_failed_stage(fake_full_output, monkeypatch):
    """报告写盘失败要记成 failed_stage="report"，不能记成上一阶段（stage7_temporal_figures）。"""
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", fake_full_output)
    calls: list[str] = []
    for name in ("run_stage1", "run_stage2", "run_stage3", "run_stage4", "run_stage5", "run_stage6", "run_stage6b"):
        monkeypatch.setattr(pipeline, name, _stub(name, calls))
    monkeypatch.setattr(pipeline, "render_static", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "render_temporal", lambda *a, **k: [])

    def boom(*args, **kwargs):
        raise RuntimeError("报告写盘炸了")

    monkeypatch.setattr(pipeline, "write_report", boom)
    with pytest.raises(RuntimeError, match="报告写盘炸了"):
        pipeline.run_all()
    timing = json.loads((fake_full_output / "logs" / report.TIMING_FILE).read_text(encoding="utf-8"))
    assert timing["failed_stage"] == "report"
