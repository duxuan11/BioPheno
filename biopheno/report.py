"""Stage 7 报告：把各阶段落盘的产物与 QC 汇总成 `output/report.md`（Spec §6-Stage7）。

本模块**只读盘**，不重算任何上游：实例数来自 Stage1 的 `stage1_qc.json` / `all_instances.csv`，
特征维度来自 Stage2/Stage3 的 QC，UMAP 参数与 K 来自 Stage5 的 `kmeans_metrics.json`，模块数来自
Stage4 的 `stage4_qc.json`，配准偏移与匹配统计来自 Stage6 的 `match_qc.json` + `tracks.csv`，时间轴口径
来自 Stage6b 的 `stage6b_qc.json`。图索引按文件名清单逐个检查**是否真的存在**，缺图写"缺失"而不是假装
有；缺必需产物则抛点名阶段的中文 `RuntimeError`（一份数字残缺的报告比没有报告更危险）。

报告是中文的，且刻意把三件事写死：

- **实例数合计 vs 数据事实 6741**：差额必须显式给出并解释成因，不许沉默；
- **`PIXEL_RES = 1.0`**：所有长度/面积都是像素单位（数据没有标尺信息，Spec §7-D1/D14）；
- **已知偏差清单**：无 motion、只有 3 个离散采样点（且两步宽度不等）、无批次校正、图与论文的对应关系。

`pipeline.run_all` 会把每阶段耗时写进 `logs/run_all_timings.json`，`logs/umap_repeat_check.json`（可选）
记录“重跑两次 UMAP 坐标是否逐位一致”的核验结果；两者存在时报告都会如实引用，不存在时写“未记录/未验证”。

三处“必须把话说清”的升级（均已测试钉住）：

- **被剔除的特征名逐类列出**（尤其 `dropped_high_variance` 里的 `area` / `convex_area`），并写清后果：
  §3 的景观与 §3.1 的天数主导读数是在**没有原始面积特征**的输入上得到的；
- **配准告警按类型计数**（`registration.warning_kinds`）：“被拒的相位相关估计”只数
  `offset_exceeds_limit`，不把空掩码/上游异常/复合偏移越界当同一件事（只要有任何一个视野缺
  `warning_kinds` 就退回“数不出来”，不拿缺项静默当 0）；零偏移回退是否无害由
  `registration_geometry`（或 `logs/registration_geometry_check.json`）的零偏移 IoU 核验回答，
  没核验过的旧产物一律写“必须去核对”，不靠猜；
- **几何核验的偏移/窗口都是像素**（`offset_unit="px"`）：与 `align_offsets` / `pair_offsets` 同单位，
  旧产物把偏移存成网格步时先乘回步长并在报告里注明。
"""

import json
import math
import statistics
from datetime import datetime
from pathlib import Path

import pandas as pd

from biopheno import config as cfg

# 数据事实：`D:/Datasets/test` 那份 9 视野掩码里应有的标签总数（Spec §10 验收标准 3）。
# 只对旧约定（`config.DATA_ROOT`）成立：命令行 `--images/--masks` 换了数据之后，这个数不属于本轮数据，
# 此时报告不做这项比对（见 `_section_instances`），否则会写出"数据事实是 6741、差额 -4318"这种自相矛盾的话。
EXPECTED_TOTAL_INSTANCES: int = 6741

# 一键运行（pipeline.run_all）的日志子目录与两份可选记录；报告读它们填"运行耗时 / 可复现性"两节。
LOG_SUBDIR: str = "logs"
TIMING_FILE: str = "run_all_timings.json"
REPEAT_CHECK_FILE: str = "umap_repeat_check.json"
# 核验 JSON 必须由这个脚本产出：报告只在 `source` 与它一致、且有 `checked_at` 时才敢写“已核验”。
REPEAT_CHECK_TOOL: str = "scripts/check_umap_repeat.py"
# 配准几何核验（“掩码本来就对齐吗”）同样是“不信自证文件”：只认这个脚本写的那份 JSON；
# Stage 6 自己算过的话优先用 `match_qc.json` 的 `registration_geometry`。
GEOMETRY_CHECK_FILE: str = "registration_geometry_check.json"
GEOMETRY_CHECK_TOOL: str = "scripts/check_registration_geometry.py"

# `temporal.WARNING_KIND_OFFSET_EXCEEDS_LIMIT` 的值（这里重写一份是为了让报告不 import 整个 Stage6；
# 两处必须一致，由 `test_report_and_temporal_agree_on_warning_kind_names` 钉住）。
_REGISTRATION_REJECTION_KIND: str = "offset_exceeds_limit"

# adjusted inertia 在相邻 k 上差小于这个值就写“几乎持平”：K 只是曲线最低点，不是唯一解。
_FLAT_ELBOW_TOLERANCE: float = 0.01

# 剔除的特征名逐条列出几个（超过时写“等共 N 个”，完整清单仍在 QC 里）。
_FEATURE_NAME_SHOW_LIMIT: int = 12

# “尺寸类”特征的命名关键词：用来证明 `area` 被剔除后尺寸信息仍从其它形状特征间接进入。
_SIZE_FEATURE_KEYWORDS: tuple[str, ...] = ("diameter", "perimeter", "axis_length", "bbox")

# 告警清单在报告里最多逐条列几条（超过时必须写明“只列前 N 条”+ 完整原文的位置）。
_WARNING_SHOW_LIMIT: int = 5

# 图与报告的落盘位置（Spec §9）。
FIGURES_SUBDIR: str = "figures"
REPORT_FILE: str = "report.md"

# —— 输入产物（与各阶段的落盘位置一致）——
_INSTANCES_SUBDIR: str = "instances"
_FEATURES_SUBDIR: str = "features"
_PREPROCESSED_SUBDIR: str = "preprocessed"
_MODULES_SUBDIR: str = "modules"
_ANALYSIS_SUBDIR: str = "analysis"
_STAGE1_QC: str = "stage1_qc.json"
_ALL_INSTANCES: str = "all_instances.csv"
_STAGE2_QC: str = "stage2_qc.json"
_FEATURE_NAMES: str = "feature_names.json"
_STAGE3_QC: str = "stage3_qc.json"
_STAGE4_QC: str = "stage4_qc.json"
_KMEANS_METRICS: str = "kmeans_metrics.json"
_CLUSTER_LABELS: str = "cluster_labels.csv"
_MATCH_QC: str = "match_qc.json"
_TRACKS_FILE: str = "tracks.csv"
_STAGE6B_QC: str = "stage6b_qc.json"

# `all_instances.csv` 里报告真正会读的列（缺列说明表被截断/换格式，必须报错而不是猜）。
_ALL_INSTANCES_COLUMNS: tuple[str, ...] = ("instance_uid", "day", "fov", "frame_idx", "label", "is_small")
# `tracks.csv` 里报告会读的列（track 长度分布、长度 ≥2 的条数、每视野 track 数都从这里算）。
_TRACKS_COLUMNS: tuple[str, ...] = ("track_id", "fov", "frame_idx")
# `cluster_labels.csv` 里报告会读的列（F6b 的读数：每簇的 day 组成，Spec §11-Q1 的证据）。
_CLUSTER_LABELS_COLUMNS: tuple[str, ...] = ("cluster", "day", "fov")

# 图索引：文件名 → (对应论文哪张图, 这张图在本地数据上是什么)。顺序 = 出图顺序（静态 F1→F6b、时间 F7→F11）。
_FIGURE_INDEX: tuple[tuple[str, str, str], ...] = (
    (
        "F1_umap_clusters_patches.png",
        "论文 Fig. 3b（表型景观 + 每簇代表形态）",
        "UMAP 景观按表型簇着色（1..K，blue→red）+ 每簇 4 张代表图（原始尺寸与 64×64 两版）",
    ),
    (
        "F2_umap_by_day_fov.png",
        "论文 Fig. 3b 的分面版（论文按条件/天数着色的景观）",
        "同一景观分别按 day、按 fov 着色，用来看景观是否被天数/视野主导",
    ),
    (
        "F3_umap_density.png",
        "论文 Fig. 3b 的密度层",
        "每簇一层、共用网格与色标的 2-D 密度场（面板之间可直接比较高低）",
    ),
    (
        "F4_contribution_scores.png",
        "论文 Fig. 2f–g / Fig. 3c 的 feature-contribution 面板",
        "shape/appearance/motion × global/regional/distribution 的 contribution score（motion 置空）",
    ),
    (
        "F5_sam_modules.png",
        "论文 Fig. 3d（SAM 模块 clustermap + 模块表达）",
        "SAM 模块 clustermap + 模块色条 + 簇 × 模块表达热图",
    ),
    (
        "F6_cluster_feature_bars.png",
        "论文 Fig. 3c（每簇可解释特征）",
        "每簇 area / eccentricity / intensity 的均值 ± SEM，motion 面板显式标注置空",
    ),
    (
        "F6b_cluster_day_composition.png",
        "论文无对应图（Spec §11-Q1 的诊断图）",
        "每簇的 day 组成堆叠条：一眼看出景观是否主要由天数/大小驱动",
    ),
    (
        "F7_stacked_cluster_proportions.png",
        "论文 Fig. 3c iii（各簇占比随时间）",
        "每个视野一个面板 + 按实例数加权的合并面板的簇占比堆叠条（{n_days} 个离散采样点）",
    ),
    (
        "F8_transition_graph.png",
        "论文 Fig. 3c iv（簇间转移）",
        "簇间转移有向图（只画观测到的边）+ 含自转移的转移概率热图（纯先验格打斜纹）",
    ),
    (
        "F9_umap_trajectories.png",
        "论文 Fig. 3c v（表型轨迹）",
        "每视野的 UMAP 表型轨迹；折线只画长度 ≥ 2 的 track",
    ),
    (
        "F10a_trajectory_affinity_fov.png",
        "论文 Fig. 4c–e 的视野级版本（Spec §6-Stage6-5 的 (a)）",
        "视野级轨迹亲和矩阵 + dendrogram（只有 {n_fovs} 个视野，叶子少、信息量有限）",
    ),
    (
        "F10b_trajectory_affinity_cluster.png",
        "论文 Fig. 4c–e 的簇级版本（Spec §6-Stage6-5 的 (b)）",
        "表型簇级轨迹亲和矩阵 + dendrogram（哪些表型簇的时间行为相似）",
    ),
    (
        "F11_match_events.png",
        "论文无对应图（QC 图）",
        "每个 (视野, 帧对) 的 one_to_one / merge / split 事件与两侧未匹配数",
    ),
)


def _read_json(path: Path, stage: str, what: str) -> dict | list:
    """读 JSON 产物；缺文件时抛点名阶段的中文错误（不回落到"空报告"）。"""
    if not path.exists():
        raise RuntimeError(f"找不到 {stage} 的产物 {path}（{what}）：请先跑该阶段，或用 run_all 一键跑完")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path, stage: str, what: str, columns: tuple[str, ...]) -> pd.DataFrame:
    """读 CSV 产物并校验必需列：列名是阶段间的契约，缺列说明产物不是这一代写的。"""
    if not path.exists():
        raise RuntimeError(f"找不到 {stage} 的产物 {path}（{what}）：请先跑该阶段，或用 run_all 一键跑完")
    frame = pd.read_csv(path)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} 缺少列 {missing}（{stage} 的 {what}）：产物格式与报告不符，请重跑该阶段")
    return frame


def _table(header: tuple[str, ...], rows: list[tuple]) -> list[str]:
    """Markdown 表格（定宽在报告里可读，窄终端也不会打乱）。"""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def _warning_lines(warnings: list[str], source: str) -> list[str]:
    """告警逐条列出；超过 `_WARNING_SHOW_LIMIT` 条时写明还有几条、去哪儿看完整原文。

    否则“共 6 条”下面只列 5 条，读者会以为漏了一条（或者以为第 6 条不重要）。
    """
    if not warnings:
        return []
    shown = warnings[:_WARNING_SHOW_LIMIT]
    lines = [f"  - {warning}" for warning in shown]
    if len(warnings) > len(shown):
        lines.append(f"  - …其余 {len(warnings) - len(shown)} 条见 {source}（本报告只列前 {len(shown)} 条）")
    return lines


def _fmt(value, digits: int = 3) -> str:
    """数值格式化：None/NaN 写成"—"，整数不带小数点，浮点保留指定位数。"""
    if value is None:
        return "—"
    if isinstance(value, float) and math.isnan(value):
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _order_key(key: str) -> tuple[int, int]:
    """按 `cfg.DAYS` / `cfg.FOVS` 的声明顺序排 `"day0/A1"` 这类键（未知键排在最后）。"""
    day, _, fov = key.partition("/")
    days = list(cfg.DAYS)
    fovs = list(cfg.FOVS)
    return (days.index(day) if day in days else len(days), fovs.index(fov) if fov in fovs else len(fovs))


def _name_list(names) -> str:
    """特征名清单：逐个列出（超过上限写“等共 N 个”），空清单写“（无）”。

    返回值自带括号/计数，调用处不要再包一层括号（否则会出现“（`a`（共 2 个））这种嵌套）。
    """
    names = [str(name) for name in names]
    if not names:
        return "（无）"
    shown = "、".join(f"`{name}`" for name in names[:_FEATURE_NAME_SHOW_LIMIT])
    if len(names) > _FEATURE_NAME_SHOW_LIMIT:
        return f"{shown} 等共 {len(names)} 个"
    return f"{shown}（共 {len(names)} 个）"


def _day_dominance(cluster_labels: pd.DataFrame) -> dict:
    """每簇的 day 组成（§3.1 与 §8 共用的一份读数，只读 `cluster_labels.csv`，不重算任何统计）。"""
    days = [str(day) for day in cfg.DAYS]
    frame = cluster_labels.copy()
    frame["day"] = frame["day"].astype(str)
    frame["cluster"] = frame["cluster"].astype(int)
    rows: list[dict] = []
    for cluster in sorted(frame["cluster"].unique()):
        members = frame[frame["cluster"] == cluster]
        shares = {day: float((members["day"] == day).mean()) for day in days}
        top_day = max(shares, key=lambda day: shares[day])
        rows.append(
            {
                "cluster": int(cluster),
                "n": len(members),
                "shares": shares,
                "top_day": top_day,
                "top_share": shares[top_day],
            }
        )
    shares_all = [row["top_share"] for row in rows]
    return {
        "days": days,
        "rows": rows,
        "n_clusters": len(rows),
        "n_dominated": sum(1 for share in shares_all if share >= 0.5),
        "mean_share": sum(shares_all) / len(shares_all) if shares_all else float("nan"),
    }


def _registration_totals(match_qc: dict) -> dict:
    """逐视野汇总配准告警：总数 + 按类型计数（缺 `warning_kinds` 时 `n_rejected=None`）。

    `n_warnings`（字符串告警数）**不能**当成“被拒的相位相关估计数”：空掩码、上游异常、
    复合偏移越界也都在同一个字符串列表里。只有 `warning_kinds["offset_exceeds_limit"]` 才是被拒次数。

    计数是**全有或全无**的：只要有一个视野没带 `warning_kinds`，就不给 `n_rejected`（那个视野会静默贡献
    0 次被拒，汇总值会偏小）。`n_fovs` / `n_fovs_with_kinds` 留给报告写明到底缺了多少。
    """
    fovs = match_qc.get("fovs", {})
    n_warnings = 0
    kinds: dict[str, int] = {}
    n_fovs = len(fovs)
    n_fovs_with_kinds = 0
    for fov in fovs:
        registration = fovs[fov].get("registration", {})
        n_warnings += len(registration.get("warnings", []))
        fov_kinds = registration.get("warning_kinds")
        if isinstance(fov_kinds, dict):
            n_fovs_with_kinds += 1
            for kind, count in fov_kinds.items():
                kinds[str(kind)] = kinds.get(str(kind), 0) + int(count)
    complete = n_fovs > 0 and n_fovs_with_kinds == n_fovs
    return {
        "n_warnings": n_warnings,
        "kinds": kinds if complete else None,
        "n_rejected": int(kinds.get(_REGISTRATION_REJECTION_KIND, 0)) if complete else None,
        "n_fovs": n_fovs,
        "n_fovs_with_kinds": n_fovs_with_kinds,
    }


def _kinds_missing_reason(n_reg: dict) -> str:
    """缺 `warning_kinds` 的原因：全缺，还是只缺一部分视野（后者会静默低估被拒次数）。"""
    if n_reg["n_fovs_with_kinds"] == 0:
        return "这份产物没有机器可读的告警类型（`warning_kinds`）"
    missing = n_reg["n_fovs"] - n_reg["n_fovs_with_kinds"]
    return (
        f"这份产物只有 {n_reg['n_fovs_with_kinds']}/{n_reg['n_fovs']} 个视野带机器可读的告警类型"
        f"（`warning_kinds`），另 {missing} 个视野没有（那部分被拒次数数不出来）"
    )


def _kinds_note(n_reg: dict) -> str:
    """局限段用的短版说明：没有可靠的告警类型计数时必须点明“只给了告警总数”。"""
    if n_reg["n_rejected"] is not None:
        return ""
    if n_reg["n_fovs_with_kinds"] == 0:
        return "（旧产物没有告警类型标签，这里只能给告警总数）"
    return (
        f"（只有 {n_reg['n_fovs_with_kinds']}/{n_reg['n_fovs']} 个视野带告警类型标签，"
        "缺标签的视野数不出被拒次数，这里只能给告警总数）"
    )


def _geometry_payload_from_qc(match_qc: dict) -> dict | None:
    """Stage6 自己算的 `registration_geometry`（有 `pairs` 且有帧对才算数）。"""
    in_qc = match_qc.get("registration_geometry")
    if isinstance(in_qc, dict) and isinstance(in_qc.get("pairs"), dict) and int(in_qc.get("n_pairs") or 0) > 0:
        return {**in_qc, "source": "match_qc.json", "checked_at": None}
    return None


def _geometry_payload_from_log(root: Path) -> dict | None:
    """核验脚本落盘的 `logs/registration_geometry_check.json`（手写的“自证”文件不认）。"""
    path = root / LOG_SUBDIR / GEOMETRY_CHECK_FILE
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(loaded, dict):
        return None
    if loaded.get("source") != GEOMETRY_CHECK_TOOL or not loaded.get("checked_at"):
        return None
    if not isinstance(loaded.get("pairs"), dict) or int(loaded.get("n_pairs") or 0) <= 0:
        return None
    return loaded


def _geometry_check(root: Path, match_qc: dict) -> dict | None:
    """取配准几何核验结果：优先用 Stage6 QC 里的，其次用核验脚本产出、带 `source`/`checked_at` 的日志。

    没沾过这两处的旧产物一律返回 `None`（宁可直接说“未核验”，也不拿“偏移全为 0”去猜几何是否对齐）。

    旧产物把 `best_offset` / `max_shift_px` 存成“每隔 `downsample_step` 个像素”的网格步，新产物带
    `offset_unit="px"`。因此优先选像素口径的那一份（旧口径仍能用，但要在报告里换算并注明）。
    """
    candidates = [
        payload
        for payload in (_geometry_payload_from_qc(match_qc), _geometry_payload_from_log(root))
        if payload is not None
    ]
    for payload in candidates:
        if payload.get("offset_unit") == "px":
            return payload
    return candidates[0] if candidates else None


def _geometry_provenance(geometry: dict) -> str:
    """写清“这份几何核验是谁、什么时候做的”（手写 JSON 不会被认）。"""
    if geometry.get("source") == "match_qc.json":
        return "来自 `analysis/match_qc.json` 的 `registration_geometry`（Stage 6 直接算）"
    return (
        f"来自 `{LOG_SUBDIR}/{GEOMETRY_CHECK_FILE}`（由 `{geometry.get('source')}` 在 "
        f"{geometry.get('checked_at')} 生成）"
    )


def _geometry_summary(geometry: dict) -> dict:
    """把逐帧对的几何核验压成几个数字（零偏移 IoU 范围/中位、最佳偏移、是否全对齐）。

    偏移一律归一到**像素**：新产物本来就在 px（`offset_unit="px"`）；旧产物存的是“每隔
    `downsample_step` 个像素”的网格步，这里乘回步长，并用 `legacy_grid_offsets` 标记让报告注明换算。
    """
    entries = [entry for _fov, pairs in geometry.get("pairs", {}).items() for entry in pairs.values()]
    zero = [float(entry["zero_offset_iou"]) for entry in entries]
    best = [float(entry["best_iou"]) for entry in entries]
    gains = [
        float(entry["iou_gain"]) if "iou_gain" in entry else float(entry["best_iou"]) - float(entry["zero_offset_iou"])
        for entry in entries
    ]
    # 旧产物（无 `offset_unit`）的偏移单位是网格步，乘回步长才是像素
    scale = 1 if geometry.get("offset_unit") == "px" else max(1, int(geometry.get("downsample_step") or 1))
    max_shift = geometry.get("max_shift_px")
    offsets = sorted({tuple(int(value) * scale for value in entry["best_offset"]) for entry in entries})
    return {
        "n_pairs": len(entries),
        "n_aligned": int(sum(1 for entry in entries if entry.get("aligned_at_zero"))),
        "all_aligned": bool(entries) and all(entry.get("aligned_at_zero") for entry in entries),
        "zero_min": min(zero),
        "zero_max": max(zero),
        "zero_median": statistics.median(zero),
        "best_min": min(best),
        "best_max": max(best),
        "max_gain": max(gains),
        "offsets": offsets,
        "tolerance": float(geometry.get("tolerance", 0.0)),
        # `max_shift_px` / `offsets` / `shift_step_px` 全部是像素
        "max_shift_px": None if max_shift is None else int(max_shift) * scale,
        "shift_step_px": int(geometry.get("shift_step_px") or scale),
        "offset_scale": scale,
        "legacy_grid_offsets": scale > 1,
    }


def _geometry_scan_text(stats: dict) -> str:
    """扫描方式的一句话（单位必须是 px；旧产物的网格步口径要注明已换算）。"""
    if stats["legacy_grid_offsets"]:
        return (
            f"以 {stats['shift_step_px']} px 为步长扫 ±{stats['max_shift_px']} px"
            "（旧产物把偏移按网格步存储，报告里已换算成 px）"
        )
    return f"逐像素扫 ±{stats['max_shift_px']} px"


def _offsets_text(offsets: list[tuple[int, ...]]) -> str:
    """把一组偏移写成 `(0, 0)、(0, 1)` 这样的紧凑形式。"""
    return "、".join("(" + ", ".join(str(int(value)) for value in offset) + ")" for offset in offsets) or "—"


def _section_instances(stage1_qc: dict, all_instances: pd.DataFrame, stage2_qc: dict, stage4_qc: dict) -> list[str]:
    """§1 实例数：逐 (day, fov) 的实测数、合计、与数据事实 6741 的差额与成因。

    命令行指定输入（`cfg.IMAGE_DIR` 非空）时不做 6741 比对：那个数属于旧约定那份数据，
    对新数据做差只会自相矛盾（见 `EXPECTED_TOTAL_INSTANCES` 的注释）。
    """
    keys = sorted(stage1_qc, key=_order_key)
    rows = [
        (
            key,
            stage1_qc[key]["n_instances"],
            f"({stage1_qc[key].get('align_dy', 0)}, {stage1_qc[key].get('align_dx', 0)})",
            stage1_qc[key].get("n_labels_discarded_contours", 0),
            stage1_qc[key].get("n_is_small", 0),
        )
        for key in keys
    ]
    total = int(sum(stage1_qc[key]["n_instances"] for key in keys))
    n_summary_rows = len(all_instances)
    lines = [
        "## 1. 实例数",
        "",
        "每个 (day, fov) 的实测实例数（Stage 1 为掩码里每个非零标签建一行：**只打标、不过滤**，",
        f"`is_small` 是面积 < {cfg.SMALL_AREA_FLAG_PX} px² 的标记而不是删除依据）：",
        "",
        *_table(("day/fov", "实例数", "掩码对齐偏移 (dy, dx)", "丢弃轮廓的标签数", "小实例数"), rows),
        "",
        (
            f"- **合计**：`stage1_qc.json` 求和 {total} 个实例；`all_instances.csv` 有 {n_summary_rows} 行"
            f"（两者{'一致' if total == n_summary_rows else '**不一致，产物可能不是同一次运行的**'}）。"
        ),
    ]
    if cfg.IMAGE_DIR is None:
        # 6741 是 Spec §2 那份 9 视野数据的标签总数。命令行 --images/--masks 换数据之后它不属于本轮数据，
        # 继续做差会写出"数据事实是 6741、差额 …"这种自相矛盾的句子，所以只在旧约定下比对。
        diff = EXPECTED_TOTAL_INSTANCES - total
        lines.append(
            f"- **与数据事实的比较**：掩码标签总数的数据事实是 {EXPECTED_TOTAL_INSTANCES}，本轮 Stage 1 入册 {total}，"
            f"差额 {diff}。"
        )
        if diff == 0:
            lines.append("  差额为 0：Stage 1 的实例数与数据事实完全一致，没有静默丢实例。")
        else:
            lines.append(
                "  差额不为 0：Stage 1 为每个非零标签都建一行，所以差额只能来自数据目录与"
                "数据事实不一致（视野缺失、掩码被重新分割、或读到的不是同一份数据），必须人工核对——"
                "本仓库不做任何静默丢弃。"
            )
    else:
        lines.append(
            f"- **与数据事实的比较**：本轮输入由命令行指定（`--images {cfg.IMAGE_DIR}`），没有外部给定的期望标签总数，"
            f"所以**不做 6741 那类比对**；实例数的唯一依据就是上面逐格实测值与 `stage1_qc.json`，"
            f"`all_instances.csv` 的 {n_summary_rows} 行已与它核对过。"
        )
    n_skipped = int(sum(entry.get("n_skipped_no_contour", 0) for entry in stage2_qc.values()))
    n_computed = int(sum(entry.get("n_instances_computed", 0) for entry in stage2_qc.values()))
    n_meta_not_in_x = int(stage4_qc.get("n_meta_not_in_X", 0))
    n_in_x = int(stage4_qc.get("n_instances_in_X", 0))
    lines.extend(
        [
            "",
            "下游口径（每个数都是可见的、可审计的）：",
            "",
            (
                f"- Stage 1 入册 {total} 个 → Stage 2 **跳过 {n_skipped} 个**（该标签没有可用轮廓，"
                f"仍留在 `all_instances.csv` 里）→ 有特征 {n_computed} 个 → 预处理矩阵 {n_in_x} 行；"
            ),
            (
                f"- Stage 4 报告还有 **{n_meta_not_in_x} 个**实例有 Stage 1 的 meta 行、但 Stage 2 没算特征"
                "（轮廓被丢弃），它们被**排除**在模块表达与代表图候选之外。"
            ),
            "",
        ]
    )
    return lines


def _section_features(stage2_qc: dict, feature_names: list, stage3_qc: dict) -> list[str]:
    """§2 特征维度：shape/appearance/motion 占位 → ECT 降维 → 预处理后的最终维度。"""
    entries = list(stage2_qc.values())
    shape_raw = max((int(entry.get("n_shape_features_raw", 0)) for entry in entries), default=0)
    shape_reduced = max((int(entry.get("n_shape_features_reduced", 0)) for entry in entries), default=0)
    appearance = max((int(entry.get("n_appearance_features", 0)) for entry in entries), default=0)
    motion = max((int(entry.get("n_motion_placeholder", 0)) for entry in entries), default=0)
    total = max((int(entry.get("n_features_total", 0)) for entry in entries), default=0)
    n_names = len(feature_names)
    n_ect_filled = max((int(entry.get("n_nonfinite_ect_filled", 0)) for entry in entries), default=0)
    rows = [
        ("shape（SAM 形状族，ECT 降维**前**）", shape_raw),
        (
            f"shape（ECT 降维**后**：ECT 家族压到 {cfg.ECC_N_DIM} 维，非 ECT 的形状列原样保留）",
            shape_reduced,
        ),
        ("appearance（外观族：亮度直方图 + SIFT 等）", appearance),
        ("motion（**置空占位**：同名 NaN 列，有意不算）", motion),
        ("Stage 2 特征表总维数", total),
        ("`feature_names.json` 的列数（交叉核对）", n_names),
        ("ECT 降维前被填 0 的非有限值个数", n_ect_filled),
    ]
    dropped = {str(step): int(value) for step, value in stage3_qc.get("n_dropped_per_step", {}).items()}
    dropped_zero = list(stage3_qc.get("dropped_zero", []))
    dropped_all_nan = list(stage3_qc.get("dropped_all_nan", []))
    dropped_partial_nan = list(stage3_qc.get("dropped_partial_nan", []))
    dropped_non_finite = list(stage3_qc.get("dropped_non_finite", []))
    dropped_high_variance = list(stage3_qc.get("dropped_high_variance", []))
    sigma = stage3_qc.get("variance_threshold_sigma")
    # 高方差剔除里最容易被误读的一步：原始面积特征（area / convex_area）就在这一步被删掉
    dropped_names = {
        str(name)
        for names in (dropped_zero, dropped_all_nan, dropped_partial_nan, dropped_non_finite, dropped_high_variance)
        for name in names
    }
    size_features = [
        str(name)
        for name in feature_names
        if str(name) not in dropped_names and any(key in str(name).lower() for key in _SIZE_FEATURE_KEYWORDS)
    ]
    if any(str(name) in ("area", "convex_area") for name in dropped_high_variance):
        consequence = (
            f"- **后果（读者最容易误读的一条）**：按论文的 σ={sigma} 高方差剔除规则，`area` / `convex_area` 在"
            f"**进入 UMAP 之前**就被删掉了，所以 §3 的景观（{stage3_qc.get('n_features_out')} 维输入）与 §3.1 的"
            "“由采样天数驱动”读数，都是**在没有原始面积特征**的情况下得到的；尺寸信息仍通过保留的形状特征"
            f"间接进入：{_name_list(size_features)}，但景观里没有 `area` 这个直接量。"
        )
    elif dropped_high_variance:
        consequence = (
            f"- **后果**：σ={sigma} 的高方差规则剔除了 {_name_list(dropped_high_variance)}，"
            f"§3 的景观是在这 {stage3_qc.get('n_features_out')} 维上跑的。"
        )
    else:
        consequence = f"- **后果**：本轮没有特征被高方差规则（σ={sigma}）剔除，§3 的景观包含了预处理后的全部维度。"
    lines = [
        "## 2. 特征维度",
        "",
        *_table(("特征块", "维度"), [(name, value) for name, value in rows]),
        "",
        (
            f"- motion 家族**有意置空**：本地数据每 2–3 天才有一帧，帧间光流没有物理意义，所以不计算、"
            f"保留 {motion} 个同名 NaN 列（而不是填 0，避免“零位移”被误读成“静止”）。这 {motion} 列在预处理"
            "的 NaN 方差那一步整体剔除，所以不进 UMAP。"
        ),
        (
            f"- Stage 3 预处理：输入 {stage3_qc.get('n_features_in')} 维 → 输出 "
            f"{stage3_qc.get('n_features_out')} 维（共剔除 {stage3_qc.get('n_dropped_features')} 维）。"
            f"剔除按原因互斥记账：零值/零方差 {stage3_qc.get('n_dropped_zero')}、全 NaN "
            f"{stage3_qc.get('n_dropped_all_nan')}、部分 NaN {stage3_qc.get('n_dropped_partial_nan')}、"
            f"非有限 {stage3_qc.get('n_dropped_non_finite')}、高方差（σ="
            f"{stage3_qc.get('variance_threshold_sigma')}）{stage3_qc.get('n_dropped_high_variance')}。"
            f"各步剔除数：{dropped}。"
        ),
        "- **被剔除的特征名（按原因逐类列出，不是只给计数）**：",
        f"  - 零值/零方差（{stage3_qc.get('n_dropped_zero', len(dropped_zero))}）：{_name_list(dropped_zero)}；",
        (
            f"  - 全 NaN（{stage3_qc.get('n_dropped_all_nan', len(dropped_all_nan))}）：{_name_list(dropped_all_nan)}"
            + (
                f"（motion 占位列共 {motion} 个，"
                + (
                    "两者相等，即这一步剔掉的就是整个 motion 家族）"
                    if motion == len(dropped_all_nan)
                    else "两者不等，说明还有别的全 NaN 列，需人工看一眼）"
                )
                if dropped_all_nan
                else "；"
            )
            + "；"
        ),
        f"  - 部分 NaN（{stage3_qc.get('n_dropped_partial_nan', len(dropped_partial_nan))}）：{_name_list(dropped_partial_nan)}；",
        f"  - 非有限（{stage3_qc.get('n_dropped_non_finite', len(dropped_non_finite))}）：{_name_list(dropped_non_finite)}；",
        (
            f"  - **高方差（σ={sigma}，{stage3_qc.get('n_dropped_high_variance', len(dropped_high_variance))}）："
            f"{_name_list(dropped_high_variance)}**"
            + (
                "——这一条最需要读者知道：原始面积特征就在这一步被剔除。"
                if any(str(name) in ("area", "convex_area") for name in dropped_high_variance)
                else "。"
            )
        ),
        consequence,
        (
            f"- 归一化方式：曲率尺度归一 {stage3_qc.get('n_curvature_scaled')} 列（必须在 power transform 之前），"
            f"随后 Yeo-Johnson + 标准化，条件数组为 `{stage3_qc.get('all_conditions')}`——"
            "即**全局单条件拟合**，不按天/视野分别 z-score（否则会抹掉要展示的天间差异）。"
        ),
        (
            f"- SAM 特征族/特征集等权后（`weight_sam_features(set_weighting=True, group_weighting=True)`）的各 set 列数："
            f"{stage3_qc.get('feature_sets')}；不属于任何 SAM 特征集的名字 {stage3_qc.get('n_other_set_features')} 个"
            "（> 0 说明有命名漂移，需人工看一眼）。"
        ),
        "",
    ]
    if n_names != total:
        lines.append(
            f"**警告**：`feature_names.json` 有 {n_names} 列而 Stage 2 QC 记的总维数是 {total}，"
            "两者应当相等——产物可能不是同一次运行的。"
        )
        lines.append("")
    return lines


def _section_umap(kmeans: dict, dominance: dict) -> list[str]:
    """§3 UMAP 与表型簇：参数、K 扫描的全部指标、选中的 K、簇大小、编号单调性、day 组成。"""
    params = kmeans.get("umap_params", {})
    ks = [str(k) for k in kmeans.get("k_scanned", [])]
    inertia = kmeans.get("inertia", {})
    silhouette = kmeans.get("silhouette", {})
    scaled = kmeans.get("scaled_inertia", {})
    param_text = "、".join(f"{key}={value}" for key, value in params.items())
    rows = [(k, _fmt(inertia.get(k)), _fmt(silhouette.get(k)), _fmt(scaled.get(k))) for k in ks]
    ecc = kmeans.get("eccentricity_score", {})
    cluster_sizes = kmeans.get("cluster_sizes", {})
    mono_native = ecc.get("monotone_native")
    mono_score = ecc.get("monotone_score")
    mono_preprocessed = ecc.get("monotone_preprocessed")
    # “预处理量纲不单调是预期的”只在该值**真的是 false** 时写：否则同一段里会既说单调又说不单调。
    numbering = (
        f"- 编号单调性检查：score 来源 `{ecc.get('source', '—')}`（论文口径是**原生量纲**的偏心率），"
        f"原生量纲单调：{_fmt(mono_native)}、score 量纲单调：{_fmt(mono_score)}、"
        f"预处理量纲单调：{_fmt(mono_preprocessed)}；各簇平均 score：{ecc.get('mean_per_cluster_score', '—')}。"
    )
    if mono_preprocessed is False:
        numbering += "预处理量纲不单调是预期的（各簇分布宽度不同），它只进 QC，不参与编号。"
    numbering += "编号以原生量纲为准。"
    size_rows = [(f"簇 {label}", size) for label, size in sorted(cluster_sizes.items(), key=lambda kv: int(kv[0]))]
    lines = [
        "## 3. UMAP 景观与表型簇（K）",
        "",
        (
            f"- **UMAP 参数**（论文默认）：{param_text}；`random_state={kmeans.get('random_state')}`，"
            f"metric 与参数都来自 `config.UMAP_PARAMS`，没有逐天/逐视野分别嵌入。"
        ),
        (
            f"- 输入：{kmeans.get('n_objects')} 个实例 × {kmeans.get('n_features')} 维预处理特征（三天实例**一次**嵌入，"
            "否则轨迹画不到同一个空间里）。"
        ),
        (
            f"- **K 的扫描与选择**：扫描范围 {kmeans.get('k_range')}（SPOT 内部是 `range(k_range[0], k_range[1])`，"
            f"**不含上界**，所以实际只扫了 {ks}）；按 adjusted inertia（`inertia_ / 总平方和 + 0.02*k`）取最小者，"
            f"选中 **K={kmeans.get('k')}**（= `chosen_k`，共 {kmeans.get('n_clusters')} 个表型簇）。"
        ),
        "",
        "elbow 与轮廓系数的证据（`kmeans_metrics.json`，同一个 2-D 嵌入、同一组 k-means 同参）：",
        "",
        *_table(("k", "inertia（簇内平方和）", "silhouette（平均轮廓系数）", "adjusted inertia"), rows),
        "",
        f"- elbow 规则原文：{kmeans.get('scan_rule', '—')}",
        *_flat_elbow_lines(kmeans),
        (
            f"- silhouette 只作旁证：选 K 的口径是论文的 adjusted inertia，不按 silhouette 最大去挑"
            f"（本轮 silhouette 最大的 k 是 "
            f"{max(silhouette, key=lambda key: silhouette[key]) if silhouette else '—'}）。"
        ),
        "",
        "簇大小（1-based，按簇内平均 eccentricity 升序编号）：",
        "",
        *(_table(("簇", "实例数"), size_rows) if size_rows else ["（`kmeans_metrics.json` 没有 `cluster_sizes`）"]),
        "",
        numbering,
        "",
        *_section_day_composition(dominance),
    ]
    return lines


def _flat_elbow_lines(kmeans: dict) -> list[str]:
    """相邻 k 的 adjusted inertia 几乎持平时提醒一句：K 是曲线最低点，不是唯一解。"""
    scaled = kmeans.get("scaled_inertia", {})
    chosen = str(kmeans.get("k"))
    if chosen not in scaled:
        return []
    neighbours = [str(int(chosen) + delta) for delta in (-1, 1) if str(int(chosen) + delta) in scaled]
    if not neighbours:
        return []
    nearest = min(neighbours, key=lambda name: abs(float(scaled[name]) - float(scaled[chosen])))
    gap = float(scaled[nearest]) - float(scaled[chosen])
    if gap >= _FLAT_ELBOW_TOLERANCE:
        return []
    return [
        (
            f"- adjusted inertia 在相邻的 k 上**几乎持平**：k={chosen}（{_fmt(scaled[chosen])}）与 k={nearest}"
            f"（{_fmt(scaled[nearest])}）只差 {gap:.3f}（< {_FLAT_ELBOW_TOLERANCE}），所以 K={chosen} 只是曲线的"
            "最低点，不宜过度解读成“唯一合理的簇数”。"
        )
    ]


def _section_day_composition(dominance: dict) -> list[str]:
    """§3.1 景观是否被天数主导：每簇的 day 组成（F6b 的读数，Spec §11-Q1 / D15 的证据）。

    只读 `cluster_labels.csv`（`_day_dominance` 已算好），不重算任何统计：每簇的 day 占比、
    占比最大的那一天与它的份额。这也是“不预先回归面积/天数”、先把混杂因素**如实报出来**的那一步。
    """
    days = dominance["days"]
    rows = [
        (f"簇 {row['cluster']}", row["n"], *(f"{row['shares'][day]:.1%}" for day in days), row["top_day"])
        for row in dominance["rows"]
    ]
    lines = [
        "### 3.1 景观是否被天数主导（F6b 的读数）",
        "",
        "每簇的 day 组成（`cluster_labels.csv` 直接读数，`n` = 簇内实例数）：",
        "",
        *_table(("簇", "n", *days, "优势 day"), rows),
        "",
        (
            f"- {dominance['n_clusters']} 个簇里有 **{dominance['n_dominated']} 个**单日占比 ≥ 50%，优势 day 占比的平均值是 "
            f"**{dominance['mean_share']:.1%}**。这就是 Spec §11-Q1 要的读数：份额越接近 100%，表型景观越可能主要由"
            "**采样天数**（而不是随时间稳定的形态表型）驱动，F6b 的堆叠条是它的证据图。"
        ),
        "- 按 D15 的口径，**没有**预先回归面积/天数；这里只如实报告，不做“去掉天数后重跑”的处理。",
        "",
    ]
    return lines


def _section_modules(stage4_qc: dict) -> list[str]:
    """§4 SAM 模块：模块数、各模块特征数、选择方式与参数、contribution。"""
    sizes = stage4_qc.get("module_sizes", {})
    features = stage4_qc.get("module_features", {})
    rows = [
        (
            f"M{module}",
            sizes.get(module, "—"),
            "、".join(features.get(module, [])[:8]) + (" …" if len(features.get(module, [])) > 8 else ""),
        )
        for module in sorted(sizes, key=int)
    ]
    lines = [
        "## 4. SAM 模块",
        "",
        (
            f"- 共 **{stage4_qc.get('n_modules')} 个模块**（选择方式 `{stage4_qc.get('module_selection')}`，"
            f"论文参数 {stage4_qc.get('module_params')}；`random_state={stage4_qc.get('random_state')}`）。"
        ),
        (
            f"- 输入 {stage4_qc.get('n_features')} 个预处理后的特征（行身份来自 "
            f"`{stage4_qc.get('module_expr_row_source')}`，不依赖表的行序）。"
        ),
        "",
        *_table(("模块", "特征数", "特征名（最多列 8 个）"), rows),
        "",
        (
            f"- contribution（对 shape/appearance/motion 三个类型、global/regional/distribution 三个 scope）："
            f"type {stage4_qc.get('type_contribution')}、scope {stage4_qc.get('scope_contribution')}；"
            "motion 那一项为 0 是**置空**造成的，不是模块里真的没有运动特征的信息。"
        ),
        f"- 模块发现耗时 {stage4_qc.get('timing_seconds', {}).get('total', '—')} s。",
        "",
    ]
    return lines


def _track_stats(tracks: pd.DataFrame) -> dict:
    """从 `tracks.csv` 直接算 track 长度分布（报告不引用 QC 的二手数字，两者在后面交叉核对）。"""
    sizes = tracks.groupby("track_id").size()
    histogram = {int(length): int(count) for length, count in sizes.value_counts().sort_index().items()}
    per_fov = {}
    for fov, group in tracks.groupby("fov", sort=True):
        fov_sizes = group.groupby("track_id").size()
        per_fov[str(fov)] = {
            "n_instances": len(group),
            "n_tracks": int(fov_sizes.size),
            "n_len_ge2": int((fov_sizes >= 2).sum()),
        }
    return {
        "n_instances": len(tracks),
        "n_tracks": int(sizes.size),
        "n_len_ge2": int((sizes >= 2).sum()),
        "track_length_histogram": histogram,
        "per_fov": per_fov,
    }


def _section_matching(match_qc: dict, tracks: pd.DataFrame, geometry: dict | None = None) -> list[str]:
    """§5 跨天配准与匹配：对齐/配准偏移、匹配率、merge/split/未匹配、track 长度分布。"""
    stats = _track_stats(tracks)
    if stats["n_len_ge2"] == 0:
        raise RuntimeError(
            f"`tracks.csv` 里 {stats['n_tracks']} 条 track 全部长度为 1（长度 ≥2 的有 {stats['n_len_ge2']} 条）："
            "帧轴编码或跨天匹配失效，这不是可以写进报告的正常结果，请先查 Stage6"
        )
    config = match_qc.get("config", {})
    totals = match_qc.get("totals", {})
    fovs = match_qc.get("fovs", {})
    lines = [
        "## 5. 跨天配准与匹配",
        "",
        (
            f"候选规则：bbox IoU ≥ {config.get('min_bbox_iou')} **或** 质心距离 ≤ {config.get('max_cent_dist_px')} px；"
            f"配准平移搜索上限 ±{config.get('align_max_shift_px')} px；帧轴 {config.get('frame_axis')}、"
            f"时间轴（小时）{config.get('time_hours')}。"
        ),
        "",
        "### 5.1 配准与对齐偏移",
        "",
    ]
    rows = []
    for fov in sorted(fovs):
        registration = fovs[fov].get("registration", {})
        rows.append(
            (
                fov,
                registration.get("reference_frame"),
                registration.get("align_offsets"),
                registration.get("offset_to_reference"),
                registration.get("pair_offsets"),
                len(registration.get("warnings", [])),
            )
        )
    lines.extend(
        [
            (
                "`align_offsets` 是 Stage 1 掩码-原图对齐时求出的平移（先按它把掩码搬到 meta 的坐标系），"
                "`offset_to_reference` 是各采样点相对本视野第一帧的配准平移，`pair_offsets` 是相邻帧对的复合偏移。"
            ),
            "",
            *_table(("视野", "参考帧", "align_offsets", "offset_to_reference", "pair_offsets", "配准告警数"), rows),
            "",
        ]
    )
    warnings = list(match_qc.get("warnings", []))
    shown_text = (
        f"下面只列前 {_WARNING_SHOW_LIMIT} 条，其余见 `match_qc.json` 的 `warnings`"
        if len(warnings) > _WARNING_SHOW_LIMIT
        else "全部列在下面，完整原文见 `match_qc.json` 的 `warnings`"
    )
    lines.append(f"- 配准告警共 {len(warnings)} 条（{shown_text}）" if warnings else "- 配准告警 0 条。")
    lines.extend(_warning_lines(warnings, "`match_qc.json` 的 `warnings`"))
    # 配准到底有没有生效：逐帧对看复合偏移是不是全 0，以及有多少次估计被上限拒掉
    n_reg = _registration_totals(match_qc)
    geometry_stats = _geometry_summary(geometry) if geometry else None
    pair_offsets_all = [
        tuple(offset) for fov in fovs for offset in fovs[fov].get("registration", {}).get("pair_offsets", {}).values()
    ]
    if pair_offsets_all and all(offset == (0, 0) for offset in pair_offsets_all):
        limit = config.get("align_max_shift_px")
        if n_reg["n_rejected"] is not None:
            rejected = (
                f"其中按告警类型计数，{n_reg['n_rejected']}/{len(pair_offsets_all)} 次相位相关估计因为超出 ±{limit} px "
                "上限被拒、退回 (0, 0)"
                if n_reg["n_rejected"] >= len(pair_offsets_all)
                else f"其中按告警类型计数，{n_reg['n_rejected']} 次相位相关估计因为超出 ±{limit} px 上限被拒、退回 (0, 0)"
            )
        else:
            rejected = (
                f"配准告警共 {n_reg['n_warnings']} 条，但{_kinds_missing_reason(n_reg)}，"
                "所以无法只数“相位相关估计被上限拒收”这一类（空掩码/上游异常/复合偏移越界也会出现在同一个列表里）"
            )
        if geometry_stats and geometry_stats["all_aligned"]:
            lines.append(
                f"- **零偏移回退已核验为无害（掩码在平移意义上本来就对齐）**：{len(pair_offsets_all)} 个相邻帧对的"
                f"复合偏移全为 (0, 0)，{rejected}。但掩码本身不需要平移：逐帧对在二值掩码上"
                f"{_geometry_scan_text(geometry_stats)}，零偏移 IoU = "
                f"{geometry_stats['zero_min']:.3f}–{geometry_stats['zero_max']:.3f}（中位 {geometry_stats['zero_median']:.3f}），"
                f"最佳 IoU = {geometry_stats['best_min']:.3f}–{geometry_stats['best_max']:.3f}（最佳偏移 "
                f"{_offsets_text(geometry_stats['offsets'])} px，与零偏移的 IoU 差最大 {geometry_stats['max_gain']:.3f} ≤ "
                f"容差 {geometry_stats['tolerance']}）→ {geometry_stats['n_aligned']}/{geometry_stats['n_pairs']} 个帧对都判为"
                "“零偏移已对齐”。这里的“对齐”只指**没有整体平移**，不等于重叠度高——零偏移 IoU 本身只有 "
                f"{geometry_stats['zero_min']:.3f}–{geometry_stats['zero_max']:.3f}，那是掩码在生长/形变，不是平移。"
                "所以退回零偏移不改变几何，跨天匹配只依赖 bbox IoU 与质心距离这一点不影响结论。"
            )
        elif geometry_stats:
            lines.append(
                f"- **配准在本数据上等价于不平移，而且几何核验不支持“本来就对齐”**：{len(pair_offsets_all)} 个相邻帧对的"
                f"复合偏移全为 (0, 0)，{rejected}；但把掩码在二值图上扫一遍（{_geometry_scan_text(geometry_stats)}）后，"
                f"零偏移 IoU 只有 {geometry_stats['zero_min']:.3f}–{geometry_stats['zero_max']:.3f}，最佳 IoU 可达 "
                f"{geometry_stats['best_min']:.3f}–{geometry_stats['best_max']:.3f}（最佳偏移 {_offsets_text(geometry_stats['offsets'])} px），"
                f"{geometry_stats['n_aligned']}/{geometry_stats['n_pairs']} 个帧对判为已对齐。未对齐的帧对必须人工核对原图/掩码。"
            )
        else:
            lines.append(
                f"- **配准在本数据上等价于不平移，且这是一个可疑的模式（需人工核对）**："
                f"{len(pair_offsets_all)} 个相邻帧对的复合偏移全为 (0, 0)，{rejected}。零偏移只说明“最终没有平移”，"
                "无法区分“掩码本来就对齐（偏移 < 1 px）”与“配准失败后退回零偏移”——本数据属于哪一种，现有 QC 判不了，"
                "必须去核对原图/掩码本身是否真有整体平移。因此跨天匹配实际是在零偏移几何上跑的，只依赖 bbox IoU 与质心距离。"
                "为什么相关估计会全被拒（例如二值掩码上的相位相关锁到了错误的峰）目前只是**假设、尚未验证**。"
            )
        if geometry_stats:
            lines.extend([""] + _geometry_table(geometry, geometry_stats) + [""])
    else:
        lines.append(f"- 配准生效情况：{len(pair_offsets_all)} 个帧对的复合偏移为 {pair_offsets_all}。")
    n_prev_total = int(sum(pair.get("n_prev", 0) for fov in fovs for pair in fovs[fov].get("pairs", {}).values()))
    n_cur_total = int(sum(pair.get("n_cur", 0) for fov in fovs for pair in fovs[fov].get("pairs", {}).values()))
    n_one_to_one = int(totals.get("n_one_to_one", 0))
    matched_prev = n_one_to_one + int(totals.get("n_merge", 0))
    matched_cur = n_one_to_one + int(totals.get("n_split", 0))
    rate_prev = matched_prev / n_prev_total if n_prev_total else float("nan")
    rate_cur = matched_cur / n_cur_total if n_cur_total else float("nan")
    rate_prev_no_merge = n_one_to_one / n_prev_total if n_prev_total else float("nan")
    lines.extend(
        [
            "",
            "### 5.2 匹配率与事件统计",
            "",
            (
                "**匹配率的定义**：分母是这些相邻帧对里**全部**实例——上一采样点数之和 `Σn_prev` / 当前采样点"
                "数之和 `Σn_cur`（**不是**“参与匹配”的那部分；`n_prev`/`n_cur` 直接取自逐帧对统计）。"
                "merge 计在上一侧、split 计在当前侧。"
            ),
            "",
            (
                f"- 上一侧匹配率（含 merge）= (one_to_one + merge) / Σn_prev = ({matched_prev}) / ({n_prev_total}) = "
                f"**{rate_prev:.1%}**；不含 merge 的 `one_to_one / Σn_prev` = ({n_one_to_one}) / ({n_prev_total}) = "
                f"**{rate_prev_no_merge:.1%}**（merge 偏多时两者差得远，逐帧对明细见下表）。"
            ),
            (
                f"- 当前侧匹配率 = (one_to_one + split) / Σn_cur = ({matched_cur}) / ({n_cur_total}) = "
                f"**{rate_cur:.1%}**。"
            ),
            "",
            *_table(
                ("指标", "数值"),
                [
                    ("one_to_one（一对一继承）", totals.get("n_one_to_one")),
                    ("merge（两个上一帧实例并进同一个当前实例）", totals.get("n_merge")),
                    ("split（一个上一帧实例分裂成多个当前实例）", totals.get("n_split")),
                    ("上一侧未匹配", totals.get("n_prev_unmatched")),
                    ("当前侧未匹配", totals.get("n_cur_unmatched")),
                    ("匹配表行数（matches.csv）", match_qc.get("match_table", {}).get("n_rows")),
                    (
                        "复合偏移超上限的帧对数（按**最终使用**的偏移统计；被拒后退回 (0, 0) 的帧对不计入）",
                        totals.get("n_pairs_offset_exceeding_limit"),
                    ),
                ],
            ),
            "",
        ]
    )
    pair_rows = []
    for fov in sorted(fovs):
        for pair, entry in sorted(fovs[fov].get("pairs", {}).items()):
            pair_rows.append(
                (
                    fov,
                    pair,
                    entry.get("n_prev"),
                    entry.get("n_cur"),
                    entry.get("n_one_to_one"),
                    entry.get("n_merge"),
                    entry.get("n_split"),
                    entry.get("n_prev_unmatched"),
                    entry.get("n_cur_unmatched"),
                    _fmt(entry.get("prev_link_rate"), 3),
                )
            )
    lines.extend(
        _table(
            (
                "视野",
                "帧对",
                "n_prev",
                "n_cur",
                "one_to_one",
                "merge",
                "split",
                "前未匹配",
                "后未匹配",
                "one_to_one/n_prev",
            ),
            pair_rows,
        )
    )
    histogram = stats["track_length_histogram"]
    hist_rows = [(f"长度 {length}", count) for length, count in sorted(histogram.items())]
    per_fov_rows = [
        (fov, entry["n_instances"], entry["n_tracks"], entry["n_len_ge2"])
        for fov, entry in sorted(stats["per_fov"].items())
    ]
    qc_len_ge2 = totals.get("n_tracks_len_ge2")
    lines.extend(
        [
            "",
            "### 5.3 track 长度分布",
            "",
            (
                f"- 从 `tracks.csv` 实测：{stats['n_instances']} 个实例 → **{stats['n_tracks']} 条 track**，其中"
                f"**长度 ≥ 2 的 {stats['n_len_ge2']} 条**（Spec §10 验收要求 > 0）；`match_qc.json` 记的是 "
                f"{qc_len_ge2} 条（{'两者一致' if qc_len_ge2 == stats['n_len_ge2'] else '两者**不一致**，需查产物代次'}）。"
            ),
            "",
            *_table(("track 长度", "条数"), hist_rows),
            "",
            *_table(("视野", "实例数", "track 数", "长度 ≥2 的 track 数"), per_fov_rows),
            "",
            (
                "长度 1 的 track 是“某个实例在相邻采样点没有匹配到继承者”（新出现、消失、或匹配失败），"
                "它们保留在 `tracks.csv` 里参与占比统计，但轨迹图（F9）不会把它们连成线——一个点连不出位移。"
            ),
            "",
        ]
    )
    return lines


def _section_temporal(stage6b_qc: dict) -> list[str]:
    """§6 时间统计：3 个离散采样点的口径、轨迹退化情况、转移矩阵的先验。"""
    axis = stage6b_qc.get("time_axis", {})
    trajectories = stage6b_qc.get("trajectories", {})
    transitions = stage6b_qc.get("transitions", {})
    subsets = stage6b_qc.get("subsets", {})
    warnings = list(stage6b_qc.get("warnings", []))
    prop_subset = subsets.get("proportions", {})
    transition_subset = subsets.get("transitions", {})
    lines = [
        "## 6. 时间统计（Stage 6b）",
        "",
        (
            f"- 时间轴：{axis.get('mode')}；时间箱边界 {axis.get('intervals')}，采样点 "
            f"{axis.get('bin_day')}（{axis.get('bin_time_hours')} h），共 {axis.get('n_bins')} 个。"
        ),
        f"- **口径**：{axis.get('discrete_days_note')}",
        (
            f"- 表型轨迹走 `{trajectories.get('path')}`（退化 {_fmt(trajectories.get('degenerate'))}），"
            f"亲和矩阵用 `{stage6b_qc.get('affinity', {}).get('affinity')}` + "
            f"`use_std_denom={stage6b_qc.get('affinity', {}).get('use_std_denom')}`。"
        ),
        (
            f"- 转移矩阵：`pseudocounts={transitions.get('pseudocounts')}`（拉普拉斯平滑，避免“从未作为起点”的簇"
            f"出现 NaN 行），朝向 {transitions.get('orientation')}；图上只画观测到的边（count ≥ 1），"
            "纯先验边打斜纹/不画，观测与先验在图上分开。"
        ),
        (
            f"- 各产物用到的实例子集：占比 {prop_subset.get('n_instances')} 个"
            f"（{prop_subset.get('track_length_filter')}）、转移 {transition_subset.get('n_instances')} 个"
            f"（{transition_subset.get('track_length_filter')}）。"
        ),
        (
            f"- Stage 6b 告警 {len(warnings)} 条"
            + (
                f"（下面只列前 {_WARNING_SHOW_LIMIT} 条，其余见 `stage6b_qc.json` 的 `warnings`）"
                if len(warnings) > _WARNING_SHOW_LIMIT
                else "（全部列在下面）"
            )
            if warnings
            else "- Stage 6b 告警 0 条。"
        ),
    ]
    lines.extend(_warning_lines(warnings, "`stage6b_qc.json` 的 `warnings`"))
    lines.append("")
    return lines


def _small_instance_counts(all_instances: pd.DataFrame, cluster_labels: pd.DataFrame) -> dict:
    """F1 代表图口径需要的两个数：小实例总数、以及“整簇都是小实例”的簇。

    只读 `all_instances.csv` 的 `is_small` 与 `cluster_labels.csv` 的簇号（按 `instance_uid` 对齐）。
    """
    flags = all_instances[["instance_uid", "is_small"]].copy()
    flags["instance_uid"] = flags["instance_uid"].astype(str)
    flags["is_small"] = flags["is_small"].astype(bool)
    labels = cluster_labels[["instance_uid", "cluster"]].copy()
    labels["instance_uid"] = labels["instance_uid"].astype(str)
    labels["cluster"] = labels["cluster"].astype(int)
    merged = labels.merge(flags, on="instance_uid", how="left", validate="one_to_one")
    merged["is_small"] = merged["is_small"].fillna(False).astype(bool)
    all_small = [
        int(cluster)
        for cluster in sorted(merged["cluster"].unique())
        if bool(merged.loc[merged["cluster"] == cluster, "is_small"].all())
    ]
    return {
        "n_instances": len(flags),
        "n_small": int(flags["is_small"].sum()),
        "clusters_all_small": all_small,
    }


def _small_instance_lines(all_instances: pd.DataFrame, cluster_labels: pd.DataFrame) -> list[str]:
    """§7 的 F1 口径：代表图跳过 `is_small`（Spec §6-Stage1-4 “供出图时排除”），并给出计数。"""
    counts = _small_instance_counts(all_instances, cluster_labels)
    n_small = counts["n_small"]
    share = f"{n_small / counts['n_instances']:.1%}" if counts["n_instances"] else "—"
    fallback = counts["clusters_all_small"]
    return [
        (
            f"- **F1 代表图的口径**：每簇代表图仍然取“离簇质心最近的 4 个实例”，但**跳过 `is_small`"
            f"（面积 < {cfg.SMALL_AREA_FLAG_PX} px²）的实例**——本轮景观里共 {n_small} 个小实例（占 {share}），"
            "它们不进 F1 的代表图候选（小实例的裁剪图只有几个像素，画出来看不出形态）"
            + (
                f"；但簇 {'、'.join(str(cluster) for cluster in fallback)} 里**没有非小实例**，这几簇只能含小实例，"
                "F1 图注会写明"
                if fallback
                else "；本轮每个簇都有非小实例，所以 F1 的代表图全部是非小实例"
            )
            + "。其余图（F6 等等）仍然把全部实例算进去，不做任何删除。"
        ),
        "",
    ]


def _section_figures(root: Path, all_instances: pd.DataFrame, cluster_labels: pd.DataFrame) -> list[str]:
    """§7 图索引：逐个检查图是否真的存在，并写明对应论文哪张图 + F1 代表图的 `is_small` 口径。

    图索引里的视野数与采样点数是**数据事实**（`all_instances` 里真实出现过的 fov / day），
    不是配置里声明的：命令行 `--images/--masks` 只喂一个视野时，写死的"3 个视野"会让报告自相矛盾。
    """
    figdir = root / FIGURES_SUBDIR
    n_fovs = int(all_instances["fov"].nunique()) if len(all_instances) else 0
    n_days = int(all_instances["day"].nunique()) if len(all_instances) else 0
    rows = []
    n_missing = 0
    for name, paper, purpose in _FIGURE_INDEX:
        present = (figdir / name).exists()
        if not present:
            n_missing += 1
        rows.append((name, paper, purpose.format(n_fovs=n_fovs, n_days=n_days), "已生成" if present else "**缺失**"))
    lines = [
        "## 7. 图索引（对应论文哪张图）",
        "",
        f"图全部落在 `{figdir}`（600 dpi PNG），共 {len(_FIGURE_INDEX) - n_missing} / {len(_FIGURE_INDEX)} 张已生成"
        + (f"，**缺失 {n_missing} 张**（见下表“缺失”行，重跑对应阶段即可）" if n_missing else "。"),
        "",
        *_table(("图文件", "对应论文", "内容", "状态"), rows),
        "",
        *_small_instance_lines(all_instances, cluster_labels),
        (
            "两处需要如实说明的对应关系：**F2 / F6b / F11 论文没有对应图**（分别是景观是否被天数/视野主导、"
            "簇的 day 组成、以及匹配事件的 QC 图）；"
            + (
                f"**F10a 只有 {n_fovs} 个视野**，{n_fovs} 个叶子的 dendrogram 本身信息量有限，"
                "真正有信息量的是 F10b 的簇级版本（Spec §6-Stage6-5）。"
                if n_fovs >= 2
                else f"**F10a 只有 {n_fovs} 个视野**，视野级 dendrogram 退化成单个叶子（没有可聚的对象），"
                "视野级相似性在本数据上无法比较，只有 F10b 的簇级版本有信息量（Spec §6-Stage6-5）。"
            )
        ),
        "",
    ]
    return lines


def _geometry_table(geometry: dict, stats: dict) -> list[str]:
    """逐帧对的几何核验表（白说一句“去核对”不如把每个帧对的数字拿出来）。

    表里的“最佳偏移”一律是**像素**（与 `align_offsets` / `pair_offsets` 同单位）：旧产物的网格步
    在这里乘回步长，并在方法那一句注明换算过。
    """
    scale = int(stats["offset_scale"])
    rows = []
    for fov in sorted(geometry.get("pairs", {})):
        for pair, entry in sorted(geometry["pairs"][fov].items()):
            dy = int(entry["best_offset"][0]) * scale
            dx = int(entry["best_offset"][1]) * scale
            rows.append(
                (
                    fov,
                    pair,
                    f"{float(entry['zero_offset_iou']):.3f}",
                    f"{float(entry['best_iou']):.3f}",
                    f"({dy}, {dx})",
                    _fmt(bool(entry.get("aligned_at_zero"))),
                )
            )
    return [
        (
            f"配准几何核验（{_geometry_provenance(geometry)}）：在二值掩码上{_geometry_scan_text(stats)}，"
            f"判“零偏移已对齐”的容差是 {geometry.get('tolerance')}（“最佳偏移”与 `align_offsets` 一样以像素为单位）："
        ),
        "",
        *_table(("视野", "帧对", "零偏移 IoU", "最佳 IoU", "最佳偏移 (dy, dx) px", "零偏移已对齐"), rows),
        "",
    ]


def _registration_limitation(match_qc: dict, geometry: dict | None = None) -> str:
    """§8 的配准条目：配准到底生效了没有，必须列进局限（而不是只在 §5.1 里说“没关系”）。"""
    config = match_qc.get("config", {})
    fovs = match_qc.get("fovs", {})
    limit = config.get("align_max_shift_px")
    pair_offsets = [
        tuple(offset) for fov in fovs for offset in fovs[fov].get("registration", {}).get("pair_offsets", {}).values()
    ]
    n_reg = _registration_totals(match_qc)
    rejected = n_reg["n_rejected"]
    n_rejected = rejected if rejected is not None else n_reg["n_warnings"]
    kinds_note = _kinds_note(n_reg)
    geometry_stats = _geometry_summary(geometry) if geometry else None
    if pair_offsets and all(offset == (0, 0) for offset in pair_offsets):
        if geometry_stats and geometry_stats["all_aligned"]:
            return (
                f"**跨天配准在本数据上未生效，但已核验为无害**：{n_rejected}/{len(pair_offsets)} 帧对的相关估计"
                f"被 ±{limit} px 阈值拒收、退回 0 偏移{kinds_note}，而几何核验表明掩码在零偏移下没有整体平移"
                f"（零偏移 IoU {geometry_stats['zero_min']:.3f}–{geometry_stats['zero_max']:.3f} 偏低是生长/形变而非平移，"
                f"最佳偏移 {_offsets_text(geometry_stats['offsets'])} px，{geometry_stats['n_aligned']}/{geometry_stats['n_pairs']} "
                "个帧对判为已对齐）；因此跨天匹配只依赖 bbox IoU 与质心距离这一点已经量化，不再是未知风险。"
            )
        if n_rejected >= len(pair_offsets):
            detail = (
                f"{n_rejected}/{len(pair_offsets)} 帧对的相关估计都被 ±{limit} px 阈值拒绝，退回 0 偏移{kinds_note}"
            )
        else:
            detail = f"{len(pair_offsets)} 个帧对的复合偏移全为 0" + (
                f"（其中 {n_rejected} 次估计被拒后退回 0 偏移{kinds_note}）" if n_rejected else ""
            )
        return (
            f"**跨天配准在本数据上未生效**（{detail}）；因此跨天匹配只依赖 bbox IoU 与质心距离，"
            "配准失效的系统性原因需人工核对。"
        )
    return (
        f"**跨天配准是单次估计、没有独立核对**（本数据上的复合偏移 {pair_offsets}）：偏移错了匹配也跟着错，"
        "跨天匹配的可信度同时依赖配准结果与 bbox IoU/质心距离阈值，需人工抽查。"
    )


def _sampling_axis_note() -> str:
    """时间轴那一条局限：采样点数、各采样点与小时数、相邻间隔是否等宽（全部按当前 config 写）。

    原先是写死的"3 个离散采样点（day0 / day3 / day5，0/72/120 h）…两步宽度不等"：换成 2 天或 6 天、
    或者别的天数标记（`--images/--masks` 从文件名推出 day 记号）之后，这句话就不再成立。
    """
    days = [str(day) for day in cfg.DAYS]
    hours = [float(cfg.TIME_HOURS.get(day, float("nan"))) for day in days]
    hours_text = " / ".join(f"{hour:g}" for hour in hours)
    gaps = [hours[i + 1] - hours[i] for i in range(len(hours) - 1)]
    tail = (
        f"而且相邻采样点的真实间隔**不等**（{'、'.join(f'{gap:g} h' for gap in gaps)}），"
        "本管线把每一步都当作“一步”，所以转移矩阵与轨迹里混着不同真实时长。"
        if len({round(gap, 6) for gap in gaps}) > 1
        else "相邻采样点等间隔，转移矩阵与轨迹里的“一步”就是同一段真实时长。"
    )
    return f"**只有 {len(days)} 个离散采样点**（{' / '.join(days)}，{hours_text} h），不是连续视频；{tail}"


def _section_limitations(
    match_qc: dict,
    dominance: dict,
    geometry: dict | None = None,
    n_fovs: int | None = None,
) -> list[str]:
    """§8 局限与已知偏差：任务说明要求逐条写明的四件事 + 复现范围、配准实况与天数主导读数。

    `n_fovs` 是数据里真实出现的视野数（`all_instances` 里数出来的）：不传时退回 `len(cfg.FOVS)`，
    但报告应当传真实值，否则"几个视野是生物学重复"这句话会在视野数与数据不符时说不通。
    """
    n_fovs = int(len(cfg.FOVS) if n_fovs is None else n_fovs)
    items = [
        (
            "**没有 motion 特征**：本地数据每 2–3 天才有一帧，帧间光流没有物理意义，因此 motion 家族整体置空"
            "（NaN 列占位）。所有图都不支持速度/迁移速率/扩散类结论，F4 的 motion 柱与 F6 的 motion 面板显式标注置空。"
        ),
        _sampling_axis_note(),
        (
            "**没有做批次校正**（`BATCH_CORRECT_FOV = False`）："
            + (
                f"{n_fovs} 个视野是生物学重复，按视野归一化会抹掉要展示的差异"
                if n_fovs >= 2
                else f"本数据只有 {n_fovs} 个视野，没有视野级重复可校正（按视野归一化更会直接抹掉全部方差）"
            )
            + "；景观是否被视野/天数主导用 F2、F6b 如实呈现，而不是预先回归掉。"
        ),
        (
            f"**表型景观部分被采样天数主导**：§3.1 的读数是 {dominance['n_clusters']} 个簇里有 "
            f"{dominance['n_dominated']} 个单日占比 ≥ 50%（优势 day 占比平均 {dominance['mean_share']:.1%}），"
            "所以簇的解释必须带上“天数混杂”这一条，不能当成随时间稳定的纯形态表型（Spec §11-Q1 / D15："
            "只如实报告，不预先回归掉）。"
        ),
        (
            f"**`PIXEL_RES = {cfg.PIXEL_RES}`**：数据没有标尺信息，所有长度/面积都是**像素单位**"
            "（µm/px 未知）。若之后拿到真实标尺，只需改 `config.PIXEL_RES` 一处并重跑。"
        ),
        (
            "**实例过滤**：只打 `is_small` 标记、不删除（day0 有大量 < 200 px² 的实例，删了会改写"
            "“实例数随时间增长”的叙述）；异常实例的处理方式在 QC 里逐个可见，不做静默剔除。"
        ),
        _registration_limitation(match_qc, geometry),
        (
            "**未复现的部分**：论文的模拟数据扰动实验（Simulated Dataset A/B、Fig. 2 的鲁棒性部分）与 pacmap 嵌入"
            "都不在本仓库范围内；`MODULE_SELECTION='legacy'` 只复现论文 Fig. 3d 那一套，v2.0.1 的 multiscale 选择器"
            "只作为开关保留（模块数会不同）。"
        ),
        "**UMAP 坐标依赖 umap-learn 版本**：固定 `random_state` 后同版本内可复现（见 §9），跨版本不能保证逐位一致。",
    ]
    lines = ["## 8. 局限与已知偏差", ""]
    lines.extend(f"{index}. {item}" for index, item in enumerate(items, start=1))
    lines.append("")
    return lines


def _umap_param_text(params: dict) -> str:
    """UMAP 参数逐项写成 `key=value`（报告里要能核对“用的是这组参数”）。"""
    return "、".join(f"{key}={value}" for key, value in params.items()) if params else "—"


def _section_timings(root: Path) -> list[str]:
    """§9 运行耗时与可复现性：读 `run_all` 写的耗时 JSON 与可选的 UMAP 重跑核验结果。"""
    timing_path = root / LOG_SUBDIR / TIMING_FILE
    lines = ["## 9. 运行耗时与可复现性", ""]
    if timing_path.exists():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        stages = timing.get("stages", {})
        rows = [(name, f"{float(seconds):.1f}") for name, seconds in stages.items()]
        total = timing.get("total_seconds")
        lines.extend(
            [
                f"**全流程耗时（wall-clock）**：`{total}` s"
                + (f" ≈ **{float(total) / 60.0:.1f} 分钟**" if total is not None else "")
                + (
                    f"（开始于 {timing.get('started_at')}，最后一阶段结束于 {timing.get('finished_at')}）"
                    if timing.get("finished_at")
                    else "（开始于 "
                    + str(timing.get("started_at"))
                    + "；本报告写盘时最后一阶段刚结束，`run_all_timings.json` 里的总数还会计入报告写盘那几秒）"
                ),
                "",
                *_table(("阶段", "耗时 (s)"), rows),
                "",
                "逐阶段的完整 stdout 日志在 `" + str(root / LOG_SUBDIR) + "/<阶段名>.log`。",
            ]
        )
        if timing.get("failed_stage"):
            lines.append(
                f"**注意**：本次运行在 `{timing['failed_stage']}` 停下来了（{timing.get('error')}），"
                "所以报告里的数字来自失败之前已经落盘的产物。"
            )
    else:
        lines.extend(
            [
                f"**运行耗时未记录**：找不到 `{timing_path}`（本报告不是由 `run_all` 生成的，或耗时文件被删掉了）。",
                "用 `uv run python scripts/run_all.py` 一键跑完整流程即可自动记录每阶段与总耗时。",
            ]
        )
    repeat_path = root / LOG_SUBDIR / REPEAT_CHECK_FILE
    lines.append("")
    check: dict | None = None
    if repeat_path.exists():
        try:
            loaded = json.loads(repeat_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = None
        check = loaded if isinstance(loaded, dict) else None
    # 只认核验脚本产出的文件（`source` + `checked_at`）：否则任何人都能手写一份“自证”JSON
    verified = check is not None and check.get("source") == REPEAT_CHECK_TOOL and bool(check.get("checked_at"))
    if verified and check is not None:
        verdict = "逐位一致" if check.get("identical") else "**不一致**"
        lines.extend(
            [
                (
                    f"- **UMAP 可复现性已核验**（`{REPEAT_CHECK_FILE}`，由 `{check['source']}` 在 "
                    f"{check['checked_at']} 生成）：把 `preprocessed/X.npy`（{check.get('n_points')} 个点 × "
                    f"{check.get('n_features', '—')} 维）用同一组 UMAP 参数（{_umap_param_text(check.get('umap_params', {}))}）"
                    f"重新嵌入一次，新 CSV 的 md5 `{check.get('rerun_md5', '—')}` 与已落盘的 "
                    f"`analysis/umap_2d.csv` 的 md5 `{check.get('shipped_md5', '—')}` {verdict}"
                    f"（最大绝对差 {_fmt(check.get('max_abs_diff'))}）。`random_state={cfg.RANDOM_STATE}` 固定了 "
                    "UMAP 与 k-means，所以同版本内重跑得到同一张景观。"
                ),
                (
                    "  - 口径限制：第一次运行的中间文件没有保留，所以这是“同一份 `X.npy` 再嵌入一次、与已落盘的 "
                    "CSV 逐字节比对”，而不是两次独立全流程的比对；上游特征/预处理是否逐字节可重复不在这次核验范围内。"
                ),
            ]
        )
        if check.get("note"):
            lines.append(f"  - {check['note']}")
    else:
        reason = (
            f"`{repeat_path}` 不是 `{REPEAT_CHECK_TOOL}` 生成的（缺 `source`/`checked_at`，因此无法确认是谁核验的）"
            if repeat_path.exists()
            else f"找不到 `{repeat_path}`"
        )
        lines.append(
            f"- **UMAP 可复现性未核验**：本次**未验证**“同一份输入重跑得到逐位相同的坐标”——{reason}。"
            f"做法：`uv run python {REPEAT_CHECK_TOOL}`（读 `preprocessed/X.npy` 与 `kmeans_metrics.json` 的 UMAP 参数，"
            "重跑一次 `landscape.embed`，把两个 md5、最大绝对差、参数与时间戳写进 "
            f"`{repeat_path.name}`）；写好之后本报告会自动引用，没有这个文件时报告不会自称核验过。"
        )
    lines.append("")
    return lines


def _header_lines(root: Path, source: str | None) -> list[str]:
    """报告头：生成时间 + **真实的生成方式**（一键运行 / 直接调用报告函数，两者不能混着说）。"""
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    if source == "run_all":
        how = (
            "**生成方式**：`uv run python scripts/run_all.py` 的最后一个阶段"
            "（同一次运行刚跑完 Stage1–6b 与出图，§9 的耗时就是这次运行）。"
        )
    else:
        how = (
            '**生成方式**：直接调用报告函数重新生成——`uv run python -c "from biopheno.report import '
            f"write_report; write_report(Path(r'{root}'))\"`。这次**只重写了本文件**：§1–§7 的数字来自上游"
            "**上一次运行**的落盘产物（那次运行的时间见 §9），本次没有重跑 Stage1–6b。"
        )
    # 数据来源的写法跟着输入方式走：命令行给了 --images/--masks 就点出两个目录（可复算），
    # 否则报旧约定的 DATA_ROOT。
    if cfg.IMAGE_DIR is not None and cfg.MASK_DIR is not None:
        data_source = f"原图目录 `{cfg.IMAGE_DIR}`；掩码目录 `{cfg.MASK_DIR}`"
    else:
        data_source = f"数据根 `{cfg.DATA_ROOT}`"
    return [
        "# SPOT 类器官复现报告（本地数据，motion 置空）",
        "",
        f"生成时间：{now}；{data_source}；输出根 `{root}`。",
        "",
        how,
        "",
        (
            "本报告只汇总各阶段**已经落盘**的产物与 QC，不重算任何上游。所有随机性统一用 "
            f"`random_state={cfg.RANDOM_STATE}`。"
        ),
        "",
    ]


def write_report(output_root: Path, *, source: str | None = None) -> Path:
    """读 `output_root` 下各阶段的产物，写 `output_root/report.md` 并返回它的路径（Spec §6-Stage7）。

    只读盘、不重算：缺任何必需产物都抛点名阶段的中文 `RuntimeError`；`tracks.csv` 里一条长度 ≥ 2 的 track
    都没有时同样报错（那是时间编码/匹配失效，不能当成正常结果写进报告）。

    `source="run_all"` 由 `pipeline.run_all` 传入：报告头据此写“一键运行的最后一个阶段”；
    不传时写“直接调用报告函数重新生成”（上游产物沿用上一次运行，不谎称是本次跑出来的）。
    """
    root = Path(output_root)
    stage1_qc = _read_json(root / _INSTANCES_SUBDIR / _STAGE1_QC, "Stage1", "逐视野实例数与对齐偏移")
    all_instances = _read_csv(root / _INSTANCES_SUBDIR / _ALL_INSTANCES, "Stage1", "实例汇总表", _ALL_INSTANCES_COLUMNS)
    stage2_qc = _read_json(root / _FEATURES_SUBDIR / _STAGE2_QC, "Stage2", "逐视野特征维度")
    feature_names = _read_json(root / _FEATURES_SUBDIR / _FEATURE_NAMES, "Stage2", "特征列名清单")
    stage3_qc = _read_json(root / _PREPROCESSED_SUBDIR / _STAGE3_QC, "Stage3", "预处理前后维度与剔除明细")
    stage4_qc = _read_json(root / _MODULES_SUBDIR / _STAGE4_QC, "Stage4", "模块数、模块大小与行身份校对")
    kmeans = _read_json(root / _ANALYSIS_SUBDIR / _KMEANS_METRICS, "Stage5", "UMAP 参数与 K 扫描指标")
    cluster_labels = _read_csv(
        root / _ANALYSIS_SUBDIR / _CLUSTER_LABELS, "Stage5", "逐实例簇标签", _CLUSTER_LABELS_COLUMNS
    )
    match_qc = _read_json(root / _ANALYSIS_SUBDIR / _MATCH_QC, "Stage6", "配准偏移与匹配统计")
    tracks = _read_csv(root / _ANALYSIS_SUBDIR / _TRACKS_FILE, "Stage6", "track 表", _TRACKS_COLUMNS)
    stage6b_qc = _read_json(root / _ANALYSIS_SUBDIR / _STAGE6B_QC, "Stage6b", "时间轴口径与轨迹退化")
    # §3.1 与 §8 共用同一份 day 组成读数；§5.1/§8 共用同一份配准几何核验
    dominance = _day_dominance(cluster_labels)
    geometry = _geometry_check(root, match_qc)

    lines = [
        *_header_lines(root, source),
        *_section_instances(stage1_qc, all_instances, stage2_qc, stage4_qc),
        *_section_features(stage2_qc, feature_names, stage3_qc),
        *_section_umap(kmeans, dominance),
        *_section_modules(stage4_qc),
        *_section_matching(match_qc, tracks, geometry),
        *_section_temporal(stage6b_qc),
        *_section_figures(root, all_instances, cluster_labels),
        *_section_limitations(match_qc, dominance, geometry, int(all_instances["fov"].nunique())),
        *_section_timings(root),
    ]

    path = root / REPORT_FILE
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_missing = sum(1 for name, _paper, _purpose in _FIGURE_INDEX if not (root / FIGURES_SUBDIR / name).exists())
    print(
        f"报告完成：{path}（实例合计 {sum(entry['n_instances'] for entry in stage1_qc.values())} 个，"
        f"K={kmeans.get('k')}，模块 {stage4_qc.get('n_modules')} 个，图缺 {n_missing} 张）"
    )
    return path
