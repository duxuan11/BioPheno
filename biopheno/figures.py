"""Stage 7 静态图（F1–F6b，Spec §6-Stage7）：形状 + 外观的表型景观，motion 已置空。

产物（写进 `outdir`，全部 600 dpi PNG）：

- `F1_umap_clusters_patches.png`：UMAP 景观按表型簇着色（簇号 1..K，blue→red）+ 每簇 4 张代表图，
  **原始尺寸**与**标准化 64×64** 两版（论文 Fig. 3b 的"景观 + 代表形态"）。左边那块景观是 **1:1 的方形**
  （`_square_view`：x/y 每单位等长且共用同一跨度），右边代表图铺进约 4 行（`_f1_patch_grid`）——
  不再"每簇一行"，因为那样簇一多就会把景观纵向拉长。
- `F2_umap_by_day_fov.png`：同一景观分别按 day、按 fov 着色（看景观是不是被天数/视野主导，§11-Q1 的辅助图）。
- `F3_umap_density.png`：每簇一层的 2-D 密度场（**所有面板共用**同一个网格范围 + 同一个高斯核 + 同一个色标，
  所以面板之间可以直接比较高低）。
- `F4_contribution_scores.png`：shape/appearance/motion × global/regional/distribution 的 contribution score。
- `F5_sam_modules.png`：SAM 模块 clustermap（相关系数矩阵 + dendrogram + 模块色条）+ 簇 × 模块表达热图。
- `F6_cluster_feature_bars.png`：每簇 area/eccentricity/intensity 的均值 ± SEM，motion 面板显式标注置空。
- `F6b_cluster_day_composition.png`：每簇的 day 组成堆叠条（景观是否主要由天数驱动）。

以及时间图（`render_temporal`，读 Stage6b 的产物；时间轴是 **3 个离散采样点**，不是连续视频，
motion 已置空，所以任何图都不支持速度类结论）：

- `F7_stacked_cluster_proportions.png`：每个视野一个面板 + 一个合并面板的簇占比堆叠条；
  合并 = **按每视野每采样点的实例数加权平均**（不是等权，小视野不会被放大）。
- `F8_transition_graph.png`：左 = 观测到的簇间转移有向图（`count >= 1` 的边才画；`pseudocounts=1`
  产生的 `count == 0` 纯先验边一条都不画，图上写明过滤了多少条），右 = 转移概率热图
  （行 = 来源、列 = 目标、行和 1，对角就是自转移，纯先验格打斜纹）。图上的边概率是
  **去掉自转移后按行重归一**的 `inter`，与右侧热图的 `probs`（含自转移）不是同一个分母，图上写明。
- `F9_umap_trajectories.png`：每视野一个面板，彩色折线**只画长度 ≥ 2 的 track**（长度 1 的 track
  只有一个采样点，连成线等于凭空造出一条位移），粗黑线是 Stage6b 的表型轨迹。
- `F10a_trajectory_affinity_fov.png` / `F10b_trajectory_affinity_cluster.png`：视野级与簇级的轨迹亲和
  + dendrogram（F10a 读 `affinity.npz`；F10b 的簇轨迹是每簇每采样点的 UMAP 质心，亲和用与 F10a 相同的
  DTW 指数核）。
- `F11_match_events.png`：每个 (视野, 帧对) 的匹配事件（one_to_one / merge / split）与未匹配数
  （读 Stage6 的 `match_qc.json`）。

两个渲染函数（`render_static` / `render_temporal`）渲染完都会在 `outdir` 里顺手写一份 `README.md`
（`write_figure_readme`）：13 张图各自"是什么 / 怎么读 / 依赖哪些上游产物"，并按目录里**实际存在**的文件
标注"已生成 / 缺失"——说明与出图代码同源，不会出现"图改了、说明忘了改"。它是自动生成物，别手改。

输入**全部**来自持久化产物，本阶段不重算任何上游：

- `analysis/`（Stage5）：`umap_2d.csv` / `cluster_labels.csv` / `per_instance_features.csv` /
  `kmeans_metrics.json`；
- `modules/`（Stage4）：`module_expr.npy` / `cluster_map.npz` / `stage4_qc.json`（contribution score）；
- `instances/`（Stage1）：`all_instances.csv` + `<day>/<fov>/patches.npz`（每簇代表图）；
- `preprocessed/`（Stage3）：**只在** `stage4_qc.json` 没落 contribution 时用来重算那两个 3 元组。

`render_static(analysis_dir, outdir, force_single_cluster=False)` 的 `analysis_dir` 是**与真实 `output/`
同构的目录树根**（不是里面的 `analysis/` 子目录）：四类输入都由它派生。所以测试可以把参数指向一棵合成
小树，本模块绝不假设真实数据集（1.2 GB）存在。`render_temporal(analysis_dir, outdir=None)` 的
`analysis_dir` 两种写法都接受（`output/` 根或直接 `output/analysis/`），因为时间图的产物全在一个目录里。

时间图的退化一律**写在图上**并配一条中文 `warnings.warn`：没有长度 ≥ 2 的 track、表型轨迹全为 NaN、
没有任何观测到的转移、亲和矩阵只有一个对象、`match_qc.json` 里一条匹配都没发生——这些都不许静默空白。

几个刻意的口径选择：

- 簇配色统一走 `matplotlib.cm.coolwarm` 的 blue→red，与 Stage 5 的编号（按平均 eccentricity 递增）同序，
  所以"越红越扁长"在所有图里都成立；
- 每簇代表图**不是**随机抽样：在簇内按"到簇质心的 UMAP 距离"由近到远排序，取最近的 4 个，
  用 `modules.representative_patches`（即 SPOT 的 `compute_most_representative_image_patches`）
  拼 2×2 马赛克。原始尺寸与 64×64 两版用**同一份 score**，所以两版是同一批实例；
- numpy 图像数组是 `(row, col)`，matplotlib 是 `(x, y)`：这个转换只发生在 `_imshow_patch` 一处
  （`origin="upper"` 让第 0 行画在最上面），其余绘图函数只吃已经就位的坐标/数值；
- F3 画的是**自己估的网格密度场**，不是 SPOT 的逐点密度：参见 `_density_fields` 的说明（SPOT 的带宽是
  每个条件各自定的，面板之间的亮度不可比，本阶段需要的是可比的热图）；
- 图注是中文，而 matplotlib 默认的 DejaVu Sans 没有中文字形：从 `_CJK_FONT_CANDIDATES` 里挑一款本机
  可用的字体（本机 = Microsoft YaHei）；一款都没有时 `_configure_fonts` 会发一条中文 `warnings.warn`，
  `render_static` 的汇总日志行里也会再记一句（不静默降级成方框图）；
- 每个绘图函数都自己开关 figure（`_open_figure`），成功/异常都不留 figure；
- 落盘不会先删旧图：先把七张图渲染到 `<outdir>/.tmp-<文件名>`，全部成功后才逐个 `replace` 成正式文件名
  （任何一步失败都不会把上一次的好图弄丢，也不会留下半成品）。
"""

import contextlib
import json
import math
import unicodedata
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import matplotlib

# 必须在任何 pyplot 导入之前定后端：本模块与 SPOT 的密度函数都会建 figure，无头环境下
# 交互式后端会真的开窗口（长跑会挂住）。force=False：调用方已经选好后端时不覆盖它。
matplotlib.use("Agg", force=False)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, font_manager
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import squareform

from biopheno import config as cfg
from biopheno import modules as mods
from biopheno import temporal_stats as tstats
from SPOT.Analysis import sam_analysis as sam

# —— 输入产物（与 Stage1/3/4/5 的落盘位置一致；根目录由调用方给）——
_ANALYSIS_SUBDIR: str = "analysis"
_MODULES_SUBDIR: str = "modules"
_INSTANCES_SUBDIR: str = "instances"
_PREPROCESSED_SUBDIR: str = "preprocessed"
_UMAP_FILE: str = "umap_2d.csv"
_LABELS_FILE: str = "cluster_labels.csv"
_PER_INSTANCE_FILE: str = "per_instance_features.csv"
_METRICS_FILE: str = "kmeans_metrics.json"
_MODULE_EXPR_FILE: str = "module_expr.npy"
_CLUSTER_MAP_FILE: str = "cluster_map.npz"
_STAGE4_QC_FILE: str = "stage4_qc.json"
_ALL_INSTANCES_FILE: str = "all_instances.csv"
_PATCH_FILE: str = "patches.npz"
_X_FILE: str = "X.npy"
_KEEP_FILE: str = "keep_features.json"

# —— 本阶段产物（Spec §6-Stage7 的静态图部分）——
_FIG_F1: str = "F1_umap_clusters_patches.png"
_FIG_F2: str = "F2_umap_by_day_fov.png"
_FIG_F3: str = "F3_umap_density.png"
_FIG_F4: str = "F4_contribution_scores.png"
_FIG_F5: str = "F5_sam_modules.png"
_FIG_F6: str = "F6_cluster_feature_bars.png"
_FIG_F6B: str = "F6b_cluster_day_composition.png"
_FIGURE_NAMES: tuple[str, ...] = (_FIG_F1, _FIG_F2, _FIG_F3, _FIG_F4, _FIG_F5, _FIG_F6, _FIG_F6B)

# —— 时间图（F7–F11）的输入（Stage5/6/6b 的落盘位置，目录由 `_locate_analysis_dir` 定位）——
_TRACKS_FILE: str = "tracks.csv"
_MATCH_QC_FILE: str = "match_qc.json"
_PROPORTIONS_FILE: str = "proportions.npz"
_TRANSITIONS_FILE: str = "transitions.npz"
_TRAJECTORIES_FILE: str = "trajectories.npz"
_AFFINITY_FILE: str = "affinity.npz"
_STAGE6B_QC_FILE: str = "stage6b_qc.json"

# 四份时间 npz 共有的标签顺序键（Stage6b 每份都写一遍，绘图层据此对齐轴，不猜顺序）。
_TEMPORAL_COMMON_KEYS: tuple[str, ...] = (
    "cluster_labels",
    "conditions",
    "bin_frame_idx",
    "bin_analysis_time",
    "bin_time_hours",
    "bin_day",
    "time_intervals",
)

# —— 本阶段产物（Spec §6-Stage7 的时间图部分）——
_FIG_F7: str = "F7_stacked_cluster_proportions.png"
_FIG_F8: str = "F8_transition_graph.png"
_FIG_F9: str = "F9_umap_trajectories.png"
_FIG_F10A: str = "F10a_trajectory_affinity_fov.png"
_FIG_F10B: str = "F10b_trajectory_affinity_cluster.png"
_FIG_F11: str = "F11_match_events.png"
_TEMPORAL_FIGURE_NAMES: tuple[str, ...] = (_FIG_F7, _FIG_F8, _FIG_F9, _FIG_F10A, _FIG_F10B, _FIG_F11)

# 时间图的落盘子目录（`render_temporal` 的 `outdir` 缺省值）：与 Spec §9 的 `output/figures/` 一致。
_FIGURES_SUBDIR: str = "figures"

# 论文图的输出分辨率（Spec §8：默认 dpi=600）。
_DPI: int = 600

# 落盘时的临时文件前缀：图先写进 `<outdir>/.tmp-<文件名>`，全部成功后才换成正式文件名。
_TMP_PREFIX: str = ".tmp-"

# F10a/F10b 的图注折行宽度（半角字符数）：中文一字算 2、ASCII 算 1；110 个半角宽在 6.8 in 宽、7 pt 字号下
# 左右各留一点空，真数据里 fov 名更长时也不会顶到画布边。
_CAPTION_WRAP_WIDTH: int = 110

# 每簇代表图张数（论文 Fig. 3b 的 2×2）：4 张 → 马赛克 2 行 × 2 列。
_PATCHES_PER_CLUSTER: int = 4

# F1 右侧代表图的排版：**不再"每簇一行"**——那样 K 一多，左边那块跨全部行的 UMAP 就被拉成
# 又高又窄的长条（x/y 每单位长度不等，"景观"的形状全是假的）。改成把簇铺进 ~4 行：
# 每行放 ceil(K/4) 个簇、每簇 2 个面板（原始尺寸 / 标准 64×64），行数压到 ~4 行，多出来的格子留白。
_F1_TARGET_ROWS: int = 4
# 右侧每个代表图面板（2×2 马赛克）的边长（英寸）：图幅按它反推，patch 才不会被格子拉变形。
_F1_PATCH_CELL_IN: float = 1.55
# UMAP 面板上簇号的字号：景观缩成方阵后，9 pt 在图里几乎读不出来。
_F1_CLUSTER_LABEL_FONTSIZE: float = 15.0
# UMAP 面板的留白（占坐标跨度的比例）：最外面的点不贴轴框。
_F1_VIEW_MARGIN: float = 0.04

# clustermap 的特征刻度上限：D 可能有上百个特征，全标上去会糊成一片。
_MAX_FEATURE_TICKS: int = 30

# F3 的密度场：所有面板共用同一个网格（`_DENSITY_BINS`×`_DENSITY_BINS`）与同一个高斯核
# （`_DENSITY_SIGMA_BINS` 格，单位是网格格数而不是数据单位，所以换数据集也不会变亮度的尺度）。
_DENSITY_BINS: int = 64
_DENSITY_SIGMA_BINS: float = 2.5
# 网格往外留白的比例（图上的点不贴边框）。
_DENSITY_MARGIN_FRACTION: float = 0.05
# 等密度线的水平（占全局最大密度的比例，跨面板一致）。
_DENSITY_CONTOUR_FRACTIONS: tuple[float, ...] = (0.15, 0.35, 0.60, 0.85)

# F4/F6 的面板标签（顺序与 SPOT 的整数编码一致：0/1/2）。
_TYPE_LABELS: tuple[str, ...] = ("shape", "appearance", "motion")
_SCOPE_LABELS: tuple[str, ...] = ("global", "regional", "distribution")

# day / fov 这类离散取值的配色：按名字排序后按固定顺序取（与 matplotlib 的定性色带等价，但写死在这里，
# 免得换 matplotlib 版本或色带口径就悄悄变颜色）。值取自 tab10 的前 8 个颜色。
_CATEGORY_COLOURS: tuple[tuple[float, float, float], ...] = (
    (0.1216, 0.4667, 0.7059),
    (1.0000, 0.4980, 0.0549),
    (0.1725, 0.6275, 0.1725),
    (0.8392, 0.1529, 0.1569),
    (0.5804, 0.4039, 0.7412),
    (0.5490, 0.3373, 0.2941),
    (0.8902, 0.4667, 0.7608),
    (0.4980, 0.4980, 0.4980),
)
# F6 的三个可解释特征列（Stage5 的 per_instance_features.csv）与第四格的 motion 说明。
_BAR_FEATURES: tuple[tuple[str, str], ...] = (
    ("area", "面积 (px²)"),
    ("eccentricity", "偏心率"),
    ("intensity", "平均强度"),
)
_MOTION_FEATURE: str = "speed"

# 各阶段缺失时的提示（错误信息里必须点出该跑哪个 Stage，别让调用方猜）。
_HINT_STAGE1: str = "请先跑 Stage1（biopheno.instances.run_stage1）"
_HINT_STAGE3: str = "请先跑 Stage3（biopheno.preprocess.run_stage3）"
_HINT_STAGE4: str = "请先跑 Stage4（biopheno.modules.run_stage4）"
_HINT_STAGE5: str = "请先跑 Stage5（biopheno.landscape.run_stage5）"
_HINT_STAGE6: str = "请先跑 Stage6（biopheno.temporal.run_stage6）"
_HINT_STAGE6B: str = "请先跑 Stage6b（biopheno.temporal_stats.run_stage6b）"

# 时间图只画"长度 ≥ 2"的 track：长度 1 的 track 只有一个采样点，连成折线等于凭空造出一条位移。
_TRACK_MIN_LENGTH: int = 2
# F11 的事件配色（写死，换 matplotlib 版本也不变色）。
_EVENT_COLOURS: dict[str, tuple[float, float, float]] = {
    "one_to_one": (0.1216, 0.4667, 0.7059),
    "merge": (1.0000, 0.4980, 0.0549),
    "split": (0.1725, 0.6275, 0.1725),
}
# F11 右面板的"未匹配"配色（上一帧 / 下一帧两套分母，分开画而不是堆叠）。
_UNMATCHED_COLOURS: tuple[tuple[float, float, float], ...] = (
    (0.5490, 0.3373, 0.2941),
    (0.5804, 0.4039, 0.7412),
)
# F11 从 `match_qc.json` 的逐帧对统计里取的键（缺一个就点出 Stage6，不画半套数）。
_MATCH_STAT_KEYS: tuple[str, ...] = (
    "n_prev",
    "n_cur",
    "n_one_to_one",
    "n_merge",
    "n_split",
    "n_prev_unmatched",
    "n_cur_unmatched",
)

# 代表图挑选的 score 来源说明（写进报告/图注时用得上，也解释了为什么同一批实例出现在两版里）。
_REPRESENTATIVE_RULE: str = "簇内到簇质心的 UMAP 距离最近的 4 个实例"
# 代表图候选的硬限制：`is_small`（面积 < SMALL_AREA_FLAG_PX）的实例不参与挑选——Spec §6-Stage1-4 说
# 小实例标记是“供出图时排除”的。整簇都是小实例时才退回包含它们（否则那一簇就没代表图了），并在图注写明。
_SMALL_SKIP_NOTE: str = f"候选跳过 is_small（面积 < {cfg.SMALL_AREA_FLAG_PX} px²）的实例"

# 图注全是中文，而 matplotlib 默认的 DejaVu Sans 没有中文字形（会渲染成方框）。按常见程度挑一款本机
# 可用的中文字体；一款都没有时发中文 warnings.warn（不静默吞掉，也不静默换成英文）。
_CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "DengXian",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
)


def _configure_fonts() -> str | None:
    """把第一款本机可用的中文字体塞进 rcParams，返回字体名（一款都没有时返回 None）。

    同时关掉 `axes.unicode_minus`：中文字体的 U+2212 字形常常缺失，负号会变成方框。
    一款候选都没有时**发中文警告**：否则整张图的图注会静默变成一串方框，而调用方看不出任何异常。
    """
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in _CJK_FONT_CANDIDATES:
        if candidate in available:
            plt.rcParams["font.sans-serif"] = [candidate, *plt.rcParams["font.sans-serif"]]
            plt.rcParams["axes.unicode_minus"] = False
            return candidate
    warnings.warn(
        f"找不到任何候选中文字体（{' / '.join(_CJK_FONT_CANDIDATES)}）：图注里的中文会渲染成方框（豆腐块），"
        "图仍然会出，但字不可读。请安装其中任意一款（如 Noto Sans CJK SC）后重跑",
        UserWarning,
        stacklevel=2,
    )
    return None


# 导入时定一次（后面的绘图函数不再碰 rcParams）：同样的输入在任何机器上只要字体一致就画出同样的图。
_CJK_FONT: str | None = _configure_fonts()


def _fail(message: str) -> NoReturn:
    """抛出一句带上下文的中文 ValueError（薄封装只为统一文案格式）。"""
    raise ValueError(message)


@contextlib.contextmanager
def _open_figure(**kwargs):
    """建一个 figure，并保证无论成功与否都关掉它（`plt.close` 在 finally 里）。

    本模块自己建的 figure 全走这里，一次 `render_static` 之后进程里不留任何 figure
    （测试里直接断言 `plt.get_fignums() == []`）。
    """
    fig = plt.figure(**kwargs)
    try:
        yield fig
    finally:
        plt.close(fig)


def _stage_and_replace(out: Path, names: Sequence[str], renderers: Sequence[Callable[[Path], object]]) -> list[Path]:
    """把每张图先渲染到 `<out>/.tmp-<文件名>`，**全部成功**后才逐个 `replace` 成正式文件名。

    静态图（F1–F6b）与时间图（F7–F11）共用这一份逻辑：渲染失败时上一次的好图原样保留，也不会
    留下半成品。

    清理**不看**已成功登记的列表：渲染器可能在 `savefig` 写到一半时抛错（临时文件已经建出来，
    却还来不及登记），那种残骸只能按临时文件前缀 glob 出来；顺带清掉以前崩掉留下的同类残骸。
    """
    if len(names) != len(renderers):
        _fail(f"图名清单有 {len(names)} 个、渲染器有 {len(renderers)} 个：两者必须一一对应")
    out.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        for name, render in zip(names, renderers):
            temporary = out / f"{_TMP_PREFIX}{name}"
            render(temporary)
            staged.append((temporary, out / name))
        for temporary, final in staged:
            temporary.replace(final)
        return [final for _temporary, final in staged]
    finally:
        for leftover in out.glob(f"{_TMP_PREFIX}*"):
            leftover.unlink(missing_ok=True)


def _require(path: Path, hint: str) -> None:
    """文件必须存在，否则给一句点名阶段的中文错误。"""
    if not path.exists():
        _fail(f"找不到 {path}：{hint}")


def _read_csv(path: Path, hint: str) -> pd.DataFrame:
    """读一张上游 csv；空文件/坏文件/非 UTF-8 都翻译成带路径的中文错误。"""
    _require(path, hint)
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        _fail(f"{path} 读不出来（{type(exc).__name__}: {exc}）：{hint}")
    if len(frame) == 0 or frame.shape[1] == 0:
        _fail(f"{path} 是空的（{len(frame)} 行 × {frame.shape[1]} 列）：{hint}")
    return frame


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], path: Path, hint: str) -> None:
    """核对必需列，缺列时把现有列名一起报出来（否则调用方只能靠猜上游改了什么）。"""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        _fail(f"{path} 缺少列 {missing}（现有列如 {list(frame.columns[:6])}）：{hint}")


def _read_json(path: Path, hint: str) -> dict:
    """读一份上游 json（Stage 4/5 的 QC 与指标）。"""
    _require(path, hint)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _fail(f"{path} 不是合法 JSON（{exc}）：{hint}")
    if not isinstance(payload, dict):
        _fail(f"{path} 不是 JSON 对象（实际是 {type(payload).__name__}）：{hint}")
    return payload


def _read_npy(path: Path, hint: str) -> np.ndarray:
    """读一份上游 .npy（模块表达 / 预处理矩阵）。"""
    _require(path, hint)
    try:
        array = np.load(path)
    except (OSError, ValueError) as exc:
        _fail(f"{path} 读不出来（{type(exc).__name__}: {exc}）：{hint}")
    return np.asarray(array, dtype=np.float64)


def _read_npz(path: Path, hint: str) -> dict[str, np.ndarray]:
    """读一份上游 .npz（clustermap 的重画状态 / patches），全部取成内存里的数组。"""
    _require(path, hint)
    try:
        with np.load(path, allow_pickle=False) as npz:
            return {key: np.array(npz[key]) for key in npz.files}
    except (OSError, ValueError) as exc:
        _fail(f"{path} 读不出来（{type(exc).__name__}: {exc}）：{hint}")


def _read_preprocessed(root: Path) -> tuple[np.ndarray, list[str]]:
    """读 Stage 3 的 `X.npy` + `keep_features.json`（只有 F4 的重算路径需要）。"""
    prep_root = root / _PREPROCESSED_SUBDIR
    X = _read_npy(prep_root / _X_FILE, _HINT_STAGE3)
    if X.ndim != 2:
        _fail(f"Stage3 的预处理矩阵 {prep_root / _X_FILE} 不是二维（形状 {X.shape}）：{_HINT_STAGE3}")
    keep_path = prep_root / _KEEP_FILE
    _require(keep_path, _HINT_STAGE3)
    try:
        recorded = json.loads(keep_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _fail(f"Stage3 的特征名清单 {keep_path} 不是合法 JSON（{exc}）：{_HINT_STAGE3}")
    if not isinstance(recorded, list):
        _fail(f"Stage3 的特征名清单 {keep_path} 不是名字列表（实际是 {type(recorded).__name__}）：{_HINT_STAGE3}")
    names = [str(name) for name in recorded]
    if len(names) != X.shape[1]:
        _fail(
            f"Stage3 的特征名清单 {keep_path}（{len(names)} 个）与预处理矩阵列数 {X.shape[1]} 不一致："
            f"列的语义无法确定，{_HINT_STAGE3}"
        )
    return X, names


def _contribution_scores(root: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """shape/appearance/motion 与 global/regional/distribution 的 contribution score。

    Stage 4 已经把这两个 3 元组写进 `stage4_qc.json` 的 `type_contribution` / `scope_contribution`，
    所以优先读它（零计算、与模块发现同一次运行）。只有 QC 里没有这两个键时才用 Stage 3 的
    `X.npy` + `keep_features.json` 重算一次（`compute_SAM_features_and_scope_contribution` 只做两个
    PCA，很便宜），**不重跑模块发现**。返回的第三项是来源标签（`stage4_qc` / `recomputed`），
    写进图注，免得读者以为永远是同一条路径。
    """
    qc_path = root / _MODULES_SUBDIR / _STAGE4_QC_FILE
    if qc_path.exists():
        qc = _read_json(qc_path, _HINT_STAGE4)
        if "type_contribution" in qc and "scope_contribution" in qc:
            type_scores = np.asarray(qc["type_contribution"], dtype=np.float64).ravel()
            scope_scores = np.asarray(qc["scope_contribution"], dtype=np.float64).ravel()
            if type_scores.shape != (len(_TYPE_LABELS),) or scope_scores.shape != (len(_SCOPE_LABELS),):
                _fail(
                    f"{qc_path} 的 contribution 不是 3 元组（type 形状 {type_scores.shape}、"
                    f"scope 形状 {scope_scores.shape}）：应该是 shape/appearance/motion 与 "
                    f"global/regional/distribution 各 3 项，{_HINT_STAGE4}"
                )
            return type_scores, scope_scores, "stage4_qc"

    # 重算路径：Stage 4 没落 contribution（或 QC 缺失），用 Stage 3 的矩阵补，不重跑模块发现
    X, names = _read_preprocessed(root)
    if X.shape[0] < 3:
        _fail(
            f"只有 {X.shape[0]} 个实例：contribution score 要对 3 列 SAM 表达做 PCA，至少需要 3 个实例，"
            f"无法重算，{_HINT_STAGE3}"
        )
    _types, _expr, (type_scores, scope_scores) = sam.compute_SAM_features_and_scope_contribution(
        X, np.asarray(names, dtype=object), random_state=cfg.RANDOM_STATE
    )
    return (
        np.asarray(type_scores, dtype=np.float64).ravel(),
        np.asarray(scope_scores, dtype=np.float64).ravel(),
        "recomputed",
    )


def _patch_index(root: Path, uids: list[str]) -> dict[str, tuple[Path, str, str]]:
    """按 `instance_uid` 定位 patch 条目的**地址**，不读图像数据。

    patch 文件按 `(day, fov)` 分文件、以 `label<label>_native` / `label<label>_sift` 为键，所以先用
    Stage 1 的 `all_instances.csv` 把 uid 映射到 `(day, fov, label)`，再只打开**用得到**的那几个文件。
    uid 到 label 的映射取持久化表里的 `label` 列（不是从 uid 字符串里拆），但**反手核对一次**：
    `instance_uid` 必须等于 `f"{fov}_{label}_{frame_idx}"`（Spec D13 的 uid 约定，Stage 4 也这么校过）。
    否则代表图会拼错到别的实例上，而这种错误在图上完全看不出来。轮廓被丢弃的实例在 patch 文件里
    没有条目，属于正常情形（它们不会出现在景观里），这里只保留有该实例**图像**映射的 uid，其余只计数。

    返回 `uid -> (patches.npz 路径, native 键, sift 键)`；这一步只看键名（`npz.files`），不把任何
    图像读进内存——真实数据集上"每个实例一张原生分辨率裁剪图"能吃掉几个 GB。
    """
    inst_root = root / _INSTANCES_SUBDIR
    meta_path = inst_root / _ALL_INSTANCES_FILE
    meta = _read_csv(meta_path, _HINT_STAGE1)
    _require_columns(meta, ("instance_uid", "day", "fov", "label"), meta_path, _HINT_STAGE1)
    if meta["instance_uid"].duplicated().any():
        _fail(f"Stage1 的实例汇总表 {meta_path} 里 instance_uid 有重复：无法一一定位代表图，{_HINT_STAGE1}")
    has_frame = "frame_idx" in meta.columns

    meta_by_uid = meta.set_index(meta["instance_uid"].astype(str))
    missing = [uid for uid in uids if uid not in meta_by_uid.index]
    if missing:
        head = "、".join(missing[:5])
        _fail(
            f"{len(missing)} 个景观里的实例在 Stage1 的 {meta_path} 里找不到（如 {head}"
            f"{' …' if len(missing) > 5 else ''}）：取不到 label 就定位不到代表图，"
            f"Stage2/5 与 Stage1 不是同一次运行的产物，{_HINT_STAGE1}"
        )

    rows = meta_by_uid.loc[uids]
    index: dict[str, tuple[Path, str, str]] = {}
    for (day, fov), group in rows.groupby(["day", "fov"], sort=False):
        npz_path = inst_root / str(day) / str(fov) / _PATCH_FILE
        _require(npz_path, _HINT_STAGE1)
        try:
            with np.load(npz_path, allow_pickle=False) as npz:
                available = set(npz.files)
        except (OSError, ValueError) as exc:
            _fail(f"Stage1 的 patch 文件 {npz_path} 读不出来（{type(exc).__name__}: {exc}）：{_HINT_STAGE1}")
        frames = group["frame_idx"].to_numpy() if has_frame else np.full(len(group), -1, dtype=np.int64)
        for uid, label, frame in zip(group.index.astype(str), group["label"].to_numpy(), frames):
            expected = f"{fov}_{int(label)}_{int(frame)}"
            if has_frame and uid != expected:
                _fail(
                    f"Stage1 的 {meta_path} 里 uid {uid!r} 与按 (fov, label, frame_idx) 重建的 {expected!r} "
                    f"不一致：代表图会拼错到别的实例上，{_HINT_STAGE1}"
                )
            native_key = f"label{int(label)}_native"
            sift_key = f"label{int(label)}_sift"
            if native_key not in available or sift_key not in available:
                continue
            index[uid] = (npz_path, native_key, sift_key)
    if not index:
        _fail(
            f"{len(uids)} 个实例全部没有可用代表图（{inst_root}/<day>/<fov>/patches.npz 一条都没匹配上）：{_HINT_STAGE1}"
        )
    if len(index) < len(uids):
        warnings.warn(
            f"{len(uids) - len(index)} 个实例在 Stage1 的 patches.npz 里没有代表图"
            f"（轮廓被丢弃的实例属于正常情形），F1 只用找到的那 {len(index)} 个做候选",
            UserWarning,
            stacklevel=2,
        )
    return index


def _small_flags(root: Path) -> dict[str, bool]:
    """从 Stage 1 的 `all_instances.csv` 读 `instance_uid -> is_small`（缺列时告警并当作全 False）。

    `is_small` 是“供出图时排除”的标记（Spec §6-Stage1-4）：景观里保留这些实例（删了会改写“实例数随时间
    增长”的叙述），但 F1 的代表图不拿它们当候选（小实例的裁剪图只有几个像素）。
    """
    meta_path = root / _INSTANCES_SUBDIR / _ALL_INSTANCES_FILE
    meta = _read_csv(meta_path, _HINT_STAGE1)
    if "is_small" not in meta.columns:
        warnings.warn(
            f"Stage1 的实例汇总表 {meta_path} 没有 is_small 列：F1 代表图无法按小实例过滤（仍照常出图），"
            f"请重跑 Stage1 让该列落盘（{_HINT_STAGE1}）",
            UserWarning,
            stacklevel=2,
        )
        return {}
    return {str(uid): bool(flag) for uid, flag in zip(meta["instance_uid"], meta["is_small"])}


def _filtered_members(
    members: list[int],
    uids: list[str],
    small_uids: frozenset[str] | set[str],
) -> tuple[list[int], bool]:
    """按 `is_small` 过滤代表图候选，返回 `(候选, 是否整簇都是小实例)`。

    整簇都是小实例时不静默出空面板：退回包含小实例（不然那一簇就没代表图了），由图注写明。
    """
    if not small_uids:
        return members, False
    kept = [index for index in members if uids[index] not in small_uids]
    if kept:
        return kept, False
    return members, bool(members)


@dataclass(frozen=True)
class _PatchSelection:
    """每簇预选的实例 uid + 过滤口径的计数（给 F1 图注与测试看）。"""

    uids_by_cluster: dict[int, list[str]]
    n_candidates: int
    n_excluded_small: int
    fallback_clusters: tuple[int, ...]


def _representative_uid_selection(
    coords: np.ndarray,
    clusters: np.ndarray,
    uids: list[str],
    index: dict[str, tuple[Path, str, str]],
    n_per_cluster: int = _PATCHES_PER_CLUSTER,
    small_uids: frozenset[str] | set[str] = frozenset(),
) -> _PatchSelection:
    """每簇预选 `n_per_cluster` 个实例（按到簇质心的 UMAP 距离由近到远，只给这些实例读图像）。

    选图口径与 `_cluster_panels` 一致：score = -到质心距离，取最大的 n 个。小实例（`is_small`）不进候选
    （Spec §6-Stage1-4 “供出图时排除”）；整簇只有小实例时退回包含它们，并记在 `fallback_clusters` 里。
    """
    coords = np.asarray(coords, dtype=np.float64)
    clusters = np.asarray(clusters, dtype=np.int64).ravel()
    if coords.shape[0] != clusters.size or len(uids) != clusters.size:
        _fail(
            f"代表图无法对齐：坐标 {coords.shape[0]} 个、簇标签 {clusters.size} 个、uid {len(uids)} 个，"
            "三者必须一一对应"
        )
    if n_per_cluster < 1:
        _fail(f"每簇代表图至少 1 张，实际 {n_per_cluster}")
    selection: dict[int, list[str]] = {}
    n_candidates = 0
    n_excluded_small = 0
    fallback_clusters: list[int] = []
    for cluster in [int(k) for k in np.unique(clusters)]:
        members = [int(position) for position in np.flatnonzero(clusters == cluster) if uids[position] in index]
        n_candidates += len(members)
        members, fell_back = _filtered_members(members, uids, small_uids)
        if fell_back:
            fallback_clusters.append(cluster)
        elif small_uids:
            n_excluded_small += sum(
                1
                for position in np.flatnonzero(clusters == cluster)
                if uids[position] in index and uids[position] in small_uids
            )
        if not members:
            selection[cluster] = []
            continue
        sub = coords[members]
        score = -np.linalg.norm(sub - sub.mean(axis=0), axis=1)
        # 反向 argsort：与 SPOT 的 `argsort(purity)[::-1]` 同一口径，否则"最近的 4 张"会差一两个
        picked = np.argsort(score)[::-1][:n_per_cluster]
        selection[cluster] = [uids[members[position]] for position in picked]
    return _PatchSelection(
        uids_by_cluster=selection,
        n_candidates=n_candidates,
        n_excluded_small=n_excluded_small,
        fallback_clusters=tuple(fallback_clusters),
    )


def _read_patch_arrays(
    index: dict[str, tuple[Path, str, str]], uids: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """只把 `uids` 这些实例的图像读进内存（每个 patch 文件最多打开一次），返回 (native, sift) 两份映射。"""
    by_file: dict[Path, list[str]] = {}
    for uid in uids:
        if uid not in index:
            _fail(f"实例 {uid!r} 没有 patch 地址：调用方与 `_patch_index` 不一致")
        by_file.setdefault(index[uid][0], []).append(uid)
    native: dict[str, np.ndarray] = {}
    sift: dict[str, np.ndarray] = {}
    for npz_path, group in by_file.items():
        try:
            with np.load(npz_path, allow_pickle=False) as npz:
                for uid in group:
                    _path, native_key, sift_key = index[uid]
                    native[uid] = np.array(npz[native_key])
                    sift[uid] = np.array(npz[sift_key])
        except (OSError, ValueError, KeyError) as exc:
            _fail(
                f"Stage1 的 patch 文件 {npz_path} 读不出 {group[:3]} 这些实例的图像（{type(exc).__name__}: {exc}）：{_HINT_STAGE1}"
            )
    return native, sift


@dataclass(frozen=True)
class _StaticInputs:
    """静态图需要的一切（都已按 `instance_uid` 对齐、行序统一到 umap_2d.csv 的行序）。"""

    frame: pd.DataFrame  # instance_uid / umap_1 / umap_2 / day / fov / cluster
    features: pd.DataFrame  # instance_uid / area / eccentricity / intensity / speed
    metrics: dict
    module_expr: np.ndarray  # (N, n_modules)，行序 = Stage2 实例序 = frame 的行序
    cluster_map: dict[str, np.ndarray]
    type_scores: np.ndarray
    scope_scores: np.ndarray
    score_source: str


def _load_inputs(root: Path) -> _StaticInputs:
    """读齐静态图的四类输入并做交叉核对（行身份、簇编号、模块行数）。

    行身份一律按 `instance_uid` 对齐：Stage 5 的三份表都带 uid，Stage 4 的 `module_expr.npy` 没有
    列名，只能靠"两边都是 Stage 2 的实例序"来对齐，所以这里显式断言行数相等（错位就报错，
    绝不静默画错）。
    """
    analysis_root = root / _ANALYSIS_SUBDIR
    umap_path = analysis_root / _UMAP_FILE
    umap = _read_csv(umap_path, _HINT_STAGE5)
    _require_columns(umap, ("instance_uid", "umap_1", "umap_2"), umap_path, _HINT_STAGE5)
    labels_path = analysis_root / _LABELS_FILE
    labels = _read_csv(labels_path, _HINT_STAGE5)
    _require_columns(labels, ("instance_uid", "day", "fov", "cluster"), labels_path, _HINT_STAGE5)

    umap_uids = [str(uid) for uid in umap["instance_uid"]]
    label_uids = [str(uid) for uid in labels["instance_uid"]]
    if len(set(umap_uids)) != len(umap_uids):
        _fail(f"Stage5 的 {umap_path} 里 instance_uid 有重复：行身份无法确定，{_HINT_STAGE5}")
    if len(set(label_uids)) != len(label_uids):
        _fail(f"Stage5 的 {labels_path} 里 instance_uid 有重复：行身份无法确定，{_HINT_STAGE5}")
    if set(umap_uids) != set(label_uids):
        only_umap = sorted(set(umap_uids) - set(label_uids))[:5]
        only_labels = sorted(set(label_uids) - set(umap_uids))[:5]
        _fail(
            f"Stage5 的 umap_2d.csv（{len(umap_uids)} 行）与 cluster_labels.csv（{len(label_uids)} 行）"
            f"实例集合不一致（只在 umap 里：{only_umap}；只在 labels 里：{only_labels}）："
            f"两份表不是同一次运行的产物，{_HINT_STAGE5}"
        )
    keep_columns = [column for column in ("instance_uid", "day", "fov", "cluster") if column in labels.columns]
    frame = umap.merge(
        labels[keep_columns], on="instance_uid", how="left", validate="one_to_one", sort=False
    ).reset_index(drop=True)
    coords = frame[["umap_1", "umap_2"]].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(coords)):
        n_bad = int((~np.isfinite(coords)).any(axis=1).sum())
        _fail(f"Stage5 的 {umap_path} 里有 {n_bad} 行非有限坐标（NaN/Inf）：画不了景观，{_HINT_STAGE5}")
    clusters = frame["cluster"].to_numpy(dtype=np.int64)
    cluster_ids = sorted(np.unique(clusters).tolist())
    if cluster_ids != list(range(1, len(cluster_ids) + 1)):
        _fail(
            f"簇编号是 {cluster_ids}，不是论文口径的 1..K 连续编号："
            f"（Stage5 的 order_clusters_by_score 负责给号）请重跑 Stage5"
        )

    metrics = _read_json(analysis_root / _METRICS_FILE, _HINT_STAGE5)
    expected = metrics.get("n_clusters")
    if expected is not None and int(expected) != len(cluster_ids):
        _fail(
            f"Stage5 的 kmeans_metrics.json 说切了 {int(expected)} 个簇，而 cluster_labels.csv 里有 "
            f"{len(cluster_ids)} 个（{cluster_ids}）：两份产物不是同一次运行的，{_HINT_STAGE5}"
        )

    features_path = analysis_root / _PER_INSTANCE_FILE
    features = _read_csv(features_path, _HINT_STAGE5)
    _require_columns(features, ("instance_uid", *(name for name, _label in _BAR_FEATURES)), features_path, _HINT_STAGE5)
    features = features.set_index(features["instance_uid"].astype(str))
    if features.index.duplicated().any():
        duplicated = sorted(set(features.index[features.index.duplicated()].tolist()))[:5]
        _fail(
            f"Stage5 的 {features_path} 里 instance_uid 有重复（如 {duplicated}）：按 uid 取每实例特征会一对多，"
            f"柱状图会对错实例，{_HINT_STAGE5}"
        )
    absent = [str(uid) for uid in frame["instance_uid"] if str(uid) not in features.index]
    if absent:
        _fail(
            f"{len(absent)} 个实例在 Stage5 的 per_instance_features.csv 里找不到（如 {absent[:5]}）："
            f"柱状图画不了，{_HINT_STAGE5}"
        )
    features = features.loc[[str(uid) for uid in frame["instance_uid"]]].reset_index(drop=True)

    module_expr = _read_npy(root / _MODULES_SUBDIR / _MODULE_EXPR_FILE, _HINT_STAGE4)
    if module_expr.ndim != 2:
        _fail(f"Stage4 的 module_expr.npy 不是二维（形状 {module_expr.shape}）：{_HINT_STAGE4}")
    if module_expr.shape[0] != len(frame):
        _fail(
            f"Stage4 的 module_expr.npy 有 {module_expr.shape[0]} 行，而 Stage5 的景观有 {len(frame)} 个实例："
            f"两者都按 Stage2 的实例序排列，行数不等说明不是同一次运行的产物，{_HINT_STAGE4}"
        )

    cluster_map_path = root / _MODULES_SUBDIR / _CLUSTER_MAP_FILE
    cluster_map = _read_npz(cluster_map_path, _HINT_STAGE4)
    required = (
        "corr",
        "linkage",
        "reordered_ind",
        "colorbar",
        "module_labels",
        "feature_type",
        "feature_scope",
        "feature_names",
    )
    missing_keys = [key for key in required if key not in cluster_map]
    if missing_keys:
        _fail(
            f"{cluster_map_path} 缺少键 {missing_keys}（现有键 {sorted(cluster_map)}）："
            f"重画 clustermap 需要完整的重排状态，{_HINT_STAGE4}"
        )

    type_scores, scope_scores, source = _contribution_scores(root)
    return _StaticInputs(
        frame=frame,
        features=features,
        metrics=metrics,
        module_expr=module_expr,
        cluster_map=cluster_map,
        type_scores=type_scores,
        scope_scores=scope_scores,
        score_source=source,
    )


# —— 配色与刻度的小工具 ——


def _cluster_colour_map(clusters: np.ndarray) -> dict[int, tuple]:
    """簇号 → 颜色：`coolwarm` 的 blue→red，按簇号升序排（1 = 最蓝 = 平均偏心率最小）。

    只有一个簇时取中间色（`coolwarm(0.5)`）：既不是"最蓝"也不是"最红"，避免读者把唯一的簇
    误读成排序的一端。
    """
    cluster_ids = [int(k) for k in np.unique(np.asarray(clusters))]
    if len(cluster_ids) == 1:
        return {cluster_ids[0]: cm.coolwarm(0.5)}
    positions = np.linspace(0.05, 0.95, len(cluster_ids))
    return {cluster_id: cm.coolwarm(float(position)) for cluster_id, position in zip(cluster_ids, positions)}


def _category_colour_map(values) -> dict[str, tuple]:
    """离散取值（day / fov）→ 颜色：按名字排序后依次取 `_CATEGORY_COLOURS`（确定性、与行序无关）。"""
    names = sorted({str(value) for value in values})
    return {name: _CATEGORY_COLOURS[index % len(_CATEGORY_COLOURS)] for index, name in enumerate(names)}


def _imshow_patch(ax, patch: np.ndarray, cmap: str | None = "gray") -> None:
    """把 numpy `(row, col)` 图像画到 matplotlib 的 `(x, y)` 坐标系。

    这是本模块**唯一**的坐标口径转换点（Spec D8 的配套约定）：`origin="upper"` 让第 0 行落在图的
    最上方、列沿 x 递增，与 Stage 1 存下来的裁剪图方向一致（不翻转、不转置）。
    """
    ax.imshow(np.asarray(patch), cmap=cmap, origin="upper", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _feature_ticks(names, max_ticks: int = _MAX_FEATURE_TICKS) -> tuple[np.ndarray, list[str]]:
    """特征刻度：名字太多时按等间隔抽稀（否则上百个特征名会糊成一条黑带）。"""
    labels = [str(name) for name in names]
    if len(labels) <= max_ticks:
        return np.arange(len(labels)), labels
    step = int(np.ceil(len(labels) / max_ticks))
    positions = np.arange(0, len(labels), step)
    return positions, [labels[position] for position in positions]


@dataclass(frozen=True)
class _ClusterPanel:
    """一个簇的代表图面板：`native` = 原始尺寸马赛克、`sift` = 64×64 马赛克、`uids` = 被选中的实例。"""

    native: np.ndarray | None
    sift: np.ndarray | None
    uids: list[str]


def _cluster_panels(
    coords: np.ndarray,
    clusters: np.ndarray,
    uids: list[str],
    patches_native: dict[str, np.ndarray],
    patches_sift: dict[str, np.ndarray],
    n_per_cluster: int = _PATCHES_PER_CLUSTER,
    small_uids: frozenset[str] | set[str] = frozenset(),
) -> dict[int, _ClusterPanel]:
    """每簇挑 `n_per_cluster` 张最"代表"的实例图，拼成马赛克（原始尺寸 + 标准化两版）。

    代表性的口径：**簇内到簇质心的 UMAP 距离越近越代表**（score = -距离，越大越代表）。选图直接交给
    `modules.representative_patches`（SPOT 的 `compute_most_representative_image_patches`：按纯度排序、
    取前 n_rows×n_cols 个拼格子），本模块不自己实现拼图。原始尺寸与 64×64 用**同一份 score**，
    SPOT 的排序是确定性的，所以两版选中的是同一批实例（这里顺带断言一次，防止将来口径漂移）。

    `is_small` 的实例不进候选（Spec §6-Stage1-4 “供出图时排除”）；整簇只有小实例时才退回包含它们，
    由图注写明。没有可用 patch 的实例先剔除（轮廓被丢弃的实例本来就没有 patch）；整簇一个都没有时返回
    `native=None`，由绘图函数画一句说明，而不是崩掉。
    """
    coords = np.asarray(coords, dtype=np.float64)
    clusters = np.asarray(clusters, dtype=np.int64).ravel()
    if coords.shape[0] != clusters.size or len(uids) != clusters.size:
        _fail(
            f"代表图无法对齐：坐标 {coords.shape[0]} 个、簇标签 {clusters.size} 个、uid {len(uids)} 个，"
            "三者必须一一对应"
        )
    if n_per_cluster < 1:
        _fail(f"每簇代表图至少 1 张，实际 {n_per_cluster}")
    n_rows = int(np.ceil(np.sqrt(n_per_cluster)))
    panels: dict[int, _ClusterPanel] = {}
    for cluster in [int(k) for k in np.unique(clusters)]:
        members = [
            int(index)
            for index in np.flatnonzero(clusters == cluster)
            if uids[index] in patches_native and uids[index] in patches_sift
        ]
        if not members:
            panels[cluster] = _ClusterPanel(native=None, sift=None, uids=[])
            continue
        members, _fell_back = _filtered_members(members, uids, small_uids)
        sub = coords[members]
        score = -np.linalg.norm(sub - sub.mean(axis=0), axis=1)
        score_matrix = score.reshape(-1, 1)
        _purity, native_panels, native_index = mods._representative_patches(
            score_matrix, [patches_native[uids[index]] for index in members], n_rows
        )
        _purity_sift, sift_panels, sift_index = mods._representative_patches(
            score_matrix, [patches_sift[uids[index]] for index in members], n_rows
        )
        if not np.array_equal(native_index[0], sift_index[0]):
            _fail(
                "原始尺寸与 64×64 两版代表图选中的实例不一致：两份 score 明明相同，说明挑选不再确定，"
                "请检查 modules.representative_patches"
            )
        # SPOT 返回的马赛克是 dtype=object 的格子里塞 uint8 值，先转成数值数组再交给 imshow
        panels[cluster] = _ClusterPanel(
            native=np.asarray(native_panels[0], dtype=np.float64),
            sift=np.asarray(sift_panels[0], dtype=np.float64),
            uids=[uids[members[index]] for index in native_index[0]],
        )
    return panels


@dataclass(frozen=True)
class _DensityField:
    """F3 的密度场：`fields` 键 = 簇号，`extent` = 所有面板共用的 `imshow` 范围，`vmax` = 共用色标上限。"""

    fields: dict[int, np.ndarray]
    extent: tuple[float, float, float, float]
    vmax: float


def _density_fields(
    coords: np.ndarray,
    clusters: np.ndarray,
    bins: int = _DENSITY_BINS,
    sigma_bins: float = _DENSITY_SIGMA_BINS,
) -> _DensityField:
    """在**全图共用**的网格上估每簇的 2-D 密度场（2-D 直方图 + 一个固定带宽的高斯核）。

    这里**刻意不用** SPOT 的 `compute_heatmap_density_image_Embedding`：它返回的是"每个点处的局部密度"
    而不是网格图像，而且带宽按条件（= 每簇）各自估，所以各簇的亮度不可比——本图要的是"哪一簇更集中、
    集中在哪"，只能用同一把尺子。因此自己搭一层：

    1. 用全部实例的 UMAP 坐标定一个**所有面板共用**的网格（`bins`×`bins`，四周留
       `_DENSITY_MARGIN_FRACTION` 的空白）；
    2. 每簇的坐标落到同一个网格上做 2-D 直方图，再用**同一个**高斯核（`sigma_bins` 格）平滑；
    3. 除以"该簇点数 × 单格面积"，得到每簇自己的概率密度（每层积分 = 1，所以大小簇的尺度一致）；
    4. `vmax` 取各层的最大值 → 所有面板用同一个色标，亮度可以直接互相比。

    密度归一化后为 0 或非有限值就报错（说明布线坏了：例如簇点没落在这个网格里），绝不画一块"平色"。
    """
    coords = np.asarray(coords, dtype=np.float64)
    clusters = np.asarray(clusters, dtype=np.int64).ravel()
    if coords.ndim != 2 or coords.shape[1] != 2:
        _fail(f"UMAP 坐标必须是 (N, 2) 的二维数组，实际形状 {coords.shape}")
    if coords.shape[0] != clusters.size or coords.shape[0] == 0:
        _fail(f"UMAP 坐标 {coords.shape[0]} 行与簇标签 {clusters.size} 个不一致：画不了密度场")
    if not np.all(np.isfinite(coords)):
        _fail("UMAP 坐标里有非有限值（NaN/Inf）：画不了密度场")
    if int(bins) < 8:
        _fail(f"密度网格至少 8×8，实际 {bins}")
    if float(sigma_bins) <= 0.0:
        _fail(f"密度场的平滑带宽必须为正，实际 {sigma_bins}")

    low = coords.min(axis=0)
    high = coords.max(axis=0)
    span = np.where(high - low > 0.0, high - low, 1.0)
    pad = _DENSITY_MARGIN_FRACTION * span
    x_edges = np.linspace(float(low[0] - pad[0]), float(high[0] + pad[0]), int(bins) + 1)
    y_edges = np.linspace(float(low[1] - pad[1]), float(high[1] + pad[1]), int(bins) + 1)
    bin_area = float((x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0]))

    fields: dict[int, np.ndarray] = {}
    for cluster in sorted(int(k) for k in np.unique(clusters)):
        mask = clusters == cluster
        counts, _x_edge, _y_edge = np.histogram2d(coords[mask, 0], coords[mask, 1], bins=(x_edges, y_edges))
        # counts[i, j] 对应 x_edges[i]:x_edges[i+1] × y_edges[j]:y_edges[j+1]，过一遍固定带宽的高斯核
        smoothed = gaussian_filter(counts, sigma=float(sigma_bins), mode="constant")
        density = smoothed / (float(mask.sum()) * bin_area)
        if not np.all(np.isfinite(density)) or float(density.max()) <= 0.0:
            _fail(
                f"簇 {cluster} 的密度场全为 0 或含非有限值：{int(mask.sum())} 个点没落在共用的 "
                f"{int(bins)}×{int(bins)} 网格上，说明密度场的布线坏了"
            )
        fields[cluster] = density

    vmax = max(float(field.max()) for field in fields.values())
    extent = (float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1]))
    return _DensityField(fields=fields, extent=extent, vmax=vmax)


# —— 各张图 ——


def _scatter_clusters(
    ax, coords: np.ndarray, clusters: np.ndarray, colours: dict[int, tuple], size: float = 18.0
) -> None:
    """按簇着色画景观，并在每个簇的中位位置标上簇号（论文 Fig. 3b 的编号）。

    点画一圈细黑边：`coolwarm` 的中点接近白色，中间那个簇在白底上会几乎看不见。
    簇号用 `_F1_CLUSTER_LABEL_FONTSIZE` 号字（景观缩成方阵后小字读不出来），底下垫一块半透明白底，
    压在密集点上也能看清。
    """
    for cluster in sorted(colours):
        mask = clusters == cluster
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=size,
            c=[colours[cluster]],
            alpha=0.9,
            edgecolors="0.15",
            linewidths=0.3,
            label=f"簇 {cluster} (n={int(mask.sum())})",
        )
        centre = np.median(coords[mask], axis=0)
        ax.text(
            centre[0],
            centre[1],
            str(cluster),
            ha="center",
            va="center",
            fontsize=_F1_CLUSTER_LABEL_FONTSIZE,
            fontweight="bold",
            color="black",
            bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "alpha": 0.80, "linewidth": 0},
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")


def _square_view(ax, coords: np.ndarray, margin: float = _F1_VIEW_MARGIN) -> None:
    """把景观的坐标轴设成 **1:1**：x/y 每单位等长，两个方向共用同一个跨度（多余的部分留白）。

    `set_aspect("equal")` 只保证单位等长，坐标范围仍由数据各自决定，方框会是数据本身的
    长宽比；这里先把 x/y 的范围都扩成同一个跨度并居中，方框才正好是正方形，
    "哪个方向的簇分得更开"这种读法才不是被画布拉出来的假象。
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] == 0:
        _fail(f"UMAP 坐标必须是 (N, 2) 且 N > 0，实际形状 {coords.shape}")
    low = coords.min(axis=0)
    high = coords.max(axis=0)
    centre = (low + high) / 2.0
    half = float(np.max(high - low)) / 2.0
    half = half * (1.0 + 2.0 * float(margin)) if half > 0.0 else 0.5
    ax.set_xlim(float(centre[0] - half), float(centre[0] + half))
    ax.set_ylim(float(centre[1] - half), float(centre[1] + half))
    ax.set_aspect("equal", adjustable="box")


def _f1_patch_grid(n_clusters: int) -> tuple[int, int, int]:
    """F1 右侧代表图的排版 → `(每行簇数, 行数, 面板列数)`。

    把簇铺进约 `_F1_TARGET_ROWS` 行：每行 `ceil(K / _F1_TARGET_ROWS)` 个簇，每簇 2 个面板
    （原始尺寸 + 标准化 64×64）。K=7 → (2, 4, 4)：正好 4 行 4 列，最后一个格子留白。
    """
    if int(n_clusters) < 1:
        _fail(f"F1 至少要有 1 个簇，实际 {n_clusters}")
    per_row = max(1, math.ceil(int(n_clusters) / _F1_TARGET_ROWS))
    rows = math.ceil(int(n_clusters) / per_row)
    return per_row, rows, 2 * per_row


def _plot_f1(
    path: Path,
    coords: np.ndarray,
    clusters: np.ndarray,
    panels: dict[int, _ClusterPanel],
    k: int | None = None,
    small_note: str | None = None,
) -> Path:
    """F1：UMAP 按簇着色（编号 + blue→red）+ 每簇 4 张代表图（原始尺寸、标准化尺寸两版）。

    版式：左边一块**方形**的 UMAP（1:1，见 `_square_view`），右边把代表图铺进约 4 行
    （`_f1_patch_grid`），图幅按 patch 面板的边长反推——不按簇数拉高，景观因此不会被纵向拉长。
    """
    colours = _cluster_colour_map(clusters)
    cluster_ids = sorted(colours)
    per_row, n_rows, n_panel_cols = _f1_patch_grid(len(cluster_ids))
    cell = _F1_PATCH_CELL_IN
    right_w = n_panel_cols * cell
    # 每个格子下面留 15% 给两行小标题（簇号 + 强度已归一化 + 张数）
    right_h = n_rows * cell * 1.15
    # UMAP 取正方形：`_square_view` 之后方框里才是满的，不是又高又窄的一条
    left_w = right_h
    with _open_figure(figsize=(left_w + right_w + 1.1, right_h + 1.0)) as fig:
        outer = fig.add_gridspec(1, 2, width_ratios=[left_w, right_w], wspace=0.06)
        ax_map = fig.add_subplot(outer[0])
        _scatter_clusters(ax_map, coords, clusters, colours)
        _square_view(ax_map, coords)
        ax_map.legend(fontsize=7, loc="best", framealpha=0.85)
        ax_map.set_title("UMAP 表型景观（按表型簇着色，坐标 1:1）", fontsize=10)
        inner = outer[1].subgridspec(n_rows, n_panel_cols, wspace=0.10, hspace=0.32)
        for index, cluster in enumerate(cluster_ids):
            row, col_block = divmod(index, per_row)
            panel = panels.get(cluster, _ClusterPanel(native=None, sift=None, uids=[]))
            cells = (
                (panel.native, "原始尺寸（几何原生）"),
                (panel.sift, "标准化 64×64"),
            )
            for offset, (image, kind) in enumerate(cells):
                ax = fig.add_subplot(inner[row, 2 * col_block + offset])
                if image is None:
                    ax.text(0.5, 0.5, "无可用代表图", ha="center", va="center", fontsize=8, transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"簇 {cluster} · {kind}\n（n=0）", fontsize=7)
                    continue
                _imshow_patch(ax, image)
                # 标题分两行：格子只有 ~1.5 in 宽，一行塞不下"原始尺寸 + 强度已归一化 + 张数"
                ax.set_title(f"簇 {cluster} · {kind}\n（强度已归一化，{len(panel.uids)} 张）", fontsize=7)
        fig.suptitle(
            f"F1 · 表型景观与每簇代表图（K={len(cluster_ids)}"
            f"{f'，elbow 选定 K={int(k)}' if k is not None else ''}；"
            f"簇号 1..K 蓝→红 = 平均偏心率递增；代表图 = {_REPRESENTATIVE_RULE}"
            f"{f'，{small_note}' if small_note else ''}）",
            fontsize=10,
        )
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    return path


def _plot_f2(path: Path, coords: np.ndarray, days, fovs) -> Path:
    """F2：同一景观分别按 day 与按 fov 着色（看景观是否被天数/视野主导）。"""
    with _open_figure(figsize=(12.0, 4.8)) as fig:
        axes = np.atleast_1d(fig.subplots(1, 2, squeeze=False)).ravel()
        for ax, values, name in ((axes[0], days, "day"), (axes[1], fovs, "fov")):
            values = np.asarray([str(value) for value in values])
            colours = _category_colour_map(values)
            for category in sorted(colours):
                mask = values == category
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=18.0,
                    c=[colours[category]],
                    alpha=0.9,
                    edgecolors="0.15",
                    linewidths=0.3,
                    label=f"{category} (n={int(mask.sum())})",
                )
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title(f"按 {name} 着色", fontsize=10)
            ax.legend(fontsize=7, loc="best", framealpha=0.85, title=name)
        fig.suptitle(f"F2 · 同一 UMAP 景观按 day / fov 着色（{len(coords)} 个实例）", fontsize=10)
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    return path


def _plot_f3(path: Path, coords: np.ndarray, clusters: np.ndarray, field: _DensityField) -> Path:
    """F3：每簇一层的 2-D 密度场热图（共用网格 / 带宽 / 色标 + 等密度线，并淡着叠上该簇的实例点）。"""
    cluster_ids = sorted(int(k) for k in np.unique(np.asarray(clusters, dtype=np.int64)))
    missing = [cluster for cluster in cluster_ids if cluster not in field.fields]
    if missing:
        _fail(
            f"簇 {missing} 没有密度场（现有 {sorted(field.fields)}）：共用的网格/核没盖住这些簇，"
            "密度场的布线坏了，不能画成一块平色"
        )
    ncols = min(3, len(cluster_ids))
    nrows = int(np.ceil(len(cluster_ids) / ncols))
    contours = _DENSITY_CONTOUR_FRACTIONS
    with _open_figure(figsize=(4.8 * ncols, 4.4 * nrows)) as fig:
        axes = np.atleast_1d(fig.subplots(nrows, ncols, squeeze=False)).ravel()
        fig.subplots_adjust(wspace=0.55, hspace=0.45)
        for ax, cluster in zip(axes, cluster_ids):
            mask = clusters == cluster
            density = np.asarray(field.fields[cluster], dtype=np.float64)
            # histogram2d 的第一个轴是 x、第二个是 y，所以 imshow 用 density.T（行 = y）
            image = ax.imshow(
                density.T,
                origin="lower",
                extent=field.extent,
                cmap=cm.coolwarm,
                vmin=0.0,
                vmax=field.vmax,
                interpolation="nearest",
                aspect="auto",
            )
            x_centres = np.linspace(field.extent[0], field.extent[1], density.shape[0])
            y_centres = np.linspace(field.extent[2], field.extent[3], density.shape[1])
            # 等密度线的水平按**全局** vmax 取（跨面板同义），本层面密度不到的水平直接跳过（避免
            # matplotlib 报 "No contour levels were found"）
            levels = [fraction * field.vmax for fraction in contours]
            levels = [level for level in levels if level < float(density.max())]
            if levels:
                ax.contour(
                    x_centres,
                    y_centres,
                    density.T,
                    levels=levels,
                    colors="0.15",
                    linewidths=0.5,
                )
            ax.scatter(coords[mask, 0], coords[mask, 1], s=4.0, c="0.25", alpha=0.45, linewidths=0)
            ax.set_xlim(field.extent[0], field.extent[1])
            ax.set_ylim(field.extent[2], field.extent[3])
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title(f"簇 {cluster} 的密度场（共用网格/核，n={int(mask.sum())}）", fontsize=9)
            fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04, label="密度（簇内概率密度）")
        for ax in axes[len(cluster_ids) :]:
            ax.set_visible(False)
        fig.suptitle(
            f"F3 · 每簇 UMAP 密度场（{int(_DENSITY_BINS)}×{int(_DENSITY_BINS)} 共用网格 + 固定核 σ={_DENSITY_SIGMA_BINS} 格，"
            f"共用色标 0–{field.vmax:.2e}，可跨面板比较；刻意不用 SPOT 的逐条件带宽；深灰点 = 该簇实例）",
            fontsize=10,
        )
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    return path


def _bar_with_error(ax, clusters: np.ndarray, values: np.ndarray, colours: dict[int, tuple], ylabel: str) -> None:
    """按簇画均值柱 + SEM 误差棒（NaN 列时给一句"无数据"而不是画空柱子）。"""
    cluster_ids = sorted(colours)
    values = np.asarray(values, dtype=np.float64)
    means, errors = [], []
    for cluster in cluster_ids:
        subset = values[clusters == cluster]
        subset = subset[np.isfinite(subset)]
        if subset.size:
            means.append(float(np.mean(subset)))
            errors.append(float(np.std(subset, ddof=1) / np.sqrt(subset.size)) if subset.size > 1 else 0.0)
        else:
            means.append(float("nan"))
            errors.append(0.0)
    if not np.any(np.isfinite(means)):
        ax.text(0.5, 0.5, "无数据\n（Stage5 未提供该列）", ha="center", va="center", fontsize=9, transform=ax.transAxes)
        ax.set_xticks([])
    else:
        ax.bar(
            np.arange(len(cluster_ids)),
            means,
            yerr=errors,
            capsize=3,
            color=[colours[cluster] for cluster in cluster_ids],
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_xticks(np.arange(len(cluster_ids)))
        ax.set_xticklabels([f"簇 {cluster}" for cluster in cluster_ids], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linewidth=0.4)


def _plot_f4(path: Path, type_scores: np.ndarray, scope_scores: np.ndarray, source: str) -> Path:
    """F4：contribution score 两个面板（类型 × scope），motion 明确标注为置空。"""
    with _open_figure(figsize=(11.0, 4.4)) as fig:
        axes = np.atleast_1d(fig.subplots(1, 2, squeeze=False)).ravel()
        for ax, labels, scores, title in (
            (axes[0], _TYPE_LABELS, type_scores, "特征类型（shape / appearance / motion）"),
            (axes[1], _SCOPE_LABELS, scope_scores, "空间 scope（global / regional / distribution）"),
        ):
            scores = np.asarray(scores, dtype=np.float64).ravel()
            colours = [cm.coolwarm(position) for position in np.linspace(0.05, 0.95, len(labels))]
            bars = ax.bar(np.arange(len(labels)), scores, color=colours, edgecolor="black", linewidth=0.4, width=0.6)
            if "motion" in labels:
                motion = labels.index("motion")
                bars[motion].set_facecolor("lightgrey")
                bars[motion].set_hatch("//")
                # 置空的那一格没有柱子可看，把说明写在柱子位置内部（放在轴上方会撞标题）
                top = max(float(np.nanmax(scores)), 0.02) * 1.15
                ax.set_ylim(0.0, top)
                ax.text(
                    motion,
                    top * 0.45,
                    "motion 置空\n（本项目只做\nshape + appearance）",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "linewidth": 0},
                )
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("contribution（PC1 载荷绝对值）")
            ax.set_title(title, fontsize=10)
            ax.grid(axis="y", alpha=0.25, linewidth=0.4)
        origin = (
            "Stage4 的 stage4_qc.json"
            if source == "stage4_qc"
            else "Stage3 的 X.npy + keep_features.json 重算（Stage4 未落 contribution）"
        )
        fig.suptitle(f"F4 · contribution score（来源：{origin}）", fontsize=10)
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    return path


@dataclass(frozen=True)
class _ClustermapState:
    """F5 的显示状态：矩阵与色条**已经是 dendrogram 列序**，这里只做核对与切片，不再重排。

    `order`：dendrogram 的叶序（与落盘的 `reordered_ind` 已核对一致）；
    `distances`：要画的特征距离矩阵（就是落盘的 `corr`，原样使用）；
    `module_of_position`：每个**显示位置**属于哪个模块；
    `strip`：模块色条（每个显示位置一行 RGB）；
    `feature_names`：已按显示序排好的特征名；
    `module_counts`：模块 → 特征数（按模块号的顺序）。
    """

    order: np.ndarray
    linkage: np.ndarray
    distances: np.ndarray
    module_of_position: np.ndarray
    strip: np.ndarray
    feature_names: list[str]
    module_counts: dict[int, int]


def _clustermap_display(cluster_map: dict[str, np.ndarray]) -> _ClustermapState:
    """把 Stage4 落盘的 clustermap 状态翻成"可以直接画"的显示状态，并做一致性核对。

    关键口径（`modules.discover_modules` 写盘时的口径）：

    - `corr` 取的是 seaborn 的 `grid.data2d`，而 seaborn 在 `plot_matrix` 里已经自己做了
      `data2d.iloc[yind, xind]`，所以**落盘矩阵已经是 dendrogram 列序**；再乘一个 `display_order` 会把
      矩阵转成 `D[P[P], P[P]]`，树与矩阵、矩阵与刻度就不对应了（块结构也会被洗掉）；
    - `colorbar` 是 SPOT 的 `features_clusters_colorbar`，同样**已经是 dendrogram 列序**的连续色块；
    - `module_labels` 则是 **original 列序**（模块号已按 dendrogram 里的出现先后重编过），
      所以显示序的模块标签 = `module_labels[order]`。

    因此这里只验证"落盘的列序 == 重新从 linkage 算出的 dendrogram 叶序"：两者不一致说明这份 npz 是
    旧产物（树或矩阵被换过），直接报中文错误，而不是硬排一个错位的热图。
    """
    corr = np.asarray(cluster_map["corr"], dtype=np.float64)
    linkage = np.asarray(cluster_map["linkage"], dtype=np.float64)
    order = np.asarray(cluster_map["reordered_ind"], dtype=np.int64).ravel()
    colorbar = np.asarray(cluster_map["colorbar"], dtype=np.float64)
    module_labels = np.asarray(cluster_map["module_labels"], dtype=np.int64).ravel()
    names = [str(name) for name in np.asarray(cluster_map["feature_names"]).ravel()]
    n_features = corr.shape[0] if corr.ndim == 2 else -1
    if corr.ndim != 2 or corr.shape != (n_features, n_features):
        _fail(f"Stage4 的 cluster_map.npz 里 corr 形状是 {corr.shape}，不是方阵：无法重画 clustermap")
    if sorted(order.tolist()) != list(range(n_features)):
        _fail(
            f"Stage4 的 cluster_map.npz 里 reordered_ind 不是 0..{n_features - 1} 的一个排列"
            f"（实际 {order.tolist()[:8]}…）：重排索引坏了，无法重画 clustermap"
        )
    if colorbar.shape != (n_features, 3) or module_labels.shape[0] != n_features or len(names) != n_features:
        _fail(
            f"Stage4 的 cluster_map.npz 内部长度不一致：corr {n_features}、colorbar {colorbar.shape}、"
            f"module_labels {module_labels.shape}、feature_names {len(names)}"
        )

    # 从存下来的 linkage 反推 dendrogram 叶序，与落盘的 reordered_ind 对一遍
    leaves = np.asarray(dendrogram(linkage, no_plot=True)["leaves"], dtype=np.int64)
    if not np.array_equal(leaves, order):
        _fail(
            f"Stage4 的 cluster_map.npz 里 reordered_ind（{order[:8].tolist()}…）与 linkage 画出的 "
            f"dendrogram 叶子顺序（{leaves[:8].tolist()}…）不一致：落盘的矩阵/色条与树不是同一个列序，"
            f"重画出来树与热图会错位，请重跑 Stage4{_HINT_STAGE4}"
        )

    module_of_position = module_labels[order]
    # 色条必须按模块连续成块（一个模块恰一段），否则图例无法与色块对应
    runs: list[tuple[int, int, int]] = []
    start = 0
    for position in range(1, n_features + 1):
        if position == n_features or module_of_position[position] != module_of_position[start]:
            runs.append((int(module_of_position[start]), start, position))
            start = position
    seen = [module for module, _start, _stop in runs]
    if len(set(seen)) != len(seen):
        _fail(
            f"Stage4 的 cluster_map.npz 里模块色条不连续（显示序的模块标签 {seen} 有重复块）："
            f"模块按 dendrogram 列序应当各自成一段连续色块，{_HINT_STAGE4}"
        )
    module_counts = {module: stop - start for module, start, stop in runs}
    return _ClustermapState(
        order=order,
        linkage=linkage,
        # 已重排：原样使用，绝不再乘一次 display_order
        distances=corr,
        module_of_position=module_of_position,
        strip=colorbar,
        feature_names=[names[index] for index in order],
        module_counts=module_counts,
    )


def _plot_f5(path: Path, module_expr: np.ndarray, clusters: np.ndarray, state: _ClustermapState) -> Path:
    """F5：SAM 模块 clustermap（dendrogram + 相关系数矩阵 + 模块色条）+ 簇 × 模块表达热图。"""
    corr = state.distances
    n_features = corr.shape[0]

    cluster_ids = sorted(int(k) for k in np.unique(clusters))
    expr = np.asarray(module_expr, dtype=np.float64)
    per_cluster = np.vstack([expr[clusters == cluster].mean(axis=0) for cluster in cluster_ids])
    module_ids = np.arange(1, expr.shape[1] + 1)

    with _open_figure(figsize=(13.0, 6.4)) as fig:
        grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 4.0, 0.5], width_ratios=[3.0, 2.0], hspace=0.32, wspace=0.5)
        ax_tree = fig.add_subplot(grid[0, 0])
        ax_map = fig.add_subplot(grid[1, 0])
        ax_strip = fig.add_subplot(grid[2, 0])
        ax_expr = fig.add_subplot(grid[0:3, 1])

        # 树的叶子顺序已在 `_clustermap_display` 里核对过；这里再确认一次"画出来的叶序 == 核对过的列序"，
        # 防止 dendrogram 的调用参数（例如 truncate）以后偷偷改变画图用的列序。
        tree = dendrogram(state.linkage, ax=ax_tree, no_labels=True, color_threshold=0.0, above_threshold_color="black")
        if not np.array_equal(np.asarray(tree["leaves"], dtype=np.int64), state.order):
            _fail(
                "F5 画出来的 dendrogram 叶序与核对过的列序不一致：树与热图会错位，"
                "请检查 `_plot_f5` 传给 dendrogram 的 linkage/参数"
            )
        ax_tree.set_xticks([])
        ax_tree.set_yticks([])
        for spine in ax_tree.spines.values():
            spine.set_visible(False)
        ax_tree.set_title("SAM 模块 clustermap（特征相关系数距离 + dendrogram）", fontsize=9)

        # 落盘的 corr / colorbar 已是 dendrogram 列序（seaborn 的 data2d 与 SPOT 的 colorbar 都是），
        # 所以这里直接画，不再按 display_order 乘一次（否则矩阵被转两次，树与矩阵、图例全不对应）。
        image = ax_map.imshow(corr, cmap="viridis", origin="upper", interpolation="nearest", aspect="auto")
        positions, labels = _feature_ticks(state.feature_names)
        # 特征名只标在 y 轴上（x 轴与下面的模块色条共用同一批位置，再标一遍会叠在一起）
        ax_map.set_xticks([])
        ax_map.set_yticks(positions)
        ax_map.set_yticklabels(labels, fontsize=max(2.5, min(6.0, 90.0 / max(len(labels), 1))))
        # 色条用 inset axes 而不是 fig.colorbar(ax=...)：后者会缩窄 ax_map，热图就和下面的色条错位了
        inset = ax_map.inset_axes([1.02, 0.0, 0.03, 1.0])
        fig.colorbar(image, cax=inset, label="特征距离（1 - 相关系数）")

        ax_strip.imshow(state.strip[None, :, :], aspect="auto", origin="upper")
        ax_strip.set_yticks([])
        ax_strip.set_xticks([])
        ax_strip.set_title("模块色条（按 dendrogram 列序，与上面的矩阵同序）", fontsize=7)
        for spine in ax_strip.spines.values():
            spine.set_visible(False)
        # 模块图例：按**显示序**的连续块取色（色条本身就是一段一段的），才能保证图例颜色是从这个
        # 模块自己的色块里取的；若拿 original 列序的 module_labels 去索引 dendrogram 列序的 colorbar，
        # 拿到的会是别的模块的颜色。
        handles = []
        position = 0
        for module, count in state.module_counts.items():
            colour = state.strip[position]
            handles.append(mpatches.Patch(color=colour, label=f"M{module} ({count} 个特征)"))
            position += count
        ax_strip.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(4, len(handles)),
            fontsize=6,
            frameon=False,
            handlelength=1.2,
            columnspacing=0.8,
        )

        limit = float(np.nanmax(np.abs(per_cluster))) if np.any(np.isfinite(per_cluster)) else 1.0
        expr_image = ax_expr.imshow(
            per_cluster, cmap=cm.coolwarm, origin="upper", aspect="auto", vmin=-limit, vmax=limit
        )
        ax_expr.set_yticks(np.arange(len(cluster_ids)))
        ax_expr.set_yticklabels([f"簇 {cluster}" for cluster in cluster_ids], fontsize=8)
        ax_expr.set_xticks(np.arange(expr.shape[1]))
        ax_expr.set_xticklabels([f"M{module}" for module in module_ids], fontsize=7)
        ax_expr.set_title(f"簇 × 模块表达（PC1 均值，{len(cluster_ids)} × {expr.shape[1]}）", fontsize=9)
        fig.colorbar(expr_image, ax=ax_expr, fraction=0.04, pad=0.02, label="模块表达 PC1 均值")
        fig.suptitle(f"F5 · SAM 模块（{n_features} 个特征 → {len(state.module_counts)} 个模块）", fontsize=10)
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    return path


def _motion_note(features: pd.DataFrame) -> str:
    """F6 的 motion 说明：把"列不存在"与"列全为 NaN"分开讲（不能对缺列声称已核对 NaN）。"""
    if _MOTION_FEATURE not in features.columns:
        return f"{_MOTION_FEATURE} 列未提供（表里没有这一列，无法核对，也无从画起）"
    values = features[_MOTION_FEATURE].to_numpy(dtype=np.float64)
    n_finite = int(np.isfinite(values).sum())
    if n_finite == 0:
        return f"{_MOTION_FEATURE} 列全部为 NaN（已核对 {len(values)} 行）"
    return f"注意：{_MOTION_FEATURE} 列有 {n_finite} 个非 NaN 值，本图仍未画 motion"


def _plot_f6(path: Path, features: pd.DataFrame, clusters: np.ndarray) -> Path:
    """F6：每簇 area / eccentricity / intensity 的均值 ± SEM，第四格显式说明 motion 已置空。"""
    colours = _cluster_colour_map(clusters)
    motion_note = _motion_note(features)
    with _open_figure(figsize=(14.0, 4.2)) as fig:
        axes = np.atleast_1d(fig.subplots(1, len(_BAR_FEATURES) + 1, squeeze=False)).ravel()
        fig.subplots_adjust(wspace=0.32)
        for ax, (column, ylabel) in zip(axes, _BAR_FEATURES):
            values = (
                features[column].to_numpy(dtype=np.float64)
                if column in features.columns
                else np.full(len(clusters), np.nan)
            )
            _bar_with_error(ax, clusters, values, colours, ylabel)
            ax.set_title(f"{column}（每簇均值 ± SEM）", fontsize=9)
        ax_motion = axes[-1]
        ax_motion.axis("off")
        ax_motion.text(
            0.5,
            0.55,
            "motion 置空",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            transform=ax_motion.transAxes,
        )
        ax_motion.text(
            0.5,
            0.28,
            f"本项目只复现 shape + appearance；\n速度/位移类特征未计算，\n{motion_note}\n（Spec §6-Stage2 / D4）",
            ha="center",
            va="center",
            fontsize=8,
            transform=ax_motion.transAxes,
        )
        fig.suptitle("F6 · 每簇可解释特征（原生量纲；motion 被有意跳过）", fontsize=10)
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    return path


def _plot_f6b(path: Path, clusters: np.ndarray, days) -> Path:
    """F6b：每簇的 day 组成堆叠条——一眼看出景观是否主要由天数（而非表型）驱动。"""
    cluster_ids = sorted(int(k) for k in np.unique(clusters))
    days = np.asarray([str(day) for day in days])
    day_names = sorted(set(days.tolist()))
    colours = _category_colour_map(days)
    proportions = np.zeros((len(cluster_ids), len(day_names)))
    totals = np.zeros(len(cluster_ids), dtype=np.int64)
    for row, cluster in enumerate(cluster_ids):
        mask = clusters == cluster
        totals[row] = int(mask.sum())
        for column, day in enumerate(day_names):
            proportions[row, column] = float(np.mean(days[mask] == day)) if totals[row] else 0.0

    with _open_figure(figsize=(max(6.0, 1.1 * len(cluster_ids) + 4.0), 4.4)) as fig:
        ax = fig.subplots(1, 1)
        bottom = np.zeros(len(cluster_ids))
        for column, day in enumerate(day_names):
            ax.bar(
                np.arange(len(cluster_ids)),
                proportions[:, column],
                bottom=bottom,
                color=[colours[day]],
                edgecolor="black",
                linewidth=0.4,
                width=0.62,
                label=day,
            )
            bottom += proportions[:, column]
        ax.set_xticks(np.arange(len(cluster_ids)))
        ax.set_xticklabels([f"簇 {cluster}\n(n={totals[row]})" for row, cluster in enumerate(cluster_ids)], fontsize=8)
        ax.set_ylabel("day 组成（簇内占比）")
        ax.set_ylim(0.0, 1.0)
        ax.legend(title="day", fontsize=8, loc="upper right", framealpha=0.85)
        ax.set_title("每簇的 day 组成（若某簇被单一天数占满，说明景观可能主要由天数驱动）", fontsize=9)
        fig.suptitle(f"F6b · 每簇 day 组成（{len(cluster_ids)} 个簇，{len(days)} 个实例）", fontsize=10)
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    return path


# ================================================================ 图说明（figures/README.md）
#
# Stage 7 的产物落进 `output/figures/`，但"这张图在说什么、能不能读出某个结论"只写在代码的 docstring
# 和图注里，打开目录的人看不到。这里把 13 张图的说明收成一张表，渲染完图顺手写一份 `README.md`
# 到图目录：**图与说明同源**，不会出现"图改了、说明忘了改"。每条说明固定三段：
# 这张图是什么（what）/ 怎么读、不能读出什么（reading）/ 吃哪些上游产物（inputs）。

_README_NAME: str = "README.md"
# 图说明的两个分组（静态图吃 Stage1/4/5，时间图吃 Stage6/6b）
_README_GROUP_STATIC: str = "静态图（F1–F6b）"
_README_GROUP_TEMPORAL: str = "时间图（F7–F11）"


@dataclass(frozen=True)
class _FigureDoc:
    """一张图的说明：文件名 + 分组 + 一句话标题 + 是什么 / 怎么读 / 依赖哪些上游产物。"""

    name: str
    group: str
    title: str
    what: str
    reading: str
    inputs: str


_FIGURE_DOCS: tuple[_FigureDoc, ...] = (
    _FigureDoc(
        name=_FIG_F1,
        group=_README_GROUP_STATIC,
        title="表型景观 + 每簇代表形态（论文 Fig. 3b）",
        what=(
            "全实例的联合 UMAP 景观按表型簇着色（簇号 1..K，配色 blue→red 与"
            "“平均偏心率递增”同序）；右侧每簇 4 张代表图，左列原始尺寸、右列标准化 64×64。"
        ),
        reading=(
            "景观坐标是 1:1（x/y 每单位等长、两个方向共用同一跨度），所以簇之间的远近可以直接比较；"
            "代表图不是随机抽样，而是簇内到簇质心 UMAP 距离最近的 4 张（`is_small` 的实例不参与），"
            "它只说明“这一簇大概长什么样”，不是统计量。簇的实例数看图例里的 n。"
        ),
        inputs="Stage5 的 `analysis/umap_2d.csv` + `cluster_labels.csv`、Stage1 的 `instances/*/patches.npz`",
    ),
    _FigureDoc(
        name=_FIG_F2,
        group=_README_GROUP_STATIC,
        title="同一景观按 day / 按 fov 着色（论文 Fig. 3b 的分面版）",
        what="同一份 UMAP 坐标画两遍：一遍按采样天数着色，一遍按视野着色。",
        reading=(
            "某一种颜色成片占据景观的一块区域 → 这个景观很大程度在编码“天数/视野”，"
            "而不是纯形态；反过来（颜色均匀混在一起）不能证明没有批次效应，只能说明没看到。"
        ),
        inputs="Stage5 的 `analysis/umap_2d.csv` + `cluster_labels.csv`",
    ),
    _FigureDoc(
        name=_FIG_F3,
        group=_README_GROUP_STATIC,
        title="每簇一层的 UMAP 密度场（论文 Fig. 3b 的密度层）",
        what="每个表型簇单独一层 2-D 密度热图，所有面板共用同一个网格、同一个高斯核、同一个色标。",
        reading=(
            "亮 = 该簇的实例在这个位置集中。每层按簇内点数归一（每层积分为 1），所以大小簇的亮度可比，"
            "面板之间可以直接比高低。注意这是本适配器自己估的核密度，**不是** SPOT 的逐点密度函数"
            "（后者带宽按条件各自估，面板之间不可比）。"
        ),
        inputs="Stage5 的 `analysis/umap_2d.csv` + `cluster_labels.csv`",
    ),
    _FigureDoc(
        name=_FIG_F4,
        group=_README_GROUP_STATIC,
        title="特征家族的 contribution score（论文 Fig. 2f–g / 3c）",
        what="shape / appearance / motion × global / regional / distribution 六个维度的贡献分数。",
        reading=(
            "看“是哪个特征家族在驱动这些簇”。motion 一栏恒为空：本数据是相隔数天的离散采样，"
            "光流速度算不出可信值，motion 特征被有意置空（不是算出来等于 0）。"
        ),
        inputs="Stage4 的 `modules/stage4_qc.json`（缺键时用 Stage3 的 `preprocessed/X.npy` 重算）",
    ),
    _FigureDoc(
        name=_FIG_F5,
        group=_README_GROUP_STATIC,
        title="SAM 模块 clustermap + 簇 × 模块表达（论文 Fig. 3d）",
        what="左侧特征相关矩阵（已按 dendrogram 叶子序重排）+ 模块色条，右侧每簇在各模块上的表达热图。",
        reading=(
            "一个模块 = 一组彼此高相关的特征（相关特征家族），**不等于**一条生物学通路；"
            "模块数取决于选择器（本适配器用论文的 `legacy` 口径，不是 v2 默认的 `multiscale`，两者模块数不同）。"
        ),
        inputs="Stage4 的 `modules/module_expr.npy` + `cluster_map.npz`",
    ),
    _FigureDoc(
        name=_FIG_F6,
        group=_README_GROUP_STATIC,
        title="每簇可解释特征（论文 Fig. 3c）",
        what="每簇 area / eccentricity / intensity 的均值 ± SEM，第四个面板显式标注 motion 置空。",
        reading=(
            "SEM 是**同一簇内实例之间**的离散度，不是重复实验的误差棒，别当置信区间读。"
            "area 在 Stage3 的 `remove_high_variance_features` 里已被剔除出景观特征，"
            "所以“面积不驱动 UMAP”是那一步过滤的结果，不是数据里没有面积差异；这里画的是原生尺度的面积。"
        ),
        inputs="Stage5 的 `analysis/per_instance_features.csv` + `cluster_labels.csv`",
    ),
    _FigureDoc(
        name=_FIG_F6B,
        group=_README_GROUP_STATIC,
        title="每簇的 day 组成（Spec §11-Q1 诊断图）",
        what="每个表型簇里各采样天数所占的比例（堆叠到 1）。",
        reading=(
            "若某簇被单一天数占满（比如 90% 来自 day0），这个簇基本等于“那天的形态”，"
            "此时所有以簇为单位的结论都要带上“簇 ≈ 天数/大小”的限定。"
        ),
        inputs="Stage5 的 `analysis/cluster_labels.csv`",
    ),
    _FigureDoc(
        name=_FIG_F7,
        group=_README_GROUP_TEMPORAL,
        title="各簇占比随时间（论文 Fig. 3c iii）",
        what="每个视野一个面板的簇占比堆叠条，外加一个所有视野合并的面板。",
        reading=(
            "合并面板是**按每视野每采样点的实例数加权平均**（不是等权，小视野不会被放大）。"
            "时间轴是离散采样点（不是连续视频），占比变化不能读成“增殖速率”——"
            "没有同一批对象的追踪，无法区分“变多了”和“换了一批”。"
        ),
        inputs="Stage6b 的 `analysis/proportions.npz` + `tracks.csv`",
    ),
    _FigureDoc(
        name=_FIG_F8,
        group=_README_GROUP_TEMPORAL,
        title="簇间转移（论文 Fig. 3c iv）",
        what="左侧观测到的簇间转移有向图，右侧含自转移的转移概率热图（行 = 来源，列 = 目标）。",
        reading=(
            "左图只画真正观测到的边（`count >= 1`），边上的概率是**去掉自转移后按行重归一**的结果；"
            "右图的行和才是 1（含自转移），两者分母不同、别混读。右图里纯先验格的格子打了斜纹。"
            "读之前先看 F11：merge/split 多的话，这里的转移很大一部分来自匹配/分割不稳定。"
        ),
        inputs="Stage6b 的 `analysis/transitions.npz`",
    ),
    _FigureDoc(
        name=_FIG_F9,
        group=_README_GROUP_TEMPORAL,
        title="每视野的 UMAP 表型轨迹（论文 Fig. 3c v）",
        what="每个视野一个面板：彩色折线是逐个 track 在景观里走过的路径，粗黑线是 Stage6b 的表型轨迹（每簇每采样点的 UMAP 质心）。",
        reading=(
            "只画长度 ≥ 2 的 track（长度 1 的只有一个采样点，连成线等于凭空造出一段位移）。"
            "折线的长短是“在 UMAP 上走了多远”，**不是**物理位移或速度：采样点相隔 2–3 天，"
            "景观的尺度也没有物理单位。"
        ),
        inputs="Stage6b 的 `analysis/trajectories.npz` + `tracks.csv` + Stage5 的 `umap_2d.csv`",
    ),
    _FigureDoc(
        name=_FIG_F10A,
        group=_README_GROUP_TEMPORAL,
        title="视野级轨迹亲和 + dendrogram（论文 Fig. 4c–e 的视野级版本）",
        what="每个视野一条表型轨迹，两两之间的 DTW 指数核亲和矩阵 + 层次聚类树。",
        reading=(
            "**视野数少时这张图的信息量很小**：只有 1 个视野时树退化成单个叶子，"
            "2 个视野时只有一对数值；真正有信息量的是 F10b 的簇级版本。"
        ),
        inputs="Stage6b 的 `analysis/affinity.npz`",
    ),
    _FigureDoc(
        name=_FIG_F10B,
        group=_README_GROUP_TEMPORAL,
        title="簇级轨迹亲和 + dendrogram（论文 Fig. 4c–e 的簇级版本）",
        what="每个表型簇一条表型轨迹（该簇在各采样点的 UMAP 质心），两两之间的亲和矩阵 + 层次聚类树。",
        reading="回答“哪些表型簇的时间行为相似”：树上的近邻簇在景观里可能离得很远，那是两回事。",
        inputs="Stage6b 的 `analysis/trajectories.npz` + Stage5 的 `umap_2d.csv`",
    ),
    _FigureDoc(
        name=_FIG_F11,
        group=_README_GROUP_TEMPORAL,
        title="跨采样点匹配事件 QC（论文无对应图）",
        what="每个 (视野, 帧对) 的 one_to_one / merge / split 事件数，以及两侧各自未匹配的实例数。",
        reading=(
            "这是读 F7/F8/F9 之前的前置 QC：merge/split/未匹配占比高，说明匹配阈值或分割稳定性有问题，"
            "那些“转移 / 轨迹”就该打折扣看；这张图不提供任何生物学结论。"
        ),
        inputs="Stage6 的 `analysis/match_qc.json`",
    ),
)


def _readme_lines(out: Path) -> list[str]:
    """把 `_FIGURE_DOCS` + 本目录的实际文件情况拼成 markdown 行（纯函数，便于测试）。"""
    present = [doc for doc in _FIGURE_DOCS if (out / doc.name).exists()]
    missing = [doc for doc in _FIGURE_DOCS if doc not in present]
    lines = [
        "# 图说明：Stage 7 出图目录",
        "",
        (
            f"本目录是复现流水线 Stage 7 的出图目录（`<output>/figures/`，默认就是 `output/figures/`），"
            f"全部 **{_DPI} dpi PNG**。本文件由"
            " `biopheno.figures.write_figure_readme` 在每次出图后自动重写（**别手改**，重跑会覆盖）。"
        ),
        "",
        "## 本目录现在有什么",
        "",
        f"- 已生成 **{len(present)} / {len(_FIGURE_DOCS)}** 张。",
    ]
    if missing:
        lines.append(
            f"- 缺失 {len(missing)} 张：{'、'.join(f'`{doc.name}`' for doc in missing)}"
            "（缺静态图重跑 Stage 7 静态部分，缺时间图重跑 `render_temporal`）。"
        )
    lines += [
        "",
        "| 文件 | 一句话 | 状态 |",
        "|---|---|---|",
    ]
    lines += [f"| `{doc.name}` | {doc.title} | {'已生成' if doc in present else '**缺失**'} |" for doc in _FIGURE_DOCS]
    lines += [
        "",
        "## 读图前先记住这三条口径",
        "",
        (
            "1. **motion 是空的，不是 0**：采样点相隔 2–3 天，帧间光流算不出有意义的速度，"
            "motion 特征被有意置空（NaN）。所有图上标“置空”的格子都不代表“速度为 0”。"
        ),
        (
            "2. **时间轴是离散采样点，不是连续视频**：任何“斜率 / 速率 / 单位时间变化”都读不出来，"
            "只能读“两个采样点之间的差异”。"
        ),
        (
            "3. **簇号 1..K 是按簇内平均偏心率递增排的**（配色 blue→red 同序，越红越扁长）："
            "簇号是排序结果，本身没有生物学含义，换一次 K 就会重排。"
        ),
        "",
    ]
    for group in (_README_GROUP_STATIC, _README_GROUP_TEMPORAL):
        docs = [doc for doc in _FIGURE_DOCS if doc.group == group]
        lines += [
            f"## {group}",
            "",
        ]
        for doc in docs:
            lines += [
                f"### `{doc.name}` — {doc.title}",
                "",
                f"- **是什么**：{doc.what}",
                f"- **怎么读**：{doc.reading}",
                f"- **依赖**：{doc.inputs}",
                "",
            ]
    lines += [
        "## 这张目录里**没有**的图",
        "",
        "- 速度 / 位移类（论文里的 motion 家族）：本数据的采样间隔不支持，未计算。",
        ("- 任何绝对尺度（µm）：`PIXEL_RES = 1.0`，所有长度/面积都以像素为单位，没有标尺信息就无法换算成物理尺寸。"),
        "- 统计检验与 p 值：这些图是描述性景观，不是假设检验的结果。",
        "",
        "## 重新生成",
        "",
        "```bash",
        "cd repro",
        "uv run python scripts/run_all.py --images <原图目录> --masks <掩码目录> --output output/<名字>",
        "```",
        "",
        (
            "只重画图、不重算上游：在项目根目录下跑 "
            '`uv run python -c "from biopheno.figures import render_static, render_temporal; '
            "render_static('output', 'output/figures'); render_temporal('output')\"`。"
        ),
        "",
    ]
    return lines


def write_figure_readme(outdir: Path) -> Path:
    """把 13 张图的说明写进 `<outdir>/README.md`，返回该文件路径。

    内容全部来自 `_FIGURE_DOCS`（与出图代码同一个模块，改图时顺手改文档），并根据**目录里实际存在的
    文件**标注“已生成 / 缺失”——所以 `render_static` 之后跑一次、`render_temporal` 之后再跑一次，
    最后留下的那份说明与实际目录一致。先写 `<outdir>/.tmp-README.md` 再 `replace`，不会留下半份文件。
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / _README_NAME
    temporary = out / f"{_TMP_PREFIX}{_README_NAME}"
    temporary.write_text("\n".join(_readme_lines(out)), encoding="utf-8")
    temporary.replace(target)
    return target


def render_static(analysis_dir: Path, outdir: Path, force_single_cluster: bool = False) -> list[Path]:
    """画 Spec §6-Stage7 的静态图 F1–F6b，返回写出的 PNG 路径（顺序与文件名清单一致）。

    Parameters
    ----------
    analysis_dir : Path
        **与真实 `output/` 同构的目录树根**（其下有 `analysis/`、`modules/`、`instances/`、
        `preprocessed/`）。测试把它指向一棵合成小树，所以本函数从不假设真实数据集存在。
    outdir : Path
        图的落盘目录（不存在时创建）。七张图先渲染到 `<outdir>/.tmp-<文件名>`，**全部成功**后才逐个
        `replace` 成正式文件名：渲染失败时上一次的好图原样保留，也不会留下半成品。
    force_single_cluster : bool
        把所有实例当成一个簇来画（退化路径）。用途是把"K=1 也不能崩"这条约束变成可测的开关：
        真实数据上 K>=2（Stage 5 的 elbow 扫描下界是 2），但簇数退化成 1 时所有图仍应出得来。

    每一步的失败都抛带阶段名的中文 ValueError（缺哪个产物就说该跑哪一阶段），不静默少画；
    降级（缺中文字体 / 部分实例没有代表图 / 强制单簇）一律走 `warnings.warn`，并在汇总日志行里重中一次。
    """
    root = Path(analysis_dir)
    out = Path(outdir)
    inputs = _load_inputs(root)

    coords = inputs.frame[["umap_1", "umap_2"]].to_numpy(dtype=np.float64)
    clusters = inputs.frame["cluster"].to_numpy(dtype=np.int64)
    uids = [str(uid) for uid in inputs.frame["instance_uid"]]
    if force_single_cluster:
        clusters = np.ones_like(clusters)
        warnings.warn(
            "force_single_cluster=True：所有实例按 1 个簇画（退化路径，用于验证 K=1 不崩），"
            "簇相关的图（F1/F3/F5/F6/F6b）不再反映真实的 K",
            UserWarning,
            stacklevel=2,
        )

    # 先确定每簇要用的那几张图，再只把这几张读进内存（原生分辨率裁剪图逐实例一张，全读会吃几 GB）
    patch_index = _patch_index(root, uids)
    small_flags = _small_flags(root)
    small_uids = {uid for uid, flag in small_flags.items() if flag}
    selection = _representative_uid_selection(coords, clusters, uids, patch_index, small_uids=small_uids)
    patches_native, patches_sift = _read_patch_arrays(
        patch_index, [uid for cluster_uids in selection.uids_by_cluster.values() for uid in cluster_uids]
    )
    panels = _cluster_panels(coords, clusters, uids, patches_native, patches_sift, small_uids=small_uids)
    density = _density_fields(coords, clusters)
    clustermap = _clustermap_display(inputs.cluster_map)
    small_note = (
        f"{_SMALL_SKIP_NOTE}（跳过 {selection.n_excluded_small} 个候选"
        + (
            f"；簇 {'、'.join(str(cluster) for cluster in selection.fallback_clusters)} 内没有非小实例，只能含小实例）"
            if selection.fallback_clusters
            else "）"
        )
        if small_uids
        else None
    )

    renderers = (
        lambda path: _plot_f1(path, coords, clusters, panels, k=inputs.metrics.get("k"), small_note=small_note),
        lambda path: _plot_f2(path, coords, inputs.frame["day"].to_numpy(), inputs.frame["fov"].to_numpy()),
        lambda path: _plot_f3(path, coords, clusters, density),
        lambda path: _plot_f4(path, inputs.type_scores, inputs.scope_scores, inputs.score_source),
        lambda path: _plot_f5(path, inputs.module_expr, clusters, clustermap),
        lambda path: _plot_f6(path, inputs.features, clusters),
        lambda path: _plot_f6b(path, clusters, inputs.frame["day"].to_numpy()),
    )
    # 先写临时文件，全部成功后才换成正式文件名（不在渲染前删旧图）
    paths = _stage_and_replace(out, _FIGURE_NAMES, renderers)
    # 图说明随图一起更新（列出的是本目录**实际存在**的图，所以时间图那部分此时可能还标“缺失”）
    readme = write_figure_readme(out)

    font_note = (
        ""
        if _CJK_FONT is not None
        else f"；警告：未找到中文字体（候选 {' / '.join(_CJK_FONT_CANDIDATES)}），图注中文会渲染成方框"
    )
    print(
        f"Stage7 静态图完成：{len(paths)} 张（{len(clusters)} 个实例、"
        f"{len(np.unique(clusters))} 个簇、contribution 来源 {inputs.score_source}；"
        f"F3 密度场用 {int(_DENSITY_BINS)}×{int(_DENSITY_BINS)} 共用网格 + 固定核，"
        f"刻意不同于 SPOT 的逐条件带宽）→ {out}；图说明 → {readme}{font_note}"
    )
    return paths


# ================================================================ 时间图（F7–F11）
#
# 时间轴是 **3 个离散采样点**（day0 / day3 / day5），不是连续视频：motion 特征被有意置空，
# 所以这里所有图的"时间"含义都只是"第 1/2/3 次采样之间的差异"，任何斜率/速度都不能读。
# 数据全部来自 Stage6b 的落盘产物（四份 npz + QC）与 Stage6 的 `tracks.csv` / `match_qc.json`：
# 本阶段只读盘，不重跑占比/转移/轨迹统计。


def _require_keys(payload: dict[str, np.ndarray], keys: tuple[str, ...], path: Path, hint: str) -> None:
    """npz 缺键时报中文错误，并把现有键一起列出来（否则调用方只能猜上游改了什么）。"""
    missing = [key for key in keys if key not in payload]
    if missing:
        _fail(f"{path} 缺少键 {missing}（现有键 {sorted(payload)}）：{hint}")


def _locate_analysis_dir(directory: Path) -> Path:
    """定位 Stage6b 的产物目录：既接受 `output/` 根（其下有 `analysis/`），也接受 `output/analysis/`。

    时间图的输入全在同一个目录里，两种写法都试一次只是为了让调用方不必记住 `render_static` 与本函数的
    口径差异；两者都不存在时给一句点名 Stage6b 的中文错误。
    """
    candidate = Path(directory)
    if (candidate / _PROPORTIONS_FILE).exists():
        return candidate
    nested = candidate / _ANALYSIS_SUBDIR
    if (nested / _PROPORTIONS_FILE).exists():
        return nested
    _fail(
        f"找不到 {candidate / _PROPORTIONS_FILE}（也没有 {nested / _PROPORTIONS_FILE}）："
        f"时间图要读 Stage6b 的占比/转移/轨迹/亲和产物，{_HINT_STAGE6B}"
    )


def _temporal_common(payload: dict[str, np.ndarray], path: Path) -> dict:
    """取出并核对四份 npz 共有的标签顺序（簇 / fov / 时间箱），返回规范化后的 dict。"""
    _require_keys(payload, _TEMPORAL_COMMON_KEYS, path, _HINT_STAGE6B)
    labels = np.asarray(payload["cluster_labels"], dtype=np.int64).ravel()
    conditions = [str(name) for name in np.asarray(payload["conditions"]).ravel()]
    bin_frame_idx = np.asarray(payload["bin_frame_idx"], dtype=np.int64).ravel()
    bin_day = [str(day) for day in np.asarray(payload["bin_day"]).ravel()]
    bin_time_hours = np.asarray(payload["bin_time_hours"], dtype=np.float64).ravel()
    bin_analysis_time = np.asarray(payload["bin_analysis_time"], dtype=np.float64).ravel()
    time_intervals = np.asarray(payload["time_intervals"], dtype=np.float64).ravel()

    if labels.size == 0 or not conditions or bin_frame_idx.size == 0:
        _fail(
            f"{path} 的标签顺序是空的（簇 {labels.size} 个、fov {len(conditions)} 个、"
            f"时间箱 {bin_frame_idx.size} 个）：时间图画不出任何轴，{_HINT_STAGE6B}"
        )
    if labels.tolist() != list(range(1, labels.size + 1)):
        _fail(f"{path} 的 cluster_labels 是 {labels.tolist()}，不是论文口径的 1..K 连续编号：{_HINT_STAGE6B}")
    if not (len(bin_day) == bin_frame_idx.size == bin_time_hours.size == bin_analysis_time.size):
        _fail(
            f"{path} 的时间箱标签长度不一致（bin_frame_idx {bin_frame_idx.size}、bin_day {len(bin_day)}、"
            f"bin_time_hours {bin_time_hours.size}、bin_analysis_time {bin_analysis_time.size}）：{_HINT_STAGE6B}"
        )
    if time_intervals.size != bin_frame_idx.size + 1 or not np.all(np.diff(time_intervals) > 0.0):
        _fail(f"{path} 的 time_intervals 不是 {bin_frame_idx.size + 1} 个严格递增的边界：{_HINT_STAGE6B}")
    return {
        "cluster_labels": labels,
        "conditions": conditions,
        "bin_frame_idx": bin_frame_idx,
        "bin_day": bin_day,
        "bin_time_hours": bin_time_hours,
        "bin_analysis_time": bin_analysis_time,
        "time_intervals": time_intervals,
    }


def _check_shared_labels(expected: dict, payload: dict[str, np.ndarray], path: Path) -> None:
    """四份 npz 必须共用同一套标签顺序，否则图会把簇或视野对错轴（Stage6b 已校验过，这里防混用旧产物）。"""
    if not np.array_equal(expected["cluster_labels"], np.asarray(payload["cluster_labels"], dtype=np.int64).ravel()):
        _fail(f"{path} 的 cluster_labels 与 {_PROPORTIONS_FILE} 不一致：两份产物不是同一次运行的，{_HINT_STAGE6B}")
    if [str(name) for name in np.asarray(payload["conditions"]).ravel()] != list(expected["conditions"]):
        _fail(f"{path} 的 conditions 与 {_PROPORTIONS_FILE} 不一致：视野顺序会对错轴，{_HINT_STAGE6B}")
    if not np.array_equal(expected["bin_frame_idx"], np.asarray(payload["bin_frame_idx"], dtype=np.int64).ravel()):
        _fail(f"{path} 的 bin_frame_idx 与 {_PROPORTIONS_FILE} 不一致：时间箱会对错位置，{_HINT_STAGE6B}")
    if [str(day) for day in np.asarray(payload["bin_day"]).ravel()] != list(expected["bin_day"]):
        _fail(f"{path} 的 bin_day 与 {_PROPORTIONS_FILE} 不一致：时间轴标签会对错箱子，{_HINT_STAGE6B}")


def _npz_scalar(payload: dict[str, np.ndarray], key: str, path: Path, hint: str = _HINT_STAGE6B):
    """取 npz 里的单值条目（`pseudocounts` / `path` 这类元数据）。

    `ravel()[0]` 会静默吞掉多值或空值：上游要是误存了一个数组，图上就会拿第一个值冒充整份配置。
    不是恰好 1 个值就报一句中文错误，并把它到底存了几个值列出来。
    """
    values = np.asarray(payload[key]).ravel()
    if values.size != 1:
        _fail(f"{path} 的 {key} 应只有 1 个值，实际有 {values.size} 个（{values.tolist()[:6]}）：{hint}")
    return values[0]


def _load_proportions(root: Path) -> tuple[np.ndarray, dict]:
    """读 `proportions.npz`（条件 × 时间箱 × 簇），并核对"每行要么整行有效、要么整行空箱、有效行和为 1"。"""
    path = root / _PROPORTIONS_FILE
    payload = _read_npz(path, _HINT_STAGE6B)
    _require_keys(payload, ("proportions", *_TEMPORAL_COMMON_KEYS), path, _HINT_STAGE6B)
    common = _temporal_common(payload, path)
    proportions = np.asarray(payload["proportions"], dtype=np.float64)
    expected = (len(common["conditions"]), common["bin_frame_idx"].size, common["cluster_labels"].size)
    if proportions.shape != expected:
        _fail(
            f"{path} 的 proportions 形状是 {proportions.shape}，按标签顺序应为 (条件, 时间箱, 簇) = {expected}："
            f"{_HINT_STAGE6B}"
        )
    rows = proportions.reshape(-1, proportions.shape[-1])
    partial = np.isfinite(rows).any(axis=1) & ~np.isfinite(rows).all(axis=1)
    if partial.any():
        _fail(
            f"{path} 有 {int(partial.sum())} 个 (视野, 采样点) 的占比只填了一部分（同一行里有的簇有值、有的 NaN）："
            f"占比要么整行有效、要么整行空箱，{_HINT_STAGE6B}"
        )
    full = np.isfinite(rows).all(axis=1)
    if full.any():
        deviation = float(np.max(np.abs(rows[full].sum(axis=1) - 1.0)))
        if deviation > 1e-6:
            _fail(
                f"{path} 有占比行不为 1（最大偏差 {deviation:.3g}）：F7 的堆叠条会凭空多出或少掉一部分，{_HINT_STAGE6B}"
            )
    return proportions, common


def _load_transitions(root: Path, common: dict) -> dict:
    """读 `transitions.npz` 并核对它的契约：`probs` 行随机、`inter` 对角为 0、计数非负。"""
    path = root / _TRANSITIONS_FILE
    payload = _read_npz(path, _HINT_STAGE6B)
    names = ("inter", "self_transitions", "probs", "counts", "pseudocounts")
    _require_keys(payload, (*names, *_TEMPORAL_COMMON_KEYS), path, _HINT_STAGE6B)
    _temporal_common(payload, path)
    _check_shared_labels(common, payload, path)

    n_clusters = common["cluster_labels"].size
    matrices = {name: np.asarray(payload[name], dtype=np.float64) for name in ("inter", "probs", "counts")}
    for name, matrix in matrices.items():
        if matrix.shape != (n_clusters, n_clusters):
            _fail(
                f"{path} 的 {name} 形状是 {matrix.shape}，按 {n_clusters} 个簇应为 "
                f"{(n_clusters, n_clusters)}：{_HINT_STAGE6B}"
            )
    self_transitions = np.asarray(payload["self_transitions"], dtype=np.float64).ravel()
    if self_transitions.size != n_clusters:
        _fail(f"{path} 的 self_transitions 有 {self_transitions.size} 个值，应有 {n_clusters} 个：{_HINT_STAGE6B}")
    if not np.allclose(np.diag(matrices["probs"]), self_transitions, atol=1e-8):
        _fail(f"{path} 的 self_transitions 与 probs 的对角不一致：自转移热图会对不上，{_HINT_STAGE6B}")
    # 单簇时 inter 的非对角全是“0 / 0” → NaN（Stage6b 的 _inter_from_probs 不会凭空造边），
    # 这不是坏产物：只校核“有值的对角必须是 0”，NaN 留给画图时当“没有边”。
    diagonal = np.diag(matrices["inter"])
    if np.any(np.isfinite(diagonal) & ~np.isclose(diagonal, 0.0, atol=1e-9)):
        _fail(f"{path} 的 inter 对角不是 0（自转移应单列在 self_transitions 里）：{_HINT_STAGE6B}")
    row_sums = matrices["probs"].sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        _fail(
            f"{path} 的 probs 不是行随机矩阵（行和 {np.round(row_sums, 6).tolist()}）："
            f"转移概率的契约是行 = 来源、行和 = 1，{_HINT_STAGE6B}"
        )
    if not np.all(np.isfinite(matrices["counts"])):
        _fail(f"{path} 的 counts 里有 NaN/Inf：未平滑原始计数必须有限，{_HINT_STAGE6B}")
    if (matrices["counts"] < 0.0).any():
        _fail(f"{path} 的 counts 里有负数：未平滑原始计数不可能为负，{_HINT_STAGE6B}")
    return {
        "inter": matrices["inter"],
        "probs": matrices["probs"],
        "counts": matrices["counts"],
        "pseudocounts": float(_npz_scalar(payload, "pseudocounts", path)),
    }


def _load_trajectories(root: Path, common: dict) -> tuple[np.ndarray, str]:
    """读 `trajectories.npz` 的表型轨迹（条件 × 时间箱 × 2）与它实际走的路径名。"""
    path = root / _TRAJECTORIES_FILE
    payload = _read_npz(path, _HINT_STAGE6B)
    _require_keys(payload, ("trajectories", "path", *_TEMPORAL_COMMON_KEYS), path, _HINT_STAGE6B)
    _temporal_common(payload, path)
    _check_shared_labels(common, payload, path)

    trajectories = np.asarray(payload["trajectories"], dtype=np.float64)
    expected = (len(common["conditions"]), common["bin_frame_idx"].size, 2)
    if trajectories.shape != expected:
        _fail(
            f"{path} 的 trajectories 形状是 {trajectories.shape}，应为 (条件, 时间箱, 2) = {expected}：{_HINT_STAGE6B}"
        )
    return trajectories, str(_npz_scalar(payload, "path", path))


def _load_affinity(root: Path, common: dict) -> tuple[np.ndarray, np.ndarray]:
    """读 `affinity.npz` 的视野级亲和矩阵与它的显示序（`order`）。"""
    path = root / _AFFINITY_FILE
    payload = _read_npz(path, _HINT_STAGE6B)
    _require_keys(payload, ("affinity", "order", *_TEMPORAL_COMMON_KEYS), path, _HINT_STAGE6B)
    _temporal_common(payload, path)
    _check_shared_labels(common, payload, path)

    affinity = np.asarray(payload["affinity"], dtype=np.float64)
    n_conditions = len(common["conditions"])
    if affinity.shape != (n_conditions, n_conditions):
        _fail(
            f"{path} 的 affinity 形状是 {affinity.shape}，按 {n_conditions} 个视野应为 "
            f"{(n_conditions, n_conditions)}：{_HINT_STAGE6B}"
        )
    order = np.asarray(payload["order"], dtype=np.int64).ravel()
    if sorted(order.tolist()) != list(range(n_conditions)):
        _fail(f"{path} 的 order {order.tolist()} 不是 0..{n_conditions - 1} 的一个排列：{_HINT_STAGE6B}")
    return affinity, order


def _load_temporal_frame(root: Path, common: dict) -> pd.DataFrame:
    """按 `instance_uid` 对齐 tracks / cluster_labels / umap_2d（F7 的权重、F9 的轨迹、F10b 的质心都要它）。"""
    tracks_path = root / _TRACKS_FILE
    tracks = _read_csv(tracks_path, _HINT_STAGE6)
    _require_columns(tracks, ("instance_uid", "track_id", "fov", "frame_idx"), tracks_path, _HINT_STAGE6)
    labels_path = root / _LABELS_FILE
    labels = _read_csv(labels_path, _HINT_STAGE5)
    _require_columns(labels, ("instance_uid", "cluster"), labels_path, _HINT_STAGE5)
    umap_path = root / _UMAP_FILE
    umap = _read_csv(umap_path, _HINT_STAGE5)
    _require_columns(umap, ("instance_uid", "umap_1", "umap_2"), umap_path, _HINT_STAGE5)

    for frame, name in ((tracks, _TRACKS_FILE), (labels, _LABELS_FILE), (umap, _UMAP_FILE)):
        if not frame["instance_uid"].is_unique:
            _fail(
                f"{name} 里 instance_uid 有重复：按 uid 对齐会把一行复制成多行"
                "（占比的权重与轨迹的长度都会被重复计入），请重跑产生该表的那一阶段"
            )
    merged = tracks.merge(labels[["instance_uid", "cluster"]], on="instance_uid", how="inner")
    if merged.empty:
        _fail(
            f"{_TRACKS_FILE} 与 {_LABELS_FILE} 的 instance_uid 没有任何交集：两者不是同一批实例的产物，"
            "请按顺序重跑 Stage5 → Stage6"
        )
    if len(merged) < len(tracks):
        warnings.warn(
            f"{len(tracks) - len(merged)} 个实例在 {_LABELS_FILE} 里没有簇标签（Stage5 只给有可用轮廓的实例聚类）："
            f"F7 的加权权重与 F9/F10b 的轨迹只用有标签的 {len(merged)} 个实例",
            UserWarning,
            stacklevel=2,
        )
    merged = merged.merge(umap[["instance_uid", "umap_1", "umap_2"]], on="instance_uid", how="left")
    coordinates = merged[["umap_1", "umap_2"]].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(coordinates)):
        _fail(f"有实例在 {_UMAP_FILE} 里没有有限坐标（NaN/Inf）：画不了 UMAP 轨迹，请重跑 Stage5")

    clusters = merged["cluster"].to_numpy(dtype=np.int64)
    unknown = sorted(set(np.unique(clusters).tolist()) - set(common["cluster_labels"].tolist()))
    if unknown:
        _fail(
            f"{_LABELS_FILE} 里的簇号 {unknown} 不在 Stage6b 的 cluster_labels "
            f"{common['cluster_labels'].tolist()} 里：两份产物不是同一次运行，{_HINT_STAGE6B}"
        )
    fovs = sorted({str(name) for name in merged["fov"]})
    if fovs != sorted(common["conditions"]):
        _fail(
            f"{_TRACKS_FILE} 的视野 {fovs} 与 Stage6b 的 conditions {common['conditions']} 不一致："
            "F7 的合并权重会对错视野，请重跑 Stage6b"
        )
    return merged.reset_index(drop=True)


def _bin_instance_counts(frame: pd.DataFrame, common: dict) -> np.ndarray:
    """每 (视野, 采样点) 的实例数 = Stage6b 占比的分母，也就是 F7 合并面板的加权权重。"""
    counts = np.zeros((len(common["conditions"]), common["bin_frame_idx"].size), dtype=np.int64)
    frames = frame["frame_idx"].to_numpy(dtype=np.int64)
    fovs = frame["fov"].astype(str).to_numpy()
    unknown = sorted(set(frames.tolist()) - set(common["bin_frame_idx"].tolist()))
    if unknown:
        _fail(
            f"{_TRACKS_FILE} 里的帧号 {unknown} 不在 Stage6b 的 bin_frame_idx "
            f"{common['bin_frame_idx'].tolist()} 里：占比的分箱与实例的帧号对不上，请重跑 Stage6b"
        )
    for condition_index, condition in enumerate(common["conditions"]):
        for bin_index, bin_frame in enumerate(common["bin_frame_idx"]):
            counts[condition_index, bin_index] = int(np.sum((fovs == condition) & (frames == bin_frame)))
    return counts


@dataclass(frozen=True)
class _TemporalInputs:
    """时间图要的一切（四份 npz 已按同一套标签顺序核对过，frame 已按 uid 对齐）。"""

    proportions: np.ndarray  # (n_conditions, n_bins, n_labels)
    proportions_labels: np.ndarray  # 1..K
    conditions: list[str]  # = fov 名，顺序与落盘产物的轴一致
    bin_frame_idx: np.ndarray
    bin_day: list[str]
    bin_time_hours: np.ndarray
    bin_counts: np.ndarray  # (n_conditions, n_bins) 实例数 = 占比分母
    inter: np.ndarray  # (K, K) 对角 0、行和 1
    probs: np.ndarray  # (K, K) 平滑后行随机
    counts: np.ndarray  # (K, K) 未平滑原始计数
    pseudocounts: float
    trajectories: np.ndarray  # (n_conditions, n_bins, 2)
    trajectory_path: str
    affinity: np.ndarray  # (n_conditions, n_conditions)
    affinity_order: np.ndarray
    frame: pd.DataFrame  # uid / track_id / fov / frame_idx / cluster / umap_1 / umap_2
    qc: dict
    match_qc: dict


def _load_temporal_inputs(directory: Path) -> _TemporalInputs:
    """读齐 F7–F11 的输入（产出目录 → 八份产物），每一步失败都点名该跑哪个 Stage。"""
    root = _locate_analysis_dir(directory)
    proportions, common = _load_proportions(root)
    transitions = _load_transitions(root, common)
    trajectories, trajectory_path = _load_trajectories(root, common)
    affinity, affinity_order = _load_affinity(root, common)
    frame = _load_temporal_frame(root, common)
    qc = _read_json(root / _STAGE6B_QC_FILE, _HINT_STAGE6B)
    match_qc = _read_json(root / _MATCH_QC_FILE, _HINT_STAGE6)
    return _TemporalInputs(
        proportions=proportions,
        proportions_labels=common["cluster_labels"],
        conditions=common["conditions"],
        bin_frame_idx=common["bin_frame_idx"],
        bin_day=common["bin_day"],
        bin_time_hours=common["bin_time_hours"],
        bin_counts=_bin_instance_counts(frame, common),
        inter=transitions["inter"],
        probs=transitions["probs"],
        counts=transitions["counts"],
        pseudocounts=transitions["pseudocounts"],
        trajectories=trajectories,
        trajectory_path=trajectory_path,
        affinity=affinity,
        affinity_order=affinity_order,
        frame=frame,
        qc=qc,
        match_qc=match_qc,
    )


def _pooled_proportions(proportions: np.ndarray, bin_counts: np.ndarray) -> np.ndarray:
    """把逐视野的簇占比合并成一条：**按每视野每采样点的实例数加权平均**，不是等权。

    等权平均在视野实例数不同时会偏（小视野被放大成和大视野一样重）。分母只累加"该行占比有效"的
    视野，所以空箱（整行 NaN）仍然得到 NaN，不会用 0 冒充"占比 0"。
    """
    values = np.asarray(proportions, dtype=np.float64)
    weights = np.asarray(bin_counts, dtype=np.float64)
    if values.ndim != 3 or weights.shape != values.shape[:2]:
        _fail(f"占比形状 {values.shape} 与实例数形状 {weights.shape} 对不上：合并面板无法加权")
    finite_rows = np.isfinite(values).all(axis=2)
    weighted = np.where(finite_rows[:, :, None], values * weights[:, :, None], 0.0).sum(axis=0)
    denominator = (weights * finite_rows).sum(axis=0)
    pooled = np.full(values.shape[1:], np.nan)
    valid = denominator > 0.0
    pooled[valid] = weighted[valid] / denominator[valid, None]
    return pooled


def _plot_f7(path: Path, inputs: _TemporalInputs) -> Path:
    """F7：每视野一个面板 + 合并面板的簇占比堆叠条（时间轴只有 3 个离散采样点）。"""
    labels = [int(label) for label in inputs.proportions_labels]
    colours = _cluster_colour_map(inputs.proportions_labels)
    pooled = _pooled_proportions(inputs.proportions, inputs.bin_counts)
    panels: list[tuple[str, np.ndarray, np.ndarray]] = [
        (f"视野 {condition}", inputs.proportions[index], inputs.bin_counts[index])
        for index, condition in enumerate(inputs.conditions)
    ]
    panels.append(("合并（按实例数加权）", pooled, inputs.bin_counts.sum(axis=0)))

    ncols = min(3, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    with _open_figure(figsize=(4.3 * ncols, 4.0 * nrows + 0.9)) as fig:
        axes = np.atleast_1d(fig.subplots(nrows, ncols, squeeze=False)).ravel()
        fig.subplots_adjust(wspace=0.3, hspace=0.5)
        for ax, (name, values, counts) in zip(axes, panels):
            bottom = np.zeros(values.shape[0])
            for column, label in enumerate(labels):
                heights = np.where(np.isfinite(values[:, column]), values[:, column], 0.0)
                ax.bar(
                    np.arange(values.shape[0]),
                    heights,
                    bottom=bottom,
                    color=[colours[label]],
                    edgecolor="black",
                    linewidth=0.4,
                    width=0.62,
                    label=f"簇 {label}",
                )
                bottom += heights
            # 空箱（整行 NaN）在图上写明"无实例"，不留一根看不见的柱子
            for bin_index in range(values.shape[0]):
                if not np.isfinite(values[bin_index]).all():
                    ax.text(bin_index, 0.5, "无实例\n（空箱）", ha="center", va="center", fontsize=7, color="0.3")
            ax.set_xticks(np.arange(values.shape[0]))
            # 横轴标签：day 名 + 真实采样小时（0/72/120 h，只作为标签，不参与分箱）+ 实例数
            ax.set_xticklabels(
                [
                    f"{day}\n{hours:.0f} h\n(n={int(count)})" if np.isfinite(hours) else f"{day}\n(n={int(count)})"
                    for day, hours, count in zip(
                        inputs.bin_day,
                        inputs.bin_time_hours,
                        np.asarray(counts, dtype=np.int64),
                    )
                ],
                fontsize=7,
            )
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("簇占比（同一采样点内归一）")
            ax.set_title(f"{name}（{int(np.sum(counts))} 个实例）", fontsize=9)
            ax.grid(axis="y", alpha=0.25, linewidth=0.4)
        for ax in axes[len(panels) :]:
            ax.set_visible(False)
        # 图例放在整张图下方：堆叠条总是顶到 y=1，图例塞在轴内会盖住最上面那段
        fig.legend(
            handles=[mpatches.Patch(color=colours[label], label=f"簇 {label}") for label in labels],
            loc="lower center",
            ncol=min(len(labels), 8),
            fontsize=7,
            title="簇",
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )
        fig.suptitle(
            f"F7 · 表型簇占比随采样点的变化（{len(inputs.conditions)} 个视野 + 合并面板；"
            f"{len(inputs.bin_frame_idx)} 个离散采样点 = {', '.join(inputs.bin_day)}）\n"
            "合并面板 = 按每视野每采样点的实例数加权平均（不是等权）；时间轴是离散采样点"
            "（一步跨 3 天或 2 天，且 motion 已置空），不能读成速率",
            fontsize=9,
        )
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    return path


def _observed_flow_edges(inter: np.ndarray, counts: np.ndarray) -> list[tuple[int, int, float, float]]:
    """只保留**观测到**的簇间转移（未平滑计数 ≥ 1），返回 (来源下标, 目标下标, 概率, 计数)。

    `pseudocounts=1` 让零观测的边也拿到非零概率（一整行没观测时就是均匀先验），把它们画成流动会把
    "没观测到"伪装成"观测到了"。对角（自转移）由热图单独承担，这里排除。
    """
    inter = np.asarray(inter, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if inter.shape != counts.shape or inter.ndim != 2 or inter.shape[0] != inter.shape[1]:
        _fail(f"inter 形状 {inter.shape} 与未平滑计数 {counts.shape} 必须同为方阵")
    edges: list[tuple[int, int, float, float]] = []
    for source in range(inter.shape[0]):
        for target in range(inter.shape[1]):
            if source == target or counts[source, target] <= 0.0:
                continue
            probability = float(inter[source, target])
            if not np.isfinite(probability) or probability <= 0.0:
                continue
            edges.append((source, target, probability, float(counts[source, target])))
    return edges


def _plot_f8(path: Path, inputs: _TemporalInputs) -> Path:
    """F8：左 = 观测到的簇间转移有向图，右 = 转移概率热图（对角 = 自转移）。"""
    labels = [int(label) for label in inputs.proportions_labels]
    n_clusters = len(labels)
    colours = _cluster_colour_map(inputs.proportions_labels)
    counts = np.asarray(inputs.counts, dtype=np.float64)
    edges = _observed_flow_edges(inputs.inter, counts)
    off_diagonal = ~np.eye(n_clusters, dtype=bool)
    n_prior_only = int(np.sum(off_diagonal & (counts <= 0.0)))

    angles = np.pi / 2.0 - 2.0 * np.pi * np.arange(n_clusters) / max(n_clusters, 1)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    with _open_figure(figsize=(13.0, 5.9)) as fig:
        axes = np.atleast_1d(fig.subplots(1, 2, squeeze=False)).ravel()
        # 顶部留白：三行 suptitle 与两个面板的两行标题要有各自的位置，否则会叠在一起
        fig.subplots_adjust(wspace=0.35, top=0.78)
        ax_graph, ax_heat = axes

        ax_graph.set_xlim(-1.5, 1.5)
        ax_graph.set_ylim(-1.5, 1.5)
        ax_graph.set_aspect("equal")
        ax_graph.axis("off")
        for index, label in enumerate(labels):
            ax_graph.scatter(
                positions[index, 0],
                positions[index, 1],
                s=300.0,
                c=[colours[label]],
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
            )
            ax_graph.text(
                positions[index, 0],
                positions[index, 1],
                str(label),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="black",
                zorder=4,
            )
        for source, target, probability, count in edges:
            start, end = positions[source], positions[target]
            arrow = mpatches.FancyArrowPatch(
                start,
                end,
                connectionstyle="arc3,rad=0.18",
                arrowstyle="-|>",
                mutation_scale=12.0,
                linewidth=0.8 + 3.0 * probability,
                color=colours[labels[source]],
                alpha=0.85,
                shrinkA=16.0,
                shrinkB=18.0,
                zorder=2,
            )
            # gid 里带上观测计数：测试与人工审计都能一眼看出"这条边是几次观测撑起来的"
            arrow.set_gid(f"edge-{labels[source]}-{labels[target]}-n{round(count)}")
            # 概率另存为 artist 的 label：gid 已被测试当作"来源-目标-计数"解析，不能再塞第四段
            arrow.set_label(f"p={probability:.6f}")
            ax_graph.add_patch(arrow)
            middle = 0.5 * (start + end) + 0.18 * np.array([end[1] - start[1], -(end[0] - start[0])])
            ax_graph.text(
                float(middle[0]),
                float(middle[1]),
                f"p={probability:.2f}\nn={round(count)}",
                ha="center",
                va="center",
                fontsize=6,
                color="0.15",
            )
        if not edges:
            ax_graph.text(
                0.0,
                0.0,
                (
                    "只有 1 个簇：没有簇间转移可画\n（自转移见右侧热图）"
                    if n_clusters < 2
                    else "没有任何观测到的簇间转移：\n所有非对角未平滑计数都是 0（只剩先验）"
                ),
                ha="center",
                va="center",
                fontsize=8,
            )
        ax_graph.set_title(
            f"观测到的簇间转移（只画 count ≥ 1 的 {len(edges)} 条边；已过滤 {n_prior_only} 条 count = 0 的纯先验边）\n"
            "边概率 = 去掉自转移后按行重归一（与右侧热图同行不同分母）",
            fontsize=9,
        )
        # 轴关掉了，轴标签不会渲染：用轴内文字写明左右两栏的归一化口径不同（放在图例空白处，不压节点）
        ax_graph.text(
            0.5,
            0.02,
            "边概率 = 去掉自转移后按行重归一，与右侧热图同行不同分母（同一 (来源, 目标) 两处数字不同）",
            transform=ax_graph.transAxes,
            ha="center",
            va="bottom",
            fontsize=7,
            color="0.25",
        )

        heatmap = np.asarray(inputs.probs, dtype=np.float64)
        image = ax_heat.imshow(
            heatmap,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
            interpolation="nearest",
            aspect="auto",
        )
        for row in range(n_clusters):
            for column in range(n_clusters):
                value = float(heatmap[row, column])
                prior_only = counts[row, column] <= 0.0
                if prior_only:
                    # 斜纹密集程度降到单道斜线：两道以上的线会穿过格内数字导致不可读
                    ax_heat.add_patch(
                        mpatches.Rectangle(
                            (column - 0.5, row - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            hatch="//",
                            edgecolor="0.7",
                            linewidth=0.4,
                        )
                    )
                if row == column:
                    ax_heat.add_patch(
                        mpatches.Rectangle(
                            (column - 0.5, row - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            edgecolor="red",
                            linewidth=1.2,
                        )
                    )
                # 纯先验格额外注明"先验"：只给数字容易被读成观测值；底色块让它盖住斜纹
                ax_heat.text(
                    column,
                    row,
                    f"{value:.2f}\n(先验)" if prior_only else f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6 if prior_only else 7,
                    color="white" if value < 0.55 else "black",
                    bbox=(
                        {"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.45, "linewidth": 0}
                        if prior_only
                        else None
                    ),
                )
        ax_heat.set_xticks(np.arange(n_clusters))
        ax_heat.set_xticklabels([f"簇 {label}" for label in labels], fontsize=8)
        ax_heat.set_yticks(np.arange(n_clusters))
        ax_heat.set_yticklabels([f"簇 {label}" for label in labels], fontsize=8)
        ax_heat.set_xlabel("目标簇")
        ax_heat.set_ylabel("来源簇")
        ax_heat.set_title("转移概率热图（含自转移，对角 = 自转移；斜纹格 = 观测计数 0 的纯先验）", fontsize=9)
        fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.03, label="转移概率（行和 1）")
        fig.suptitle(
            f"F8 · 相邻采样点之间的簇转移（行 = 来源、列 = 目标；pseudocounts={inputs.pseudocounts:g} 平滑）\n"
            "左侧图 = 去掉自转移后按行重归一的边概率，右侧热图 = 含自转移的行随机概率：两者分母不同，"
            "同一 (来源, 目标) 的数字不可互换\n"
            "一步 = day0→day3 或 day3→day5（两步真实宽度是 72 h 与 48 h，且 motion 已置空）：这是频率，不是速率",
            fontsize=9,
        )
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    if not edges and n_clusters >= 2:
        warnings.warn(
            f"F8：{n_clusters} 个簇之间没有任何观测到的转移（所有非对角未平滑计数都是 0）："
            "图上只画了节点与自转移热图，并写明这一退化",
            UserWarning,
            stacklevel=2,
        )
    return path


@dataclass(frozen=True)
class _TrackPath:
    """一条长度 ≥ 2 的 track：`coords` 已按采样点先后排序，`cluster` 取首个采样点的簇（线色）。"""

    fov: str
    track_id: int
    coords: np.ndarray
    cluster: int


def _track_paths(frame: pd.DataFrame, min_length: int = _TRACK_MIN_LENGTH) -> list[_TrackPath]:
    """把 `frame` 按 (fov, track_id) 拆成轨迹，只留长度 ≥ `min_length` 的，按采样点先后排序。

    长度 1 的 track 只有一个采样点，画不出"轨迹"：把它连成一条线等于凭空造出一条位移。
    """
    if int(min_length) < 2:
        _fail(f"轨迹的最短长度至少 2（实际 {min_length}）：长度 1 画不出轨迹")
    paths: list[_TrackPath] = []
    for (fov, track_id), group in frame.groupby(["fov", "track_id"], sort=True):
        if len(group) < int(min_length):
            continue
        ordered = group.sort_values("frame_idx", kind="mergesort")
        paths.append(
            _TrackPath(
                fov=str(fov),
                track_id=int(track_id),
                coords=ordered[["umap_1", "umap_2"]].to_numpy(dtype=np.float64),
                cluster=int(ordered["cluster"].iloc[0]),
            )
        )
    paths.sort(key=lambda track: (track.fov, track.track_id))
    return paths


def _track_summary(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    """每视野的 track 总数 / 长度 ≥ 2 的 track 数 / 长度 1 的 track 数（F9 的图注要如实报数）。"""
    sizes = frame.groupby(["fov", "track_id"], sort=True).size()
    summary: dict[str, dict[str, int]] = {}
    for fov, group in sizes.groupby(level=0, sort=True):
        values = group.to_numpy()
        summary[str(fov)] = {
            "n_tracks": int(values.size),
            "n_long": int((values >= _TRACK_MIN_LENGTH).sum()),
            "n_single": int((values < _TRACK_MIN_LENGTH).sum()),
        }
    return summary


def _plot_f9(path: Path, inputs: _TemporalInputs) -> Path:
    """F9：UMAP 背景 + 长度 ≥ 2 的 track 折线 + Stage6b 的表型轨迹（每视野一个面板）。"""
    frame = inputs.frame
    colours = _cluster_colour_map(frame["cluster"].to_numpy(dtype=np.int64))
    paths = _track_paths(frame)
    summary = _track_summary(frame)
    total_tracks = int(sum(item["n_tracks"] for item in summary.values()))
    total_long = int(sum(item["n_long"] for item in summary.values()))
    by_fov: dict[str, list[_TrackPath]] = {}
    for track in paths:
        by_fov.setdefault(track.fov, []).append(track)

    conditions = list(inputs.conditions)
    ncols = min(3, len(conditions))
    nrows = int(np.ceil(len(conditions) / ncols))
    empty_trajectories: list[str] = []
    sparse_trajectories: list[str] = []
    with _open_figure(figsize=(4.8 * ncols, 4.4 * nrows + 1.0)) as fig:
        axes = np.atleast_1d(fig.subplots(nrows, ncols, squeeze=False)).ravel()
        # 底部留出图例的位置，否则图例会压到面板的 UMAP 1 轴标签上
        fig.subplots_adjust(wspace=0.28, hspace=0.42, bottom=0.22)
        for ax, (condition_index, condition) in zip(axes, enumerate(conditions)):
            information = summary.get(condition, {"n_tracks": 0, "n_long": 0, "n_single": 0})
            # 背景：**全部视野**的实例（淡灰），给轨迹一个共用参照系：面板标题写的是单个视野，
            # 但淡灰点的数量远大于该视野的实例数，所以图例与图注都点明这一点
            ax.scatter(frame["umap_1"], frame["umap_2"], s=5.0, c="0.65", alpha=0.35, linewidths=0, zorder=0)
            for track in by_fov.get(condition, []):
                colour = colours.get(track.cluster, (0.35, 0.35, 0.35))
                line = ax.plot(
                    track.coords[:, 0],
                    track.coords[:, 1],
                    color=colour,
                    linewidth=1.1,
                    alpha=0.85,
                    marker="o",
                    markersize=2.2,
                    zorder=2,
                )[0]
                line.set_gid(f"track-{condition}-{track.track_id}")
                # 末端画方块：一眼看出这条轨迹往哪个方向走（时间轴只有 3 个点，不画箭头）
                ax.scatter(
                    track.coords[-1, 0],
                    track.coords[-1, 1],
                    marker="s",
                    s=12.0,
                    color=colour,
                    edgecolors="black",
                    linewidths=0.3,
                    zorder=3,
                )
            if information["n_long"] == 0:
                ax.text(
                    0.5,
                    0.93,
                    f"本视野没有长度 ≥ 2 的 track（{information['n_tracks']} 条 track 全是单点）",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=7,
                    color="0.25",
                )

            trajectory = np.asarray(inputs.trajectories[condition_index], dtype=np.float64)
            finite = np.isfinite(trajectory).all(axis=1)
            if finite.any():
                line = ax.plot(
                    trajectory[finite, 0],
                    trajectory[finite, 1],
                    color="black",
                    linewidth=2.0,
                    marker="D",
                    markersize=4.0,
                    zorder=4,
                )[0]
                line.set_gid(f"trajectory-{condition}")
                for bin_index in np.flatnonzero(finite):
                    # 标签写成 day 名（真实小时数 0/72/120 h 只作为标签，不参与分箱）
                    ax.annotate(
                        inputs.bin_day[bin_index],
                        (float(trajectory[bin_index, 0]), float(trajectory[bin_index, 1])),
                        textcoords="offset points",
                        xytext=(4.0, 4.0),
                        fontsize=6,
                        color="black",
                    )
                if len(finite) < inputs.bin_frame_idx.size:
                    sparse_trajectories.append(condition)
                    ax.text(
                        0.5,
                        0.06,
                        f"表型轨迹缺 {inputs.bin_frame_idx.size - int(finite.sum())} 个采样点（NaN）",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=7,
                        color="0.25",
                    )
            else:
                empty_trajectories.append(condition)
                ax.text(
                    0.5,
                    0.06,
                    "该视野的表型轨迹全为 NaN（Stage6b 退化）",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=7,
                    color="0.25",
                )
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title(
                f"视野 {condition}：长度 ≥ 2 的 track {information['n_long']} / {information['n_tracks']} 条",
                fontsize=9,
            )
        for ax in axes[len(conditions) :]:
            ax.set_visible(False)
        fig.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    markersize=4.0,
                    color="0.65",
                    label="全部视野的实例（淡灰点，共用参照系）",
                ),
                *[mpatches.Patch(color=colours[label], label=f"簇 {label}") for label in sorted(colours)],
            ],
            loc="lower center",
            ncol=min(len(colours) + 1, 8),
            fontsize=7,
            title="track 首个采样点的簇（线色）",
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
        subset_note = str(
            inputs.qc.get("subsets", {}).get("trajectories", {}).get("track_length_filter", "QC 未记录子集")
        )
        fig.suptitle(
            f"F9 · UMAP 表型轨迹（彩色折线只画长度 ≥ 2 的 track：{total_long} / {total_tracks} 条；"
            "长度 1 的 track 只有单个采样点，画不出轨迹）\n"
            "淡灰点 = 全部视野的实例（共用参照系，不是该面板那个视野的实例数）；"
            f"粗黑线 = Stage6b 的表型轨迹（path={inputs.trajectory_path}；用全部有簇标签的实例，{subset_note}）；"
            f"时间轴只有 {len(inputs.bin_frame_idx)} 个离散采样点，不是连续视频，不能读成速度",
            fontsize=9,
        )
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")

    if total_long == 0:
        warnings.warn(
            f"F9：{total_tracks} 条 track 全部只有 1 个采样点（没有长度 ≥ 2 的 track）："
            "图上只画了 UMAP 背景与表型轨迹，并在每个面板写明这一退化",
            UserWarning,
            stacklevel=2,
        )
    if empty_trajectories:
        warnings.warn(
            f"Stage6b 的表型轨迹在视野 {empty_trajectories} 上全为 NaN（退化）："
            "F9 只画出这些视野的 UMAP 背景与 track 折线，面板上已标注退化",
            UserWarning,
            stacklevel=2,
        )
    if sparse_trajectories:
        warnings.warn(
            f"Stage6b 的表型轨迹在视野 {sparse_trajectories} 上缺采样点（NaN）：折线只连有效采样点，面板上已标注",
            UserWarning,
            stacklevel=2,
        )
    if bool(inputs.qc.get("trajectories", {}).get("degenerate", False)):
        warnings.warn(
            f"Stage6b 的 QC 标注表型轨迹退化（path={inputs.trajectory_path}）：F9 的粗黑线是兜底路径的产物，"
            "不能当成 SPOT 密度轨迹",
            UserWarning,
            stacklevel=2,
        )
    return path


@dataclass(frozen=True)
class _AffinityDisplay:
    """亲和矩阵的显示状态：矩阵与标签**已按 dendrogram 叶序排好**，`linkage` 给树（单对象时 None）。"""

    matrix: np.ndarray
    order: np.ndarray
    linkage: np.ndarray | None
    note: str | None


def _affinity_display(
    affinity: np.ndarray, stored_order: np.ndarray | None = None, method: str = "average"
) -> _AffinityDisplay:
    """把亲和矩阵摆成 dendrogram 叶序，并给出一棵可画的树（F10a/F10b 共用）。

    距离用 `1 - 亲和`：亲和 = `exp(-DTW / std)`，所以 1 减它就是归一化后的 DTW 距离，满足"越像越近"。
    只有 1 个对象时没有树可画（返回 `linkage=None`），调用方要在图上写明退化，而不是画一棵空树。
    `stored_order` 给了就与本地叶序对一遍：不一致说明落盘的 order 不是"1 - 亲和"的叶序
    （SPOT 的 hcluster 是对亲和矩阵的行做距离），此时按本地叶序显示并把差异写进 `note`。
    """
    matrix = np.asarray(affinity, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        _fail(f"亲和矩阵必须是方阵，实际形状 {matrix.shape}")
    n_objects = matrix.shape[0]
    if n_objects == 0:
        _fail("亲和矩阵是空的：没有任何可聚类的对象")
    if not np.all(np.isfinite(matrix)):
        _fail("亲和矩阵含非有限值（NaN/Inf）：画不了亲和热图，请检查 Stage6b 的 affinity.npz")
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        _fail("亲和矩阵不对称：按序号画出来的图会误导，请检查 Stage6b 的 affinity.npz")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-8):
        _fail(f"亲和矩阵的对角不是 1（{np.round(np.diag(matrix), 6).tolist()}）：请检查 Stage6b 的 affinity.npz")
    if n_objects < 2:
        return _AffinityDisplay(
            matrix=matrix.copy(),
            order=np.zeros(n_objects, dtype=np.int64),
            linkage=None,
            note="只有 1 个对象：退化成单个格子，无法做层次聚类",
        )

    distance = np.clip(1.0 - matrix, 0.0, None)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    tree = linkage(squareform(distance, checks=False), method=str(method))
    order = np.asarray(dendrogram(tree, no_plot=True)["leaves"], dtype=np.int64)
    note = None
    if stored_order is not None:
        stored = np.asarray(stored_order, dtype=np.int64).ravel()
        if sorted(stored.tolist()) != list(range(n_objects)):
            _fail(f"落盘的 order {stored.tolist()} 不是 0..{n_objects - 1} 的一个排列：无法重排亲和矩阵")
        if not np.array_equal(stored, order):
            note = (
                f"affinity.npz 的 order {stored.tolist()} 与本地 1-亲和 距离的叶序 {order.tolist()} 不一致，"
                "已按本地叶序显示（SPOT 的 hcluster 对亲和矩阵的行做距离，口径不同）"
            )
    return _AffinityDisplay(matrix=matrix[np.ix_(order, order)], order=order, linkage=tree, note=note)


def _wrap_caption(text: str, width: int = _CAPTION_WRAP_WIDTH) -> list[str]:
    """按**显示宽度**折行（中文/全角算 2、ASCII 算 1）。

    `textwrap` 只在空格处折行，中文长句里几乎没有空格，所以一段中文要么被当成一个超长单词直接
    撑成一行，要么在仅有的几个空格（如 "DTW 对"）处断得莫明其妙。这里逐字累加宽度，够了就换行。
    """
    lines: list[str] = []
    current: list[str] = []
    used = 0
    for char in str(text):
        char_width = 2 if unicodedata.east_asian_width(char) in "WF" else 1
        if used + char_width > width and current:
            lines.append("".join(current))
            current, used = [], 0
        current.append(char)
        used += char_width
    if current:
        lines.append("".join(current))
    return lines


def _plot_affinity_figure(
    path: Path,
    display: _AffinityDisplay,
    names: list[str],
    suptitle: str,
    notes: Sequence[str],
) -> Path:
    """F10a/F10b 共用：上方 dendrogram、下方亲和热图（矩阵已按叶序排好，格子标数值）。

    `notes` 每项一句图注，这里按显示宽度逐句折行后一行一行画在坐标轴下方：直接把一整段交给
    `wrap=True` 时，matplotlib 会按整图宽折行但每行都从同一个 x 起算，中文长句会冲出画布两侧。
    """
    n_objects = display.matrix.shape[0]
    with _open_figure(figsize=(6.8, 6.6)) as fig:
        if display.linkage is not None:
            grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.2], hspace=0.06)
            ax_tree = fig.add_subplot(grid[0])
            ax_map = fig.add_subplot(grid[1])
            tree = dendrogram(
                display.linkage,
                ax=ax_tree,
                no_labels=True,
                color_threshold=0.0,
                above_threshold_color="black",
            )
            if not np.array_equal(np.asarray(tree["leaves"], dtype=np.int64), display.order):
                _fail("dendrogram 画出来的叶序与显示序不一致：树与热图会错位，请检查 `_affinity_display`")
            ax_tree.set_xticks([])
            ax_tree.set_yticks([])
            for spine in ax_tree.spines.values():
                spine.set_visible(False)
            ax_tree.set_title("层次聚类 dendrogram（average linkage，距离 = 1 - 亲和）", fontsize=9)
        else:
            ax_map = fig.subplots(1, 1)
        image = ax_map.imshow(
            display.matrix,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
            interpolation="nearest",
        )
        ticks = np.arange(n_objects)
        ordered_names = [names[index] for index in display.order]
        ax_map.set_xticks(ticks)
        ax_map.set_xticklabels(ordered_names, fontsize=8, rotation=45, ha="right")
        ax_map.set_yticks(ticks)
        ax_map.set_yticklabels(ordered_names, fontsize=8)
        for row in range(n_objects):
            for column in range(n_objects):
                value = float(display.matrix[row, column])
                ax_map.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value < 0.55 else "black",
                )
        fig.colorbar(image, ax=ax_map, fraction=0.046, pad=0.03, label="轨迹亲和（exp(-DTW 距离 / std)）")
        # 图注逐句折行后画在坐标轴下方（负 y 也会被 bbox_inches="tight" 收进画布）：放在热图标题位置会
        # 顶到 dendrogram，放在图内底部又会压到 x 轴刻度标签。左对齐才能让折行读起来是同一句。
        fig.subplots_adjust(bottom=0.10)
        lines = [line for note in notes for line in _wrap_caption(note)]
        fig.text(0.02, -0.06, "\n".join(lines), ha="left", va="top", fontsize=7)
        fig.suptitle(suptitle, fontsize=9)
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    return path


def _plot_f10a(path: Path, inputs: _TemporalInputs) -> Path:
    """F10a：视野级的表型轨迹亲和（读 `affinity.npz` 的 DTW 指数核）+ 层次聚类。"""
    display = _affinity_display(inputs.affinity, inputs.affinity_order)
    if display.note is not None:
        warnings.warn(f"F10a：{display.note}", UserWarning, stacklevel=2)
    names = list(inputs.conditions)
    # 轨迹来源与退化状态必须与 F9 一样明写：退化兜底路径算出来的亲和看起来和真轨迹一模一样
    degenerate = bool(inputs.qc.get("trajectories", {}).get("degenerate", False))
    subset_note = str(inputs.qc.get("subsets", {}).get("trajectories", {}).get("track_length_filter", "QC 未记录子集"))
    if degenerate:
        warnings.warn(
            f"Stage6b 的 QC 标注表型轨迹退化（path={inputs.trajectory_path}）：F10a 的亲和是兜底路径的产物，"
            "不能当成 SPOT 密度轨迹的亲和",
            UserWarning,
            stacklevel=2,
        )
    notes = [
        "行/列 = 视野（按 dendrogram 叶序）；对角 = 1（自身）",
        (
            "亲和 = exp(-DTW / std)：DTW 对时间伸缩不敏感，但包含位置/偏移差异（没有先做平移归一），"
            "所以同形但整体位移的两条轨迹也拿不到高亲和"
        ),
        f"轨迹来源 path={inputs.trajectory_path}；Stage6b QC degenerate={degenerate}；子集口径：{subset_note}",
        "F10a 与 F10b 各自用自己那批轨迹的距离 std 归一（本图 = 视野级），两者的亲和数值不可直接比较",
    ]
    return _plot_affinity_figure(
        path,
        display,
        names,
        suptitle=f"F10a · 视野级表型轨迹亲和（{len(names)} 个视野；DTW 指数核，读盘自 affinity.npz）",
        notes=notes,
    )


def _cluster_trajectories(frame: pd.DataFrame, labels: np.ndarray, bin_frame_idx: np.ndarray) -> np.ndarray:
    """每簇在每个采样点的 UMAP 质心构成一条"簇轨迹"（K, n_bins, 2），给 F10b 的簇级亲和用。

    坐标来自读盘的 `umap_2d.csv` + `cluster_labels.csv`（不重跑 UMAP），只按 (簇, 采样点) 求均值。
    某簇在某采样点没有实例时该行留 NaN（不拿别的采样点冒充），F10b 的图注会写明缺覆盖。
    """
    labels = np.asarray(labels, dtype=np.int64).ravel()
    bins = np.asarray(bin_frame_idx, dtype=np.int64).ravel()
    clusters = frame["cluster"].to_numpy(dtype=np.int64)
    frame_values = frame["frame_idx"].to_numpy(dtype=np.int64)
    coords = frame[["umap_1", "umap_2"]].to_numpy(dtype=np.float64)
    out = np.full((labels.size, bins.size, 2), np.nan)
    for label_index, label in enumerate(labels):
        for bin_index, bin_frame in enumerate(bins):
            members = (clusters == label) & (frame_values == bin_frame)
            if members.any():
                out[label_index, bin_index] = coords[members].mean(axis=0)
    return out


def _plot_f10b(path: Path, inputs: _TemporalInputs) -> Path:
    """F10b：簇级的轨迹亲和（簇轨迹 = 每簇每采样点的 UMAP 质心）+ 层次聚类。"""
    labels = [int(label) for label in inputs.proportions_labels]
    names = [f"簇 {label}" for label in labels]
    trajectories = _cluster_trajectories(inputs.frame, inputs.proportions_labels, inputs.bin_frame_idx)
    coverage = np.isfinite(trajectories).all(axis=2)
    missing = [names[index] for index in range(len(names)) if not coverage[index].any()]
    sparse = [names[index] for index in range(len(names)) if 0 < int(coverage[index].sum()) < inputs.bin_frame_idx.size]
    affinity, _labels, info = tstats.trajectory_affinity(
        {"trajectories": trajectories, "conditions": names}, return_meta=True
    )
    if info["fallback"]:
        warnings.warn(f"F10b 的簇级轨迹亲和退化：{info['fallback']}", UserWarning, stacklevel=2)
    if missing:
        warnings.warn(
            f"{missing} 在所有采样点都没有实例（UMAP 质心全 NaN）：簇级亲和会整体退化，图上已标注",
            UserWarning,
            stacklevel=2,
        )

    display = _affinity_display(affinity)
    if display.note is not None:
        warnings.warn(f"F10b：{display.note}", UserWarning, stacklevel=2)
    notes = [
        (
            f"簇轨迹 = 每簇在每个采样点的 UMAP 质心（读盘 umap_2d.csv + cluster_labels.csv），共 {len(names)} 个簇；"
            "质心轨迹的 3 点主要差在位置上，所以这里的亲和很大程度上反映质心间距，而不是轨迹形状"
        ),
        "亲和 = 与 F10a 相同的 DTW 指数核（Stage6b 的实现）：DTW 对时间伸缩不敏感，但包含位置/偏移差异（未做平移归一）",
        "F10a 与 F10b 各自用自己那批轨迹的距离 std 归一（本图 = 簇质心），两者的亲和数值不可直接比较",
    ]
    if display.note is not None:
        notes.append(display.note)
    if info["fallback"]:
        notes.append(f"亲和退化：{info['fallback']}")
    if missing:
        notes.append(f"质心全 NaN 的簇：{'、'.join(missing)}")
    if sparse:
        notes.append(f"部分采样点缺实例的簇：{'、'.join(sparse)}")
    return _plot_affinity_figure(
        path,
        display,
        names,
        suptitle=f"F10b · 簇级表型轨迹亲和（{len(names)} 个簇；质心轨迹 + DTW 指数核）",
        notes=notes,
    )


def _pair_sort_key(pair: str) -> tuple:
    """帧对名（`"0->1"`）的排序键：能解析成数字就按数字排，否则退到字符串（确定性、不抛错）。"""
    left, separator, right = str(pair).partition("->")
    if separator:
        try:
            return (0, int(left), int(right), str(pair))
        except ValueError:
            pass
    return (1, 0, 0, str(pair))


def _match_event_table(match_qc: dict, conditions: Sequence[str]) -> pd.DataFrame:
    """把 `match_qc.json` 的逐帧对统计摊成一张表（一行 = 一个 (视野, 帧对)）。

    只读 QC，不重读 `matches.csv`（Spec §6-Stage7 的 F11 口径）：事件数、未匹配数与配准告警都在这里。
    `fovs` 必须与 Stage6b 的 `conditions` 是同一批视野：两份产物不是同一次运行时，F11 会把帧对画到
    错的视野上（或者静默少画一个视野），所以直接报错而不是靠"帧对数 ÷ 视野数"去猜。
    """
    fovs = match_qc.get("fovs")
    if not isinstance(fovs, dict) or not fovs:
        _fail(f"match_qc.json 里没有 fovs 段（现有键 {sorted(match_qc)}）：F11 要按帧对画匹配事件，{_HINT_STAGE6}")
    recorded = sorted(str(name) for name in fovs)
    expected = sorted(str(name) for name in conditions)
    if recorded != expected:
        _fail(
            f"match_qc.json 的 fovs {recorded} 与 Stage6b 的 conditions {expected} 不一致："
            f"两份产物不是同一次运行的，F11 的帧对会画到错的视野上，{_HINT_STAGE6}"
        )
    rows: list[dict] = []
    for fov in sorted(fovs):
        pairs = fovs[fov].get("pairs") if isinstance(fovs[fov], dict) else None
        if not isinstance(pairs, dict) or not pairs:
            _fail(
                f"match_qc.json 的 fovs[{fov!r}] 没有 pairs 段：该视野可能只有 1 个采样点，"
                f"F11 画不出帧对匹配（不默默省略这个视野），{_HINT_STAGE6}"
            )
        for pair in sorted(pairs, key=_pair_sort_key):
            stats = pairs[pair]
            missing = [key for key in _MATCH_STAT_KEYS if key not in stats]
            if missing:
                _fail(f"match_qc.json 的 fovs[{fov!r}].pairs[{pair!r}] 缺少 {missing}：{_HINT_STAGE6}")
            rows.append(
                {
                    "fov": fov,
                    "pair": pair,
                    **{key: int(stats[key]) for key in _MATCH_STAT_KEYS},
                    "offset_exceeds_limit": bool(stats.get("offset_exceeds_limit", False)),
                }
            )
    return pd.DataFrame(rows)


def _plot_f11(path: Path, inputs: _TemporalInputs) -> Path:
    """F11：每个 (视野, 帧对) 的匹配事件（one_to_one / merge / split）与两侧未匹配数。"""
    table = _match_event_table(inputs.match_qc, inputs.conditions)
    # 每视野的帧对数直接按 QC 的 fovs 数，不用"总帧对数 ÷ 视野数"去猜：各视野帧对数可以不同
    pairs_per_fov = table.groupby("fov", sort=True).size()
    pairs_note = (
        f"每视野 {int(pairs_per_fov.iloc[0])} 个帧对"
        if pairs_per_fov.nunique() == 1
        else f"每视野 {int(pairs_per_fov.min())}–{int(pairs_per_fov.max())} 个帧对（各视野不同）"
    )
    positions = np.arange(len(table), dtype=np.float64)
    tick_labels = [f"{row.fov}\n{row.pair}{' *' if row.offset_exceeds_limit else ''}" for row in table.itertuples()]
    events = tuple(_EVENT_COLOURS)
    total_events = int(table[[f"n_{event}" for event in events]].to_numpy().sum())
    with _open_figure(figsize=(max(7.5, 1.6 * len(table) + 3.5), 5.6)) as fig:
        axes = np.atleast_1d(fig.subplots(1, 2, squeeze=False)).ravel()
        # 三行 suptitle 需要独立的顶部留白，否则会压到两个面板的标题上
        fig.subplots_adjust(wspace=0.28, top=0.76, bottom=0.16)
        ax_events, ax_unmatched = axes

        bottom = np.zeros(len(table))
        for event in events:
            heights = table[f"n_{event}"].to_numpy(dtype=np.float64)
            ax_events.bar(
                positions,
                heights,
                bottom=bottom,
                color=[_EVENT_COLOURS[event]],
                edgecolor="black",
                linewidth=0.4,
                width=0.62,
                label=event,
            )
            bottom += heights
        if total_events == 0:
            ax_events.text(
                0.5,
                0.9,
                "没有任何 one_to_one / merge / split 事件\n（全部实例都进了 unmatched）",
                transform=ax_events.transAxes,
                ha="center",
                va="top",
                fontsize=8,
            )
        for index, row in enumerate(table.itertuples()):
            ax_events.text(
                float(positions[index]),
                float(bottom[index]) + 0.15,
                f"{int(row.n_prev)}→{int(row.n_cur)}",
                ha="center",
                va="bottom",
                fontsize=6,
                color="0.3",
            )
        ax_events.set_xticks(positions)
        ax_events.set_xticklabels(tick_labels, fontsize=7)
        ax_events.set_ylabel("匹配表行数")
        # 顶部留白：帧对标注（n_prev→n_cur）与图例都要有地方放，否则会压到柱子上
        ax_events.set_ylim(0.0, max(float(bottom.max()) * 1.45, 1.0))
        ax_events.set_title("每个帧对的匹配事件（读盘自 match_qc.json）", fontsize=9)
        ax_events.legend(fontsize=7, title="event", loc="upper right", framealpha=0.85)
        ax_events.grid(axis="y", alpha=0.25, linewidth=0.4)

        width = 0.36
        ax_unmatched.bar(
            positions - width / 2.0,
            table["n_prev_unmatched"].to_numpy(dtype=np.float64),
            width=width,
            color=[_UNMATCHED_COLOURS[0]],
            edgecolor="black",
            linewidth=0.4,
            label="上一帧未匹配",
        )
        ax_unmatched.bar(
            positions + width / 2.0,
            table["n_cur_unmatched"].to_numpy(dtype=np.float64),
            width=width,
            color=[_UNMATCHED_COLOURS[1]],
            edgecolor="black",
            linewidth=0.4,
            label="下一帧未匹配",
        )
        ax_unmatched.set_xticks(positions)
        ax_unmatched.set_xticklabels(tick_labels, fontsize=7)
        tallest = float(max(table["n_prev_unmatched"].max(), table["n_cur_unmatched"].max(), 1.0))
        ax_unmatched.set_ylim(0.0, tallest * 1.3)
        ax_unmatched.set_ylabel("没有进匹配表的实例数")
        ax_unmatched.set_title("两侧未匹配实例数（分母不同，不能相加当事件数）", fontsize=9)
        ax_unmatched.legend(fontsize=7, loc="upper right", framealpha=0.85)
        ax_unmatched.grid(axis="y", alpha=0.25, linewidth=0.4)

        fig.suptitle(
            f"F11 · 帧对匹配事件（{len(inputs.bin_frame_idx)} 个离散采样点 → {pairs_note}；"
            f"{len(inputs.conditions)} 个视野共 {len(table)} 个 (视野, 帧对)；数据 = Stage6 的 match_qc.json）\n"
            "one_to_one 才延续轨迹；merge = 被吸收进面积最大的父实例（另记一行，不延续轨迹）；"
            "split = 父实例已有继承人后新出现的子实例（另起新 track）\n"
            "* = 复合配准偏移超过上限（QC 告警，匹配仍按该偏移进行）",
            fontsize=8,
        )
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    if total_events == 0:
        warnings.warn(
            f"F11：{len(table)} 个 (视野, 帧对) 里没有任何 one_to_one / merge / split 事件"
            "（跨天匹配整体失效）：图上的事件柱全为 0，右面板是两侧未匹配数",
            UserWarning,
            stacklevel=2,
        )
    if bool(table["offset_exceeds_limit"].any()):
        flagged = [f"{row.fov} {row.pair}" for row in table.itertuples() if row.offset_exceeds_limit]
        warnings.warn(
            f"F11：{flagged} 的复合配准偏移超过上限（match_qc.json 的告警）：匹配仍按该偏移进行，图上用 * 标出",
            UserWarning,
            stacklevel=2,
        )
    return path


def render_temporal(analysis_dir: Path, outdir: Path | None = None) -> list[Path]:
    """画 Spec §6-Stage7 的时间图 F7–F11（含 F10a/F10b），返回写出的 PNG 路径（顺序 = 文件名清单）。

    Parameters
    ----------
    analysis_dir : Path
        Stage6b 的产物目录：既可以指向 `output/analysis/`，也可以指向 `output/` 根（其下有 `analysis/`，
        与 `render_static` 的口径一致）。本函数**只读盘**：占比/转移/轨迹/亲和全部来自 Stage6b 的 npz，
        匹配事件全部来自 Stage6 的 `match_qc.json`；F10b 的"簇轨迹"只是把 `umap_2d.csv` 的坐标按
        (簇, 采样点) 求均值，不重跑任何上游统计。
    outdir : Path | None
        图的落盘目录（缺省 `output/figures/`，即 Spec §9 的产物位置）。六张图先渲染到
        `<outdir>/.tmp-<文件名>`，**全部成功**后才逐个 `replace` 成正式文件名：渲染失败时上一次的好图
        原样保留，也不会留下半成品。

    退化一律写在图上并配中文 `warnings.warn`：没有长度 ≥ 2 的 track、表型轨迹全为 NaN、没有观测到的
    转移、亲和矩阵只有一个对象、一条匹配都没发生。缺产物抛带阶段名的中文 ValueError（Stage6 / Stage6b）。
    """
    root = _locate_analysis_dir(analysis_dir)
    out = Path(outdir) if outdir is not None else Path(cfg.OUTPUT_ROOT) / _FIGURES_SUBDIR
    inputs = _load_temporal_inputs(root)

    renderers = (
        lambda path: _plot_f7(path, inputs),
        lambda path: _plot_f8(path, inputs),
        lambda path: _plot_f9(path, inputs),
        lambda path: _plot_f10a(path, inputs),
        lambda path: _plot_f10b(path, inputs),
        lambda path: _plot_f11(path, inputs),
    )
    # 先写临时文件，全部成功后才换成正式文件名（不在渲染前删旧图）；与 render_static 共用 _stage_and_replace
    paths = _stage_and_replace(out, _TEMPORAL_FIGURE_NAMES, renderers)
    # 图说明与图同源：这次跑完，静态图与时间图都在的目录会得到一份“全部已生成”的 README
    readme = write_figure_readme(out)

    n_long = len(_track_paths(inputs.frame))
    print(
        f"Stage7 时间图完成：{len(paths)} 张（{len(inputs.conditions)} 个视野 × "
        f"{len(inputs.bin_frame_idx)} 个离散采样点、{inputs.proportions_labels.size} 个簇、"
        f"长度 ≥ 2 的 track {n_long} 条；时间轴是离散采样点，不是连续视频）→ {out}；图说明 → {readme}"
    )
    return paths
