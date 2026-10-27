"""复现用的全部固定参数。

这里只放"论文写死"或"数据事实"决定的常量，不放运行时状态；每个常量上方注释写明它是什么、
以及出自论文/参考脚本的哪一处。所有随机性统一用 RANDOM_STATE。
"""

from pathlib import Path

# 已分割类器官数据的根目录（不是本仓库的 data/）：其下 day0/、day3/、day5/ 各有 A1/A2/A3 三个视野，
# 原图为 <day>/<FOV>_多聚焦融合.png，标签图为 <day>/Output/<FOV>_多聚焦融合.tif（Spec §2）。
# 这是**旧约定**：只在没有从命令行给 --images/--masks（见 inputs.py）时使用。
DATA_ROOT: Path = Path("D:/Datasets/test")

# 命令行显式给的输入目录（inputs.configure() 写入；None 表示走上面的旧约定）。
# 一旦设置，原图/掩码一律按 INPUT_FILES 里的配对路径读，不再拼 DATA_ROOT 的模板。
IMAGE_DIR: Path | None = None
MASK_DIR: Path | None = None

# 输入清单：(day, fov) → (原图路径, 掩码路径)。空 dict = 没有命令行输入，走旧约定。
# 用 dict 而不是列表：下游只需要按 (day, fov) 取路径，重复键本身就该在发现阶段报错。
INPUT_FILES: dict[tuple[str, str], tuple[Path, Path]] = {}

# 一天算多少小时：从文件名里的天数推出 TIME_HOURS（day3 → 72h）。真实时间间隔不均也照此换算。
HOURS_PER_DAY: float = 24.0

# 所有产物（实例目录、特征表、嵌入坐标、图表、报告）都落在这里；.gitignore 已忽略 output/。
OUTPUT_ROOT: Path = Path(__file__).resolve().parents[1] / "output"

# 真实天数 → 连续帧号。SPOT 的 construct_obj_traj_from_uniq_obj_ids 用标量 d_Frame 判连续性，
# 真实天数 0/3/5 的差是 3 和 2，无法用单一 d_Frame 表达，必须换成 0/1/2（Spec §6-Stage1-6）。
# 这是旧约定的默认值；用 --images/--masks 时由 inputs.configure() 按文件名里的天数重写。
DAYS: dict[str, int] = {"day0": 0, "day3": 1, "day5": 2}

# 三个视野名，与文件名前缀 A1/A2/A3 一致（同样会被 inputs.configure() 重写）。
FOVS: tuple[str, ...] = ("A1", "A2", "A3")

# 时间轴（小时）：0 天 = 0h、3 天 = 72h、5 天 = 120h。出时间图时用它而不是帧号。
TIME_HOURS: dict[str, float] = {"day0": 0.0, "day3": 72.0, "day5": 120.0}

# 每个像素的物理边长（µm/px）。这份数据没有标尺信息，取 1.0 表示所有面积/长度按像素计。
PIXEL_RES: float = 1.0

# 轮廓等弧长重采样后的点数，对应论文 Stage1 的边界采样口径。
N_CONTOUR_POINTS: int = 100

# 裁剪代表图时 bbox 向外扩的像素数，保证实例周围留一点背景。
PATCH_PAD_PX: int = 5

# SIFT 输入小图尺寸 (h, w)：与论文 Stage3 代表图/描述子所用 64×64 一致。
PATCH_SIZE: tuple[int, int] = (64, 64)

# 面积小于该值的实例只打 is_small 标记、不删除（day0 有 39–42% 实例 < 500 px²，删了会改写
# "实例数随时间增长"的叙述，Spec §6-Stage1-4）。
SMALL_AREA_FLAG_PX: int = 200

# 掩码与原图尺寸不一致时，±该像素范围内搜索平移做兜底对齐（Spec §6-Stage1-1）。
ALIGN_MAX_SHIFT_PX: int = 8

# 全局随机种子：UMAP、k-means、模块 bootstrap 全部用它，保证图可离线复现。
RANDOM_STATE: int = 0

# 联合 UMAP 参数，出自论文默认设置：n_neighbors=100, min_dist=0.5, spread=0.5, metric='euclidean'。
UMAP_PARAMS: dict = {
    "n_neighbors": 100,
    "min_dist": 0.5,
    "spread": 0.5,
    "metric": "euclidean",
    "random_state": RANDOM_STATE,
}

# k-means 的 K 搜索范围（肘部法），论文 Stage3 在 2–12 之间扫描。注意 SPOT 内部是
# `range(k_range[0], k_range[1])`（**不含上界**），所以 (2, 12) 实际只扫 2–11。
KMEANS_K_RANGE: tuple[int, int] = (2, 12)

# remove_high_variance_features 的阈值（以 sigma 为单位），论文用 2 而不是 3。
VARIANCE_THRESHOLD_SIGMA: float = 2.0

# ECT/shape-context 家族在 UMAP 前先降到 100 维（论文 SPOT_Stage3:224 的 kernel_dim_reduction_ECC_features）。
ECC_N_DIM: int = 100

# 模块选择方式：显式用 'legacy' 复现论文 Fig. 3d；v2.0.1 默认的 'multiscale' 是另一套
# null 模型 + bootstrap 共识方法，模块数不同且明显更慢，只作为实验开关保留。
MODULE_SELECTION: str = "legacy"

# 论文 SAM 模块层次聚类的参数（max_clust_len=None, lam_smooth=1e0, buffer=30）。
MODULE_PARAMS: dict = {"max_clust_len": None, "lam_smooth": 1.0, "buffer": 30}

# 跨天匹配候选对的 bbox IoU 下限。
MIN_BBOX_IOU: float = 0.05

# 跨天匹配候选对的质心距离上限（像素）。
MAX_CENT_DIST_PX: float = 60.0

# 传给 SPOT 特征函数的并行度：-1 表示用全部 CPU 核。
N_JOBS: int = -1

# 跨 (天, 视野) 的进程级并行度（Stage 1 / Stage 2 用，见 parallel.py）：
# 0 = 自动（min(视野数, max(1, 逻辑核 // 2))）、1 = 顺序执行、>1 = 固定上限。
# 默认 1（顺序）：P0–P3 之后单帧成本只剩几秒，而每个子进程要重新 import numpy/skimage 并解码自己的大图，
# 实测 9 帧的 Stage2 顺序 41.8 s、jobs=3 87.1 s、jobs=6 61.6 s、jobs=9 42.8 s —— **并行不赚**。
# 帧数很多（≳24，例如多个时间点 × 多个视野）或单帧很重时才值得 `--jobs N` 手动打开。
PARALLEL_JOBS: int = 1

# 是否按视野做批次校正：False，因为三个视野本身就是生物学重复，按视野归一化会抹掉要展示的差异（Spec D5）。
BATCH_CORRECT_FOV: bool = False

# 注意（预处理阶段必读）：power_transform_and_scale_features 会为 all_conditions 的每个唯一取值
# 分别拟合 PowerTransformer + StandardScaler。论文脚本走的是全局拟合（sepcond_norm=False），
# 所以调用时 all_conditions 必须传单一常量（如 np.zeros(n)）：传 day 等于按天各自 z-score、
# 恰好抹掉天间差异；传 FOV 等于隐式按视野做批次归一化，与 BATCH_CORRECT_FOV=False 冲突。
