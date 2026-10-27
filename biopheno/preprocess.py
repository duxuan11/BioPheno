"""Stage 3：预处理与 SAM 特征族等权（Spec §6-Stage3）。

整个过程是"全局拟合"的：所有实例一起做 power transform + 标准化，再按 SAM 特征族等权。
**不做**按天/按视野分别归一（`cfg.BATCH_CORRECT_FOV=False`），否则恰好抹掉本项目要测量的天间差异。

上游函数（`SPOT/Analysis/preprocessing.py`）的真实返回契约，逐条对照源码记下：

- `remove_zero_features(X, names, return_index=False)` → `(X, names)`（`return_index=True` 时再给
  `select_index`）：剔掉 max==0 与 std==0 两类列。它用 `np.nanmax`，全 NaN 列得到 NaN、不会被这一步
  剔除，但会发一条 "All-NaN slice encountered" 的 RuntimeWarning（已在调用处静音）。
- `remove_nan_variance_features(X, names, return_index=False)` → `(X, names)`（+ `select_index`）：
  剔掉 `np.isnan(std)` 的列，也就是**任何含 NaN 的列**。`np.std` 对含 ±Inf 的列也返回 NaN，所以
  全 NaN 的 motion 占位列与含 Inf 的列都是**被这一步**物理剔除的；第 3 步的显式剔除因此是幂等
  安全网（正常路径上剔除 0 列，见 `_drop_non_finite`）。
- `remove_high_variance_features(X, names, variance_threshold_sigma=3, return_index=False)` →
  `(X, names)`（+ `select_index`）：阈值 = `mean(std) + sigma * std(std)`，论文用 sigma=2。
- `scale_normalize_curvature_features(X, names)` → `(X, names)`：名字含 `'curvature'` 的列乘
  `names == 'equivalent_diameter'` 那一列（曲率的量纲是 length^-1，乘等效直径才可跨尺寸比较）。
  没有 `equivalent_diameter` 列时上游会拿空索引广播、抛一句看不出原因的 numpy 错误，这里提前自查。
- `power_transform_and_scale_features(X, names, all_conditions)` →
  `(X, names, pow_tformer_dict, std_tformer_dict)`：对 `all_conditions` 的每个唯一值各拟合一套
  PowerTransformer + StandardScaler。注意源码里两个字典的键是反的（`pow_tformer_dict` 里存的是
  StandardScaler 对象）——本项目不持久化 transformer，这里只是记录事实。
- `weight_sam_features(X, names, group_weighting=True, set_weighting=True, feature_sets=None,
  return_weights=False)` → **只返回 X**（不返回名字）：set/group 由 `sam_feature_groups` 按名字推断，
  每个 group 在 set 内总方差相等、每个 set 总方差为 1；未识别的名字会落进 'other' set 并发警告。
"""

import json
import warnings
from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd

from biopheno import config as cfg
from SPOT.Analysis import preprocessing as spot_prep

# 落盘目录与文件名（Spec §6-Stage3）。Stage 3 只读 Stage 2 的产物，自己不重算特征。
_FEATURES_SUBDIR: str = "features"
_PREPROCESSED_SUBDIR: str = "preprocessed"
_ALL_FEATURES_FILE: str = "all_features.csv"
_NAMES_FILE: str = "feature_names.json"
_X_FILE: str = "X.npy"
_KEEP_FILE: str = "keep_features.json"
_CONDITIONS_FILE: str = "all_conditions.npy"
_QC_FILE: str = "stage3_qc.json"

# 曲率尺度归一依赖的那一列（上游硬编码按名取列）。
_DIAMETER_NAME: str = "equivalent_diameter"

# 剔除记账按"原因"分成**互斥**类别：每类都记名字，并集恰好等于被剔除的全部列。
# 不按"哪一步删了什么"再累加一遍（那会双计：全 NaN 列既属于"NaN 方差"又属于"非有限"），
# `n_dropped_features` 只由维度相减得出。
_DROP_CATEGORIES: tuple[str, ...] = (
    "dropped_zero",
    "dropped_all_nan",
    "dropped_partial_nan",
    "dropped_non_finite",
    "dropped_high_variance",
)
# 各步物理剔除的列数（按执行顺序）。每列只会在某一步被删一次，所以合计同样 = n_dropped_features；
# 它回答"哪一步删的"，与上面的"为什么删"正交。
_DROP_STEPS: tuple[str, ...] = (
    "remove_zero_features",
    "remove_nan_variance_features",
    "remove_non_finite_features",
    "remove_high_variance_features",
)

# 打印被剔除列名时最多列几个（完整清单落在 stage3_qc.json）。
_PRINT_NAMES_LIMIT: int = 20


def _fail(message: str) -> NoReturn:
    """抛出一句带上下文的中文 ValueError（薄封装只为统一文案格式）。"""
    raise ValueError(message)


def _as_names(names, n_cols: int) -> np.ndarray:
    """特征名统一成 str 的 numpy 数组，并核对个数与列数一致。

    上游全部按布尔掩码取列（`feature_names[mask]`），传 list 会静默变成花式索引，
    所以必须先转 ndarray。
    """
    arr = np.asarray([str(n) for n in names], dtype=str)
    if arr.shape != (n_cols,):
        _fail(f"特征名个数 {arr.size} 与特征矩阵列数 {n_cols} 不一致：列的语义无法确定，拒绝预处理")
    return arr


def _curvature_scale(X: np.ndarray, names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """曲率特征按等效直径做尺度归一（第 5 步），缺列时给中文错误。

    上游 `scale_normalize_curvature_features` 用 `'curvature' in feature_name` 判定曲率列，
    我们去核对的是同一件事：有曲率列就必须有 `equivalent_diameter`，否则上游会抛
    `all_feats[:, 空索引].ravel()` 的广播错误（shape (0,) → (n,)），看不出是缺列。
    """
    names = np.asarray(names)
    has_curvature = np.array(["curvature" in str(n) for n in names]) if names.size else np.zeros(0, dtype=bool)
    if has_curvature.any() and not bool((names == _DIAMETER_NAME).any()):
        _fail(
            f"特征里有 {int(has_curvature.sum())} 个 curvature 列，但没有 {_DIAMETER_NAME} 列："
            "SPOT 的曲率尺度归一要乘等效直径，缺这一列无法做，请检查 Stage2 的形状特征表"
        )
    return spot_prep.scale_normalize_curvature_features(X, names)


def _dropped_positions(n_cols: int, select_index: np.ndarray) -> np.ndarray:
    """把上游 `return_index=True` 给的保留下标换成被剔除列的**位置**（按名字做差不精确：会因重名错位）。"""
    keep = np.zeros(n_cols, dtype=bool)
    keep[np.asarray(select_index, dtype=int)] = True
    return np.flatnonzero(~keep)


def _split_by_index(before: np.ndarray, select_index: np.ndarray) -> list[str]:
    """把上游 `return_index=True` 给的保留下标换成被剔除的列名。"""
    return [str(before[i]) for i in _dropped_positions(len(before), select_index)]


def _drop_non_finite(X: np.ndarray, names: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """第 3 步：显式剔除非有限列（幂等安全网，正常路径上剔除 0 列）。

    事实：`np.std` 对任何含 NaN 或 ±Inf 的列都返回 NaN，所以这些列在第 2 步（NaN 方差）就已经
    出局，到不了这里。这一步保留为幂等保险，防止上游 `remove_nan_variance_features` 的语义将来被
    改窄（那时漏下的非有限列就必须由我们自己兜住）。

    判据是逐列 `np.isfinite`，不是 `std == inf`：全有限但 std 溢出到 inf 的列（例如 ±1e308 交替）
    必须**保留**——它是量化尺度问题，不是坏数据。
    """
    finite_ok = np.isfinite(X).all(axis=0)
    dropped = [str(n) for n in names[~finite_ok]]
    return X[:, finite_ok], names[finite_ok], dropped


def _finish(info: dict, X: np.ndarray, names: np.ndarray) -> tuple[np.ndarray, list[str], dict]:
    """收尾记账并返回 `(X, kept_names, info)`。

    `n_dropped_features` 只由维度相减得出（不累加各步），并用两条恒等式自检：
    原因分类合计、各步剔除合计都必须等于实际剔除数。对不上说明实现漏记/双计了列，
    宁可在这里报错，也不给下游一份自相矛盾的审计表。
    """
    info["n_features_out"] = int(X.shape[1])
    info["n_dropped_features"] = info["n_features_in"] - info["n_features_out"]
    info["degenerate_no_features"] = X.shape[1] == 0
    n_classified = sum(int(info[f"n_{category}"]) for category in _DROP_CATEGORIES)
    n_per_step = sum(int(v) for v in info["n_dropped_per_step"].values())
    if n_classified != info["n_dropped_features"] or n_per_step != info["n_dropped_features"]:
        _fail(
            f"剔除记账不自洽：原因分类合计 {n_classified}、各步合计 {n_per_step}，而实际剔除"
            f" {info['n_dropped_features']} 列（输入 {info['n_features_in']} 列 → 输出 {info['n_features_out']} 列）："
            "这是实现 bug（某类/某步删了列却漏记名字），请报告"
        )
    return np.asarray(X, dtype=np.float64), [str(n) for n in names], info


def preprocess(X: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str], dict]:
    """Stage 3 的预处理链：剔除退化列 → 全局归一 → SAM 特征族等权。

    顺序是硬约束（与论文脚本一致）：
    零值/零方差 → NaN 方差 → 显式非有限列 → 高方差（σ=`cfg.VARIANCE_THRESHOLD_SIGMA`）→
    曲率尺度归一 → 全局 power transform + 标准化 → SAM set/group 等权。

    曲率尺度归一必须在 power transform **之前**：它作用在原始量纲上（曲率 × 等效直径），
    归一之后就找不到可乘的等效直径尺度了。

    `all_conditions` 传**单一常量** `np.zeros(n)`：`power_transform_and_scale_features` 会为每个唯一
    取值分别拟合一套 PowerTransformer + StandardScaler，传 day 就是按天各自 z-score（正好抹掉天间
    差异），传 FOV 就是隐式按视野做批次归一（与 `cfg.BATCH_CORRECT_FOV=False` 冲突）。
    该函数接受数值数组（内部只用 `np.unique` + 相等比较），不需要字符串。

    返回 `(X_processed, kept_names, info)`；`info` 逐步记录剔除的列名与个数、阈值、最终维度、
    feature set/group 的计数（`weight_sam_features` 只返回矩阵，这部分记账是我们自己补的）。
    剔除记账分两套互补、互不双计的口径：
    - `dropped_*`（`dropped_zero` / `dropped_all_nan` / `dropped_partial_nan` / `dropped_non_finite` /
      `dropped_high_variance`）回答“**为什么**删”：五类互斥，并集 = 被剔除的全部列；
    - `n_dropped_per_step`（`remove_zero_features` / `remove_nan_variance_features` /
      `remove_non_finite_features` / `remove_high_variance_features`）回答“**哪一步**删的”。
    `n_dropped_features` 只由维度相减得出（`n_features_in - n_features_out`），不累加。
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        _fail(f"特征矩阵必须是二维 (n_objects, n_feats)，实际 {X.ndim} 维、形状 {X.shape}")
    if X.shape[0] < 2:
        # StandardScaler / PowerTransformer 要估均值和方差；0–1 个实例只会给出全 0 或直接崩
        _fail(f"预处理至少需要 2 个实例（要估每列的均值/方差），实际只有 {X.shape[0]} 个")
    names = _as_names(names, X.shape[1])
    sigma = float(cfg.VARIANCE_THRESHOLD_SIGMA)

    info: dict = {
        "n_objects": int(X.shape[0]),
        "n_features_in": int(X.shape[1]),
        "n_features_out": 0,
        "n_dropped_features": 0,
        "variance_threshold_sigma": sigma,
        # 全局拟合的标记：下游据此确认没有按天/按视野分别归一
        "all_conditions": "single global condition (np.zeros(n))",
        "n_dropped_per_step": dict.fromkeys(_DROP_STEPS, 0),
        "n_curvature_scaled": 0,
        "n_other_set_features": 0,
        "feature_sets": {},
        "feature_groups": {},
        "degenerate_no_features": False,
    }
    for category in _DROP_CATEGORIES:
        info[category] = []
        info[f"n_{category}"] = 0

    # —— 第 1 步：全零 / 零方差列 ——
    # 全 NaN 列会让上游的 np.nanmax 发 "All-NaN slice encountered"；这些列马上就会被第 2 步剔除，
    # 警告本身没有信息量，这里只静音这一条。
    before = names
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        X, names, select_index = spot_prep.remove_zero_features(X, before, return_index=True)
    info["dropped_zero"] = _split_by_index(before, select_index)
    info["n_dropped_zero"] = len(info["dropped_zero"])
    info["n_dropped_per_step"]["remove_zero_features"] = info["n_dropped_zero"]

    # —— 第 2 步：std 为 NaN 的列（任何含 NaN 或 ±Inf 的列都在这里出局）——
    # 这一步的剔除按“NaN 的形态”拆成三类互斥原因，名字全部留给审计：
    # 全 NaN（motion 占位列）、部分 NaN、含 Inf 但无 NaN。
    before, before_X = names, X
    X, names, select_index = spot_prep.remove_nan_variance_features(X, before, return_index=True)
    dropped_positions = _dropped_positions(len(before), select_index)
    nan_any = np.isnan(before_X).any(axis=0)
    nan_all = np.isnan(before_X).all(axis=0)
    has_inf = ~nan_any & ~np.isfinite(before_X).all(axis=0)
    info["dropped_all_nan"] = [str(before[i]) for i in dropped_positions if nan_all[i]]
    info["dropped_partial_nan"] = [str(before[i]) for i in dropped_positions if nan_any[i] and not nan_all[i]]
    info["dropped_non_finite"] = [str(before[i]) for i in dropped_positions if has_inf[i]]
    info["n_dropped_all_nan"] = len(info["dropped_all_nan"])
    info["n_dropped_partial_nan"] = len(info["dropped_partial_nan"])
    info["n_dropped_non_finite"] = len(info["dropped_non_finite"])
    info["n_dropped_per_step"]["remove_nan_variance_features"] = len(dropped_positions)

    # —— 第 3 步：显式剔除非有限列（幂等安全网，正常路径上剔除 0 列，见 _drop_non_finite）——
    X, names, guard_dropped = _drop_non_finite(X, names)
    info["dropped_non_finite"] += guard_dropped
    info["n_dropped_non_finite"] = len(info["dropped_non_finite"])
    info["n_dropped_per_step"]["remove_non_finite_features"] = len(guard_dropped)

    if X.shape[1] == 0:
        # 所有列都退化（例如输入只有常数列 + 全 NaN 列）。sklearn 的 PowerTransformer 在 0 列时
        # 直接 ValueError，这里显式记成“退化”并原样返回：调用方（run_stage3）会据此报错，
        # 而 preprocess 本身不制造一个假的成功。
        return _finish(info, X, names)

    # —— 第 4 步：高方差列（论文用 sigma=2，不是上游默认的 3）——
    before = names
    X, names, select_index = spot_prep.remove_high_variance_features(
        X, before, variance_threshold_sigma=sigma, return_index=True
    )
    info["dropped_high_variance"] = _split_by_index(before, select_index)
    info["n_dropped_high_variance"] = len(info["dropped_high_variance"])
    info["n_dropped_per_step"]["remove_high_variance_features"] = info["n_dropped_high_variance"]
    if X.shape[1] == 0:
        return _finish(info, X, names)

    # —— 第 5 步：曲率尺度归一（必须在 power transform 之前）——
    X, names = _curvature_scale(X, names)
    info["n_curvature_scaled"] = int(sum("curvature" in str(n) for n in names))

    # —— 第 6 步：全局 power transform（Yeo-Johnson）+ 标准化 ——
    # 单一常量 condition = 全局只拟合一套变换（见模块 docstring 与 cfg 的说明）
    X, names, _pow_tformer, _std_tformer = spot_prep.power_transform_and_scale_features(X, names, np.zeros(X.shape[0]))

    # —— 第 7 步：SAM 特征族 / 特征集等权 ——
    # 上游只返回矩阵；名字不变，所以 kept_names 直接用第 6 步的 names。
    X = spot_prep.weight_sam_features(np.asarray(X, dtype=np.float64), names, group_weighting=True, set_weighting=True)

    sets, groups = spot_prep.sam_feature_groups(names)
    info["feature_sets"] = {s: int((sets == s).sum()) for s in dict.fromkeys(sets)}
    info["feature_groups"] = {g: int((groups == g).sum()) for g in dict.fromkeys(groups)}
    info["n_other_set_features"] = int((sets == "other").sum())
    return _finish(info, X, names)


def _read_stage2_features(feat_root: Path) -> tuple[np.ndarray, list[str]]:
    """读 Stage 2 的特征表，返回 `(X, names)`，并与 `feature_names.json` 交叉核对列名顺序。"""
    csv_path = feat_root / _ALL_FEATURES_FILE
    if not csv_path.exists():
        _fail(
            f"找不到 Stage2 的特征表 {csv_path}：Stage3 只做预处理、不重算特征，请先跑 Stage2"
            "（biopheno.features.run_stage2，产出 output/features/all_features.csv）再跑 Stage3"
        )
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        _fail(
            f"Stage2 的特征表 {csv_path} 读不出来（空文件、被截断或不是 UTF-8 文本："
            f"{type(exc).__name__}: {exc}）；请重跑 Stage2"
        )
    if len(df) == 0 or df.shape[1] == 0:
        _fail(f"Stage2 的特征表 {csv_path} 是空的（{len(df)} 行 × {df.shape[1]} 列）：请重跑 Stage2")
    if df.index.name != "instance_uid":
        # 没有 uid 列时 pandas 会把第一个数据列当成索引，特征矩阵会静默少一列、且行身份不明
        _fail(
            f"Stage2 的特征表 {csv_path} 的索引列不是 instance_uid（实际为 {df.index.name!r}）："
            "表可能被重写过，请重跑 Stage2"
        )
    names = [str(c) for c in df.columns]

    names_path = feat_root / _NAMES_FILE
    if names_path.exists():
        try:
            recorded = json.loads(names_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _fail(f"Stage2 的列名清单 {names_path} 不是合法 JSON（{exc}）：请重跑 Stage2")
        if not isinstance(recorded, list):
            # dict/标量也能被迭代（dict 迭代出键）或直接崩在 str(n) 上，先看一眼形状再比较
            _fail(
                f"Stage2 的列名清单 {names_path} 不是一个名字列表（实际是 {type(recorded).__name__}）："
                "文件可能被重写过，请重跑 Stage2"
            )
        if [str(n) for n in recorded] != names:
            _fail(
                f"Stage2 的列名清单 {names_path}（{len(recorded)} 个）与特征表 {csv_path} 的列"
                f"（{len(names)} 个）不一致：按名取列会静默错位，请重跑 Stage2"
            )
    try:
        X = df.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        _fail(f"Stage2 的特征表 {csv_path} 里有无法转成浮点数的值（{exc}）：请重跑 Stage2")
    return X, names


def run_stage3() -> None:
    """跑完 Stage 3 并落盘（Spec §6-Stage3）：读 `output/features/`，写 `output/preprocessed/`。

    产物（`output/preprocessed/`）：
    - `X.npy`：等权后的特征矩阵，float64（下游 UMAP / 聚类 / 模块识别全用 float64，
      这里不做 float32 降精度，免得把 1e-8 级别的等权归零误差放大进距离矩阵）；
    - `keep_features.json`：保留列的名字，顺序与 `X.npy` 的列完全一致；
    - `all_conditions.npy`：`np.zeros(n)`，即"全局单一条件"的事实记录（下游据此确认没有按天/按视野拆分）；
    - `stage3_qc.json`：`preprocess` 返回的完整 `info`——每一类被剔除列的名字（零值 / 全 NaN /
      部分 NaN / 非有限 / 高方差）、每步剔除个数、阈值、输入输出维度、实际用的 condition。
      剔除清单必须落盘：`dropped_high_variance` 决定了哪些特征根本进不了下游，不能只在 stdout 里一闪而过。

    失败时的磁盘状态是故意的：任何在落盘前失败的路径（读表失败、预处理报错）**都不会动**上一轮的
    `X.npy` / `keep_features.json`——旧的矩阵与旧的列名仍然成对可用，下游不会读到残缺产物；
    只有当新产物确定要写时才先删旧的三份 + 旧 QC（避免“新矩阵 + 旧列名”的混合态）。
    """
    feat_root = Path(cfg.OUTPUT_ROOT) / _FEATURES_SUBDIR
    X, names = _read_stage2_features(feat_root)

    X_processed, kept, info = preprocess(X, names)
    if X_processed.shape[1] == 0:
        per_step = "、".join(f"{step} {n}" for step, n in info["n_dropped_per_step"].items())
        _fail(
            f"预处理后没有任何特征剩下（输入 {info['n_features_in']} 列；各步剔除：{per_step}；"
            f"合计剔除了 {info['n_dropped_features']} 列）：Stage2 产物可能已损坏，请重跑 Stage2"
        )

    out_root = Path(cfg.OUTPUT_ROOT) / _PREPROCESSED_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)
    # 与 Stage 1/2 同样的理由：先删上一代产物，中途失败时磁盘上不会出现"新矩阵 + 旧列名"的混合态
    for stale in (_X_FILE, _KEEP_FILE, _CONDITIONS_FILE, _QC_FILE):
        (out_root / stale).unlink(missing_ok=True)

    np.save(out_root / _X_FILE, np.asarray(X_processed, dtype=np.float64))
    (out_root / _KEEP_FILE).write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(out_root / _CONDITIONS_FILE, np.zeros(X_processed.shape[0], dtype=np.float64))
    qc_path = out_root / _QC_FILE
    qc_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Stage3 完成：{info['n_features_in']} → {len(kept)} 维（剔除：零值 {info['n_dropped_zero']}、"
        f"全 NaN {info['n_dropped_all_nan']}、部分 NaN {info['n_dropped_partial_nan']}、"
        f"非有限 {info['n_dropped_non_finite']}、高方差 {info['n_dropped_high_variance']}；"
        f"各步：{info['n_dropped_per_step']}），特征集 {info['feature_sets']}，"
        f"全局拟合（{info['all_conditions']}）→ {out_root / _X_FILE}，剔除明细 → {qc_path}"
    )
    high_var = info["dropped_high_variance"]
    if high_var:
        # 高方差剔除最容易被忽略：它删的是“有值且变异大”的列，不是明显的坏数据
        head = "、".join(high_var[:_PRINT_NAMES_LIMIT])
        tail = "" if len(high_var) <= _PRINT_NAMES_LIMIT else f" …（共 {len(high_var)} 个，完整清单见 {qc_path}）"
        print(f"高方差剔除 {len(high_var)} 列（σ={info['variance_threshold_sigma']}）：{head}{tail}")
    if info["n_other_set_features"]:
        # 未识别的名字会被上游当成额外的 'other' set 参与等权，是个必须被看见的命名漂移
        print(f"警告：{info['n_other_set_features']} 个特征名不属于任何 SAM 特征集，已作为 'other' set 参与等权")
