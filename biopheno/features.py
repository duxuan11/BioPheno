"""Stage 2：单帧 SAM 形状 + 外观特征（Spec §6-Stage2）。

坐标约定沿用 `instances.py`：轮廓是 (row, col) 即 numpy 的 yx 顺序，与
`SPOT.Features.features.contour_to_binary` 的约定一致，这里不做任何转置。

**motion 特征被有意跳过**（人类决定：本地数据每 2–3 天才有一帧，帧间光流没有物理意义）。
为了让特征表与论文 2185 维的命名口径对齐、且下游一眼能看出"漏"是故意的，motion 家族以
同名 **NaN 列**占位（`motion_placeholder_names()`），Stage 5 会按 NaN 把它们剔除并记录。

本模块只读 `SPOT/`，不修改它。
"""

import functools
import inspect
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import exposure

from biopheno import config as cfg
from biopheno import instances as inst
from biopheno import parallel
from SPOT.Analysis import preprocessing
from SPOT.Features import features as spot_features

# 落盘目录与文件名（Spec §6-Stage2-3）。
_FEATURES_SUBDIR: str = "features"
_SHAPE_FILE: str = "sam_shape.npz"
_APPEARANCE_FILE: str = "sam_appearance.npz"
_NAMES_FILE: str = "feature_names.json"
_ALL_FEATURES_FILE: str = "all_features.csv"
_QC_FILE: str = "stage2_qc.json"

# 交叉核对要从 Stage 1 汇总表里读的三列；缺任何一列说明表被截断／换了格式，直接报错。
_STAGE1_KEY_COLUMNS: tuple[str, ...] = ("day", "fov", "instance_uid")

# motion 特征名静态解析用的常量。展开规则对应 SPOT 的
# `_motion_features_sample_fast`（label 段）与 `_directional_motion_descriptors`（描述子家族）：
# - 方向 SIFT / 向量 SIFT：8 个方向bin × 4 ring × 4 sector = 128 维；
# - 带符号标量描述子（curl/div）：2 个通道 × 16 个空间格 = 32 维；
# - 未做 masked 处理的标准 SIFT（非方向分支）：128 维。
_MOTION_SIFT_DIMS: dict[str, int] = {
    "flow_vector_sift": 128,
    "curl_signed": 32,
    "div_signed": 32,
    "shear_axis_sift": 128,
    "sift_motion": 128,
    "curl_sift": 128,
    "div_sift": 128,
    "shear_sift": 128,
}
# 全局运动量：8 个无条件输出的 label + 2 个 shear 专属 label（`shear_features=True` 时）
_MOTION_GLOBAL_LABELS: tuple[str, ...] = (
    "mean_speed_global",
    "stddev_speed_global",
    "mean_speed_global_flow",
    "stddev_speed_global_flow",
    "mean_curl_global_flow",
    "stddev_curl_global_flow",
    "mean_divergence_global_flow",
    "stddev_divergence_global_flow",
    "mean_shear_global_flow",
    "stddev_shear_global_flow",
)
_MOTION_REGION_HIST_TEMPLATE: str = "hist_bin_%d_region_%s"
_MOTION_REGION_LABEL_TEMPLATES: tuple[str, ...] = (
    "mean_speed_flow_region_%d",
    "stddev_speed_flow_region%d",
)
# `_motion_features_sample_fast` 里 `all_feats_labels.append(` 的出现次数。
# 用来把"SPOT 改了 label 段（增删家族）"这件事变成一个可检测的事实，而不是静默错位。
_MOTION_N_LABEL_BLOCKS: int = 9
# 默认参数下 motion 占位列的总数（10 全局 + 3 区域 ×(8 bin+2 统计) + 128+32+32+128 描述子）。
# 这是个钉死的期望值：静态展开的个数对不上，就说明描述子维度或 bin 数漂移了，此时宁可不给
# motion 列（可见的退化）也不给一份可能错位的清单。
_MOTION_N_PLACEHOLDER_COLS: int = 360


def _fail(message: str) -> None:
    """抛出一句带上下文的 ValueError（薄封装只为统一文案格式）。"""
    raise ValueError(message)


def _is_ect_feature(name: str) -> bool:
    """ECT 家族判定，必须与上游逐字一致：子串 `'ECT_' in name`。

    上游 `SPOT/Analysis/preprocessing.py:245`（`kernel_dim_reduction_ECC_features`）用的就是子串匹配。
    这里若改成 `startswith("ECT_")`，被上游选中而这里没选中的列会带着 NaN/Inf 进 `Nystroem.fit_transform`
    （sklearn 硬报错），反过来则会留下一份没被降维的 ECT 家族——两种都是静默错误。
    """
    return "ECT_" in str(name)


def _as_boundaries(contours: list[np.ndarray]) -> np.ndarray:
    """把轮廓列表转成 SPOT 要的 (n_objects, n_frames, n_points, 2) 单帧张量。

    轴序是硬约束：SPOT 对每个 (object, frame) 独立算特征，`np.array([contours])` 会得到
    (1, N, P, 2)，即"1 个实例、N 帧"——外观特征按帧索引 `vid[tt]`，单帧 vid 会在 tt>=1 时
    IndexError（已实测）。所以实例必须放在 object 轴、单帧放在 frame 轴：(N, 1, P, 2)。
    """
    if len(contours) == 0:
        return np.zeros((0, 1, 0, 2), dtype=np.float64)
    # 先自查点数是否等长：点数不同时 np.stack 抛的是裸 numpy 错误（"all input arrays must have the
    # same shape"），既没说是哪个模块、也看不出是本模块的等长重采样契约被破坏。
    shapes = {np.shape(c) for c in contours}
    if len(shapes) != 1:
        _fail(
            f"轮廓点数不一致，无法堆成同一张张量：出现 {sorted(shapes, key=str)} 种形状（应为等长重采样后的 (n_points, 2)）"
        )
    arr = np.stack([np.asarray(c, dtype=np.float64) for c in contours])
    if arr.ndim != 3 or arr.shape[2] != 2:
        _fail(f"轮廓数组形状应为 (n_points, 2)，实际 {arr.shape}")
    return np.ascontiguousarray(arr[:, None, :, :])


def _collapse_single_frame(X: np.ndarray, n: int) -> np.ndarray:
    """把 SPOT 的 (n_objects, n_frames, n_feats) 折叠成 (n_objects, n_feats)。

    实测（SPOT 2.x fast 路径）：`compute_boundary_morphology_features` 返回 (3, 1, 1601)、
    `compute_boundary_texture_features` 返回 (3, 1, 149)，即带一个 n_frames 轴。
    我们只喂单帧，所以这个轴必须正好是 1，否则下游 (N, D) 的契约就被悄悄破坏。
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == n and arr.shape[1] == 1:
        return arr[:, 0, :]
    if arr.ndim == 2 and arr.shape[0] == n:
        return arr
    _fail(f"SPOT 返回的特征形状 {arr.shape} 与预期的 ({n}, 1, D) 不符，无法折叠成 (N, D)")


def compute_shape_features(
    contours: list[np.ndarray], imshape: tuple[int, int], pixel_res: float
) -> tuple[np.ndarray, list[str]]:
    """单帧形状特征集（Spec §6-Stage2-1）：曲率 / 几何 / 区域属性 / Fourier / ECT / shape-context。

    `all_feats_compute=True` 会输出 ≈1600 列，其中 ECT 家族（实测 1152 列）与 shape-context
    家族（300 列）占绝大部分，由此产生的"形状特征上千维"是真实情况，交给
    `reduce_ecc_features` 压 ECT，不在这里裁剪。
    """
    boundaries = _as_boundaries(contours)
    X, names, _norm = spot_features.compute_boundary_morphology_features(
        boundaries,
        imshape,
        all_feats_compute=True,
        pixel_xy_res=pixel_res,
        fast=True,
        n_jobs=cfg.N_JOBS,
    )
    return _collapse_single_frame(X, len(contours)), [str(n) for n in names]


def compute_appearance_features(
    image: np.ndarray, contours: list[np.ndarray], pixel_res: float
) -> tuple[np.ndarray, list[str]]:
    """单帧外观特征集（Spec §6-Stage2-2）：区域内强度统计 + Haralick 纹理 + 掩码内 SIFT。

    `pixel_res` 只是为了与 `compute_shape_features` 的调用形状一致而保留：
    强度/纹理特征没有长度量纲，SPOT 的纹理函数也不接受像素分辨率参数。

    `masks=None` 的理由：轮廓本来就是 label 掩码的边界，`contour_to_binary` 重建的前景与真值掩码
    等价，再传一遍 masks 只会多一次「轮廓栅格化 vs 掩码」的尺寸对齐风险。

    `vid` 必须是 (T, H, W)：单帧就是 `image[None]`；强度先 `rescale_intensity` 到 [0,1] 的 float，
    这样 masked Haralick 的分级区间（由数据 min/max 推得）与人眼看的对比度一致，也不受 8/16 bit 影响。
    """
    vid = exposure.rescale_intensity(np.asarray(image, dtype=np.float64), in_range="image", out_range=(0.0, 1.0))
    boundaries = _as_boundaries(contours)
    X, names, _norm = spot_features.compute_boundary_texture_features(
        vid=vid[None],
        boundaries=boundaries,
        use_mask=True,
        masks=None,
        compute_all_feats=True,
        fast=True,
        n_jobs=cfg.N_JOBS,
    )
    return _collapse_single_frame(X, len(contours)), [str(n) for n in names]


def reduce_ecc_features(X: np.ndarray, names: list[str], n_dim: int) -> tuple[np.ndarray, list[str]]:
    """把 ECT 家族用 RBF 核（Nystroem）压到 `n_dim` 维，其它家族原样保留（Spec §6-Stage2-4）。

    对应论文脚本 `SPOT_Stage3...py:224` 的 `kernel_dim_reduction_ECC_features`。
    随机种子取 `cfg.RANDOM_STATE`（=0）：SPOT 自己的默认是 1，这里按计划统一成全局种子。
    """
    X = np.asarray(X, dtype=np.float64)
    names_list = [str(n) for n in names]
    if X.shape[1] != len(names_list):
        _fail(f"特征矩阵列数 {X.shape[1]} 与特征名个数 {len(names_list)} 不一致")
    if not any(_is_ect_feature(n) for n in names_list):
        return X, names_list  # 没有 ECT 家族时 Nystroem 无输入可拟合，原样返回
    # Nystroem 的 n_components 不能超过样本数（sklearn 会按不放回抽样直接报错），
    # 所以实例数少于目标维数时按实例数封顶；真实数据 6741 个实例时这就是 n_dim 本身。
    n_components = int(min(n_dim, X.shape[0]))
    if n_components < 1:
        _fail(f"ECC 降维至少需要 1 个实例与 1 维输出（n_dim={n_dim}, n_objects={X.shape[0]}）")
    Xr, names_r, _tformer = preprocessing.kernel_dim_reduction_ECC_features(
        X,
        np.array(names_list),
        n_dim=n_components,
        gamma=None,
        random_state=cfg.RANDOM_STATE,
    )
    return np.asarray(Xr, dtype=np.float64), [str(n) for n in names_r]


def motion_placeholder_names() -> list[str]:
    """静态解析 SPOT 源码得到 motion 特征名清单，用于生成 NaN 占位列（Spec §6-Stage2-3）。

    motion **不计算**（人类决定），这里只对齐命名口径。实现方式是「源码里存在性检查 + 模板展开」：
    1. 用 `inspect.getsource` 取 `_motion_features_sample_fast` 的源码，确认所有 label 模板/SIFT 家族名
       仍然在里面，且 `all_feats_labels.append(` 的段数仍是 9；
    2. 展开计数全部来自 `compute_boundary_motion_features_fast` 自己的签名默认值
       （`n_contour_feat_bins` / `n_contours` / `n_angles` / `shear_features` / `directional_motion_sift`）。

    任一条对不上（上游重命名、增删特征家族）就返回空列表并告警，而不是给出一份可能错位的清单：
    下游看到"没有 motion 列"是可见的事实，错位的名字则是静默错误。
    """
    try:
        src = inspect.getsource(spot_features._motion_features_sample_fast)
        # SIFT 家族的 label 前缀写在 _directional_motion_descriptors 里，两个函数一起看
        src_desc = inspect.getsource(spot_features._directional_motion_descriptors)
        params = inspect.signature(spot_features.compute_boundary_motion_features_fast).parameters
    except (OSError, TypeError, AttributeError) as exc:  # 源码不可读（打包成 pyc 等）
        warnings.warn(f"无法读取 SPOT motion 特征源码，motion 占位列被跳过：{exc}", stacklevel=2)
        return []

    expected_tokens = (
        *_MOTION_GLOBAL_LABELS,
        _MOTION_REGION_HIST_TEMPLATE,
        *_MOTION_REGION_LABEL_TEMPLATES,
        *_MOTION_SIFT_DIMS,
    )
    missing = [token for token in expected_tokens if token not in src and token not in src_desc]
    n_blocks = src.count("all_feats_labels.append(")
    if missing or n_blocks != _MOTION_N_LABEL_BLOCKS:
        warnings.warn(
            "SPOT motion 特征的 label 段与本模块的静态假设不一致"
            f"（缺失模板 {missing}，label 段数 {n_blocks} != {_MOTION_N_LABEL_BLOCKS}），"
            "motion 占位列被跳过；请核对 SPOT 版本后更新 features.py",
            stacklevel=2,
        )
        return []

    n_bins = int(params["n_contour_feat_bins"].default)
    # 径向窗口（region）数 = n_contours × n_angles；SPOT 的 `_uniq_tile_regions` 在窗口数不足时
    # 会补到 n_contours 个，所以下限取 n_contours，两边取大才不会少算
    n_regions = max(
        int(params["n_contours"].default), int(params["n_contours"].default) * int(params["n_angles"].default)
    )
    shear = bool(params["shear_features"].default)
    directional = bool(params["directional_motion_sift"].default)

    names = list(_MOTION_GLOBAL_LABELS[:8])
    if shear:
        names += list(_MOTION_GLOBAL_LABELS[8:])
    # 区域段：每个径向窗口先出 n_bins 个速度直方图 bin，再出 2 个速度统计量
    for reg in range(1, n_regions + 1):
        names += [_MOTION_REGION_HIST_TEMPLATE % (bin_i + 1, str(reg).zfill(3)) for bin_i in range(n_bins)]
        names += [template % reg for template in _MOTION_REGION_LABEL_TEMPLATES]
    # 描述子段：方向分支（默认）与经典 SIFT 分支的 label 前缀不同
    if directional:
        families = ["flow_vector_sift", "curl_signed", "div_signed"]
        families += ["shear_axis_sift"] if shear else []
    else:
        families = ["sift_motion", "curl_sift", "div_sift"]
        families += ["shear_sift"] if shear else []
    for family in families:
        names += [f"{family}_{i + 1}" for i in range(_MOTION_SIFT_DIMS[family])]
    # 个数钉死：描述子维度表或 bin 数漂移时，展开个数会变（例如上游把 16 个空间格改成别的），
    # 此时给一份长度不对的"motion 名"只会让下游的列对齐错位，退化成空列表才是有事可查的失败。
    if len(names) != _MOTION_N_PLACEHOLDER_COLS:
        warnings.warn(
            f"motion 占位列应为 {_MOTION_N_PLACEHOLDER_COLS} 个，静态展开却得到 {len(names)} 个"
            "（描述子维度或直方图 bin 数已漂移），motion 占位列被跳过；请核对 SPOT 版本后更新 features.py",
            stacklevel=2,
        )
        return []
    return names


def _fill_nonfinite_ect(shape: np.ndarray, names: list[str]) -> int:
    """把 ECT 列里的非有限值填 0，返回被填的个数。

    非有限值（退化轮廓的分割结果）会让 Nystroem 的 sklearn 输入检查直接报错。ECT 列之后会被
    核投影整体压掉，所以填 0 只是为了让这一步能跑通，个数记进 QC 供人工核对；其它家族的
    非有限值一律原样保留，交给 Stage 5 的 NaN 列剔除（那才是它们的归宿）。
    """
    cols = [i for i, name in enumerate(names) if _is_ect_feature(name)]
    if not cols:
        return 0
    block = shape[:, cols]
    bad = ~np.isfinite(block)
    n_bad = int(bad.sum())
    if n_bad:
        shape[:, cols] = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
    return n_bad


def _count_nonfinite(X: np.ndarray) -> int:
    """数一个特征块里的非有限值个数（NaN / ±Inf）。不填、不丢、不插值，只计数。"""
    return int((~np.isfinite(np.asarray(X, dtype=np.float64))).sum())


def _load_stage1_uid_sets() -> dict[str, set[str]] | None:
    """读 Stage 1 的 `all_instances.csv`，返回 {"day/fov": {instance_uid, ...}}。

    文件不存在时返回 None（并由调用方把"未核对"记进 QC）：Stage 2 的测试可以在没有真实
    Stage 1 产物的前提跑合成数据，但运行时必须能从 QC 看到这件事没被核对过。

    文件存在但**读不出来**（空文件、被截断的表头、行内字段数不足）则是硬错误：空文件抛的
    `EmptyDataError`、缺列抛的 `KeyError`、截断行静默变 NaN 都是"看不出是哪个产物坏了"的坑，
    这里一律换成带路径的中文错误。
    """
    path = Path(cfg.OUTPUT_ROOT) / inst._INSTANCES_SUBDIR / inst._ALL_INSTANCES_FILE
    if not path.exists():
        return None
    try:
        stage1 = pd.read_csv(path, dtype=str)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        _fail(
            f"Stage1 汇总表 {path} 读不出来（空文件或被截断：{type(exc).__name__}: {exc}）；"
            "Stage2 无法核对实例，请重跑 Stage1"
        )
    missing_cols = [col for col in _STAGE1_KEY_COLUMNS if col not in stage1.columns]
    if missing_cols:
        _fail(
            f"Stage1 汇总表 {path} 缺少必需列 {missing_cols}（实际列 {list(stage1.columns)}）："
            "文件可能不完整，请重跑 Stage1"
        )
    # 行内字段数不足时 pandas 不报错，而是补 NaN（写到一半被杀就会这样），
    # 这里把它变成可见的错误，否则会冒出一个 uid 为 NaN 的假实例。
    n_incomplete = int(stage1[list(_STAGE1_KEY_COLUMNS)].isna().any(axis=1).sum())
    if n_incomplete:
        _fail(
            f"Stage1 汇总表 {path} 里有 {n_incomplete} 行缺少 day/fov/instance_uid（文件可能被截断）；"
            "Stage2 拒绝据一份残缺的实例清单做核对，请重跑 Stage1"
        )
    return {
        f"{day}/{fov}": set(group["instance_uid"].astype(str))
        for (day, fov), group in stage1.groupby(["day", "fov"], sort=False)
    }


def _stage1_catalog(day: str, fov: str) -> tuple[pd.DataFrame, dict[int, np.ndarray], np.ndarray]:
    """读 Stage 1 的产物：实例 meta、**已对齐坐标下**的轮廓、原图亮度图。

    为什么不再调 `inst.build_catalog`：那一步是 Stage 1 的活（±8 px 对齐搜索、逐标签求外轮廓、
    裁剪代表图），对 15 MP 的图每个视野约 80 s；而 Stage 2 只需要"已对齐的轮廓"（Stage 1 已落盘到
    `contours.npz`）和原图（读文件即可）。于是 Stage 2 从"重算一遍 Stage 1"变成"读盘 + 算特征"，
    一个字节的数字都不会变（轮廓、图像、画布尺寸三者都与重算得到的一致）。

    画布尺寸取原图前两维：`align_mask_to_image` 返回的标签图就铺在 `image.shape[:2]` 上，
    与原先 `catalog["labels"].shape` 完全一致（即使掩码比原图小）。
    """
    stage1_dir = Path(cfg.OUTPUT_ROOT) / inst._INSTANCES_SUBDIR / str(day) / str(fov)
    try:
        catalog = inst.load_catalog(stage1_dir)
    except FileNotFoundError as exc:
        _fail(
            f"读不到 {day}/{fov} 的 Stage1 产物（{stage1_dir}：{exc}）。Stage2 直接复用 Stage1 的"
            "已对齐轮廓与原图，不再重算对齐/轮廓；请先跑 Stage1，并确认 --output 指向同一棵产物树"
        )
    image = inst._to_luma(inst._read_image(str(day), str(fov)))
    return catalog["meta"], catalog["contours"], image


def _cross_check_stage1_uids(day: str, fov: str, meta: pd.DataFrame, stage1: dict[str, set[str]] | None) -> str:
    """把本视野读回来的 uid 集合与 Stage 1 汇总表的 uid 集合对齐，返回 QC 里的核对结果。

    为什么要有这一步：Stage 2 现在**直接复用** Stage 1 落盘的 `contours.npz`（见 `_stage1_catalog`），
    但 `cfg.N_CONTOUR_POINTS` / `ALIGN_MAX_SHIFT_PX` 之类的参数可能在两次运行之间被改过——那会让
    Stage 1 的产物与当前参数不一致。uid 集合（fov_标签_帧号）能捕捉"实例集/对齐变化"，
    对不上就直接报错，而不是让特征静默挂到别的轮廓上。

    返回写进 QC 的 `uid_cross_check`，取值：

    - `"ok"`：两侧 uid 集合相同；
    - `"ok_empty"`：本视野 0 个实例且汇总表里也没有这个键（零实例的正常写法，见下面的注释）；
    - `"not_checked_no_all_instances"`：汇总表根本不存在（调用方已发出可见警告）；
    - 真正的错配则抛 `RuntimeError`。
    """
    if stage1 is None:
        return "not_checked_no_all_instances"
    key = f"{day}/{fov}"
    expected = stage1.get(key)
    actual = {str(uid) for uid in meta["instance_uid"]} if len(meta) else set()
    if expected is None:
        # Stage 1 用 `pd.concat(metas)` 写汇总表：某视野的 meta 是 0 行时，它在 all_instances.csv 里
        # 连一行都没有（键直接不存在）。所以"键缺失 + 本视野也 0 个实例"是两侧对同一事实（零实例）
        # 达成一致，不是错——正是空视野跳过逻辑要容忍的情形。只有 meta 非空才算真对不上。
        if len(meta) == 0:
            return "ok_empty"
        raise RuntimeError(
            f"Stage1 的 all_instances.csv 里没有 {key} 的实例行（本次重建出 {len(actual)} 个实例），"
            "Stage2 拒绝把特征接到未经 Stage1 落盘的实例上；"
            "请先跑 Stage1（或核对 days/fovs 是否与 Stage1 一致）"
        )
    if actual == expected:
        return "ok"
    raise RuntimeError(
        f"{key} 的实例 uid 集合与 Stage1 落盘的 all_instances.csv 不一致"
        f"（本次重建 {len(actual)} 个、Stage1 落盘 {len(expected)} 个；"
        f"仅重建侧 {len(actual - expected)} 个、仅落盘侧 {len(expected - actual)} 个）："
        "cfg（N_CONTOUR_POINTS/ALIGN_MAX_SHIFT_PX）或 instances.py 可能在两次运行之间被改动，"
        "特征会挂到不同的轮廓上；请重跑 Stage1 后再跑 Stage2"
    )


def _unusable_block_reason(
    day: str,
    fov: str,
    shape: np.ndarray,
    names_s: list[str],
    appearance: np.ndarray,
    names_a: list[str],
) -> str | None:
    """返回该视野不可用的原因（可直接拼进表）或 None。

    宽度为 0 的块（轮廓全部退化、SPOT 什么都不给）没有任何列可拼，返回原因让调用方整帧跳过；
    宽度与特征名个数不一致则是契约被破坏（无法判断列的含义），直接报错。
    """
    for kind, block, names in (("形状", shape, names_s), ("外观", appearance, names_a)):
        width = int(block.shape[1])
        if width == 0 and len(names) == 0:
            return "empty_feature_block"
        if width != len(names):
            _fail(
                f"{day}/{fov} 的{kind}特征块宽 {width} 与特征名个数 {len(names)} 不匹配："
                "列的含义无法确定，拒绝拼进总表（SPOT 版本或参数被改动？）"
            )
    return None


def _stage2_frame(day: str, fov: str, *, stage1_uids: dict[str, set[str]] | None) -> dict:
    """算一个 (day, fov) 的形状/外观特征块，交给父进程拼表（`parallel.map_frames` 的任务函数）。

    必须是**模块级**函数（要 pickle 到子进程），而且不碰全局拟合：返回 shape / appearance 块与
    `sub_meta`（行身份），全实例一次拟合的 ECT 降维仍由父进程做。

    `line` 是交给父进程按顺序打印的日志行：子进程的 print 不会被父进程的 `redirect_stdout` 捕获，
    不这样带回来，并行时逐阶段日志会丢掉每个视野那一行。
    """
    meta, contours_by_label, image = _stage1_catalog(day, fov)
    imshape = image.shape
    # 先核对"复用到的实例"与 Stage 1 汇总表是同一批，再往下算（对不上直接 RuntimeError）
    uid_check = _cross_check_stage1_uids(day, fov, meta, stage1_uids)
    # 只算有轮廓的实例；contours 里没有键的标签照样留在 meta 里，但这里数一下跳过多少
    keep_labels = [int(label) for label in meta["label"] if int(label) in contours_by_label]
    entry: dict = {
        "n_instances_meta": len(meta),
        "n_instances_computed": len(keep_labels),
        "n_skipped_no_contour": len(meta) - len(keep_labels),
        "uid_cross_check": uid_check,
        "skipped_reason": None,
    }
    result: dict = {
        "day": day,
        "fov": fov,
        "entry": entry,
        "skipped": True,
        "meta": None,
        "shape": None,
        "appearance": None,
        "names_s": None,
        "names_a": None,
    }
    if not keep_labels:
        # 整个视野一个可用轮廓都没有（分割失败，或标签全是退化小块）：宽度为 0 的块拼进
        # vstack 会让整轮在最后一步崩掉，所以整帧跳过、记账、继续。
        entry["skipped_reason"] = "no_usable_contour"
        entry["n_shape_features_raw"] = 0
        entry["n_appearance_features"] = 0
        entry["n_nonfinite_shape"] = 0
        entry["n_nonfinite_appearance"] = 0
        entry["shape_seconds"] = 0.0
        entry["appearance_seconds"] = 0.0
        result["line"] = f"[{day}/{fov}] 无可用轮廓（meta {len(meta)} 个实例全部跳过），跳过该视野"
        return result

    row_of_label = {int(label): row for row, label in enumerate(meta["label"])}
    sub_meta = meta.iloc[[row_of_label[label] for label in keep_labels]].reset_index(drop=True)
    contours = [contours_by_label[label] for label in keep_labels]

    t0 = time.perf_counter()
    shape, names_s = compute_shape_features(contours, imshape, cfg.PIXEL_RES)
    shape_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    appearance, names_a = compute_appearance_features(image, contours, cfg.PIXEL_RES)
    appearance_seconds = time.perf_counter() - t0
    # 每家族的非有限值计数：只记不填（ECT 会在拟合前填 0，那是 sklearn 的硬要求），
    # Task 5 按 NaN 剔列时才有据可查
    entry.update(
        {
            "n_shape_features_raw": int(shape.shape[1]),
            "n_appearance_features": int(appearance.shape[1]),
            "n_nonfinite_shape": _count_nonfinite(shape),
            "n_nonfinite_appearance": _count_nonfinite(appearance),
            "shape_seconds": round(shape_seconds, 2),
            "appearance_seconds": round(appearance_seconds, 2),
        }
    )
    # 宽度为 0 的块（轮廓全退化）整帧跳过；宽度与名数不符则报错（列的含义不确定）
    skip_reason = _unusable_block_reason(day, fov, shape, names_s, appearance, names_a)
    if skip_reason:
        entry["skipped_reason"] = skip_reason
        result["line"] = f"[{day}/{fov}] 特征块宽度为 0（轮廓全部退化？），跳过该视野"
        return result

    result.update(
        {
            "skipped": False,
            "meta": sub_meta,
            "shape": shape,
            "appearance": appearance,
            "names_s": names_s,
            "names_a": names_a,
            "line": (
                f"[{day}/{fov}] 实例 {len(keep_labels)} 个（无轮廓跳过 {len(meta) - len(keep_labels)} 个），"
                f"形状 {shape.shape[1]} 维 {shape_seconds:.1f}s，外观 {appearance.shape[1]} 维 {appearance_seconds:.1f}s"
            ),
        }
    )
    return result


def run_stage2(days: list[str] | None = None, fovs: list[str] | None = None) -> pd.DataFrame:
    """跑完 Stage 2 并落盘（Spec §6-Stage2）：逐视野算形状+外观，汇总成实例 × 特征表。

    days/fovs 缺省时用 `cfg.DAYS` 的全部天数与 `cfg.FOVS` 的全部视野（与 `run_stage1` 一致；
    测试靠显式传参，不依赖真实数据）。`cfg.*` 一律在**调用时**读取，测试才能 monkeypatch。

    ECT 降维在**全部视野拼起来之后一次性拟合**：逐视野各拟合一个 Nystroem 会把 9 个互不相同的
    投影空间拼进同一张表，列名相同、语义不同，是静默错误。

    只对 meta 里有轮廓的实例算特征（Task 2 的已知契约：没有可用轮廓的标签在 contours 里没有键），
    跳过数逐视野记进 QC，行数变化是可见的、可审计的。整个视野一个可用轮廓都没有、或特征块宽度为 0
    （列全退化）时，该视野整帧跳过并记进 QC：这种块拼进 `vstack` 会让整轮在最后一步崩掉，
    而且看不出是哪个视野的问题。

    开跑时先交叉核对每个视野重建出的 uid 集合与 Stage 1 落盘的 `instances/all_instances.csv`：
    两者不一致就会把特征挂到不同的轮廓上，必须用 `RuntimeError` 拦住（核对结果记进 QC）。

    产物（`output/features/`）：
    - `sam_shape.npz`：降维后的形状特征块（键 `X` / `names` / `uid`）
    - `sam_appearance.npz`：外观特征块（同上）
    - `feature_names.json`：**全部列名，顺序与矩阵列完全一致**（形状 → 外观 → motion 占位）
    - `all_features.csv`：最终特征表，首列为 `instance_uid`
    - `stage2_qc.json`：每视野的实例数、跳过数与原因、uid 核对结果、每家族非有限值计数、维数与耗时
    """
    day_list = list(days) if days is not None else list(cfg.DAYS)
    fov_list = list(fovs) if fovs is not None else list(cfg.FOVS)
    out_root = Path(cfg.OUTPUT_ROOT) / _FEATURES_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)
    # 与 run_stage1 同样的理由：先删上一代产物，中途失败时磁盘上不会出现"新数组 + 旧列名"的混合态
    for stale in (_SHAPE_FILE, _APPEARANCE_FILE, _NAMES_FILE, _ALL_FEATURES_FILE, _QC_FILE):
        (out_root / stale).unlink(missing_ok=True)

    stage1_uids = _load_stage1_uid_sets()
    if stage1_uids is None:
        warnings.warn(
            "找不到 Stage1 的 instances/all_instances.csv，本次无法核对复用到的实例与 Stage1 汇总表是否一致"
            "（QC 里记为 not_checked_no_all_instances）",
            stacklevel=2,
        )

    shape_blocks: list[np.ndarray] = []
    appearance_blocks: list[np.ndarray] = []
    meta_blocks: list[pd.DataFrame] = []
    shape_names: list[str] = []
    appearance_names: list[str] = []
    qc: dict[str, dict] = {}
    ordered_frames: list[tuple[str, str]] = []

    # 输入清单（--images/--masks）里没有的 (day, fov) 格子直接不跑：真实数据不保证每个视野每天都有图，
    # 硬按笛卡尔积跑会在缺失格子上报错、或在 Stage2 的 uid 核对里对不上 Stage1 的清单。
    frames = set(inst.available_frames(day_list, fov_list))
    if not frames:
        _fail(
            f"days={day_list} × fovs={fov_list} 与输入清单没有交集"
            f"（清单里有 {sorted(cfg.INPUT_FILES)}），Stage2 拒绝产出空特征表"
        )

    for day in day_list:
        for fov in fov_list:
            if (str(day), str(fov)) not in frames:
                continue  # 输入里没有这一格（Stage1 也没跑过它）
            ordered_frames.append((str(day), str(fov)))

    # 各 (day, fov) 彼此独立（子进程只读自己那一对图），所以按视野并行；全实例一次拟合的
    # ECT 降维仍在父进程对拼接后的矩阵做，口径不变。日志行随结果带回、按顺序打，行序同顺序执行。
    results = parallel.map_frames(
        functools.partial(_stage2_frame, stage1_uids=stage1_uids),
        ordered_frames,
    )
    for result in results:
        day, fov = result["day"], result["fov"]
        print(result["line"])
        qc[f"{day}/{fov}"] = result["entry"]
        if result["skipped"]:
            continue
        if not shape_names:
            shape_names, appearance_names = result["names_s"], result["names_a"]
        elif result["names_s"] != shape_names or result["names_a"] != appearance_names:
            _fail(f"{day}/{fov} 的特征名与前面视野不一致，无法拼成同一张表（SPOT 版本或参数被改动？）")
        shape_blocks.append(result["shape"])
        appearance_blocks.append(result["appearance"])
        meta_blocks.append(result["meta"])

    shape_all = np.vstack(shape_blocks) if shape_blocks else np.zeros((0, 0))
    appearance_all = np.vstack(appearance_blocks) if appearance_blocks else np.zeros((0, 0))
    uids = (
        np.concatenate([block["instance_uid"].to_numpy(dtype=str) for block in meta_blocks])
        if meta_blocks
        else np.zeros(0, dtype=str)
    )

    n_ect_filled = _fill_nonfinite_ect(shape_all, shape_names)
    shape_reduced, names_reduced = reduce_ecc_features(shape_all, shape_names, cfg.ECC_N_DIM)

    # motion 故意不计算：以同名 NaN 列占位，让下游一眼看出是被有意置空（而不是算失败）
    motion_names = motion_placeholder_names()
    blocks = [shape_reduced, appearance_all]
    if motion_names:
        blocks.append(np.full((len(uids), len(motion_names)), np.nan))
    names = [*names_reduced, *appearance_names, *motion_names]
    # 落盘前最后一道关卡：列名重复会让下游按名取列（df[motion] 之类）静默取错。
    # 用 _fail 而不是 assert：`python -O` 会把 assert 整条删掉，这道关卡就不存在了。
    n_duplicate = len(names) - len(set(names))
    if n_duplicate:
        _fail(f"特征名有 {n_duplicate} 个重复（共 {len(names)} 列），按名取列会静默取错，拒绝落盘")
    X = np.hstack(blocks) if blocks else np.zeros((len(uids), 0))

    df = pd.DataFrame(X, index=pd.Index(uids, name="instance_uid"), columns=names)

    np.savez_compressed(
        out_root / _SHAPE_FILE,
        X=shape_reduced,
        # 字符串用 numpy 的 unicode dtype（不是 object）：这样下游 np.load 不需要 allow_pickle
        names=np.asarray(names_reduced, dtype=str),
        uid=np.asarray(uids, dtype=str),
    )
    np.savez_compressed(
        out_root / _APPEARANCE_FILE,
        X=appearance_all,
        names=np.asarray(appearance_names, dtype=str),
        uid=np.asarray(uids, dtype=str),
    )
    (out_root / _NAMES_FILE).write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(out_root / _ALL_FEATURES_FILE)

    for row in qc.values():
        row["n_shape_features_reduced"] = len(names_reduced)
        row["n_motion_placeholder"] = len(motion_names)
        row["n_features_total"] = len(names)
        # 全视野汇总值（ECT 降维只在拼接后拟合一次，所以这个计数天然是全局的，不是逐视野的）
        row["n_nonfinite_ect_filled"] = n_ect_filled
    (out_root / _QC_FILE).write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Stage2 完成：{len(df)} 个实例 × {len(names)} 维特征 → {out_root / _ALL_FEATURES_FILE}")
    return df
