"""掩码 → 实例目录：对齐、外轮廓重采样、裁剪代表图（Spec §6-Stage1）。

坐标约定：本模块全程用 (row, col) 即 numpy 的 yx 顺序，与
`skimage.measure.find_contours` 及 `SPOT.Features.features.contour_to_binary`
的约定一致；本任务不做 (x, y) 转换，下游 Task 4/8/10 直接沿用。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.transform as sktform
import tifffile
from skimage import io
from skimage.measure import find_contours
from skimage.util import img_as_ubyte

from biopheno import config as cfg

# 真实数据的文件名后缀：原图 <day>/<fov>_多聚焦融合.png，掩码 <day>/Output/<fov>_多聚焦融合.tif。
_NAME_SUFFIX: str = "_多聚焦融合"

# 少于这么多点的候选轮廓直接丢弃：说明标签只有一两像素或者退化，重采样没有意义。
_MIN_CONTOUR_POINTS: int = 6

# 落盘文件名（Spec §6-Stage1-7）。
_META_FILE: str = "instances.csv"
_CONTOUR_FILE: str = "contours.npz"
_PATCH_FILE: str = "patches.npz"

# Stage 1 全部产物落在这里的子目录，汇总表与 QC 表同层（Spec §6-Stage1-7 + 风险表）。
_INSTANCES_SUBDIR: str = "instances"
_ALL_INSTANCES_FILE: str = "all_instances.csv"
_QC_FILE: str = "stage1_qc.json"

# meta 表的列顺序，与 Spec 保持一致，方便人工核对。
_META_COLUMNS: tuple[str, ...] = (
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
    "n_contours_discarded",
)


def _to_luma(image: np.ndarray) -> np.ndarray:
    """转 ITU-R 601 亮度作为 appearance 输入通道（Spec §6-Stage1-2）；已是二维则原样返回。"""
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[..., 0]
    rgb = image[..., :3].astype(np.float64)
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    if image.dtype == np.uint8:
        return np.clip(np.rint(luma), 0, 255).astype(np.uint8)
    return luma


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """统一成 uint8：uint8 原样返回；浮点按 [0,1] 或 [0,255] 量级归一后量化。"""
    if arr.dtype == np.uint8:
        return arr
    a = np.asarray(arr, dtype=np.float64)
    if a.size and a.max() > 1.0:
        a = a / 255.0  # skimage.transform.resize(preserve_range=True) 保留 0–255 量级
    return img_as_ubyte(np.clip(a, 0.0, 1.0))


def _place(labels: np.ndarray, dy: int, dx: int, shape: tuple[int, int]) -> np.ndarray:
    """把 labels 平移到 shape 大小的全零画布上，偏移 (dy, dx)；越界部分直接裁掉。"""
    out = np.zeros(shape, dtype=labels.dtype)
    h, w = labels.shape
    # 源窗口与目标窗口的重叠区域（dy>0 表示向下移，dy<0 表示裁掉上面几行）
    src_r0, dst_r0 = max(0, -dy), max(0, dy)
    src_c0, dst_c0 = max(0, -dx), max(0, dx)
    n_r = min(h - src_r0, shape[0] - dst_r0)
    n_c = min(w - src_c0, shape[1] - dst_c0)
    if n_r > 0 and n_c > 0:
        out[dst_r0 : dst_r0 + n_r, dst_c0 : dst_c0 + n_c] = labels[src_r0 : src_r0 + n_r, src_c0 : src_c0 + n_c]
    return out


def align_mask_to_image(image: np.ndarray, labels: np.ndarray, max_shift: int) -> tuple[np.ndarray, tuple[int, int]]:
    """±max_shift 平移搜索，最大化"掩码内均值 − 掩码外均值"的**绝对值**（Spec §6-Stage1-1）。

    取绝对值而不是有符号差：本地数据的实例比背景**更暗**（内部 147–189 vs 外部 181–212，Spec §2），
    有符号差在正确对齐处是负的最大的（约 −60），而把掩码推出组织让内外均值趋同反而"更大"（趋近 0），
    搜索会主动错位；取绝对值既兼容更暗/更亮两种极性，又保证对齐位置对比最强。

    掩码与原图的画布尺寸可能差几行几列，先各自裁到公共最小尺寸再搜索平移。返回
    （补齐到 `image.shape[:2]` 的 labels，(dy, dx) 偏移）；正常数据偏移应为 0，非零时
    Task 3 会记录告警（Spec 风险表：>= 8 px 即触到搜索边界、需人工确认）。

    画布契约：labels 永远是二维 (H, W) 标签图，即使传入 RGB 三通道原图（`_read_image` 的返回类型），
    也按前两维 `image.shape[:2]` 分配画布，绝不把三维 shape 传给 `_place`。
    """
    img = _to_luma(image)
    canvas = (int(img.shape[0]), int(img.shape[1]))  # 二维画布：RGB 输入取前两维
    common = (min(canvas[0], labels.shape[0]), min(canvas[1], labels.shape[1]))
    img_c = img[: common[0], : common[1]].astype(np.float64)
    lab_c = labels[: common[0], : common[1]]

    best_key: tuple[float, int] | None = None  # (目标函数, -|dy|-|dx|)：同分时取更小的平移
    dy = dx = 0
    for cand_dy in range(-max_shift, max_shift + 1):
        for cand_dx in range(-max_shift, max_shift + 1):
            shifted = _place(lab_c, cand_dy, cand_dx, common)
            inside = shifted > 0
            n_in = int(inside.sum())
            if n_in == 0 or n_in == inside.size:
                continue  # 掩码被移出画布或占满画布，目标函数无定义
            objective = abs(float(img_c[inside].mean() - img_c[~inside].mean()))
            key = (objective, -(abs(cand_dy) + abs(cand_dx)))
            if best_key is None or key > best_key:
                best_key, dy, dx = key, cand_dy, cand_dx

    return _place(lab_c, dy, dx, canvas), (dy, dx)


def _resample_closed(contour: np.ndarray, n_points: int) -> np.ndarray:
    """把 (n,2) 轮廓按弧长等间距重采样成 n_points 个闭合点（不重复首尾点）。"""
    pts = np.asarray(contour, dtype=np.float64)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])  # find_contours 对贴边实例不给闭合折线，这里显式闭合
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:  # 所有点重合的退化轮廓
        return np.repeat(pts[:1], n_points, axis=0)
    targets = np.linspace(0.0, total, n_points, endpoint=False)
    return np.column_stack([np.interp(targets, s, pts[:, axis]) for axis in (0, 1)])


def extract_contours(labels: np.ndarray, n_points: int) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """每个标签的最长闭合外轮廓，等弧长重采样到 n_points 点，坐标为 (row, col)。

    返回（label -> (n_points, 2) 轮廓，label -> 被丢弃的候选轮廓数）。凹形轮廓保留
    （find_contours 直接给出凹凸一致的边界），点数 < 6 的候选被丢弃并计数。
    """
    contours: dict[int, np.ndarray] = {}
    discarded: dict[int, int] = {}
    for label in np.unique(labels):
        if label == 0:
            continue  # 0 是背景
        candidates = find_contours(labels == label, 0.5)
        kept = [c for c in candidates if len(c) >= _MIN_CONTOUR_POINTS]
        discarded[int(label)] = len(candidates) - len(kept)
        if not kept:
            continue  # 全部候选都太小：不进 contours，但 discarded 仍计数，build_catalog 会补上 meta 行
        contours[int(label)] = _resample_closed(max(kept, key=len), n_points)
    return contours, discarded


def bbox_of_contour(contour: np.ndarray) -> tuple[int, int, int, int]:
    """轮廓的闭区间包围盒 (r0, c0, r1, c1)，整数像素下标（含端点）。"""
    return (
        int(np.floor(contour[:, 0].min())),
        int(np.floor(contour[:, 1].min())),
        int(np.ceil(contour[:, 0].max())),
        int(np.ceil(contour[:, 1].max())),
    )


def crop_patches(
    image: np.ndarray, contours: dict[int, np.ndarray], pad: int, size: tuple[int, int]
) -> dict[int, dict[str, np.ndarray]]:
    """按轮廓包围盒外扩 pad 后裁图，返回 label -> {"native": 原分辨率, "sift": size 尺寸}。

    顺序是"先外扩再裁剪"：`native` 的形状恒为 `(bbox_h + 2*pad, bbox_w + 2*pad)`，贴边实例超出
    图像画布的那几条边填 0（黑）。填 0 而不是背景均值：0 表示"图像之外的未知区域"，
    填背景均值会伪造出一片看起来很正常的组织间质，且 0 不与任何真实像素亮度混淆。
    """
    img = _to_uint8(image)
    h, w = img.shape
    patches: dict[int, dict[str, np.ndarray]] = {}
    for label, contour in contours.items():
        r0, c0, r1, c1 = bbox_of_contour(contour)
        r0, c0, r1, c1 = r0 - pad, c0 - pad, r1 + pad, c1 + pad  # 先外扩
        native = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.uint8)
        # 只有落在图像内的窗口需要拷贝，越界的那部分保持 0（黑）——也就是"越界用 0 填充"
        rs, cs = max(r0, 0), max(c0, 0)
        re, ce = min(r1 + 1, h), min(c1 + 1, w)
        if re > rs and ce > cs:
            native[rs - r0 : re - r0, cs - c0 : ce - c0] = img[rs:re, cs:ce]
        resized = sktform.resize(native, size, preserve_range=True, anti_aliasing=True)
        patches[int(label)] = {"native": native, "sift": _to_uint8(resized)}
    return patches


def _paths(day: str, fov: str) -> tuple[Path, Path]:
    """返回 (原图, 掩码) 路径：命令行给的 `cfg.INPUT_FILES` 优先，否则退回 Spec §2 的旧约定。

    `cfg.INPUT_FILES` 非空时**不允许**回退：那说明本次是命令行指定了输入目录，
    某个 (day, fov) 不在清单里是调用方的错，静默去读 `DATA_ROOT` 会把两套数据混进同一轮结果。
    """
    configured = cfg.INPUT_FILES.get((str(day), str(fov)))
    if configured is not None:
        return Path(configured[0]), Path(configured[1])
    if cfg.INPUT_FILES:
        raise RuntimeError(
            f"本次输入清单里没有 {day}/{fov}（命令行 --images/--masks 只认出 {len(cfg.INPUT_FILES)} 个采样点）："
            "请核对 days/fovs 是否来自同一份输入目录，或重跑上一次 Stage1 的产物是否被复用"
        )
    return (
        cfg.DATA_ROOT / day / f"{fov}{_NAME_SUFFIX}.png",
        cfg.DATA_ROOT / day / "Output" / f"{fov}{_NAME_SUFFIX}.tif",
    )


def _require(path: Path, what: str, *, legacy: bool) -> None:
    """确认要读的那个文件存在；不存在时报一个能照做的错（而不是裸的 io.imread 路径）。

    只检查**本次要读的那一个**文件：`_read_labels` 不该因为同一视野的原图不在就拒绝读掩码
    （配准核验这类调用只读掩码）。`legacy=True` 说明路径是按旧约定拼出来的，报错里才该提示
    `--images/--masks`；命令行给的路径消失则说明文件在开跑后被挪走/删了。
    """
    if path.exists():
        return
    if legacy:
        raise FileNotFoundError(
            f"旧约定的{what} {path} 不存在。若数据是「扁平布局」（原图目录 + 掩码目录，文件名里带天数，"
            f"如 A1_day0.png / A1_day0.tif），请用 `--images <原图目录> --masks <掩码目录>` 指定；"
            f"否则请核对 config.DATA_ROOT（当前 {cfg.DATA_ROOT}）"
        )
    raise FileNotFoundError(
        f"命令行给定的{what} {path} 不存在：`--images/--masks` 指向的文件在开跑后消失了（被移动/改名？）"
    )


def _read_image(day: str, fov: str) -> np.ndarray:
    """读原图（RGB uint8）；路径见 `_paths`（命令行目录优先，其次 Spec §2 旧约定）。"""
    image = _paths(day, fov)[0]
    _require(image, "原图", legacy=cfg.IMAGE_DIR is None)
    return io.imread(image)


def _read_labels(day: str, fov: str) -> np.ndarray:
    """读掩码（单页 uint16 标签图）；路径见 `_paths`。

    按扩展名选 reader：TIFF 用 `tifffile`（16 位标签的正确通道数/位深由它保证），
    PNG/BMP 等走 skimage——`tifffile.imread` 读不了 PNG（直接抛 `TiffFileError: not a TIFF file`）。
    """
    path = _paths(day, fov)[1]
    _require(path, "掩码", legacy=cfg.MASK_DIR is None)
    if path.suffix.lower() in (".tif", ".tiff"):
        return tifffile.imread(path)
    return io.imread(path)


def available_frames(days: list[str] | None = None, fovs: list[str] | None = None) -> list[tuple[str, str]]:
    """要跑的 (day, fov) 清单（按 `cfg.DAYS` × `cfg.FOVS` 的声明顺序）。

    `cfg.INPUT_FILES` 非空时以它为准：真实数据不保证"每个视野每天都有图"（少一天、少一个视野都正常），
    直接按笛卡尔积去跑会在缺失的格子上报错或造出空视野。空 dict 时退回完整笛卡尔积（旧约定）。
    """
    day_list = list(days) if days is not None else list(cfg.DAYS)
    fov_list = list(fovs) if fovs is not None else list(cfg.FOVS)
    if cfg.INPUT_FILES:
        return [
            (str(day), str(fov))
            for day in day_list
            for fov in fov_list
            if (str(day), str(fov)) in cfg.INPUT_FILES
        ]
    return [(str(day), str(fov)) for day in day_list for fov in fov_list]


def build_catalog(
    day: str,
    fov: str,
    image: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> dict:
    """把一个 (day, fov) 变成实例目录：meta 表 + 对齐后的 labels + 轮廓 + 裁剪图。

    image/labels 为 None 时才读真实文件；测试直接传入合成图。
    """
    frame_idx = int(cfg.DAYS[day])
    time_hours = float(cfg.TIME_HOURS[day])
    if image is None:
        image = _read_image(day, fov)
    if labels is None:
        labels = _read_labels(day, fov)

    luma = _to_luma(image)
    labels, offset = align_mask_to_image(luma, labels, cfg.ALIGN_MAX_SHIFT_PX)
    contours, discarded = extract_contours(labels, cfg.N_CONTOUR_POINTS)
    patches = crop_patches(luma, contours, cfg.PATCH_PAD_PX, cfg.PATCH_SIZE)

    h, w = labels.shape
    # "只打标不删"：出现在掩码里的每个标签都要有 meta 行。候选轮廓全被丢弃的标签在 contours /
    # patches 里没有条目，但仍要留下 is_small 与 n_contours_discarded 的记录。
    label_ids = sorted(int(k) for k in np.unique(labels) if int(k) != 0)
    rows = []
    for label in label_ids:
        rows_ii, cols_ii = np.nonzero(labels == label)
        if label in contours:
            r0, c0, r1, c1 = bbox_of_contour(contours[label])
        else:
            # 没有可用轮廓：退回掩码自身的包围盒（闭区间）
            r0, c0, r1, c1 = int(rows_ii.min()), int(cols_ii.min()), int(rows_ii.max()), int(cols_ii.max())
        area_px = int(rows_ii.size)
        rows.append(
            {
                # 帧号必须在最后一段、分隔符必须是 '_'：SPOT 内部按 '*' + ff.split('_')[-1] 解析（Spec D13）
                "instance_uid": f"{fov}_{int(label)}_{frame_idx}",
                "day": day,
                "fov": fov,
                "frame_idx": frame_idx,
                "time_hours": time_hours,
                "label": int(label),
                "centroid_row": float(rows_ii.mean()),
                "centroid_col": float(cols_ii.mean()),
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "area_px": area_px,
                "touches_border": bool(
                    rows_ii.min() == 0 or cols_ii.min() == 0 or rows_ii.max() == h - 1 or cols_ii.max() == w - 1
                ),
                "is_small": bool(area_px * cfg.PIXEL_RES**2 < cfg.SMALL_AREA_FLAG_PX),
                "n_contours_discarded": int(discarded.get(int(label), 0)),
            }
        )

    return {
        "meta": pd.DataFrame(rows, columns=list(_META_COLUMNS)),
        "image": luma,
        "labels": labels,
        "contours": contours,
        "patches": patches,
        # 对齐偏移 (dy, dx)：run_stage1 用它做 ALIGN_MAX_SHIFT_PX 的风险检查（Spec §9 风险表）
        "align_offset": (int(offset[0]), int(offset[1])),
    }


def save_catalog(catalog: dict, outdir: Path) -> None:
    """落盘 instances.csv / contours.npz / patches.npz（Spec §6-Stage1-7）。

    原图与标签图体积太大（每张 ≈ 15 MB）不落盘：下游需要时用同参数重新 build_catalog。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    catalog["meta"].to_csv(outdir / _META_FILE, index=False)
    np.savez_compressed(
        outdir / _CONTOUR_FILE,
        **{f"label{int(label)}_contour": np.asarray(c) for label, c in catalog["contours"].items()},
    )
    flat: dict[str, np.ndarray] = {}
    for label, patch in catalog["patches"].items():
        flat[f"label{int(label)}_native"] = patch["native"]
        flat[f"label{int(label)}_sift"] = patch["sift"]
    np.savez_compressed(outdir / _PATCH_FILE, **flat)


def load_catalog(outdir: Path) -> dict:
    """读回落盘的三件套；image/labels 不落盘，这两个键返回 None。"""
    outdir = Path(outdir)
    meta = pd.read_csv(outdir / _META_FILE)

    contours: dict[int, np.ndarray] = {}
    with np.load(outdir / _CONTOUR_FILE) as npz:
        for key in npz.files:
            contours[int(key.split("_")[0].removeprefix("label"))] = npz[key]

    patches: dict[int, dict[str, np.ndarray]] = {}
    with np.load(outdir / _PATCH_FILE) as npz:
        for key in npz.files:
            stem, kind = key.rsplit("_", 1)
            patches.setdefault(int(stem.removeprefix("label")), {})[kind] = npz[key]

    return {"meta": meta, "image": None, "labels": None, "contours": contours, "patches": patches}


def run_stage1(days: list[str] | None = None, fovs: list[str] | None = None) -> pd.DataFrame:
    """跑完 Stage 1 并汇总（Spec §6-Stage1）：逐 (day, fov) 建目录、落盘、返回合并后的 meta 表。

    days/fovs 缺省时用 `cfg.DAYS` 的全部天数与 `cfg.FOVS` 的全部视野。`cfg.DATA_ROOT` /
    `cfg.OUTPUT_ROOT` 在**调用时**从模块读取（不缓存），测试才能 monkeypatch 到 tmp_path。

    产物：
    - `output/instances/<day>/<fov>/{instances.csv, contours.npz, patches.npz}`
    - `output/instances/all_instances.csv`：全部视野的 meta 拼接（下游 Stage 2/5/6 的入口）
    - `output/instances/stage1_qc.json`：每个视野的实例数、对齐偏移、丢弃轮廓标签数、小实例数

    风险兜底：对齐偏移**摸到或超过** `cfg.ALIGN_MAX_SHIFT_PX` 时直接报错停下（Spec §9 风险表：
    需人工确认），而不是带着错位的掩码往下跑。

    失败时不留半套结果：开跑前先删掉上一代的 `all_instances.csv` / `stage1_qc.json`。汇总与 QC 只在
    全部视野跑完后才写，而分视野目录在循环内即时落盘，所以若第 N 个视野抛错，磁盘上会是
    "前 N−1 个新目录 + 无汇总表"——下游看到汇总表不存在就知道这一代没跑完；不删旧表就会出现
    "新目录 + 旧汇总表"的混合态，下游无从分辨它是陈的。
    """
    day_list = list(days) if days is not None else list(cfg.DAYS)
    fov_list = list(fovs) if fovs is not None else list(cfg.FOVS)
    frames = available_frames(day_list, fov_list)
    if not frames:
        # 空清单会让下面的 pd.concat 产出空表并"成功"落盘，看起来像跑完了（Spec 反目标 A5）
        raise RuntimeError(
            f"days={day_list} × fovs={fov_list} 与输入清单没有交集"
            f"（清单里有 {sorted(cfg.INPUT_FILES)}），Stage1 拒绝产出空结果"
        )
    out_root = Path(cfg.OUTPUT_ROOT) / _INSTANCES_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)
    # 先删上一代汇总产物（本函数写出的汇总类产物就这两个），让中途失败不会看起来像成功
    for stale_name in (_ALL_INSTANCES_FILE, _QC_FILE):
        (out_root / stale_name).unlink(missing_ok=True)

    metas: list[pd.DataFrame] = []
    qc: dict[str, dict[str, int]] = {}
    for day, fov in frames:
        catalog = build_catalog(day, fov)
        dy, dx = catalog["align_offset"]
        max_shift = int(cfg.ALIGN_MAX_SHIFT_PX)
        offset = max(abs(dy), abs(dx))
        # 真正的探测器是 `>=`：build_catalog 把 max_shift 当搜索半径传给 align_mask_to_image，
        # 最优解按构造必然落在 ±max_shift 之内，单靠 `>` 永远不触发（写在那里的检查形同虚设）。
        # 反过来，偏移**正好落在边界**恰恰是坏消息：真正的最优解可能在边界之外被搜索范围截断了
        # （例：真需要 12 px 时搜索只能给出 (8, 3)，max(...) 仍是 8）。所以边界解同样要人工确认。
        # `>` 保留为防御性兜底：非默认调用者（不经过 build_catalog 自带的搜索）可能直接给出越界偏移。
        if offset >= max_shift:
            why = "摸到了搜索边界，真正的最优解可能被搜索范围截断" if offset == max_shift else "超过了上限"
            raise RuntimeError(
                f"{day}/{fov} 的掩码与原图对齐偏移 (dy={dy}, dx={dx}) {why}（±{max_shift} px），"
                "请先人工确认对齐再继续（Spec §9 风险表）"
            )
        save_catalog(catalog, out_root / day / fov)
        meta = catalog["meta"]
        # 只打标不删：没有可用轮廓的标签仍在 meta 里，这里只数数，不假设 contours/patches 有条目
        n_discarded = int((meta["n_contours_discarded"] > 0).sum())
        n_small = int(meta["is_small"].sum())
        qc[f"{day}/{fov}"] = {
            "n_instances": len(meta),
            "align_dy": int(dy),
            "align_dx": int(dx),
            "n_labels_discarded_contours": n_discarded,
            "n_is_small": n_small,
        }
        print(
            f"[{day}/{fov}] 实例 {len(meta)} 个，对齐偏移 (dy={dy}, dx={dx})，"
            f"丢弃轮廓的标签 {n_discarded} 个，小实例 {n_small} 个"
        )
        metas.append(meta)

    summary = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame(columns=list(_META_COLUMNS))
    summary.to_csv(out_root / _ALL_INSTANCES_FILE, index=False)
    (out_root / _QC_FILE).write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Stage1 完成：{len(summary)} 个实例，汇总表 → {out_root / _ALL_INSTANCES_FILE}")
    return summary
