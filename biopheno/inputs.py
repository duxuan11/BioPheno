"""输入数据集发现：把「原图目录 + 掩码目录」配成 (day, fov) 清单，并写进 `config`。

为什么需要它：Stage 1/2 原先只会拼两条写死的路径（`<DATA_ROOT>/<day>/<FOV>_多聚焦融合.png` 与
`<DATA_ROOT>/<day>/Output/<FOV>_多聚焦融合.tif`），天数与视野来自 `config.DAYS` / `config.FOVS`
这两个常量。真实数据换成扁平布局后（`images/A1_day0.png` + `masks/A1_day0.tif`），天数与视野只能
**从文件名里读出来**，而且帧轴（`frame_idx` 0/1/2…）必须按真实天数排序后重新编号——SPOT 的
`construct_obj_traj_from_uniq_obj_ids` 只认连续整数帧号（Spec §6-Stage1-6）。

约定（发现规则）：
- 原图与掩码按**同名 stem** 配对（扩展名各自独立，`.png/.tif/.tiff/...` 都收）；
  掩码额外容忍 `_mask` / `_masks` / `_seg` / `_segmentation` / `_label` / `_labels` 后缀。
- 文件名里必须有一个 day 记号：`day0`、`D3`、`day_5`、`day-5`（大小写不敏感，需落在分隔符边界上）。
  记号以外剩下的部分就是视野名（`A1_day0` → FOV `A1`，`day0_A1` → FOV `A1`）。
- 天数按**数字**排序（不是字符串：`day10` 要排在 `day3` 之后），再压成连续帧号 0..N-1。
- 时间轴按 `config.HOURS_PER_DAY` 换算：day3 → 72h。

配对失败一律报错而不是跳过：少一个掩码还继续跑，会得到一份"实例数比图像少一帧"的结果，
而报告上看不出缺了哪一帧（Spec 反目标 A5：静默成功）。掩码目录里**多出来**的文件不影响配对
（只按原图清单驱动），但会记进 `Dataset.unmatched_masks` 供人工核对。
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from biopheno import config as cfg

# 认作原图的扩展名（大小写不敏感）。
IMAGE_EXTS: tuple[str, ...] = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")

# 认作掩码的扩展名（TIFF 走 tifffile，其余走 skimage；见 `instances._read_labels`）。
MASK_EXTS: tuple[str, ...] = (".tif", ".tiff", ".png", ".bmp")

# 掩码文件名可能带的额外后缀（`A1_day0_mask.tif` 配 `A1_day0.png`）。
MASK_STEM_SUFFIXES: tuple[str, ...] = ("_mask", "_masks", "_seg", "_segmentation", "_label", "_labels")

# day 记号：必须落在字符串边界或分隔符上，避免把 `A1d3` 这种视野名误当天数。
# 例：`A1_day0` → ("day0", 0)；`day0_A1` → ("day0", 0)；`plate-D3` → ("D3", 3)。
_DAY_TOKEN = re.compile(r"(?i)(?:^|[_\-.\s])(day|d)[_\-]?(\d+)(?=$|[_\-.\s])")

# 目录名/文件名两侧的分隔符，剥掉后再当视野名用。
_SEPARATORS = "_-.\t "


@dataclass(frozen=True)
class Frame:
    """一个采样点：某个视野在某一天的一张原图 + 一张掩码。"""

    day: str  # 规范化后的天数名，如 "day3"（= 文件名里的天数记号）
    fov: str  # 视野名，如 "A1"
    day_number: int  # 真实天数（文件名里的数字），只用来排序与换算小时
    image: Path
    mask: Path


@dataclass(frozen=True)
class Dataset:
    """一次发现的结果：帧清单 + 没配上原图的掩码（供人工核对）。"""

    frames: tuple[Frame, ...]
    images_dir: Path
    masks_dir: Path
    unmatched_masks: tuple[Path, ...] = field(default=())

    @property
    def days(self) -> tuple[str, ...]:
        """去重后的天数名，按帧号顺序（即真实天数升序）。"""
        return tuple(dict.fromkeys(frame.day for frame in self.frames))

    @property
    def fovs(self) -> tuple[str, ...]:
        """去重后的视野名，按字母序（多个视野时报告与出图的顺序靠它固定）。"""
        return tuple(sorted({frame.fov for frame in self.frames}))

    @property
    def day_numbers(self) -> dict[str, int]:
        """天数名 → 真实天数（day3 → 3）。"""
        return {frame.day: frame.day_number for frame in self.frames}


def _list_images(directory: Path, exts: tuple[str, ...], what: str) -> list[Path]:
    """列出目录下的图像文件（按名字排序，稳定可复现）；目录不存在或为空都直接报错。"""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(
            f"{what}目录 {directory} 不是一个目录：--images/--masks 需要传目录（原图与掩码各一个），"
            "单个文件请先放进目录再指过来"
        )
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in exts and not path.name.startswith(".")
    )
    if not files:
        raise FileNotFoundError(
            f"{what}目录 {directory} 里没有可识别的图像文件（支持的扩展名：{', '.join(exts)}）；"
            "只扫该目录顶层，不会递归进子目录"
        )
    return files


def parse_stem(stem: str) -> tuple[str, str, int]:
    """从文件名 stem 里解析 (视野名, 天数名, 真实天数)。

    `A1_day0` → ("A1", "day0", 0)；`day3_A2` → ("A2", "day3", 3)；`D5` → ("", "day5", 5)。
    没有 day 记号时抛 `ValueError`（带上期望的命名与看到的 stem，方便直接改文件名）。
    """
    match = _DAY_TOKEN.search(stem)
    if match is None:
        raise ValueError(
            f"文件名 {stem!r} 里认不出天数记号：需要一个 `day<N>` 或 `d<N>` 的片段"
            "（例：A1_day0.png、A1_day3.tif），大小写不敏感，且前后要有 _ - . 空格 或字符串边界隔开"
        )
    day_number = int(match.group(2))
    day = f"day{day_number}"  # 规范化：d3 / day_3 / DAY-3 → day3
    fov = (stem[: match.start()] + stem[match.end() :]).strip(_SEPARATORS)
    if not fov:
        raise ValueError(
            f"文件名 {stem!r} 里只有天数、没有视野名：实例 uid、图标题与报告都按视野组织，空视野名会到处留下 `{day}/`。"
            "请把视野写进文件名（例：A1_day0.png / A1_day0.tif）"
        )
    return fov, day, day_number


def _mask_index(masks_dir: Path) -> dict[str, Path]:
    """掩码 stem → 路径（同 stem 出现两次直接报错，避免悄悄用错一张掩码）。"""
    index: dict[str, Path] = {}
    for path in _list_images(masks_dir, MASK_EXTS, "掩码"):
        if path.stem in index:
            raise ValueError(
                f"掩码目录 {masks_dir} 里有两张同名掩码 {index[path.stem].name} 与 {path.name}："
                "无法判断该用哪一张，请删掉或改名"
            )
        index[path.stem] = path
    return index


def _find_mask(index: dict[str, Path], stem: str) -> Path | None:
    """按 stem 找掩码：先精确，再试常见的 `_mask` / `_seg` 等后缀。"""
    if stem in index:
        return index[stem]
    for suffix in MASK_STEM_SUFFIXES:
        if f"{stem}{suffix}" in index:
            return index[f"{stem}{suffix}"]
    return None


def discover(images_dir: Path, masks_dir: Path) -> Dataset:
    """扫两个目录、按 stem 配对，返回按 (真实天数, 视野名) 排好序的 `Dataset`。

    原图有一个配不上掩码就抛 `FileNotFoundError`（静默跳过会少算一帧）；掩码多出来只记录不报错。
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    image_paths = _list_images(images_dir, IMAGE_EXTS, "原图")
    mask_index = _mask_index(masks_dir)

    frames: list[Frame] = []
    missing: list[str] = []
    used_masks: set[Path] = set()
    seen: dict[tuple[str, str], Path] = {}
    for image_path in image_paths:
        fov, day, day_number = parse_stem(image_path.stem)
        mask_path = _find_mask(mask_index, image_path.stem)
        if mask_path is None:
            missing.append(image_path.name)
            continue
        key = (day, fov)
        if key in seen:
            # 两个文件名推出同一个 (day, fov)：后一个会覆盖前一个的路径，帧轴上也会撞车
            raise ValueError(
                f"原图 {seen[key].name} 与 {image_path.name} 都解析成 {day}/{fov or '（空视野名）'}："
                "请让每个 (天数, 视野) 只有一张原图（视野名写进文件名，如 A1_day0.png / A2_day0.png）"
            )
        seen[key] = image_path
        used_masks.add(mask_path)
        frames.append(Frame(day=day, fov=fov, day_number=day_number, image=image_path, mask=mask_path))

    if missing:
        available = sorted(f"{path.stem}{path.suffix}" for path in mask_index.values())[:8]
        raise FileNotFoundError(
            f"{len(missing)} 张原图在掩码目录 {masks_dir} 里找不到同名掩码：{'、'.join(missing[:8])}"
            f"{'…' if len(missing) > 8 else ''}。掩码目录里前几个文件：{'、'.join(available)}"
            "（配对规则：同名 stem；掩码可带 _mask/_seg 等后缀）"
        )

    frames.sort(key=lambda frame: (frame.day_number, frame.fov))
    unmatched = tuple(path for path in mask_index.values() if path not in used_masks)
    return Dataset(frames=tuple(frames), images_dir=images_dir, masks_dir=masks_dir, unmatched_masks=unmatched)


def configure(images_dir: Path, masks_dir: Path, *, output_root: Path | None = None) -> Dataset:
    """发现输入并写进 `config`（DAYS / FOVS / TIME_HOURS / INPUT_FILES / IMAGE_DIR / MASK_DIR）。

    天数被压成连续帧号：按真实天数升序编号 0,1,2…，因为 SPOT 的时间函数只认连续整数帧号。
    `output_root` 给了就一并改写 `config.OUTPUT_ROOT`（默认不动，仍是 `output`）。
    """
    data = discover(images_dir, masks_dir)
    if not data.frames:
        raise ValueError(f"{images_dir} 与 {masks_dir} 没有配出任何 (天数, 视野) 采样点")

    cfg.DAYS = {day: index for index, day in enumerate(data.days)}
    cfg.FOVS = data.fovs
    cfg.TIME_HOURS = {day: float(number) * float(cfg.HOURS_PER_DAY) for day, number in data.day_numbers.items()}
    cfg.INPUT_FILES = {(frame.day, frame.fov): (frame.image, frame.mask) for frame in data.frames}
    cfg.IMAGE_DIR = data.images_dir
    cfg.MASK_DIR = data.masks_dir
    if output_root is not None:
        cfg.OUTPUT_ROOT = Path(output_root)
    return data
