import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from biopheno import config as cfg  # sys.path 补好之后才能 import biopheno
from biopheno import instances as inst

# 真实数据的文件名后缀，与 biopheno.instances._NAME_SUFFIX 一致（这里故意硬编码成数据契约）。
_NAME_SUFFIX: str = "_多聚焦融合"


@pytest.fixture
def make_mask_and_image():
    """合成 (亮度图 uint8, 标签图 uint16) 的工厂 fixture，避免测试依赖 1.2 GB 真实数据。

    与真实数据一致：类器官内部比背景**暗**（对齐目标函数靠这个性质，Spec §2）。
    - n_blobs：类器官个数，大小相当（半径 18 px，面积约 1000 px²）；
    - tiny_area：给定时第 1 个类器官正好这么多像素（用矩形保证面积精确）；
    - touch_border：为真时第 1 个类器官贴左边界（圆心落在第 0 列）。
    """

    def _make(n_blobs: int = 2, tiny_area: int | None = None, touch_border: bool = False):
        """按上面的参数造一对 240×320 的 (uint8 亮度图, uint16 标签图) 并返回。"""
        h, w = 240, 320
        radius = 18
        rng = np.random.default_rng(0)
        # 背景约 200 灰阶 + 轻微噪声，类器官约 140 灰阶（更暗）
        image = (200 + rng.integers(-6, 7, size=(h, w))).astype(np.int16)
        labels = np.zeros((h, w), dtype=np.uint16)
        yy, xx = np.mgrid[0:h, 0:w]

        for i in range(n_blobs):
            label = i + 1
            center_r, center_c = 70 + (i % 2) * 100, 50 + i * 80
            if touch_border and i == 0:
                center_c = 0
            if tiny_area is not None and i == 0:
                # 用小矩形保证面积精确等于 tiny_area（边长取上取整后按行填满）
                side = int(np.ceil(np.sqrt(tiny_area)))
                block_r = np.repeat(np.arange(side), side)[:tiny_area] + center_r
                block_c = np.tile(np.arange(side), side)[:tiny_area] + center_c
                labels[block_r, block_c] = label
                image[block_r, block_c] = 140 + rng.integers(-5, 6, size=tiny_area)
                continue
            blob = (yy - center_r) ** 2 + (xx - center_c) ** 2 <= radius**2
            labels[blob] = label
            image[blob] = 140 + rng.integers(-5, 6, size=int(blob.sum()))

        return np.clip(image, 0, 255).astype(np.uint8), labels

    return _make


def _disk_labels(shape: tuple[int, int], blobs: list[tuple[int, int]], radius: int | list[int]) -> np.ndarray:
    """在空白画布上按顺序画实心圆，返回 uint16 标签图（第 i 个圆 = 标签 i+1）。

    radius 可以是单个半径，也可以逐圆给（合并出的实例要更大一圈）。
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    radii = [radius] * len(blobs) if isinstance(radius, int) else list(radius)
    labels = np.zeros(shape, dtype=np.uint16)
    for label, ((center_r, center_c), r) in enumerate(zip(blobs, radii), start=1):
        labels[(yy - center_r) ** 2 + (xx - center_c) ** 2 <= r**2] = label
    return labels


def _shift_labels(labels: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """把标签图整体平移 (dy, dx)，越界部分直接丢弃（模拟掩码相对原图整体错位）。

    dy > 0 表示内容向下移、dy < 0 表示向上移，与 `instances.align_mask_to_image` 的偏移符号一致：
    掩码上移 8 行时，对齐搜索应该给出 (dy=+8) 把它推回原位。
    """
    out = np.zeros_like(labels)
    h, w = labels.shape
    # 越界部分靠切片夹掉、不留残余，模拟的是整幅掩码相对原图错位，而不是局部残缺。
    out[max(0, dy) : h - max(0, -dy), max(0, dx) : w - max(0, -dx)] = labels[
        max(0, -dy) : h - max(0, dy), max(0, -dx) : w - max(0, dx)
    ]
    return out


def _write_fov(root: Path, day: str, image: np.ndarray, labels: np.ndarray) -> None:
    """按真实数据的目录布局写一个视野：<root>/<day>/A1_多聚焦融合.png + <day>/Output/*.tif。"""
    outdir = root / day / "Output"
    outdir.mkdir(parents=True, exist_ok=True)
    rgb = np.stack([image, image, image], axis=-1)  # 真实原图是三通道 PNG
    Image.fromarray(rgb).save(root / day / f"A1{_NAME_SUFFIX}.png")
    tifffile.imwrite(outdir / f"A1{_NAME_SUFFIX}.tif", labels)


@pytest.fixture
def fake_two_frames():
    """Stage 6 用的两帧 meta：上一帧 A/B 合并成下一帧 M，C 跨帧一对一保留。

    上一帧（day0 → frame_idx 0）：A=(60,60) 与 B=(60,105) 半径 20，C=(160,220) 半径 20；
    下一帧（day3 → frame_idx 1）：M=(60,82) 半径 26 盖住 A∪B（两者的 bbox IoU 都远高于 0.05，
    质心距离也都小于 60 px，所以两者都是 M 的候选），C2 与 C 完全相同（bbox IoU = 1）。
    贪心会把 IoU=1 的 (C, C2) 与 IoU≈0.30 的 (A, M) 配成一对一，剩下的 B→M 就是一次合并。

    直接走 `inst.build_catalog` 造 meta，是为了让 bbox / 质心 / area_px 与真实 Stage 1 的列名
    和数值口径完全一致（不是手写的假表）。
    """
    shape = (240, 320)
    prev_labels = _disk_labels(shape, [(60, 60), (60, 105), (160, 220)], radius=20)
    cur_labels = _disk_labels(shape, [(60, 82), (160, 220)], radius=[26, 20])
    # 类器官比背景暗，让 build_catalog 的掩码-原图对齐搜索落在 (0, 0)
    prev_meta = inst.build_catalog(
        "day0", "A1", image=np.where(prev_labels > 0, 140, 200).astype(np.uint8), labels=prev_labels
    )["meta"]
    cur_meta = inst.build_catalog(
        "day3", "A1", image=np.where(cur_labels > 0, 140, 200).astype(np.uint8), labels=cur_labels
    )["meta"]
    return prev_meta, cur_meta


@pytest.fixture
def fake_dataset(tmp_path, make_mask_and_image):
    """伪造 Stage 1 需要的 DATA_ROOT 结构：day0 / day3 各一个 A1 视野（原图 + 掩码）。

    目录布局与真实数据一致：<root>/<day>/A1_多聚焦融合.png（RGB uint8）与
    <root>/<day>/Output/A1_多聚焦融合.tif（uint16 标签图）。落 tmp_path 是为了不依赖 1.2 GB 真实数据。
    注意调用方要 monkeypatch `cfg.DATA_ROOT`，所以路径必须能在运行时被替换。
    """
    root = tmp_path / "raw"
    for day in ("day0", "day3"):
        image, labels = make_mask_and_image(n_blobs=2)
        _write_fov(root, day, image, labels)
    return root


@pytest.fixture
def fake_dataset_boundary_shift(tmp_path, make_mask_and_image):
    """掩码相对原图整体上移正好 `cfg.ALIGN_MAX_SHIFT_PX` 行。

    这样对齐搜索的最优解恰好落在 ±上限的**边界**上（搜索半径就是上限），用来验证 run_stage1
    把边界解判定为"需人工确认"，而不是靠注入偏移的 monkeypatch 走假路径。
    """
    root = tmp_path / "raw_boundary"
    shift = int(cfg.ALIGN_MAX_SHIFT_PX)
    for day in ("day0", "day3"):
        image, labels = make_mask_and_image(n_blobs=2)
        _write_fov(root, day, image, _shift_labels(labels, -shift, 0))
    return root


@pytest.fixture
def fake_flat_dataset(tmp_path, make_mask_and_image):
    """伪造「扁平」输入布局：`images/A1_day0.png` + `masks/A1_day0.tif`（`D:\\Datasets\\test` 的实际形态）。

    与 `fake_dataset` 的区别只有目录结构：天数与视野写在文件名里（`<FOV>_day<N>`），
    而不是 `<root>/<day>/`。命令行 `--images`/`--masks` 指向这两个目录。
    """
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for day in ("day0", "day3"):
        image, labels = make_mask_and_image(n_blobs=2)
        rgb = np.stack([image, image, image], axis=-1)  # 与原图一样是三通道
        Image.fromarray(rgb).save(images / f"A1_{day}.png")
        tifffile.imwrite(masks / f"A1_{day}.tif", labels)
    return {"images": images, "masks": masks}
