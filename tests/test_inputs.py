"""命令行输入（`--images` / `--masks` / `--output`）的发现、配对与写 config 全链测试。

背景：Stage 1/2 原来只会按 `<DATA_ROOT>/<day>/<FOV>_多聚焦融合.png` 与
`<DATA_ROOT>/<day>/Output/<FOV>_多聚焦融合.tif` 两条写死的路径读数据。真实数据换成
`images/<FOV>_day<N>.png` + `masks/<FOV>_day<N>.tif` 之后，必须能在命令行上直接指目录，
由文件名推出 (day, FOV)、排好帧轴，并让下游（Stage 1/2/6 与报告）都跟着这份清单走。
"""

import argparse

import numpy as np
import pytest

from biopheno import cli
from biopheno import config as cfg
from biopheno import inputs as inp
from biopheno import instances as inst

# configure() 会写的 config 全局；测试先用 monkeypatch 记一份原值，测完自动还原。
_CONFIG_KEYS = ("DAYS", "FOVS", "TIME_HOURS", "INPUT_FILES", "IMAGE_DIR", "MASK_DIR", "OUTPUT_ROOT")


def _arm(monkeypatch):
    """让 monkeypatch 记住这几个 config 全局的原值（之后 configure() 直接赋值也能被还原）。"""
    for name in _CONFIG_KEYS:
        monkeypatch.setattr(cfg, name, getattr(cfg, name))


def _touch_pairs(tmp_path, stems, image_ext=".png", mask_ext=".tif"):
    """造一对空文件目录（discover 只列路径、不读内容，所以零字节足够）。"""
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for stem in stems:
        (images / f"{stem}{image_ext}").write_bytes(b"")
        (masks / f"{stem}{mask_ext}").write_bytes(b"")
    return images, masks


# ---------------------------------------------------------------- 发现与配对


def test_discover_pairs_images_with_masks_by_stem(fake_flat_dataset):
    """守配对：按同名 stem 配 images/masks，并由文件名推出 (day, fov) 与 day_number。"""
    data = inp.discover(fake_flat_dataset["images"], fake_flat_dataset["masks"])
    assert [(frame.day, frame.fov) for frame in data.frames] == [("day0", "A1"), ("day3", "A1")]
    assert data.frames[0].image.name == "A1_day0.png"
    assert data.frames[0].mask.name == "A1_day0.tif"
    # day_number 由文件名里的 day<N> 解析而来，供帧轴与 time_hours 使用
    assert data.frames[0].day_number == 0
    assert data.frames[1].day_number == 3


def test_discover_orders_days_numerically_not_lexically(tmp_path):
    """守排序：帧轴按天数数值递增（day0, day3, day10），不是字典序。"""
    # 字符串排序会给出 day0/day10/day3；帧轴必须按真实天数递增
    images, masks = _touch_pairs(tmp_path, ["A1_day0", "A1_day10", "A1_day3"])
    assert [frame.day for frame in inp.discover(images, masks).frames] == ["day0", "day3", "day10"]


def test_discover_handles_multiple_fovs_and_days(tmp_path):
    """守排序：多视野多天数时按 (day, fov) 升序铺开帧轴，与文件枚举顺序无关。"""
    images, masks = _touch_pairs(tmp_path, ["A2_day3", "A1_day0", "A1_day3", "A2_day0"])
    assert [(frame.day, frame.fov) for frame in inp.discover(images, masks).frames] == [
        ("day0", "A1"),
        ("day0", "A2"),
        ("day3", "A1"),
        ("day3", "A2"),
    ]


def test_discover_reports_image_without_mask(fake_flat_dataset):
    """守报错：有原图没掩码时抛 FileNotFoundError，且报错里点名那个 stem。"""
    (fake_flat_dataset["masks"] / "A1_day3.tif").unlink()
    with pytest.raises(FileNotFoundError, match="A1_day3"):
        inp.discover(fake_flat_dataset["images"], fake_flat_dataset["masks"])


def test_discover_rejects_stem_without_day_token(tmp_path):
    """守命名：文件名里没有 day 记号时抛 ValueError——推不出帧轴就不许往下走。"""
    images, masks = _touch_pairs(tmp_path, ["A1"])
    with pytest.raises(ValueError, match="day"):
        inp.discover(images, masks)


def test_discover_rejects_stem_without_fov(tmp_path):
    """只有天数、没有视野名（如 `day0.png`）时拒绝：uid 与出图都按 fov 组织，空视野名会到处留下 `day0/`。"""
    images, masks = _touch_pairs(tmp_path, ["day0"])
    with pytest.raises(ValueError, match="视野"):
        inp.discover(images, masks)


def test_discover_ignores_non_image_files(tmp_path):
    """守发现：非图像文件忽略，但有掩码没原图的文件要记进 unmatched_masks 供人工核对。"""
    images, masks = _touch_pairs(tmp_path, ["A1_day0"])
    (images / "notes.txt").write_text("不是图", encoding="utf-8")
    (masks / "Thumbs.db").write_bytes(b"")  # 不是图像扩展名：当垃圾文件忽略，连提醒都不需要
    # 认得出是掩码、但没有对应原图的文件：不影响配对，但要记进清单供人工核对
    (masks / "B2_day0.tif").write_bytes(b"")
    data = inp.discover(images, masks)
    assert [frame.day for frame in data.frames] == ["day0"]
    assert [path.name for path in data.unmatched_masks] == ["B2_day0.tif"]


def test_discover_accepts_mask_suffix_variants(tmp_path):
    """守命名：掩码带 _mask / _seg 后缀时仍能与原图配对。"""
    # 常见的另一种命名：掩码带 _mask / _seg 后缀
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    (images / "A1_day0.png").write_bytes(b"")
    (masks / "A1_day0_mask.tif").write_bytes(b"")
    data = inp.discover(images, masks)
    assert data.frames[0].mask.name == "A1_day0_mask.tif"


def test_discover_rejects_directory_as_input(tmp_path, fake_flat_dataset):
    """守入口：--images 指到文件而不是目录时抛 NotADirectoryError，不静默给出空清单。"""
    with pytest.raises(NotADirectoryError, match="目录"):
        inp.discover(fake_flat_dataset["images"] / "A1_day0.png", fake_flat_dataset["masks"])


# ---------------------------------------------------------------- 写 config


def test_configure_sets_days_fovs_time_hours_and_input_files(monkeypatch, fake_flat_dataset):
    """守写 config：DAYS 压成连续帧号、TIME_HOURS=天数×24、INPUT_FILES 是 (day, fov) 清单。"""
    _arm(monkeypatch)
    inp.configure(fake_flat_dataset["images"], fake_flat_dataset["masks"])
    assert cfg.DAYS == {"day0": 0, "day3": 1}  # 真实天数被压成连续帧号（SPOT 的硬要求）
    assert cfg.FOVS == ("A1",)
    assert cfg.TIME_HOURS == {"day0": 0.0, "day3": 72.0}  # 天数 × 24h
    assert set(cfg.INPUT_FILES) == {("day0", "A1"), ("day3", "A1")}
    assert cfg.IMAGE_DIR == fake_flat_dataset["images"]
    assert cfg.MASK_DIR == fake_flat_dataset["masks"]


def test_configured_paths_win_over_legacy_data_root(monkeypatch, fake_flat_dataset, tmp_path):
    """读图/读掩码必须走命令行给的路径，不再拼 `<DATA_ROOT>/<day>/...` 的旧约定。"""
    _arm(monkeypatch)
    inp.configure(fake_flat_dataset["images"], fake_flat_dataset["masks"])
    monkeypatch.setattr(cfg, "DATA_ROOT", tmp_path / "nope")  # 旧约定在这里根本不存在
    assert inst._read_image("day3", "A1").shape[-1] == 3
    assert int(inst._read_labels("day3", "A1").max()) == 2


def test_configure_sets_output_root(monkeypatch, fake_flat_dataset, tmp_path):
    """守写 config：给了 --output 时覆盖 cfg.OUTPUT_ROOT。"""
    _arm(monkeypatch)
    out = tmp_path / "out"
    inp.configure(fake_flat_dataset["images"], fake_flat_dataset["masks"], output_root=out)
    assert cfg.OUTPUT_ROOT == out


def test_configure_without_output_root_keeps_current_root(monkeypatch, fake_flat_dataset):
    """守默认：不给 --output 时保留原有 OUTPUT_ROOT，不被 configure 清空或改成 DATA_ROOT。"""
    _arm(monkeypatch)
    before = cfg.OUTPUT_ROOT
    inp.configure(fake_flat_dataset["images"], fake_flat_dataset["masks"])
    assert cfg.OUTPUT_ROOT == before


def test_run_stage1_uses_configured_input_files(monkeypatch, fake_flat_dataset, tmp_path):
    """守下游跟随：Stage1 只跑 INPUT_FILES 里的帧，产物按新布局落到 OUTPUT_ROOT 下。"""
    _arm(monkeypatch)
    out = tmp_path / "out"
    inp.configure(fake_flat_dataset["images"], fake_flat_dataset["masks"], output_root=out)
    meta = inst.run_stage1()
    assert set(meta["day"]) == {"day0", "day3"}
    # 真实天数被压成连续帧号 0/1（SPOT 的硬要求），Stage1 的产物目录也跟着这份清单走
    assert list(meta["frame_idx"].unique()) == [0, 1]
    assert (out / "instances" / "day3" / "A1" / "instances.csv").exists()
    assert (out / "instances" / "all_instances.csv").exists()


def test_available_frames_follows_input_files(monkeypatch, fake_flat_dataset):
    """INPUT_FILES 已给出时，帧清单以它为准，不再假设 DAYS × FOVS 的笛卡尔积齐全。"""
    _arm(monkeypatch)
    inp.configure(fake_flat_dataset["images"], fake_flat_dataset["masks"])
    monkeypatch.setattr(cfg, "FOVS", ("A1", "A2"))  # 上游多声明了一个没有数据的视野
    assert inst.available_frames() == [("day0", "A1"), ("day3", "A1")]


def test_read_labels_supports_png_masks(monkeypatch, tmp_path):
    """掩码目录允许 .png：读的时候得按扩展名选 reader（tifffile 读不了 PNG）。"""
    from skimage import io as skio

    _arm(monkeypatch)
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    labels = np.zeros((20, 20), dtype=np.uint16)
    labels[5:10, 5:10] = 3
    skio.imsave(images / "A1_day0.png", np.zeros((20, 20, 3), dtype=np.uint8), check_contrast=False)
    skio.imsave(masks / "A1_day0.png", labels, check_contrast=False)  # 16 位 PNG 标量标签
    inp.configure(images, masks)
    back = inst._read_labels("day0", "A1")
    assert back.dtype == np.uint16
    assert int(back.max()) == 3


def test_missing_legacy_files_hint_the_cli_options(monkeypatch, tmp_path):
    """没有给 --images/--masks、旧约定路径又不存在时，报错要直接指向命令行开关。

    数据换成扁平布局（images/ + masks/）之后，不带参数的 `run_all.py` 会按旧约定去
    `<DATA_ROOT>/<day>/<FOV>_多聚焦融合.png` 找图——只抛一个裸的 io.imread 报错，人根本看不出
    该传什么开关。
    """
    _arm(monkeypatch)
    monkeypatch.setattr(cfg, "DATA_ROOT", tmp_path / "empty")
    monkeypatch.setattr(cfg, "INPUT_FILES", {})
    with pytest.raises(FileNotFoundError, match="--images"):
        inst._read_image("day0", "A1")


def test_available_frames_without_input_files_uses_day_fov_product(monkeypatch):
    """守回退：没有 INPUT_FILES 时退回 DAYS × FOVS 的笛卡尔积，旧布局行为不变。"""
    _arm(monkeypatch)
    monkeypatch.setattr(cfg, "DAYS", {"day0": 0, "day3": 1})
    monkeypatch.setattr(cfg, "FOVS", ("A1", "A2"))
    monkeypatch.setattr(cfg, "INPUT_FILES", {})
    assert inst.available_frames() == [("day0", "A1"), ("day0", "A2"), ("day3", "A1"), ("day3", "A2")]


def test_temporal_stats_time_bins_follow_current_days(monkeypatch):
    """时间箱边界必须按**当前** cfg.DAYS 的长度算：写死 len(DAYS)+1 会在两天数据上多出一个空箱。"""
    from biopheno import temporal_stats as ts

    _arm(monkeypatch)
    monkeypatch.setattr(cfg, "DAYS", {"day0": 0, "day3": 1})
    assert ts.time_intervals() == (0.0, 1.0, 2.0)


# ---------------------------------------------------------------- 命令行


def _parser() -> argparse.ArgumentParser:
    """返回只挂了 --images/--masks/--output 三个开关的 run_all 解析器。"""
    parser = argparse.ArgumentParser(prog="run_all")
    cli.add_input_args(parser)
    return parser


def test_cli_parser_accepts_images_masks_output():
    """守命令行：三个开关都能解析成 Path，且取到的是目录名而非整串。"""
    args = _parser().parse_args(["--images", "in/images", "--masks", "in/masks", "--output", "out"])
    assert args.images.name == "images"
    assert args.masks.name == "masks"
    assert args.output.name == "out"


def test_cli_requires_images_and_masks_together(fake_flat_dataset):
    """守用法：只给 --images 不给 --masks 时 argparse 直接退出（error 的退出码 2）。"""
    parser = _parser()
    with pytest.raises(SystemExit):  # argparse.error 的退出码 2
        cli.configure_from_args(parser, parser.parse_args(["--images", str(fake_flat_dataset["images"])]))


def test_cli_configure_from_args_applies_config(monkeypatch, fake_flat_dataset, tmp_path, capsys):
    """守全链：configure_from_args 把命令行落到 config，并把读的目录与采样点数打到控制台。"""
    _arm(monkeypatch)
    out = tmp_path / "out"
    parser = _parser()
    args = parser.parse_args(
        [
            "--images",
            str(fake_flat_dataset["images"]),
            "--masks",
            str(fake_flat_dataset["masks"]),
            "--output",
            str(out),
        ]
    )
    data = cli.configure_from_args(parser, args)
    assert cfg.DAYS == {"day0": 0, "day3": 1}
    assert cfg.OUTPUT_ROOT == out
    assert data is not None
    # 让人在长跑前就能从控制台看到\"读的是哪两个目录、认出几个采样点\"
    printed = capsys.readouterr().out
    assert str(fake_flat_dataset["images"]) in printed
    assert "2" in printed


def test_cli_without_args_keeps_legacy_layout(monkeypatch):
    """守兼容：不带参数时不 configure，INPUT_FILES 保持为空、继续走旧约定布局。"""
    _arm(monkeypatch)
    parser = _parser()
    assert cli.configure_from_args(parser, parser.parse_args([])) is None
    assert cfg.INPUT_FILES == {}
