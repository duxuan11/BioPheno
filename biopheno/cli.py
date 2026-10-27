"""命令行输入参数（`--images` / `--masks` / `--output`）：`run_all.py` 与 `01_extract_instances.py` 共用。

不传这三个参数时行为与原来完全一致（走 `config.DATA_ROOT` 的旧目录约定），所以既有的
`uv run python scripts/run_all.py` 调用方式不会失效。
"""

import argparse
from pathlib import Path

from biopheno import config as cfg
from biopheno import inputs as input_mod


def add_input_args(parser: argparse.ArgumentParser) -> None:
    """把输入/输出参数挂到脚本的 parser 上（两个脚本共用，保证口径一致）。"""
    parser.add_argument(
        "--images",
        type=Path,
        metavar="目录",
        help="原始图像目录，文件名需含天数记号（如 images/A1_day0.png）；与 --masks 同时给出",
    )
    parser.add_argument(
        "--masks",
        type=Path,
        metavar="目录",
        help="分割掩码目录，与原图同名（如 masks/A1_day0.tif）；与 --images 同时给出",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="目录",
        help="产物根目录（实例/特征/图/报告都写在这里，默认 output）",
    )


def configure_from_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> input_mod.Dataset | None:
    """按命令行参数写好 `config`，并把"读的是哪两个目录、认出几个采样点"打到控制台。

    - 两个输入目录都没给：返回 None（保持旧约定）；只给了 `--output` 就只改输出根。
    - 只给其中一个：`parser.error`（退出码 2）——原图与掩码必须成对，猜不得。
    """
    if args.images is None and args.masks is None:
        if args.output is not None:
            cfg.OUTPUT_ROOT = Path(args.output)
            print(f"输出根 → {cfg.OUTPUT_ROOT}（输入仍按 config.DATA_ROOT 的旧约定）")
        return None
    if args.images is None or args.masks is None:
        parser.error("--images 与 --masks 必须同时给出：原图与掩码要一一配对，只给一个无法配对")

    data = input_mod.configure(args.images, args.masks, output_root=args.output)
    print(
        f"输入：原图目录 {cfg.IMAGE_DIR}\n"
        f"      掩码目录 {cfg.MASK_DIR}\n"
        f"认出 {len(data.frames)} 个采样点：{len(cfg.DAYS)} 个时间点 × {len(cfg.FOVS)} 个视野"
        f"（{'、'.join(cfg.FOVS)}），天数 → 帧号 {cfg.DAYS}，时间轴 {cfg.TIME_HOURS}\n"
        f"输出根 → {cfg.OUTPUT_ROOT}"
    )
    if data.unmatched_masks:
        print(
            f"注意：掩码目录里有 {len(data.unmatched_masks)} 个文件没有对应原图，已忽略"
            f"（例如 {data.unmatched_masks[0].name}）"
        )
    return data
