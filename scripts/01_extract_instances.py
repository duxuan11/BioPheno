"""Stage 1 薄 CLI：落盘实例目录与汇总表。

    uv run python scripts/01_extract_instances.py \
        --images D:/Datasets/test/images --masks D:/Datasets/test/masks --output D:/tmp/spot_out

不传 `--images/--masks` 时按 `config.DATA_ROOT` 的旧约定遍历全部天数 × 视野。
"""

import argparse
import sys
from pathlib import Path

# scripts/ 没有 __init__.py，直接跑脚本时把项目根目录加进 sys.path 才能 import biopheno
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biopheno import cli
from biopheno.instances import run_stage1


def build_parser() -> argparse.ArgumentParser:
    """命令行接口：`--images` / `--masks` / `--output`（与 run_all.py 完全一致）。"""
    parser = argparse.ArgumentParser(
        prog="01_extract_instances.py",
        description="Stage 1：掩码 → 实例目录（原图/掩码可由 --images/--masks 指定）",
    )
    cli.add_input_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数 → 写 config → 跑 Stage 1；返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    cli.configure_from_args(parser, args)
    run_stage1()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
