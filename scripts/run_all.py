"""Stage 7 一键运行薄 CLI：Stage1→2→3→4→5→6→6b→静态图→时间图→报告。

    # 旧约定（config.DATA_ROOT 下的 <day>/<FOV>_多聚焦融合.png 布局）
    uv run python scripts/run_all.py

    # 命令行指定输入：原图目录 + 掩码目录（文件名承载天数与视野，如 A1_day0.png / A1_day0.tif）
    uv run python scripts/run_all.py --images D:/Datasets/test/images --masks D:/Datasets/test/masks

`--output` 可以把产物写到别的目录（默认 `output`），换数据测试时不会覆盖上一次的结果。
任一阶段失败都会带着非零退出码停下（异常原样抛出），逐阶段日志留在 `<输出根>/logs/`。
真实数据实测 **25.2 分钟**（2026-10-22，9 个视野；Stage 1 ≈11.6 min、Stage 2 ≈12.3 min，共 1.2 GB 图）。
换更慢的机器（或冷缓存/机械硬盘）请预留到 ~1 小时；具体耗时看 `output/logs/run_all_timings.json`。
"""

import argparse
import sys
from pathlib import Path

# scripts/ 没有 __init__.py，直接跑脚本时把项目根目录加进 sys.path 才能 import biopheno
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biopheno import cli
from biopheno.pipeline import run_all


def build_parser() -> argparse.ArgumentParser:
    """命令行接口：`--images` / `--masks` / `--output`（见 `biopheno.cli`）。"""
    parser = argparse.ArgumentParser(
        prog="run_all.py",
        description="SPOT 复现一键运行（Stage1→7 + 报告）；不传 --images/--masks 时走 config.DATA_ROOT 的旧约定",
    )
    cli.add_input_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数 → 写 config → 跑完整条流水线；返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    cli.configure_from_args(parser, args)
    report_path = run_all()
    print(f"报告已写出：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
