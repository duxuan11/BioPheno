"""pytest 的项目根配置：把项目根目录放进 sys.path。

为什么需要：`tests/` 里的用例 `import biopheno...`，而在项目根直接跑
`uv run pytest tests` 时，pytest 的 rootdir 虽然就是这里，但并不保证把根目录加进 sys.path
（尤其是加了 `tests/__init__.py` 之后改用包路径导入时）。显式插入一行，任何启动方式都能 import。
"""

import sys
from pathlib import Path

# 本文件所在目录 = 项目根（BioPheno/），其中既有 biopheno/ 也有内联的 SPOT/
sys.path.insert(0, str(Path(__file__).resolve().parent))
