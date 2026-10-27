# BioPheno

**BioPheno = 类器官影像的 SAM 表型组分析流水线。** 输入「原图 + 分割掩码」，一条命令跑到底：
实例 → SAM 特征（形状 / 外观 / 运动）→ 预处理 → 特征模块 → UMAP/k-means 形态景观 → 跨天配准与
tracklet → 时序统计 → 图表 → 文本报告。

分析框架来自论文 *Identifying phenotype-genotype-function coupling in 3D organoid imaging using SPOT*
（Nat. Comms, 2026），代码基于 SPOT 库（SAM-SPOT 2.0.1）；设计文档见
`docs/superpowers/specs/2026-10-22-spot-organoid-repro-design.md`，实施计划见
`docs/superpowers/plans/2026-10-22-spot-organoid-repro.md`。

本项目**自包含**：运行所需的 SPOT 库源码内联在 `SPOT/`，不需要另外安装 `SAM-SPOT`。

---

## 目录结构

```
BioPheno/
├─ README.md                  ← 本文件
├─ pyproject.toml             ← 项目与依赖定义（uv / pip 都读它）
├─ uv.lock                    ← 锁定版本（uv 专用，135 个包，保证每次装出来一致）
├─ requirements.txt           ← 运行时依赖（pip 写法，与 pyproject 同一份清单）
├─ requirements-dev.txt       ← 开发/测试依赖（运行时 + pytest）
├─ conftest.py                ← pytest 用：把项目根放进 sys.path
├─ ruff.toml                  ← 代码风格（行宽 120，排除 docs/）
├─ biopheno/                  ← 流水线代码（本项目的主体）
│   ├─ config.py              ←   全部固定参数（阈值、随机种子、路径约定），每个常量上方都有出处说明
│   ├─ inputs.py / cli.py     ←   输入发现（原图+掩码配对、天数→帧号）与命令行接口
│   ├─ instances.py           ←   Stage 1：掩码 → 实例（轮廓、属性、代表图）
│   ├─ features.py            ←   Stage 2：SAM 形状/外观/运动特征
│   ├─ preprocess.py          ←   Stage 3：归一化、等权、去混杂、降维
│   ├─ modules.py             ←   Stage 4：SAM 模块（共变特征的层次聚类）
│   ├─ landscape.py           ←   Stage 5：UMAP + k-means 切表型簇，再按原生离心率给簇编号
│   ├─ temporal.py            ←   Stage 6：跨天配准与 tracklet 匹配
│   ├─ temporal_stats.py      ←   Stage 6b：簇占比、转移矩阵、表型轨迹
│   ├─ figures.py             ←   Stage 7：静态图 F1–F6b + 时间图 F7–F11（含 figures/README.md）
│   ├─ report.py              ←   report.md：每个数字的来源与口径、局限
│   └─ pipeline.py            ←   一键运行：按固定顺序串起所有阶段，逐阶段写日志
├─ SPOT/                      ← 内联（vendor）的 SPOT 库源码，供 `import SPOT...` 使用
├─ scripts/                   ← 可单独执行的入口脚本（run_all.py 与三个核验脚本）
├─ tests/                     ← pytest 用例（合成数据，不依赖真实数据）
├─ docs/                      ← 设计文档与实施计划
└─ output/                    ← 产物根（默认，已 gitignore；可用 --output 改到别处）
```

## 环境要求

- **Python 3.10 – 3.13**（开发与验证用的是 3.11 / 3.12）
- 依赖清单以 `pyproject.toml` 为准，`requirements*.txt` 是同一份清单的 pip 写法
- 只用 CPU 即可跑完全流程；`torch` 只在少数 GPU 版特征函数与预训练网络里用到

**推荐 uv（会自动建 `.venv` 并装好依赖）：**

```bash
cd D:/Code/myself/BioPheno
uv sync                 # 装运行时 + dev（pytest），严格按 uv.lock 的锁定版本
uv sync --no-dev        # 只要运行时（不装 pytest）
```

**或者 pip：**

```bash
cd D:/Code/myself/BioPheno
python -m venv .venv
.venv/Scripts/activate        # Windows；Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
```

> Windows 上 pip 可能现场编译 `python-annoy`（PaCMAP 需要它）。卡住的话先
> `conda install -c conda-forge python-annoy`，或直接用 uv。

## 快速开始

数据是「原图目录 + 掩码目录」时（文件名里带天数，如 `A1_day0.png` / `A1_day0.tif`）：

```bash
uv run python scripts/run_all.py \
    --images D:/Datasets/test/images \
    --masks  D:/Datasets/test/masks \
    --output output/test_3frames
```

不带参数时沿用 `config.DATA_ROOT` 下的**旧目录约定**：`<day>/<FOV>_多聚焦融合.png` +
`<day>/Output/<FOV>_多聚焦融合.tif`（若这些路径已不存在，报错会提示改用 `--images/--masks`）。

| 参数 | 作用 |
| --- | --- |
| `--images 目录` | 原始图像目录（只扫顶层，不递归；收 png/tif/tiff/jpg/jpeg/bmp） |
| `--masks 目录` | 分割掩码目录，与原图同名 stem 配对 |
| `--output 目录` | 产物根目录（默认 `output`；换数据测试时用它避免覆盖上一次结果） |

`--images` 与 `--masks` 必须成对出现：原图与掩码一一配对，只给一个无法判断该用哪张。

## 文件命名与配对规则

- 文件名里必须有 day 记号：`day0`、`D3`、`day_5`（大小写不敏感，前后要有 `_ - . 空格` 或字符串边界）；
- 记号以外的部分是**视野名**：`A1_day0` → FOV `A1`，`day3_A2` → FOV `A2`；视野名不能为空；
- 掩码按 **同名 stem** 配对（扩展名各自独立），并容忍 `_mask` / `_masks` / `_seg` / `_segmentation` /
  `_label` / `_labels` 后缀，例如 `A1_day0.png` ↔ `A1_day0_mask.tif`；
- 天数按**数字**排序（`day10` 在 `day3` 之后）后压成连续帧号 `0..N-1`（SPOT 的轨迹函数只认连续整数帧号），
  时间轴按 `天数 × config.HOURS_PER_DAY`（默认 24 h）换算；
- 原图找不到掩码 → **直接报错**（静默跳过会少算一帧且报告看不出来）；掩码多出来的文件 → 忽略但会提示；
- 两个文件名解析成同一个 (天数, 视野) → 报错（否则后一个会静默覆盖前一个）。

## 处理流程（Stage 1 → 7）

| 阶段 | 代码 | 做什么 |
| --- | --- | --- |
| Stage 1 | `instances.py` | 每个 (天, 视野) 的掩码 → 实例：等弧长重采样轮廓、面积/离心率等属性、代表图，落 `instances/` |
| Stage 2 | `features.py` | 用 SPOT 算 SAM 特征：形状、掩码内外观、光流运动，落 `features/` |
| Stage 3 | `preprocess.py` | 剔除高方差/非有限列 → power transform + 标准化 → SAM 集合与特征族等权 → 去混杂 → ECT 降维，落 `preprocessed/` |
| Stage 4 | `modules.py` | SAM 模块：共变特征的层次聚类，给出模块标签与模块表达，落 `modules/` |
| Stage 5 | `landscape.py` | UMAP 嵌入 + k-means（K 由 elbow/adjusted inertia 选）**切表型簇** → 再按簇内平均**原生**离心率给簇**编号** 1..K（离心率只决定编号顺序，不参与划分），落 `analysis/` |
| Stage 6 | `temporal.py` | 跨天配准（对齐标签图）+ tracklet 匹配（bbox IoU + 质心距离），落 `analysis/` |
| Stage 6b | `temporal_stats.py` | 簇占比、转移矩阵、表型轨迹，落 `analysis/` |
| Stage 7 | `figures.py` + `report.py` | 静态图 F1–F6b、时间图 F7–F11、`figures/README.md`、`report.md` |

一键运行时（`pipeline.py`）的约定：**顺序固定**、**逐阶段写日志**、**失败就停**（后面的阶段一个都不跑，
也不写报告），每阶段耗时写进 `logs/run_all_timings.json`。

> **关于 Stage 5 的簇编号**（易误读）：簇的构成完全由「Stage 3 预处理矩阵 → UMAP 2D → k-means」决定，
> 离心率不参与划分；它只用来给**已经切好的**簇排编号（1 = 簇内平均偏心率最低、最圆；K = 最扁长），
> 以对齐论文 Fig. 3 的口径。另外 `moment_eccentricity` 本身也是 Stage 3 特征矩阵里的一个 shape 特征，
> 因此它**间接**影响嵌入与聚类，但没有被当成划分依据、也没有被单独加权。
> `cluster_labels.csv` 里 `cluster` 是 1-based 编号、`cluster_raw` 是 k-means 原始 0-based 标签，
> 两者的映射与排序 QC 都在 `kmeans_metrics.json`。

## 单阶段脚本

| 脚本 | 参数 |
| --- | --- |
| `scripts/run_all.py` | `--images/--masks/--output`（Stage1→7 + 报告） |
| `scripts/01_extract_instances.py` | `--images/--masks/--output`（只跑 Stage 1） |
| `scripts/check_registration_geometry.py` | `--images/--masks/--output`（配准几何核验，读 Stage1 的产物 + 掩码） |
| `scripts/check_umap_repeat.py` | 只需 `--output`（UMAP 重跑核验，只读产物） |

## 产物

```
<output>/
├─ instances/{day}/{fov}/{instances.csv, contours.npz, patches.npz} + all_instances.csv + stage1_qc.json
├─ features/      sam_shape.npz, sam_appearance.npz, all_features.csv, feature_names.json, stage2_qc.json
├─ preprocessed/  X.npy, std_pow_transformer.joblib, keep_features.json, stage3_qc.json
├─ modules/       模块标签、模块表达、stage4_qc.json
├─ analysis/      umap_2d.csv, cluster_labels.csv, tracks.csv, matches.csv, *.npz, *_qc.json
├─ figures/       F1–F6b（静态）+ F7–F11（时间）+ README.md（每张图是干什么/怎么读，自动生成）
├─ logs/          逐阶段日志、run_all_timings.json（有核验时还有两份 check json）
└─ report.md      全部数字的来源与口径、图索引、局限
```

`figures/README.md` 由 Stage 7 出图时自动重写（`biopheno.figures.write_figure_readme`）：13 张图逐条
写"是什么 / 怎么读 / 依赖哪些上游产物"，并按目录里实际存在的文件标"已生成 / 缺失"——看图前先读它。

## 性能

3 视野 / 2423 实例的实测（同一台 28 逻辑核机器、同一个 Python 环境）：

| 阶段 | 优化前 | 优化后 |
| --- | --- | --- |
| Stage 1 实例 | 255.0 s | **8.6 s** |
| Stage 2 SAM 特征 | 259.4 s | **21.7 s** |
| Stage 3–7 + 报告 | 48.2 s | 48–66 s（与优化无关，受磁盘缓存影响） |
| **合计** | **562.7 s（9.4 min）** | **96.1 s（1.6 min）** |

三处关键优化（数值结果与优化前**逐位相同**）：

1. **Stage 2 复用 Stage 1 的产物**：不再重算对齐/轮廓/代表图，直接 `load_catalog` + 读原图
   （原实现每个视野多花 ~78 s 重复 Stage 1 的活）；
2. **轮廓提取按标签包围盒裁剪**：只在 ~50×50 的窗口里跑 `find_contours`，
   而不是每个标签都对 15 MP 整图做一次 `labels == k`（83.9 s → 0.2 s/视野）；
3. **对齐搜索改“FFT 粗筛 + 原写法精确重算”**：一次互相关给出 289 个平移的候选排序，
   只对靠前的候选做全图均值（21 s → 1.7 s/视野）；meta 的像素坐标改在包围盒内取（27 s → 0.02 s）。

剩下的耗时主要在 Stage 2 的特征计算（shape 5.6 s + appearance 1.3 s/视野）与 Stage 3–7 的固定开销；
要进一步跑快得靠**跨视野并行**（各 (天, 视野) 相互独立）。

## 测试

```bash
uv run pytest tests -q          # 或 python -m pytest tests -q
```

测试全部用 `tmp_path` 里的合成数据（240×320 的小掩码），不依赖真实数据与已跑完的产物；
输入发现与命令行参数的用例在 `tests/test_inputs.py`，各阶段 `*_qc.json` 的口径由对应的 `tests/test_*.py` 守。

## 代码约定

- **注释与文档字符串一律中文**，专有名词（SAM、UMAP、PaCMAP、SIFT、ECT、bbox IoU…）与函数名保持原文；
- 所有"论文写死"或"数据事实"决定的常量集中在 `biopheno/config.py`，每个常量上方注明它是什么、出自何处；
- 随机性统一用 `config.RANDOM_STATE`，保证图和聚类可离线复现；
- 新增阶段或改口径时，同步更新 `report.md` 的口径说明与 `figures/README.md` 的图说明（两者都是自动生成的）。

## 许可

本项目的流水线代码用于论文复现；内联的 `SPOT/` 库沿用上游 **LICR Software License**
（学术非营利用途，详见 `D:/Code/myself/SPOT` 仓库中的许可文件）。
