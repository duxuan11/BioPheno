# SPOT 类器官复现 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用本地 3 天 × 3 视野的已分割类器官数据，复现 SPOT 论文的静态 SAM 表型景观与跨天动态图（shape + appearance，motion 置空）。

**Architecture:** 在 `D:\Code\myself\SPOT\repro` 下新建一个薄适配层包 `biopheno`：把 label mask 变成 SPOT 需要的 `boundaries` 列表，调用 `SPOT` 库已有的特征/预处理/模块/聚类函数，产出落盘的表与图。SPOT 源码与本仓库 `pyproject.toml` 一律不改；每个阶段一个模块 + 一个薄 CLI 脚本 + 一个 `pipeline.run_all`。

**Tech Stack:** Python 3.12, uv, numpy, pandas, scikit-image, scikit-learn, umap-learn, matplotlib, pytest, ruff；`SPOT`（本地仓库，v2.0.1，`uv pip install -e .`）。

**Spec:** `D:\Code\myself\SPOT\repro\docs\superpowers\specs\2026-10-22-spot-organoid-repro-design.md`

## Global Constraints

- Python 3.12；包管理只用 uv，环境在 `D:\Code\myself\SPOT\.venv`。
- 不修改 `SPOT/` 下任何文件，不修改 `D:\Code\myself\SPOT\pyproject.toml`。
- 不引入 `pyproject.toml` 之外的依赖；测试只额外需要 `pytest`，格式只额外需要 `ruff`。
- 行长 120（`repro/ruff.toml`），格式化以 `uv run ruff format` 结果为准，不做手工审美调整。
- 关键代码处写中文注释说明"这是什么"；提交信息用中文 `feat:/fix:/test:/docs:`。
- 所有随机性固定：`random_state=0`（UMAP、k-means、模块 bootstrap）。
- 内部坐标约定：轮廓为 numpy `(row, col)` 顺序——`skimage.measure.find_contours` 与 `SPOT.Features.features.contour_to_binary` 都用这个约定；`(x, y)` 只在出图/IO 边界处转换，转换点集中在一处。
- 特征提取调用 `fast=True`（v2 默认实现），`n_jobs` 从 `config.N_JOBS` 读。
- 所有产物落 `repro/output/`，不进 git。
- 硬断言：长度 ≥ 2 的 track 数必须 > 0，否则 `RuntimeError`（见 Spec §6-Stage6-3、§9）。

## Review Focus

以下五类输入/失败模式最容易伤到使用者，各任务必须用测试钉住：

1. **掩码尺寸与原图不一致**（各图 3572–3580 × 4284–4288 各不相同）：期望是自动裁/补并对齐，而不是抛 `ValueError` 或静默错位。
2. **SPOT 的 uid 解析**（写死 `split('_')`、帧号取最后一段、`d_Frame` 标量比较）：期望是传错格式时立刻报错，而不是所有 track 静默退化成单点。
3. **单个实例的退化几何**（1–2 像素、贴边、多段轮廓）：期望是跳过并记数，而不是整批崩在某个实例上。
4. **特征矩阵含 NaN/Inf/常数列**（motion 置空列、ECT 降维后）：期望是预处理阶段剔除并记录名字，而不是让 UMAP 报错。
5. **空/单条件的时间轴**（某视野只有 1 帧有效）：期望是出图时跳过该视野并在报告里注明，而不是 IndexError。

---

### Task 1: 环境、配置与包骨架

**Files:**
- Create: `repro/ruff.toml`
- Create: `repro/conftest.py`（把 `repro/` 加进 `sys.path`，让 `biopheno` 可 import）
- Create: `repro/biopheno/__init__.py`
- Create: `repro/biopheno/config.py`
- Test: `repro/tests/test_config.py`

**Interfaces:**
- Produces: `biopheno.config` 里的全部常量（下文 `cfg` 指这个模块）：
  `DATA_ROOT: Path`, `OUTPUT_ROOT: Path`, `DAYS: dict[str,int]`（`{"day0":0,"day3":1,"day5":2}`）,
  `FOVS: tuple[str,...]`（`("A1","A2","A3")`）, `TIME_HOURS: dict[str,float]`（`{"day0":0.0,"day3":72.0,"day5":120.0"}` 见下）,
  `PIXEL_RES: float = 1.0`, `N_CONTOUR_POINTS: int = 100`, `PATCH_PAD_PX: int = 5`,
  `PATCH_SIZE: tuple[int,int] = (64,64)`, `SMALL_AREA_FLAG_PX: int = 200`, `ALIGN_MAX_SHIFT_PX: int = 8`,
  `UMAP_PARAMS: dict`, `KMEANS_K_RANGE: tuple[int,int] = (2,12)`, `VARIANCE_THRESHOLD_SIGMA: float = 2.0`,
  `ECC_N_DIM: int = 100`, `MODULE_SELECTION: str = "legacy"`,
  `MODULE_PARAMS: dict`（`max_clust_len=None, lam_smooth=1.0, buffer=30`）,
  `MIN_BBOX_IOU: float = 0.05`, `MAX_CENT_DIST_PX: float = 60.0`, `N_JOBS: int = -1`,
  `BATCH_CORRECT_FOV: bool = False`, `RANDOM_STATE: int = 0`

- [ ] **Step 1: 建环境并安装**

```bash
cd /d/Code/myself/SPOT
uv venv --python 3.12
uv pip install -e .
uv pip install pytest ruff
```

Expected: `uv pip install -e .` 成功（若 `pacmap`/`annoy` 构建失败，改用
`uv pip install -e . --no-deps` 再手动装除 pacmap/annoy 外的依赖，把这一步的实际命令记进
`repro/docs/superpowers/specs/` 的附录；`pacmap` 非必需）。

- [ ] **Step 2: 写失败测试**

```python
# repro/tests/test_config.py
import numpy as np
from biopheno import config as cfg

def test_days_and_time_axis_are_consistent():
    assert cfg.DAYS == {"day0": 0, "day3": 1, "day5": 2}
    assert [cfg.TIME_HOURS[d] for d in ("day0", "day3", "day5")] == [0.0, 72.0, 120.0]

def test_paper_parameters_are_pinned():
    assert cfg.UMAP_PARAMS == dict(n_neighbors=100, min_dist=0.5, spread=0.5,
                                   metric="euclidean", random_state=0)
    assert cfg.VARIANCE_THRESHOLD_SIGMA == 2.0
    assert cfg.MODULE_SELECTION == "legacy"
    assert cfg.PIXEL_RES == 1.0
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest repro/tests/test_config.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'biopheno'`

- [ ] **Step 4: 实现 `config.py`、`conftest.py`、`ruff.toml`**

`conftest.py` 只做一件事：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
```
`ruff.toml`：`line-length = 120`, `target-version = "py312"`。
`config.py` 里每个常量上方一行中文注释说明它是什么、来自论文哪里（含 `all_conditions` 必须传
单一常量的原因、`MODULE_SELECTION="legacy"` 的原因）。

- [ ] **Step 5: 运行测试确认通过并提交**

Run: `uv run pytest repro/tests/test_config.py -q && uv run ruff format repro && uv run ruff check repro`
Expected: PASS，ruff 无报错
```bash
git add repro/ && git commit -m "feat: 初始化 repro 包骨架与论文参数配置"
```

---

### Task 2: 掩码 → 实例目录（实例、轮廓、裁剪图）

**Files:**
- Create: `repro/biopheno/instances.py`
- Test: `repro/tests/test_instances.py`
- Test fixture: `repro/tests/conftest.py`（追加 `make_mask_and_image()`）

**Interfaces:**
- Consumes: `biopheno.config`（Task 1）
- Produces:
  - `align_mask_to_image(image: np.ndarray, labels: np.ndarray, max_shift: int) -> tuple[np.ndarray, tuple[int,int]]`
    返回（补齐到 `image.shape` 的 labels，`(dy, dx)` 偏移）；偏移由"掩码内均值 − 掩码外均值"最大化确定。
  - `extract_contours(labels: np.ndarray, n_points: int) -> tuple[dict[int, np.ndarray], dict[int, int]]`
    返回 `label -> (n_points, 2) float (row, col)` 的闭合外轮廓（等弧长重采样；取最长轮廓，丢弃 < 6 点的轮廓），
    以及 `label -> 被丢弃的候选轮廓数`。
  - `bbox_of_contour(contour: np.ndarray) -> tuple[int,int,int,int]`（`r0, c0, r1, c1`，闭区间）
  - `crop_patches(image: np.ndarray, contours: dict[int,np.ndarray], pad: int, size: tuple[int,int]) -> dict[int, dict[str, np.ndarray]]`
    返回 `label -> {"native": (h,w) uint8, "sift": (64,64) uint8}`
  - `build_catalog(day: str, fov: str, image: np.ndarray | None = None, labels: np.ndarray | None = None) -> dict`，键：
    `meta: pd.DataFrame`（列见下）、`image: np.ndarray`、`labels: np.ndarray`、`contours: dict`、`patches: dict`
    （`image`/`labels` 为 None 时才从 `cfg.DATA_ROOT/<day>/` 读文件；测试直接传入）
    meta 列：`instance_uid, day, fov, frame_idx, time_hours, label, centroid_row, centroid_col, r0, c0, r1, c1, area_px, touches_border, is_small, n_contours_discarded`
  - `instance_uid = f"{fov}_{label}_{frame_idx}"`（**帧号最后一段、分隔符 `_`**，Spec D13）
  - `save_catalog(catalog: dict, outdir: Path) -> None` / `load_catalog(outdir: Path) -> dict`

- [ ] **Step 1: 写失败测试**

```python
# repro/tests/test_instances.py
import numpy as np, pandas as pd
from biopheno import instances as inst

def test_contours_are_row_col_and_resampled(make_mask_and_image):
    image, labels = make_mask_and_image(n_blobs=3)
    contours, _ = inst.extract_contours(labels, n_points=100)
    assert len(contours) == 3
    for c in contours.values():
        assert c.shape == (100, 2)
        # 行坐标在 [0, H)、列坐标在 [0, W) —— 若把 (row,col) 和 (x,y) 弄反，列坐标会越界
        assert c[:, 0].max() < labels.shape[0] and c[:, 1].max() < labels.shape[1]

def test_mask_smaller_than_image_is_padded_not_crashing(make_mask_and_image):
    image, labels = make_mask_and_image(n_blobs=2)
    small = labels[:-3, :-5]
    aligned, (dy, dx) = inst.align_mask_to_image(image, small, max_shift=8)
    assert aligned.shape == image.shape
    assert abs(dy) <= 8 and abs(dx) <= 8

def test_build_catalog_uid_has_frame_last(make_mask_and_image, tmp_path):
    image, labels = make_mask_and_image(n_blobs=4)
    cat = inst.build_catalog("day3", "A1", image=image, labels=labels)
    uid = cat["meta"]["instance_uid"].iloc[0].split("_")
    assert uid[0] == "A1" and int(uid[-1]) == 1      # frame_idx: day3 -> 1
    assert uid[-2].isdigit()                          # label 在倒数第二段
    assert cat["meta"]["time_hours"].unique().tolist() == [72.0]

def test_tiny_object_is_flagged_not_dropped(make_mask_and_image):
    image, labels = make_mask_and_image(n_blobs=2, tiny_area=20)
    cat = inst.build_catalog("day0", "A1", image=image, labels=labels)
    assert len(cat["meta"]) == 2
    assert cat["meta"]["is_small"].sum() == 1


def test_border_object_is_flagged_not_dropped(make_mask_and_image):
    image, labels = make_mask_and_image(n_blobs=2, touch_border=True)
    cat = inst.build_catalog("day0", "A1", image=image, labels=labels)
    assert cat["meta"]["touches_border"].sum() == 1
    assert len(cat["meta"]) == 2


def test_save_and_load_roundtrip(make_mask_and_image, tmp_path):
    image, labels = make_mask_and_image(n_blobs=2)
    cat = inst.build_catalog("day0", "A2", image=image, labels=labels)
    inst.save_catalog(cat, tmp_path)
    back = inst.load_catalog(tmp_path)
    pd.testing.assert_frame_equal(cat["meta"], back["meta"], check_dtype=False)
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest repro/tests/test_instances.py -q`
Expected: FAIL，`ImportError`/`AttributeError: module 'biopheno.instances' has no attribute ...`

- [ ] **Step 3: 实现 `instances.py`**

要点（签名见 Interfaces，不写代码全文）：
- `align_mask_to_image`：两图先裁到公共最小尺寸；在 `±max_shift` 网格内枚举 `(dy,dx)`，
  目标函数 = 掩码内均值 − 掩码外均值（实例比背景暗，见 Spec §2），取最大者；返回补齐后的 labels 与实际偏移。
- `extract_contours`：对每个 label 取 `labels == k` 二值，`skimage.measure.find_contours(binary, 0.5)`
  取最长的一条，重采样到 `n_points`（按弧长参数化，闭合），丢弃点数 < 6 的候选并计数。
- `crop_patches`：bbox 外扩 `pad` 后裁 `image`（越界用 0 填充）得 `native`，
  再 `skimage.transform.resize` 到 `size`、`skimage.util.img_as_ubyte` 得 `sift`。
- `build_catalog` 支持直接传入 `image`/`labels`（测试用）或从 `cfg.DATA_ROOT/<day>/` 读真实文件。
- 所有面积/长度用 `cfg.PIXEL_RES` 换算（当前 1.0，数值不变）。

- [ ] **Step 4: 运行确认通过并提交**

Run: `uv run pytest repro/tests/test_instances.py -q`
Expected: PASS（5 passed）
```bash
git add repro/ && git commit -m "feat: 掩码转实例目录（轮廓重采样、裁剪图、uid 约定）"
```

---

### Task 3: Stage 1 批处理与汇总（`instances.run_stage1`）

**Files:**
- Modify: `repro/biopheno/instances.py`（追加 `run_stage1`）
- Create: `repro/scripts/01_extract_instances.py`（薄 CLI 包装）
- Test: `repro/tests/test_stage1_cli.py`

**Interfaces:**
- Consumes: `instances.build_catalog`, `instances.save_catalog`
- Produces: `instances.run_stage1(days: list[str] | None = None, fovs: list[str] | None = None) -> pd.DataFrame`
  —— 遍历 `cfg.DAYS × cfg.FOVS`，落盘 `output/instances/<day>/<fov>/`，
  返回汇总 meta（含 `instance_uid, day, fov, frame_idx, time_hours, label, area_px, is_small`）
  并写 `output/instances/all_instances.csv`。
  （放 `instances.py` 而不是 `scripts/` 下，因为 `scripts/` 无 `__init__.py` 且文件名不能作模块名。）

- [ ] **Step 1: 写失败测试**（用 `tmp_path` 伪造 `DATA_ROOT` 结构：2 天 × 1 视野的小掩码，避免依赖真实数据）

```python
def test_run_stage1_writes_catalogs_and_summary(monkeypatch, fake_dataset, tmp_path):
    from biopheno import config as cfg
    monkeypatch.setattr(cfg, "DATA_ROOT", fake_dataset)
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    from biopheno.instances import run_stage1
    meta = run_stage1(days=["day0", "day3"], fovs=["A1"])
    assert (tmp_path / "instances" / "day0" / "A1" / "instances.csv").exists()
    assert (tmp_path / "instances" / "all_instances.csv").exists()
    assert set(meta["day"]) == {"day0", "day3"}
    assert meta["instance_uid"].is_unique
```

- [ ] **Step 2: 运行确认失败** — Expected: FAIL（`ImportError: cannot import name 'run_stage1'`）

- [ ] **Step 3: 实现 `run_stage1` 与薄 CLI**

`run_stage1` 打印每个视野的实例数、对齐偏移、丢弃轮廓数。
`repro/scripts/01_extract_instances.py` 只做 bootstrap + 调用：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biopheno.instances import run_stage1

if __name__ == "__main__":
    run_stage1()
```

- [ ] **Step 4: 运行确认通过并提交**

Run: `uv run pytest repro/tests/test_stage1_cli.py -q`
Expected: PASS
```bash
git add repro/ && git commit -m "feat: Stage1 批处理脚本与汇总表"
```

---

### Task 4: Stage 2 — shape + appearance 特征

**Files:**
- Create: `repro/biopheno/features.py`
- Test: `repro/tests/test_features.py`

**Interfaces:**
- Consumes: `instances.build_catalog` 的 `contours`/`image`/`labels`; `SPOT.Features.features`
- Produces:
  - `compute_shape_features(contours: list[np.ndarray], imshape: tuple[int,int], pixel_res: float) -> tuple[np.ndarray, list[str]]`
    单帧调用 `compute_boundary_morphology_features(np.array([contours]), imshape, all_feats_compute=True,
    pixel_xy_res=pixel_res, fast=True, n_jobs=cfg.N_JOBS)`，返回 `(N, D_shape)` 与特征名。
  - `compute_appearance_features(image: np.ndarray, contours: list[np.ndarray], pixel_res: float) -> tuple[np.ndarray, list[str]]`
    调用 `compute_boundary_texture_features(vid=image[None], boundaries=np.array([contours]),
    use_mask=True, masks=None, compute_all_feats=True, fast=True, n_jobs=cfg.N_JOBS)`。
   `image` 先 `skimage.exposure.rescale_intensity` 到 `[0,1]` 并转 `float`；`masks=None` 的理由：
   轮廓就是 label 掩码的边界，`contour_to_binary` 重建的前景与真值掩码等价。
  - `reduce_ecc_features(X: np.ndarray, names: list[str], n_dim: int) -> tuple[np.ndarray, list[str]]`
    用 `preprocessing.kernel_dim_reduction_ECC_features(..., n_dim=n_dim, random_state=0)`（Spec §6-Stage2-4）
  - `motion_placeholder_names() -> list[str]` —— 从 `features.py` 里静态解析出 motion 特征名（`*_motion*` 相关的
    `all_feats_labels` 字符串，实现方式：正则扫源码里的 `'...' %(...)` label 模板并按已知个数展开），
    用于生成 NaN 列。若静态解析不可靠，退化为返回空列表并写警告（报告里注明无法对齐命名）。
  - `run_stage2() -> pd.DataFrame`：逐视野算特征，拼接成 `(6741, D)` 表 + `motion` NaN 列，
    落盘 `output/features/{sam_shape.npz,sam_appearance.npz,feature_names.json,all_features.csv}`。

- [ ] **Step 1: 写失败测试**

```python
def test_shape_features_have_expected_families(make_mask_and_image):
    image, labels = make_mask_and_image(n_blobs=3)
    contours, _ = inst.extract_contours(labels, 100)
    X, names = feat.compute_shape_features(list(contours.values()), labels.shape, pixel_res=1.0)
    assert X.shape == (3, len(names))
    assert np.isfinite(X).all()
    assert any(n.startswith("ECT_") for n in names)
    assert any(n == "area" or n.startswith("area") for n in names)

def test_appearance_features_are_finite(make_mask_and_image):
    image, labels = make_mask_and_image(n_blobs=2)
    contours, _ = inst.extract_contours(labels, 100)
    X, names = feat.compute_appearance_features(image, list(contours.values()), pixel_res=1.0)
    assert X.shape[0] == 2 and len(names) > 0
    assert np.isfinite(X).all()

def test_ecc_reduction_shrinks_ect_family():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 200)).astype(np.float64)
    names = [f"ECT_{i+1}" for i in range(150)] + [f"other_{i}" for i in range(50)]
    Xr, nr = feat.reduce_ecc_features(X, names, n_dim=10)
    assert sum(n.startswith("ECT_") for n in nr) <= 10
    assert sum(n.startswith("other_") for n in nr) == 50
```

- [ ] **Step 2: 运行确认失败** — Expected: FAIL（`compute_shape_features` 不存在）

- [ ] **Step 3: 实现 `features.py`**（按 Interfaces；关键处中文注释说明"这是形状/外观特征集"，
  并注明 motion 故意不计算、以 NaN 列占位）

- [ ] **Step 4: 运行确认通过并提交**

Run: `uv run pytest repro/tests/test_features.py -q`
Expected: PASS
```bash
git add repro/ && git commit -m "feat: shape+appearance 特征提取（motion 置空占位）"
```

---

### Task 5: Stage 3 — 预处理与等权

**Files:**
- Create: `repro/biopheno/preprocess.py`
- Test: `repro/tests/test_preprocess.py`

**Interfaces:**
- Consumes: `SPOT.Analysis.preprocessing.*`
- Produces:
  - `preprocess(X: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str], dict]`
    顺序：`remove_zero_features` → `remove_nan_variance_features` → 显式剔除 `motion` NaN 列
    （记录名字）→ `remove_high_variance_features(variance_threshold_sigma=cfg.VARIANCE_THRESHOLD_SIGMA)`
    → `scale_normalize_curvature_features` → `power_transform_and_scale_features(X, names,
    all_conditions=np.zeros(len(X)))` → `weight_sam_features(X, names, group_weighting=True,
    set_weighting=True)`。返回 `(X_processed, kept_names, info)`，`info` 含每步剔除的名字与最终维度。
  - `run_stage3() -> None`：读 `output/features/`，落盘 `output/preprocessed/{X.npy,keep_features.json,all_conditions.npy}`。

- [ ] **Step 1: 写失败测试**

```python
def test_global_scaling_keeps_between_day_difference():
    # 若 all_conditions 误传 day，两组会被各自 z-score，天间均值差会被抹平
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, size=(50, 8)), rng.normal(3, 1, size=(50, 8))])
    names = [f"f{i}" for i in range(8)]
    Xp, kept, info = prep.preprocess(X, names)
    gap = abs(Xp[:50].mean() - Xp[50:].mean())
    assert gap > 0.5, "天间差异被抹平：all_conditions 很可能被按天拆分了"

def test_nan_motion_columns_are_dropped_and_recorded():
    X = np.vstack([np.ones((10, 3)), np.full((10, 2), np.nan)], axis=1)
    names = ["a", "b", "c", "motion_speed", "motion_flow"]
    Xp, kept, info = prep.preprocess(X, names)
    assert np.isfinite(Xp).all()
    assert "motion_speed" in info["dropped_nan"]
    assert all("motion" not in n for n in kept)
```

- [ ] **Step 2: 运行确认失败** — Expected: FAIL（`prep.preprocess` 不存在）

- [ ] **Step 3: 实现 `preprocess.py`**（含 `info` 字典的每步记录）

- [ ] **Step 4: 运行确认通过并提交**

Run: `uv run pytest repro/tests/test_preprocess.py -q`
Expected: PASS
```bash
git add repro/ && git commit -m "feat: 预处理与 SAM 特征族等权（全局拟合）"
```

---

### Task 6: Stage 4 — SAM 模块（legacy 选择器）

**Files:**
- Create: `repro/biopheno/modules.py`
- Test: `repro/tests/test_modules.py`

**Interfaces:**
- Consumes: `SPOT.Analysis.sam_analysis.compute_SAM_features_and_scope_contribution`,
  `hierarchical_cluster_features_into_SAM_modules`, `compute_individual_feature_contributions_in_SAM_modules`,
  `compute_most_representative_image_patches`
- Produces:
  - `discover_modules(X: np.ndarray, names: list[str]) -> dict`，键：`module_labels`（长度 D 的数组）、
    `feature_indices`（list[array]）、`feature_names`（list[list[str]]）、`contribution`（每模块的贡献向量）、
    `type_contribution`/`scope_contribution`、`cluster_map_data`
  - `module_expression(X: np.ndarray, modules: dict) -> np.ndarray`（实例 × 模块）
  - `representative_patches(expr: np.ndarray, patches: list[np.ndarray], n_rows: int) -> tuple[np.ndarray, list[np.ndarray]]`
    包装 `compute_most_representative_image_patches`
  - `run_stage4() -> None`：落盘 `output/modules/`（`modules_labels.csv`, `module_expr.npy`,
    `cluster_map.npz`, `module_patches.npz`）

- [ ] **Step 1: 写失败测试**

```python
def test_legacy_selection_is_used_and_modules_found():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(60, 1))
    X = np.hstack([base + 0.01 * rng.normal(size=(60, 1)) for _ in range(6)]
                  + [rng.normal(size=(60, 1)) for _ in range(6)])
    names = [f"f{i}" for i in range(12)]
    mods = mod.discover_modules(X, names)
    assert mods["module_labels"].shape == (12,)
    assert 1 < len(np.unique(mods["module_labels"])) <= 12
    # 前 6 个强共变特征应落在同一个模块里
    assert len(set(mods["module_labels"][:6])) == 1

def test_module_expression_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 9))
    names = [f"f{i}" for i in range(9)]
    mods = mod.discover_modules(X, names)
    expr = mod.module_expression(X, mods)
    assert expr.shape[0] == 40
```

- [ ] **Step 2: 运行确认失败** — Expected: FAIL

- [ ] **Step 3: 实现 `modules.py`**，显式传
  `module_selection=cfg.MODULE_SELECTION`（`"legacy"`）与 `**cfg.MODULE_PARAMS`；
  在函数 docstring 里写明"为什么不使用 v2 默认的 multiscale"。

- [ ] **Step 4: 运行确认通过并提交**

Run: `uv run pytest repro/tests/test_modules.py -q`
Expected: PASS
```bash
git add repro/ && git commit -m "feat: SAM 模块发现（legacy 选择器，论文参数）"
```

---

### Task 7: Stage 5 — UMAP 景观与表型簇

**Files:**
- Create: `repro/biopheno/landscape.py`
- Test: `repro/tests/test_landscape.py`

**Interfaces:**
- Consumes: `umap.UMAP`, `sam_analysis.KMeans_cluster_low_dimensional_embedding_with_scan`,
  `sam_analysis.reorder_and_relabel_clusters`, `sam_analysis.compute_heatmap_density_image_Embedding`
- Produces:
  - `embed(X: np.ndarray) -> np.ndarray`（(N,2)，`umap.UMAP(**cfg.UMAP_PARAMS)`）
  - `cluster(embedding: np.ndarray, k_range=cfg.KMEANS_K_RANGE) -> tuple[np.ndarray, dict]`
    返回簇标签（0-based）与 `{"k": int, "inertia": list, "silhouette": list}`
  - `order_clusters_by_score(clusters: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
    包装 `reorder_and_relabel_clusters(clusters, score, order="ascending")`，返回（新簇标签 1-based、映射表）
  - `run_stage5() -> None`：读预处理的 X 与 `all_instances.csv`，落盘
    `output/analysis/{umap_2d.csv,cluster_labels.csv,kmeans_metrics.json}`

- [ ] **Step 1: 写失败测试**

```python
def test_clusters_recover_two_blobs_and_are_ordered_by_score():
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(0, 0.3, (60, 2)), rng.normal(6, 0.3, (60, 2))])
    labels, metrics = land.cluster(emb, k_range=(2, 4))
    assert metrics["k"] == 2
    assert len(np.unique(labels)) == 2
    score = np.where(labels == labels[0], 0.2, 0.9)
    ordered, mapping = land.order_clusters_by_score(labels, score)
    assert ordered.min() == 1 and ordered[score > 0.5].max() == 2

def test_embedding_is_deterministic():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 12))
    a = land.embed(X)
    b = land.embed(X)
    assert a.shape == (120, 2)
    assert np.allclose(a, b, atol=1e-6)
```

- [ ] **Step 2: 运行确认失败** — Expected: FAIL

- [ ] **Step 3: 实现 `landscape.py`**（UMAP 参数从 cfg 读并原样传给 `umap.UMAP`；
  k-means 用 `ploton=False` 避免各任务里弹图）

- [ ] **Step 4: 运行确认通过并提交**

Run: `uv run pytest repro/tests/test_landscape.py -q`
Expected: PASS
```bash
git add repro/ && git commit -m "feat: UMAP 景观与按 eccentricity 排序的表型簇"
```

---

### Task 8: Stage 6 — 跨天配准、tracklet 匹配与 SPOT uid

**Files:**
- Create: `repro/biopheno/temporal.py`
- Test: `repro/tests/test_temporal.py`

**Interfaces:**
- Consumes: `instances` 的 meta；`SPOT.Registration.registration_2D.translation_register_correlation`
- Produces:
  - `register_frame(prev: np.ndarray, cur: np.ndarray, max_shift: int) -> tuple[int,int]`
    在掩码二值上用互相关求平移；失败（异常/偏移超限）时返回 `(0,0)` 并记录告警
  - `match_frames(prev_meta: pd.DataFrame, cur_meta: pd.DataFrame, offset: tuple[int,int],
     min_iou: float, max_cent_dist: float) -> pd.DataFrame`
    返回匹配表，列：`prev_uid, cur_uid, iou, cent_dist, event`（`event ∈ {"one_to_one","merge","split"}`，
    一对多/多对一的附加行以 `prev_uid=None` 或 `cur_uid=None` 表示被吸收/新生成）
  - `assign_tracks(matches: list[pd.DataFrame], all_meta: pd.DataFrame) -> pd.DataFrame`
    给每个实例分配 `track_id`，返回含 `instance_uid, track_id, frame_idx, time_hours, fov` 的表，
    并生成 SPOT 专用列 `spot_uid = f"{fov}_{track_id}_{frame_idx}"`
  - `spot_uid_from(meta: pd.DataFrame) -> np.ndarray`
    由 `fov/track_id/frame_idx` 列拼 `spot_uid`，并校验：3 段、末段可转 int、不含 `-` 等非法字符
  - `build_tracks(metas: dict[str, pd.DataFrame]) -> pd.DataFrame`
  - `run_stage6() -> None`：落盘 `output/analysis/tracks.csv`、`output/analysis/match_qc.json`
    并断言 `(tracks.groupby("track_id").size() >= 2).sum() > 0`（否则 `RuntimeError`）

- [ ] **Step 1: 写失败测试**

```python
def test_one_to_one_and_merge_are_distinguished(fake_two_frames):
    prev_meta, cur_meta = fake_two_frames      # 2 个实例 -> 1 个（合并）+ 1 个（一对一保留）
    matches = temp.match_frames(prev_meta, cur_meta, offset=(0, 0),
                                min_iou=0.05, max_cent_dist=60.0)
    assert (matches["event"] == "one_to_one").sum() >= 1
    assert (matches["event"] == "merge").sum() >= 1

def test_spot_uid_has_frame_as_last_field_and_tracks_are_not_all_singletons():
    tracks = temp.assign_tracks([matches], all_meta)
    uid = tracks["spot_uid"].iloc[0]
    assert uid.split("_")[-1].isdigit()        # SPOT 内部写死 split('_')[-1] 取帧号
    assert (tracks.groupby("track_id").size() >= 2).sum() > 0

def test_bad_uid_format_is_rejected():
    tracks = pd.DataFrame({"fov": ["A1"], "track_id": [0], "frame_idx": [0]})
    with pytest.raises(ValueError):
        temp.spot_uid_from(tracks.assign(track_id="t-0"))   # 含 '-' 会被 SPOT 解析错，必须早失败
```

- [ ] **Step 2: 运行确认失败** — Expected: FAIL

- [ ] **Step 3: 实现 `temporal.py`**

匹配贪心：候选对按 `iou` 降序、一对一优先；残余多对一 → `merge`（保留面积最大者为父），
一对多 → `split`。`spot_uid_from` 里校验：字段数 = 3、最后一段可转 int、不含 `_` 以外的分隔符。

- [ ] **Step 4: 运行确认通过并提交**

Run: `uv run pytest repro/tests/test_temporal.py -q`
Expected: PASS
```bash
git add repro/ && git commit -m "feat: 跨天配准与 tracklet 匹配（含 SPOT uid 校验）"
```

---

### Task 9: Stage 6b — 时间统计（占比、转移、轨迹）

**Files:**
- Create: `repro/biopheno/temporal_stats.py`
- Test: `repro/tests/test_temporal_stats.py`

**Interfaces:**
- Consumes: `sam_analysis.compute_temporal_cluster_proportions`,
  `compute_phenotype_cluster_transitions_from_label_trajectories`,
  `construct_obj_traj_from_uniq_obj_ids`, `compute_phenotypic_trajectory`,
  `construct_affinity_matrix_phenotypic_trajectory`, `hcluster_phenotypic_trajectory_affinity_matrix`
- Produces:
  - `cluster_proportions(clusters: np.ndarray, time_hours: np.ndarray, conditions: np.ndarray, time_intervals: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  - `transitions(tracks: pd.DataFrame, clusters: np.ndarray) -> dict`
    （键 `inter`, `self`, `inter_graph_data`；内部先 `construct_obj_traj_from_uniq_obj_ids(tracks["spot_uid"],
    separator="_", wanted_entries=[0,1], d_Frame=1)`）
  - `phenotype_trajectories(embedding: np.ndarray, time_hours: np.ndarray, conditions: np.ndarray, time_intervals) -> dict`
  - `trajectory_affinity(trajectories: dict) -> tuple[np.ndarray, list[str]]`
  - `run_stage6b() -> None`：落盘 `output/analysis/{proportions.npz,transitions.npz,trajectories.npz,affinity.npz}`

- [ ] **Step 1: 写失败测试**

```python
def test_proportions_sum_to_one_per_time_and_condition():
    clusters = np.array([1, 1, 2, 2, 1, 2])
    times = np.array([0.0, 0.0, 0.0, 72.0, 72.0, 72.0])
    conds = np.array(["A1"] * 6)
    props, uniq_labels, uniq_cond = temp_stats.cluster_proportions(
        clusters, times, conds, time_intervals=[0.0, 72.0])
    assert props.shape[0] == 1
    assert np.allclose(props.sum(axis=-1), 1.0, atol=1e-6)

def test_transitions_are_row_stochastic():
    tracks = pd.DataFrame({
        "spot_uid": ["A1_0_0", "A1_0_1", "A2_0_0", "A2_0_1"],
        "fov": ["A1", "A1", "A2", "A2"], "track_id": [0, 0, 0, 0],
        "frame_idx": [0, 1, 0, 1]})
    clusters = np.array([1, 2, 1, 1])
    out = temp_stats.transitions(tracks, clusters)
    assert np.allclose(out["inter"].sum(axis=1), 1.0, atol=1e-6)
```

- [ ] **Step 2: 运行确认失败** — Expected: FAIL

- [ ] **Step 3: 实现 `temporal_stats.py`**

`time_intervals` 用真实小时（`[0, 72, 120]` 的细分箱或直接 3 个箱），`conditions` 用 `fov`。
docstring 写明：这里的时间轴是"3 个离散时间点"，不是连续帧，速度类结论不成立（motion 已置空）。

- [ ] **Step 4: 运行确认通过并提交**

Run: `uv run pytest repro/tests/test_temporal_stats.py -q`
Expected: PASS
```bash
git add repro/ && git commit -m "feat: 时间统计（簇占比、转移矩阵、表型轨迹）"
```

---

### Task 10: 静态图 F1–F6b

**Files:**
- Create: `repro/biopheno/figures.py`
- Test: `repro/tests/test_figures_static.py`

**Interfaces:**
- Consumes: `output/analysis/*`、`output/modules/*`、`instances` patches
- Produces: `render_static(analysis_dir: Path, outdir: Path, force_single_cluster: bool = False) -> list[Path]`，产出
  `F1_umap_clusters_patches.png`、`F2_umap_by_day_fov.png`、`F3_umap_density.png`、
  `F4_contribution_scores.png`、`F5_sam_modules.png`、`F6_cluster_feature_bars.png`、
  `F6b_cluster_day_composition.png`（Spec §6-Stage7）
- 所有绘图函数接受数据数组、不读全局状态；`matplotlib.use("Agg")` 在模块顶部设置。

- [ ] **Step 1: 写失败测试**

```python
def test_render_static_writes_seven_figures(fake_analysis_dir, tmp_path):
    paths = figs.render_static(fake_analysis_dir, outdir=tmp_path)
    names = {p.name for p in paths}
    assert {"F1_umap_clusters_patches.png", "F6b_cluster_day_composition.png"} <= names
    assert all(p.stat().st_size > 1000 for p in paths)

def test_render_static_tolerates_single_cluster(fake_analysis_dir, tmp_path):
    # 只有一个簇时不应 IndexError（Review Focus #5）
    paths = figs.render_static(fake_analysis_dir, outdir=tmp_path, force_single_cluster=True)
    assert paths
```

- [ ] **Step 2: 运行确认失败** — Expected: FAIL

- [ ] **Step 3: 实现 `figures.py`**（每张图一个 `_plot_fN(...)` 私有函数；
  簇配色用 `matplotlib.cm.coolwarm` 的 blue→red；每簇代表图用 `modules.representative_patches`；
  坐标转换 `(row, col) → (x, y)` 在这一处完成）

- [ ] **Step 4: 运行确认通过并提交**

Run: `uv run pytest repro/tests/test_figures_static.py -q`
Expected: PASS
```bash
git add repro/ && git commit -m "feat: 静态景观图 F1-F6b"
```

---

### Task 11: 时间图 F7–F11

**Files:**
- Modify: `repro/biopheno/figures.py`
- Test: `repro/tests/test_figures_temporal.py`

**Interfaces:**
- Consumes: Task 9 的 `proportions/transitions/trajectories/affinity`
- Produces: `render_temporal(outdir: Path) -> list[Path]`，产出
  `F7_stacked_cluster_proportions.png`、`F8_transition_graph.png`、`F9_umap_trajectories.png`、
  `F10_trajectory_affinity.png`（FOV 级与簇级两版：`F10a`/`F10b`）、`F11_match_events.png`

- [ ] **Step 1: 写失败测试**

```python
def test_render_temporal_writes_all_figures(fake_temporal_dir, tmp_path):
    paths = figs.render_temporal(fake_temporal_dir, outdir=tmp_path)
    names = {p.name for p in paths}
    assert {"F7_stacked_cluster_proportions.png", "F10a_trajectory_affinity_fov.png",
            "F10b_trajectory_affinity_cluster.png", "F11_match_events.png"} <= names

def test_trajectory_plot_skips_singleton_tracks(fake_temporal_dir, tmp_path):
    paths = figs.render_temporal(fake_temporal_dir, outdir=tmp_path)   # fixture 里全是长度 1 的 track
    assert any(p.name == "F9_umap_trajectories.png" for p in paths)     # 出图但不崩
```

- [ ] **Step 2: 运行确认失败** — Expected: FAIL

- [ ] **Step 3: 实现 `render_temporal`**（长度 < 2 的 track 不画轨迹，图注写"仅长度 ≥ 2 的 track"）

- [ ] **Step 4: 运行确认通过并提交**

Run: `uv run pytest repro/tests/test_figures_temporal.py -q`
Expected: PASS
```bash
git add repro/ && git commit -m "feat: 时间动态图 F7-F11"
```

---

### Task 12: 报告、一键运行与真实数据验收

**Files:**
- Create: `repro/biopheno/report.py`
- Create: `repro/biopheno/pipeline.py`
- Create: `repro/scripts/run_all.py`
- Test: `repro/tests/test_report.py`

**Interfaces:**
- Consumes: 所有 Stage 的 `run_stageN()` 产物
- Produces:
  - `write_report(output_root: Path) -> Path`：生成 `output/report.md`，含 Spec §6-Stage7 要求的全部数字
    （实例数、特征维度含置空数/ECT 降维前后、预处理后维度、UMAP 参数、K、elbow/轮廓、模块数、
    匹配率与 merge/split 统计、每张图对应论文哪张图、已知偏差清单、`PIXEL_RES=1.0` 声明）
  - `run_all() -> Path`：顺序调用 `run_stage1/2/3/4/5/6/6b` + `render_static/temporal` + `write_report`

- [ ] **Step 1: 写失败测试**

```python
def test_report_contains_required_sections(fake_full_output):
    p = report.write_report(fake_full_output)
    text = p.read_text(encoding="utf-8")
    for key in ["实例数", "特征维度", "UMAP", "K", "模块", "匹配率", "局限"]:
        assert key in text
    assert "6741" in text
```

- [ ] **Step 2: 运行确认失败** — Expected: FAIL

- [ ] **Step 3: 实现 `report.py`、`pipeline.py`、`run_all.py`**

`fake_full_output` fixture 在 `tmp_path` 下造出 `instances/features/preprocessed/modules/analysis` 的最小结构，
供 `write_report` 读取。**

- [ ] **Step 4: 在真实数据上跑通并验收**

```bash
cd /d/Code/myself/SPOT
uv run python repro/scripts/run_all.py
uv run pytest repro/tests -q
```
Expected:
- `run_all.py` 无异常结束，`repro/output/figures/` 里有 F1–F6b、F7–F11（含 F10a/F10b）
- `output/report.md` 中实例数合计与 6741 的差额有明确解释
- `output/analysis/tracks.csv` 里 `track_id` 分组长度 ≥ 2 的数目 > 0
- 全部测试 PASS

- [ ] **Step 5: 提交**

```bash
git add repro/ && git commit -m "feat: 报告生成与一键运行，真实数据全流程跑通"
```
