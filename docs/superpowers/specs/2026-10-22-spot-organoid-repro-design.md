# SPOT 复现设计文档（本地 3 天类器官数据）

日期：2026-10-22（rev.2，已并入独立构建者审查意见）
状态：待人类合作者批准（批准前不写实现代码）

---

## 1. 意图（这是什么，为什么做）

用本地的、已经分割好的类器官数据，复现 Nature Communications 2026 论文
《Identifying phenotype-genotype-function coupling in 3D organoid imaging using Shape,
Appearance and Motion Phenotype Observation Tool (SPOT)》(DOI 10.1038/s41467-026-75506-7)
里的 **SAM 表型组分析**：

1. **静态表型景观**：把每个类器官实例变成高维 SAM 特征向量 → UMAP 铺成 2D 景观 →
   k-means 聚出表型簇 → 出论文 Fig. 2d–g / Fig. 3b–d 那套图（景观图、每簇代表图、
   contribution score、SAM 模块热图与模块表达）。
2. **时间动态**：把三天的实例串成 tracklet → 各簇占比随时间、簇间转移、UMAP 表型轨迹、
   轨迹相似度层次聚类（论文 Fig. 3c iii–v / Fig. 4c–e 那套图）。

参考实现是本地仓库 `D:\Code\myself\SPOT`（PyPI 包名 `SAM-SPOT`，版本 2.0.1）。
**只调用它，不改它。**

## 2. 数据事实（已实测，不是假设）

数据根目录：`D:\Datasets\test`

| 项 | 实测值 |
|---|---|
| 结构 | `day0/`、`day3/`、`day5/` 各含 `A1/A2/A3` 三个视野，每视野一帧 |
| 原图 | `<day>/<FOV>_多聚焦融合.png`，RGB uint8，(H=3576, W=4284)，三通道略有差异（融合彩图） |
| 掩码 | `<day>/Output/<FOV>_多聚焦融合.tif`，单页 uint16 标签图，标签 1…N |
| 掩码尺寸 | 各图不一致：3572–3580 × 4284–4288（与原图也不完全一致，最多差 4 行/4 列） |
| 实例数 | day0: 1145/984/917，day3: 697/768/582，day5: 581/614/453，合计 **6741** |
| 中位面积 | day0 ≈ 590–657 px²，day3 ≈ 2114–2640，day5 ≈ 3881–4440 |
| 面积分布 | day0 有 **39–42% 实例 < 500 px²**；day3/day5 只有 6–13% |
| 掩码覆盖率 | day0 5.5–8.1%，day3 20–28.5%，day5 32.8–37.7% |
| 掩码-原图对齐 | 有机体区域显著暗于背景（内部均值 147–189 vs 外部 181–212），**已对齐** |
| 三天是否同视野 | 是。配准后掩码 IoU：day3–day5 ≈ 0.445–0.470；day0 覆盖率过低所以 day0–day5 只有 0.10–0.12 |
| 生物学含义 | 类器官生长 + 大量合并：实例数 1145→581，面积 ×6，覆盖率 ×5 |

**关键推论**：有显著跨天生长与合并，tracklet 匹配必然出现多对一（merge）与一对多（split），
不能假设一对一。另外三天面积中位数相差 6 倍，**联合 UMAP 很可能主要按"天数=大小"分开**（见 §11-Q1）。

## 3. 目标 / 非目标 / 反目标

### 目标
- G1 为 6741 个实例计算 shape + appearance 两类 SAM 特征，落盘为可复用的特征表。
- G2 产出联合 UMAP 景观（三天一次投影）、k-means 表型簇（按平均 eccentricity 编号）、
  每簇代表图、contribution score、SAM 模块热图与模块表达表。
- G3 产出跨天 tracklet、各簇占比随时间、簇间/自簇转移矩阵、UMAP 表型轨迹与轨迹聚类。
- G4 全流程可一条命令重跑；中间产物（实例目录、特征表、预处理矩阵、嵌入坐标）都落盘。
- G5 有 pytest 覆盖关键逻辑（掩码→轮廓、跨天匹配、簇重排、时间编码、出图冒烟）。

### 非目标
- N1 不做检测/分割/追踪模型训练（掩码已给；不训练 YOLOv3/UNet）。
- N2 不做 motion 特征（人类合作者已决定：motion 全部置空）。
- N3 不复现 Fig. 5 / Fig. 6（scRNA-seq、WNT3A 恢复、基因型比较）——数据里没有这些条件。
- N4 不复现模拟数据（Simulated Dataset A/B、Fig. 2 的扰动鲁棒性）。
- N5 不做 GUI、不下载预训练权重、不改 `SPOT/` 源码、不改仓库 `pyproject.toml` 的依赖列表。
- N6 不做批次校正为主结果（见 §7 D5）。

### 反目标（即使跑通也算失败）
- A1 把 motion 用"跨天位移/72h"硬算进去（人类合作者明确否决）。
- A2 为了让图好看而调 UMAP 参数直到出现想要的结构；参数照论文写死。
- A3 改 SPOT 源码里的函数来迁就本地数据（应在 `repro/` 里做适配层）。
- A4 输出只存在于 notebook 里的内存中，没有落盘 CSV。
- A5 静默成功：任何"函数跑完但结果退化成空/单点"的情况必须报错或显式告警（见 §9）。

## 4. 约束

- 平台：Windows，Python 3.12，包管理用 **uv**，虚拟环境建在 `D:\Code\myself\SPOT`。
- 依赖：以 `D:\Code\myself\SPOT\pyproject.toml` 为准（`numpy/pandas/scikit-image/scikit-learn/
  umap-learn/pacmap/torch/mahotas/...`）。**若 `pacmap`/`annoy` 在 Windows+py3.12 装不上，
  跳过 pacmap**（论文只用 UMAP，pacmap 不是必需品），并在报告里注明。
- 代码风格：Ruff 格式化结果为准，行长 120；关键处写中文注释说明"这是什么"。
- 不改 SPOT 仓库已有文件；所有新代码在 `repro/` 下。
- 运行时间预算：特征提取单进程 ≤ 2 小时（`n_jobs` 并行后 ≤ 30 分钟）；UMAP（6741 × ~600 维）≤ 10 分钟。

## 5. 输入、输出与目录布局

```
D:\Code\myself\SPOT\
├─ repro\
│  ├─ biopheno\                 # 可 import 的包（逻辑都在这）
│  │  ├─ __init__.py
│  │  ├─ config.py                # 路径、论文参数常量（不硬编码在脚本里）
│  │  ├─ instances.py             # 掩码→实例目录/轮廓/patches
│  │  ├─ features.py              # shape+appearance 特征提取（调 SPOT）
│  │  ├─ preprocess.py            # 预处理+等权（调 SPOT.Analysis.preprocessing）
│  │  ├─ modules.py               # SAM 模块发现与表达
│  │  ├─ landscape.py             # UMAP + k-means + 簇重排 + density
│  │  ├─ temporal.py              # 跨天配准+tracklet 匹配+转移/占比/轨迹
│  │  ├─ figures.py               # 所有出图（单一入口）
│  │  └─ report.py                # QC/指标 markdown
│  ├─ scripts\
│  │  ├─ 01_extract_instances.py
│  │  ├─ 02_compute_sam.py
│  │  ├─ 03_preprocess.py
│  │  ├─ 04_modules.py            # 模块先于 UMAP（与论文脚本顺序一致）
│  │  ├─ 05_landscape.py
│  │  ├─ 06_temporal.py
│  │  ├─ 07_figures.py
│  │  └─ run_all.py               # 一条命令跑完 01→07
│  ├─ tests\
│  ├─ output\                     # 全部产物（gitignore）
│  │  ├─ instances\{day}\{fov}\   # instances.csv, contours.npz, patches.npz
│  │  ├─ features\                # sam_shape.npz/.csv, sam_appearance.npz, feature_names.json
│  │  ├─ preprocessed\            # X.npy, std_pow_transformer.joblib, keep_features.json
│  │  ├─ modules\                 # modules_labels.csv, module_expr.npy, clustermap_data.npz
│  │  ├─ analysis\                # umap_2d.csv, cluster_labels.csv, tracks.csv
│  │  └─ figures\                 # 所有 png/pdf
│  └─ docs\superpowers\specs\     # 本文档
└─ pyproject.toml                 # 不改
```

## 6. 流程（七个阶段）

### Stage 1 — 实例目录（`instances.py` / `01`）

1. 读 `<day>/Output/<fov>_多聚焦融合.tif`；按需裁/补到与原图共同尺寸（取两者最小值），
   并做 **±8 px 平移搜索**，取使"掩码内均值 − 掩码外均值"最大的偏移（对齐兜底，正常情况偏移应为 0）。
2. 原图转亮度（ITU-R 601 luma）作为 appearance 输入通道。
3. 每个标签 → 二值 → `skimage.measure.find_contours` 取最长闭合外轮廓（凹形保留）；
   轮廓点做等弧长重采样到 N=100，并记录 `label, centroid, bbox, area_px, touches_border`。
4. **不做面积过滤**（`MIN_AREA_PX = 0`）：已核对参考脚本 `Stage1_Step3/Step4`、`Stage2_Step3`、
   `Stage3`，论文论文口径是 tracklet 质量过滤，没有面积阈值；而本地 day0 有 39–42% 实例
   < 500 px²，任何统一阈值都会单方面删掉 day0 大部分实例、改写"实例数随时间增长"的叙述。
   小实例（< `SMALL_AREA_FLAG_PX`，默认 200）只**打标**不删除，供出图时排除。
5. `touches_border` 为真者同样只打标不删除。
6. **时间编码（关键，审查命中）**：`SPOT.Analysis.sam_analysis.construct_obj_traj_from_uniq_obj_ids`
   从 id 字符串末尾字段解析帧号，并用标量相等判断连续性；真实天数 0/3/5 的差是 3 和 2，
   任何单一 `d_Frame` 都无法表达，会让**所有 track 静默退化成长度 1**。
   因此：`instance_uid = {fov}_{label}_{frame_idx}`（真实天数另存 `day`，`time_hours = 0/72/120`），
   `frame_idx` 为连续整数 0/1/2（day0→0, day3→1, day5→2），传给
   `compute_temporal_cluster_proportions` / `compute_phenotypic_trajectory` 的
   `objects_time` 用 `time_hours`，`time_intervals=[0, 72, 120]` 或细分箱。
   **实测到的硬编码**：`construct_obj_traj_from_uniq_obj_ids` 内部分隔符写死为 `'_'`
   （`ff.split('_')[-1]` 取帧号），所以 uid 必须用 `_` 分隔且**帧号在最后一段**；
   跨天匹配得到的 `track_id` 因此不能在 Stage 1 确定：Stage 1 只存原始 uid，
   Stage 6 重新拼成 `{fov}_{track_id}_{frame_idx}` 再交给 SPOT（`separator='_'`,
   `wanted_entries=[0,1]`, `d_Frame=1`）。
7. 落盘：`instances.csv`、`contours.npz`、`patches.npz`（bbox 外扩 5 px 裁剪，
   原始尺寸 + 64×64 两版，供代表图与 SIFT 使用）。

### Stage 2 — SAM 特征（`features.py` / `02`）

1. 形状：`SPOT.Features.features.compute_boundary_morphology_features(boundaries, imshape,
   all_feats_compute=True, pixel_res=PIXEL_RES)`。
2. 外观：`compute_boundary_texture_features(vid, boundaries, use_mask=True, masks=..., compute_all_feats=True, ...)`。
   每个 (day, fov) 只有一帧，按 T=1 传入；`timesample` 只对时间相关项生效，单帧时无影响。
3. **motion 不计算**：不调用 `compute_boundary_motion_features`。特征表里额外写出
   `motion_feature_names.json`（从代码里静态提取的名称清单）与对应 **NaN 列**，
   以便与论文 2185 维命名对齐、且下游看到 NaN 就知道是被有意置空。
4. **ECT/ECC 降维（审查命中，原文档遗漏）**：`compute_boundary_morphology_features` 会输出
   `ECT_*`（36 方向 × 32 阈值 ≈ 1152 列）+ shape-context（300 列），形状特征实际 ≈ 1700 维，
   不是"数百维"。按论文脚本 `SPOT_Stage3...py:224`，在形状特征上先跑
   `preprocessing.kernel_dim_reduction_ECC_features(all_feats, feature_names, n_dim=100,
   gamma=None, random_state=1)`，把 ECT 家族压到 100 维。
5. 维度以实测为准；实际数字（形状 / 外观 / 置空 motion）写进 QC 报告。

### Stage 3 — 预处理与等权（`preprocess.py` / `03`）

按论文脚本（`SPOT_Stage3_analyze-SAM-phenomes.py`）的顺序与参数：
`remove_zero_features` → `remove_nan_variance_features` → NaN 列剔除（记录名字）→
`remove_high_variance_features(variance_threshold_sigma=2)`（**论文用 2，不是 3**）→
`scale_normalize_curvature_features` → `power_transform_and_scale_features` →
`weight_sam_features(group_weighting=True, set_weighting=True)`。

**`all_conditions` 必须是单一常量（审查命中）**：`power_transform_and_scale_features` 会为
`all_conditions` 里每个唯一取值分别拟合 `PowerTransformer` + `StandardScaler`。论文脚本
用 `sepcond_norm = False` 走全局拟合。若误传 `day`，等于按天各自 z-score，**恰好抹掉
G3/F7 要展示的天间差异**；若传 `FOV`，等于隐式按视野做批次归一化，与 D5 冲突。
故统一传 `np.zeros(n)`（单一条件，全局拟合），并在 `config.py` 里写明原因。

保存变换器以便对新数据复用。

### Stage 4 — SAM 模块（`modules.py` / `04`）

顺序与论文一致：**模块先于 UMAP**（参考脚本 `SPOT_Stage3:305–336`）。

- `sam_analysis.select_SAM_module_scales` / `hierarchical_cluster_features_into_SAM_modules`，
  **显式指定 `module_selection='legacy'`**，并传入论文参数 `max_clust_len=None, lam_smooth=1e0,
  buffer=30`。理由：v2.0.1 默认的 `'multiscale'`（null 模型 + bootstrap 共识）是另一种方法、
  会给出不同模块数、也明显更慢；而复现目标是论文 Fig. 3d。`config.py` 留
  `MODULE_SELECTION: str = "legacy"` 开关，切到 `multiscale` 只是实验，不作主结果。
- 产出：`SAM_modules` 标签、模块色条、`compute_individual_feature_contributions_in_SAM_modules`、
  `compute_most_representative_image_patches`（每模块代表图）、模块表达矩阵（实例 × 模块）。

### Stage 5 — 静态景观（`landscape.py` / `05`）

- UMAP：`n_neighbors=100, min_dist=0.5, spread=0.5, metric='euclidean', random_state=0`（论文默认）。
- k-means：`KMeans_cluster_low_dimensional_embedding_with_scan`（elbow），K 范围 2–12。
- 簇重排：`reorder_and_relabel_clusters` 按平均 eccentricity 从蓝到红编号。
- 一并算：每实例的 4 个可解释特征（area / eccentricity / intensity；speed 置空）用于簇排序校验与条形图。
- 保存 `umap_2d.csv`、`cluster_labels.csv`、`kmeans_metrics.csv`（inertia/轮廓系数）。

### Stage 6 — 时间动态（`temporal.py` / `06`）

1. 跨天配准：`SPOT.Registration.registration_2D` 的平移互相关，基准 = day0；
   用掩码二值做输入（比强度图稳），失败则退化为偏移 0 并记录告警。
2. 跨天匹配 day0→day3、day3→day5：
   - 候选对：bbox IoU ≥ `MIN_BBOX_IOU`（默认 0.05）**或** 质心距离 ≤ `MAX_CENT_DIST_PX`；
   - 按 IoU 降序贪心分配（一对一优先）；
   - 残余多对一 → 记为 **merge**（父 track 取面积最大者，其余子实例标记 `merged_into`）；
   - 一旧对多新 → 记为 **split**（新 track 共享父 id）。
3. `construct_obj_traj_from_uniq_obj_ids`（用 `frame_idx` 0/1/2）生成 track → `tracks.csv`
   （track_id, instance_uid, frame_idx, time_hours, cluster）。
   **断言：长度 ≥ 2 的 track 占比 > 0**，否则报错（防止 §6-Stage1-6 的静默退化复发）。
4. 计算：各簇占比（按视野 + 合并，`objects_time=time_hours`、3 个时间点）、
   转移矩阵（inter / self）、`compute_phenotypic_trajectory` 轨迹、轨迹亲和矩阵 + 层次聚类。
5. **F10 比什么（审查命中）**：没有任何处理组，"条件"轴只有 3 个视野，做视野级 3 叶聚类
   没有信息量。改为**双版本**：
   (a) FOV 级亲和矩阵 + 3 叶 dendrogram（忠实复刻管线，图注说明样本只有 3 个视野）；
   (b) 表型簇级轨迹亲和矩阵 + dendrogram（信息量在"哪些表型簇的时间行为相似"）。
6. QC 指标：匹配率、merge/split 数、未匹配数、track 长度分布。

### Stage 7 — 出图与报告（`figures.py` / `07`）

静态图（对应 Fig. 3b–d）：
- F1 UMAP 按簇着色（编号 + blue→red）+ 每簇 4 张代表图（原始尺寸、标准化尺寸两版）
- F2 UMAP 按 frame/day / 按 FOV 着色（分面）
- F3 每簇 UMAP 密度热图
- F4 contribution score 条形图（shape/appearance × global/regional/distribution 两个面板）
- F5 SAM 模块 clustermap + 模块色条 + 模块表达热图（簇 × 模块）
- F6 每簇可解释特征 barplot（area/eccentricity/intensity，motion 标注为置空）
- **F6b 每簇的 day 组成堆叠条**：一眼看出景观是否主要由天数/大小驱动（§11-Q1 的证据图）

时间图（对应 Fig. 3c iii–v）：
- F7 各簇占比 stacked bar（3 个时间点，按视野分面 + pooled）
- F8 簇间转移热图 + 自簇转移热图 + 转移有向图
- F9 UMAP 表型轨迹（按 FOV 分面，3 点 + 时间箭头）
- F10 轨迹亲和矩阵 + dendrogram（FOV 级 / 簇级两版）
- F11 merge/split 事件数随帧变化（QC 图）

报告：`output\report.md`——实例数、特征维度（含置空数/ECT 降维前后）、预处理后维度、UMAP 参数、
选定 K、elbow/轮廓指标、模块数、匹配率与 merge/split 统计、每张图对应论文哪张图的说明、
以及所有已知偏差（无 motion、3 个时间点、无批次校正、pixel_res=1）。

## 7. 我（构建者）替人类合作者做的决定

- **D1 像素分辨率**：未知，`PIXEL_RES = 1.0`，所有长度/面积单位为"像素单位"，
  在报告里显式声明；若之后给出真实 µm/px，只需改 `config.py` 一处重跑。
- **D2 外观通道**：三通道取亮度（luma），不用 RGB 多通道（论文是单通道）。
- **D3 联合景观**：三天实例一次性做同一个 UMAP，不分天建模——否则轨迹无法画在同一空间。
- **D4 motion 置空方式**：不计算 + 保留 NaN 列（而非填 0），避免"零位移"被误当成"静止"。
- **D5 不做批次校正**为实测主结果；`config.py` 留 `BATCH_CORRECT_FOV: bool = False` 开关，
  打开时用 `preprocessing.regress_out_confounders(categorical=...)`。
- **D6 实例过滤**：见 §6-Stage1-4，只打标不过滤。
- **D7 代表图来源**：用 `patches.npz` 里保存的原始裁剪图，不用重新读原图。
- **D8 坐标系**：全部内部用一个约定——`(x, y)` 与 numpy `(row, col)` 的转换集中在 `instances.py`
  一处完成，避免散落各处（SPOT 里 contour 是 `(x, y)`，skimage 是 `(row, col)`，这是本地复现最容易踩的坑）。
- **D9 输出目录**：`repro\output`，不进 git。
- **D10 环境**：`uv venv` + `uv pip install -e .`（本仓库本体）+ 额外装 `pytest`（测试用）。
- **D11 全局拟合**：`all_conditions` 传单一常量（§6-Stage3）。
- **D12 模块选择器**：`legacy` + 论文参数（§6-Stage4）。
- **D13 时间轴**：`frame_idx`=0/1/2 作帧轴，`time_hours`=0/72/120 作时间轴（§6-Stage1-6）；
  SPOT 用的 uid 在 Stage 6 拼成 `{fov}_{track_id}_{frame_idx}`（帧号必须在最后一段）。
- **D14 像素分辨率**：`PIXEL_RES = 1.0`（人类合作者确认），长度/面积单位为像素单位。
- **D15 景观是否被天数/大小主导**：不预先回归面积，照论文原样如实报告，用 F6b
  （每簇 day 组成）作为证据图；若人类合作者之后要求，再开 `regress_out_confounders(continuous=[area])` 重跑。

## 8. 留给构建者的部分

- 具体的函数签名适配（v2.0.1 API 与论文版 0.1.1 脚本的差异）。
- 图的美术细节（字号、配色、dpi、图幅；默认 dpi=600，字体用默认，簇色用论文的 blue→red 渐变）。
- 特征提取的并行策略与缓存粒度。
- 测试用例的具体切分（用合成小掩码，避免测试依赖 1.2 GB 真实数据）。

## 9. 已知风险与退化路径

| 风险 | 检测方式 | 退化路径 |
|---|---|---|
| 掩码与原图错位 > 8 px | Stage 1 的偏移搜索输出 | 记录偏移量；若 > 8 px 则报错停下，人工确认 |
| `pacmap`/`annoy` 装不上 | `uv pip install` 失败 | 跳过 pacmap，只用 umap |
| 特征提取太慢（> 2 h） | 单视野计时 | 先跑单视野冒烟，再扩到全部；`n_jobs` 调整 |
| **track 全退化成长度 1** | Stage 6 的硬断言 | 立即报错（这是时间编码错误，不是数据问题） |
| 6741 实例的 UMAP 无结构 | 轮廓系数 / 视觉 | 按论文允许范围调 `n_neighbors`（35–100），记录每次尝试 |
| 跨天匹配率过低（< 30%） | Stage 6 QC | 输出诊断图（匹配前后质心叠加），bbox IoU 阈值扫描，图上标注 |
| 合并导致 track 极短（长度 1–2） | track 长度分布 | 保留但单独统计；轨迹图只画长度 ≥ 2 的 track，图注说明 |
| **景观主要由天数/大小驱动** | F6b（每簇 day 组成） | 先如实报告；若人类合作者要求，再开 `regress_out_confounders(continuous=[area])` 重跑 |

## 10. 验收标准

1. `python repro/scripts/run_all.py` 从零到全部产物无人工干预（除第一次装依赖）。
2. `pytest repro/tests` 全绿。
3. `output/report.md` 里数字自洽：实例数之和 = 6741（减去零面积/解析失败者，且报告写明差额）。
4. `output/figures/` 至少包含 F1–F11（含 F6b、F10 两版）各一张，能在报告里对上"这是论文哪张图"。
5. 簇编号（1…K）按平均 eccentricity 单调递增，报告里给出该单调性检查结果。
6. 所有图能离线复现（固定 `random_state`）：重跑两次 UMAP 坐标一致（写进测试或验收脚本）。
7. 长度 ≥ 2 的 track 数量 > 0，且报告给出具体数字。
