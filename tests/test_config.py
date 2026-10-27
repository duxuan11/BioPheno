from biopheno import config as cfg


def test_days_and_time_axis_are_consistent():
    """守口径：day→连续帧号的映射与 TIME_HOURS 必须对齐（day3=72h、day5=120h）。"""
    # 真实天数 0/3/5 的差是 3 和 2，SPOT 的标量 d_Frame 表达不了，只能压成 0/1/2 这三个连续帧号。
    assert cfg.DAYS == {"day0": 0, "day3": 1, "day5": 2}
    # 采样点间隔是 72h / 48h（不等宽）：时间图用小时轴，若误当等距帧轴会画错横轴刻度。
    assert [cfg.TIME_HOURS[d] for d in ("day0", "day3", "day5")] == [0.0, 72.0, 120.0]


def test_paper_parameters_are_pinned():
    """守口径：UMAP、方差阈值、模块选择、像素边长都钉在论文值上，漂了就整条流水线不可复现。"""
    # 与论文默认一致：n_neighbors=100, min_dist=0.5, spread=0.5, metric='euclidean', random_state=0
    assert cfg.UMAP_PARAMS == {
        "n_neighbors": 100,
        "min_dist": 0.5,
        "spread": 0.5,
        "metric": "euclidean",
        "random_state": 0,
    }
    assert cfg.VARIANCE_THRESHOLD_SIGMA == 2.0  # 论文用 2σ 不是 3σ：改大只会少删特征、改小会删掉 shape 信息
    assert cfg.MODULE_SELECTION == "legacy"  # v2.0.1 默认的 'multiscale' 是另一套方法，这里是论文 Fig.3d 口径
    assert cfg.PIXEL_RES == 1.0  # 这份数据没有标尺，面积/长度一律按像素计，不能悄悄改成真实 µm
