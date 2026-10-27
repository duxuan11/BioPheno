from biopheno import config as cfg


def test_days_and_time_axis_are_consistent():
    assert cfg.DAYS == {"day0": 0, "day3": 1, "day5": 2}
    assert [cfg.TIME_HOURS[d] for d in ("day0", "day3", "day5")] == [0.0, 72.0, 120.0]


def test_paper_parameters_are_pinned():
    # 与论文默认一致：n_neighbors=100, min_dist=0.5, spread=0.5, metric='euclidean', random_state=0
    assert cfg.UMAP_PARAMS == {
        "n_neighbors": 100,
        "min_dist": 0.5,
        "spread": 0.5,
        "metric": "euclidean",
        "random_state": 0,
    }
    assert cfg.VARIANCE_THRESHOLD_SIGMA == 2.0
    assert cfg.MODULE_SELECTION == "legacy"
    assert cfg.PIXEL_RES == 1.0
