import numpy as np
import pandas as pd

from biopheno import instances as inst


def test_contours_are_row_col_and_resampled(make_mask_and_image):
    """守口径：轮廓点是 (row, col) 顺序且等弧长重采样到 n_points 个 float64 点。"""
    # n_blobs=4：实例横向铺到第 309 列（> 250），纵向最深只到第 188 行（< 200），
    # 两个阈值不对称，才能把 (row, col) 与 (x, y) 的互换顶出来。
    _image, labels = make_mask_and_image(n_blobs=4)
    contours, _ = inst.extract_contours(labels, n_points=100)
    assert len(contours) == 4
    for c in contours.values():
        assert c.shape == (100, 2)
        assert c.dtype == np.float64
        # 行坐标在 [0, H)、列坐标在 [0, W)
        assert c[:, 0].max() < labels.shape[0] and c[:, 1].max() < labels.shape[1]
    # 若实现把 (row, col) 和 (x, y) 弄反：第 0 列会拿到列坐标（最大 308.5 > 200）
    # 而第 1 列会拿到行坐标（最大 188.5 < 250），下面两条断言同时失败。
    all_points = np.vstack(list(contours.values()))
    assert all_points[:, 0].max() < 200
    assert all_points[:, 1].max() > 250


def test_mask_smaller_than_image_is_padded_not_crashing(make_mask_and_image):
    """守边界：掩码比原图小时补边到原图尺寸而不崩，偏移仍在 max_shift 窗口内。"""
    image, labels = make_mask_and_image(n_blobs=2)
    small = labels[:-3, :-5]
    aligned, (dy, dx) = inst.align_mask_to_image(image, small, max_shift=8)
    # 画布尺寸回到原图，不被小掩码带偏
    assert aligned.shape == image.shape
    # 偏移必须落在搜索窗口里，不能报一个超限的估计
    assert abs(dy) <= 8 and abs(dx) <= 8


def test_align_accepts_rgb_image_and_returns_2d_labels(make_mask_and_image):
    """守回归：RGB 三通道图也能配准，返回仍是二维标签画布（前两维尺寸）。"""
    # 回归：这个模块自己的 _read_image 返回的就是 RGB 三通道图，若把 image.shape（三维）
    # 拿去分配画布，赋值二维 labels 时会 ValueError: could not broadcast (40,50) into (40,50,3)。
    image, labels = make_mask_and_image(n_blobs=2)
    rgb = np.stack([image, image, image], axis=-1)
    assert rgb.shape == (*image.shape, 3)
    aligned, (dy, dx) = inst.align_mask_to_image(rgb, labels, max_shift=8)
    # 契约：返回值是二维标签画布，尺寸等于原图的前两维
    assert aligned.shape == rgb.shape[:2] == labels.shape
    assert aligned.dtype == labels.dtype
    # 标签 1、2 都要保留、背景为 0（不许整幅清零或串到第三维）
    assert np.array_equal(np.unique(aligned), [0, 1, 2])
    assert abs(dy) <= 8 and abs(dx) <= 8


def test_build_catalog_uid_has_frame_last(make_mask_and_image, tmp_path):
    """守契约：instance_uid 末段是 frame_idx（day3 → 1），label 在倒数第二段。"""
    image, labels = make_mask_and_image(n_blobs=4)
    cat = inst.build_catalog("day3", "A1", image=image, labels=labels)
    uid = cat["meta"]["instance_uid"].iloc[0].split("_")
    assert uid[0] == "A1" and int(uid[-1]) == 1  # frame_idx: day3 -> 1
    assert uid[-2].isdigit()  # label 在倒数第二段
    # day3 → 72.0 h，必须沿用 config.TIME_HOURS 的口径
    assert cat["meta"]["time_hours"].unique().tolist() == [72.0]


def test_build_catalog_records_alignment_offset(make_mask_and_image):
    """守口径：build_catalog 必须回带 (dy, dx) 对齐偏移，与单独调用 align_mask_to_image 一致。"""
    # Spec 风险表要求 Stage 1 记录对齐偏移（> 8 px 需人工确认），所以 build_catalog 必须把它带出来。
    image, labels = make_mask_and_image(n_blobs=2)
    cat = inst.build_catalog("day0", "A1", image=image, labels=labels)
    _aligned, offset = inst.align_mask_to_image(image, labels, max_shift=8)
    assert isinstance(cat["align_offset"], tuple) and len(cat["align_offset"]) == 2
    # 内部算出的偏移必须与单独调用的结果完全相同（不能跳过配准直接给 (0, 0)）
    assert cat["align_offset"] == offset


def test_tiny_object_is_flagged_not_dropped(make_mask_and_image):
    """守口径：过小实例只打 is_small 标记不删行，meta 仍保留两颗。"""
    image, labels = make_mask_and_image(n_blobs=2, tiny_area=20)
    cat = inst.build_catalog("day0", "A1", image=image, labels=labels)
    assert len(cat["meta"]) == 2
    # tiny_area=20 那颗被判小，另一颗正常：恰好 1 个 True
    assert cat["meta"]["is_small"].sum() == 1


def test_label_without_usable_contour_is_still_in_meta(make_mask_and_image):
    """守口径：轮廓被丢的实例仍在 meta 里，带 is_small / n_contours_discarded 痕迹但不进 contours。"""
    # 单像素标签的候选轮廓只有 5 个点，会被丢弃；但"只打标不删"要求它仍出现在 meta 里，
    # 带着 is_small 与 n_contours_discarded 的痕迹（contours/patches 无条目）。
    image, labels = make_mask_and_image(n_blobs=2, tiny_area=1)
    cat = inst.build_catalog("day0", "A1", image=image, labels=labels)
    assert sorted(cat["meta"]["label"].tolist()) == [1, 2]
    tiny = cat["meta"].set_index("label").loc[1]
    assert int(tiny["n_contours_discarded"]) >= 1
    assert bool(tiny["is_small"])
    assert int(tiny["area_px"]) == 1
    # 没有可用轮廓时退回掩码包围盒：单像素实例在 fixture 里落在 (70, 50)
    assert (int(tiny["r0"]), int(tiny["c0"]), int(tiny["r1"]), int(tiny["c1"])) == (70, 50, 70, 50)
    assert 1 not in cat["contours"] and 1 not in cat["patches"]


def test_border_patch_is_padded_with_zeros(make_mask_and_image):
    """守契约：native 补丁恒为 bbox + 2*pad，越界处填 0；SIFT 补丁固定 64×64。"""
    # 契约：先外扩 bbox 再裁剪，native 形状恒为 bbox + 2*pad；越界部分填 0（黑）。
    image, labels = make_mask_and_image(n_blobs=2, touch_border=True)
    contours, _ = inst.extract_contours(labels, n_points=100)
    pad = 5
    patches = inst.crop_patches(image, contours, pad=pad, size=(64, 64))
    r0, c0, r1, c1 = inst.bbox_of_contour(contours[1])
    native = patches[1]["native"]
    assert native.shape == (r1 - r0 + 1 + 2 * pad, c1 - c0 + 1 + 2 * pad)
    # 贴左边界的实例：外扩出的 pad 列完全在图像之外，必须整列为 0
    assert not native[:, :pad].any()
    # 图像内的部分仍然有真实像素（否则说明裁空了）
    assert native[:, pad:].any()
    # SIFT 补丁尺寸恒定，与 native 不同
    assert patches[1]["sift"].shape == (64, 64)


def test_border_object_is_flagged_not_dropped(make_mask_and_image):
    """守口径：贴边实例只置 touches_border，不删行；另一颗不被误标。"""
    image, labels = make_mask_and_image(n_blobs=2, touch_border=True)
    cat = inst.build_catalog("day0", "A1", image=image, labels=labels)
    # 只有贴边那颗被标记
    assert cat["meta"]["touches_border"].sum() == 1
    assert len(cat["meta"]) == 2


def test_save_and_load_roundtrip(make_mask_and_image, tmp_path):
    """守口径：save_catalog → load_catalog 后 meta 列名、列序与取值不变。"""
    image, labels = make_mask_and_image(n_blobs=2)
    cat = inst.build_catalog("day0", "A2", image=image, labels=labels)
    inst.save_catalog(cat, tmp_path)
    back = inst.load_catalog(tmp_path)
    # CSV 往返不保证 dtype（int64 可能读成 object），比的是取值与列序，不是 dtype
    pd.testing.assert_frame_equal(cat["meta"], back["meta"], check_dtype=False)
