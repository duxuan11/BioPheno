"""Stage 6（跨天配准 + tracklet 匹配 + SPOT uid）的测试：全部合成数据。

三个契约测试（一对一/合并可区分、SPOT uid 的帧号在最后一段且 track 不全退化成长度 1、
非法 uid 早失败）取自任务说明。其余测试盯住实现里最容易悄悄错的地方：

- 配准方向：`register_frame` 返回的偏移是**加到下一帧坐标上**得到上一帧坐标系坐标，
  符号搞反时匹配会整体错位，所以用已知平移的合成掩码钉住符号；
- 偏移超限/上游抛异常时必须退回 (0, 0) 并留告警，不能中断整轮；
- 匹配必须与输入行序无关（并列 IoU 用 uid 定序），否则同一份数据两次跑出不同 track；
- 合并里被吸收的实例**不**继承幸存者的轨迹（否则同一 (fov, track, frame) 会出现两次，
  SPOT 的 `construct_obj_traj_from_uniq_obj_ids` 会把它当成两段），split 的新生实例同理；
- 三帧链（day0→day3→day5）必须靠并查集串成一条 track；
- day3→day5 的偏移用 day0 参考系做**复合**（O5 − O3），而不是直接拿 day5→day0；
- `run_stage6` 的硬断言：track 全退化成长度 1 时报中文错误、且不落盘半套结果。
"""

import json

import numpy as np
import pandas as pd
import pytest
import tifffile

from biopheno import config as cfg
from biopheno import instances as inst
from biopheno import temporal as temp

# 真实数据的文件名后缀（数据契约，与 biopheno.instances._NAME_SUFFIX 一致）
_NAME_SUFFIX: str = "_多聚焦融合"

# 匹配表必须有的列（顺序也钉住：下游 QC/报告按这个顺序读）
_MATCH_COLUMNS: tuple[str, ...] = ("prev_uid", "cur_uid", "iou", "cent_dist", "event")


# ---------------------------------------------------------------- 合成 meta 工具


def _meta(day: str, fov: str, boxes: dict[int, tuple[int, int, int, int]]) -> pd.DataFrame:
    """按 Stage 1 的真实列名造一帧 meta：boxes = {label: (r0, c0, r1, c1)}（闭区间）。

    instance_uid 沿用 `f"{fov}_{label}_{frame_idx}"`，质心取 bbox 中心，area_px 取 bbox 面积。
    """
    frame_idx = int(cfg.DAYS[day])
    rows = []
    for label, (r0, c0, r1, c1) in sorted(boxes.items()):
        rows.append(
            {
                "instance_uid": f"{fov}_{int(label)}_{frame_idx}",
                "day": day,
                "fov": fov,
                "frame_idx": frame_idx,
                "time_hours": float(cfg.TIME_HOURS[day]),
                "label": int(label),
                "centroid_row": (r0 + r1) / 2.0,
                "centroid_col": (c0 + c1) / 2.0,
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "area_px": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                "is_small": False,
                "touches_border": False,
            }
        )
    return pd.DataFrame(rows)


def _shift_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """把布尔掩码整体平移 (dy, dx)（越界部分丢弃），用来造"内容真的挪了位置"的下一帧。"""
    out = np.zeros_like(mask)
    h, w = mask.shape
    out[max(0, dy) : h - max(0, -dy), max(0, dx) : w - max(0, -dx)] = mask[
        max(0, -dy) : h - max(0, dy), max(0, -dx) : w - max(0, dx)
    ]
    return out


def _blob_mask(shape: tuple[int, int], centers: list[tuple[int, int]], radius: int = 18) -> np.ndarray:
    """按 (row, col) 圆心与半径画实心圆的并集，返回同形状的布尔前景掩码。"""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros(shape, dtype=bool)
    for r, c in centers:
        mask |= (yy - r) ** 2 + (xx - c) ** 2 <= radius**2
    return mask


# ---------------------------------------------------------------- fixture 组合


@pytest.fixture
def matches(fake_two_frames):
    """`fake_two_frames` 两帧的匹配表（偏移 (0, 0)：两帧本来就同坐标）。"""
    prev_meta, cur_meta = fake_two_frames
    return temp.match_frames(
        prev_meta, cur_meta, offset=(0, 0), min_iou=cfg.MIN_BBOX_IOU, max_cent_dist=cfg.MAX_CENT_DIST_PX
    )


@pytest.fixture
def all_meta(fake_two_frames):
    """两帧 meta 纵向拼接（index 重置），作为 `assign_tracks` 的"全集"输入。"""
    prev_meta, cur_meta = fake_two_frames
    return pd.concat([prev_meta, cur_meta], ignore_index=True)


# ---------------------------------------------------------------- 任务说明里的三个契约测试


def test_one_to_one_and_merge_are_distinguished(fake_two_frames):
    """守契约：合并（多对一）与一对一必须分成两种 event，且列顺序钉住 _MATCH_COLUMNS。"""
    prev_meta, cur_meta = fake_two_frames  # 2 个实例 -> 1 个（合并）+ 1 个（一对一保留）
    matches = temp.match_frames(
        prev_meta, cur_meta, offset=(0, 0), min_iou=cfg.MIN_BBOX_IOU, max_cent_dist=cfg.MAX_CENT_DIST_PX
    )
    assert list(matches.columns) == list(_MATCH_COLUMNS)
    # 两种事件都要出现：全塞进同一种就等于没区分开
    assert (matches["event"] == "one_to_one").sum() >= 1
    assert (matches["event"] == "merge").sum() >= 1
    # 被合并的上一帧实例必须指向同一颗下一帧实例（多对一），并且是上一帧里剩下的那个
    merges = matches[matches["event"] == "merge"]
    one = matches[matches["event"] == "one_to_one"].set_index("prev_uid")["cur_uid"]
    assert len(merges) == 1
    assert merges["cur_uid"].iloc[0] in set(one.values)


def test_spot_uid_has_frame_as_last_field_and_tracks_are_not_all_singletons(matches, all_meta):
    """守契约：spot_uid 末段是帧号、track_id 是整数，且不许全部退化成单帧 track。"""
    tracks = temp.assign_tracks([matches], all_meta)
    uid = tracks["spot_uid"].iloc[0]
    assert uid.split("_")[-1].isdigit()  # SPOT 内部写死 split('_')[-1] 取帧号
    # 至少一条长度 ≥2 的 track：顶住"每帧都另开新 track"的退化
    assert (tracks.groupby("track_id").size() >= 2).sum() > 0
    # 三列齐全且 track_id 必须是整数（SPOT 的 uid 里不能出现 '0.0' 这种）
    assert {"instance_uid", "spot_uid", "track_id", "fov", "frame_idx", "time_hours", "day"} <= set(tracks.columns)
    assert pd.api.types.is_integer_dtype(tracks["track_id"])
    assert tracks["spot_uid"].is_unique


def test_bad_uid_format_is_rejected():
    """守边界：track_id 是非数字串时 spot_uid_from 必须早失败，别让 SPOT 解析出脏 uid。"""
    tracks = pd.DataFrame({"fov": ["A1"], "track_id": [0], "frame_idx": [0]})
    with pytest.raises(ValueError):
        temp.spot_uid_from(tracks.assign(track_id="t-0"))  # 含 '-' 会被 SPOT 解析错，必须早失败


# ---------------------------------------------------------------- uid 校验的其余分支


def test_spot_uid_from_rejects_fov_with_separator():
    """守边界：fov 含 '_' 会多切出一段导致 SPOT 字段串位，必须抛 ValueError。"""
    # fov 里带 '_' 会多出一段，SPOT 按 '_' 切完取 [0,1,2] 就会串位
    bad = pd.DataFrame({"fov": ["A_1"], "track_id": [3], "frame_idx": [1]})
    with pytest.raises(ValueError, match="fov"):
        temp.spot_uid_from(bad)


def test_spot_uid_from_rejects_non_integer_track_id():
    """守边界：小数 track_id（1.5）非法，抛 ValueError 而不是四舍五入成 1。"""
    bad = pd.DataFrame({"fov": ["A1"], "track_id": [1.5], "frame_idx": [1]})
    with pytest.raises(ValueError, match="track_id"):
        temp.spot_uid_from(bad)


def test_spot_uid_from_accepts_integer_float_and_builds_three_fields():
    """正向：整数值的 7.0 与 np.int64(8) 都要接受，拼成 fov_track_frame 三段。"""
    ok = pd.DataFrame({"fov": ["A1", "A2"], "track_id": [7.0, np.int64(8)], "frame_idx": [2, 0]})
    # 落成 "7"/"8"，不许出现 "7.0" 这种 SPOT 认不出的段
    assert list(temp.spot_uid_from(ok)) == ["A1_7_2", "A2_8_0"]


def test_spot_uid_from_rejects_bool_track_id():
    """守边界：True 也算非法值，必须在 int 判断之前排除并点明"布尔"。"""
    # 契约：布尔值也是非法值，与其它非整数一样抛 ValueError（不另开一个 TypeError 分支）
    bad = pd.DataFrame({"fov": ["A1"], "track_id": [True], "frame_idx": [1]})
    with pytest.raises(ValueError, match="布尔"):
        temp.spot_uid_from(bad)


# ---------------------------------------------------------------- track 分配：合并 / 分裂 / 链


def test_merge_absorbed_instance_gets_its_own_single_frame_track(matches, all_meta):
    """守口径：被吸收的实例自成长度 1 的孤立 track，不继承幸存者轨迹、不与它重号。"""
    # 被吸收的上一帧实例（B，uid=A1_2_0）不能继承合并后实例（M）的轨迹：
    # 否则 (fov, track_id, frame_idx) 会与 A 的条目重复，SPOT 的轨迹解析会错。
    absorbed_uid = matches.loc[matches["event"] == "merge", "prev_uid"].iloc[0]
    tracks = temp.assign_tracks([matches], all_meta)
    by_uid = tracks.set_index("instance_uid")
    survivor_uid = matches.loc[matches["event"] == "one_to_one"].set_index("prev_uid").index[0]
    absorbed = by_uid.loc[absorbed_uid]
    # 被吸收的实例自己是长度为 1 的孤立 track（它的旧轨迹到此为止，不再延续到 M）
    assert (tracks["track_id"] == absorbed["track_id"]).sum() == 1
    assert absorbed["track_id"] != by_uid.loc[survivor_uid, "track_id"]
    assert tracks["spot_uid"].is_unique


def test_split_child_starts_new_track_and_uids_stay_unique():
    """守口径：split 的主 child 继承父轨迹，新生那颗另起单帧轨迹且 spot_uid 不重复。"""
    prev = _meta("day0", "A1", {1: (20, 20, 80, 100)})
    cur = _meta("day3", "A1", {1: (20, 20, 50, 60), 2: (50, 60, 80, 100)})
    table = temp.match_frames(prev, cur, offset=(0, 0), min_iou=cfg.MIN_BBOX_IOU, max_cent_dist=cfg.MAX_CENT_DIST_PX)
    assert (table["event"] == "split").sum() == 1
    tracks = temp.assign_tracks([table], pd.concat([prev, cur], ignore_index=True))
    assert tracks["spot_uid"].is_unique
    # 主 child（iou 更高/uid 更小）继承父轨迹；新生的那颗另起一条单帧轨迹
    by_uid = tracks.set_index("instance_uid")
    assert by_uid.loc["A1_1_1", "track_id"] == by_uid.loc["A1_1_0", "track_id"]
    child = by_uid.loc["A1_2_1"]
    assert (tracks["track_id"] == child["track_id"]).sum() == 1
    assert child["track_id"] != by_uid.loc["A1_1_0", "track_id"]


def test_three_frame_chain_becomes_one_track():
    """守口径：day0→day3→day5 必须并查集串成一条长度 3 的 track，帧号/时间严格递增。"""
    meta0 = _meta("day0", "A1", {1: (100, 100, 140, 140)})
    meta1 = _meta("day3", "A1", {1: (100, 100, 140, 140)})
    meta2 = _meta("day5", "A1", {1: (100, 100, 140, 140)})
    tables = [
        temp.match_frames(a, b, offset=(0, 0), min_iou=cfg.MIN_BBOX_IOU, max_cent_dist=cfg.MAX_CENT_DIST_PX)
        for a, b in ((meta0, meta1), (meta1, meta2))
    ]
    tracks = temp.assign_tracks(tables, pd.concat([meta0, meta1, meta2], ignore_index=True))
    sizes = tracks.groupby("track_id").size()
    # 只能有唯一一条 track、长度正好 3（并查集不许拆成 1+2 或 2+1）
    assert sizes.tolist() == [3]
    track_id = int(tracks["track_id"].iloc[0])
    assert list(tracks["frame_idx"]) == [0, 1, 2]
    assert list(tracks["time_hours"]) == [0.0, 72.0, 120.0]
    assert list(tracks["spot_uid"]) == [f"A1_{track_id}_{f}" for f in (0, 1, 2)]


def test_tracks_never_span_two_fovs():
    """守口径：track 不跨 fov；track_id 全局唯一，两个视野各一条长度 2 的轨迹。"""
    a0 = _meta("day0", "A1", {1: (100, 100, 140, 140)})
    a1 = _meta("day3", "A1", {1: (100, 100, 140, 140)})
    b0 = _meta("day0", "A2", {1: (100, 100, 140, 140)})
    b1 = _meta("day3", "A2", {1: (100, 100, 140, 140)})
    tables = [
        temp.match_frames(a0, a1, offset=(0, 0), min_iou=cfg.MIN_BBOX_IOU, max_cent_dist=cfg.MAX_CENT_DIST_PX),
        temp.match_frames(b0, b1, offset=(0, 0), min_iou=cfg.MIN_BBOX_IOU, max_cent_dist=cfg.MAX_CENT_DIST_PX),
    ]
    tracks = temp.assign_tracks(tables, pd.concat([a0, a1, b0, b1], ignore_index=True))
    assert tracks.groupby("track_id")["fov"].nunique().max() == 1
    # 两个视野各自一条长度 2 的轨迹（track_id 全局唯一，不按视野重号）
    assert sorted(tracks.groupby("track_id").size().tolist()) == [2, 2]


# ---------------------------------------------------------------- 匹配的确定性与偏移


def test_match_is_deterministic_and_ties_break_on_uid():
    """守确定性：并列 IoU 时按 prev_uid 定序，同一份数据换行序结果必须逐格相等。"""
    # 上一帧两颗实例 bbox 完全重叠（并列 IoU=1），必须按 (-iou, prev_uid, cur_uid) 定序：
    # 否则同一份数据跑两次可能把不同的上一帧实例认成"幸存者"。
    prev = _meta("day0", "A1", {2: (20, 20, 60, 60), 1: (20, 20, 60, 60)})
    cur = _meta("day3", "A1", {1: (20, 20, 60, 60)})
    baseline = temp.match_frames(prev, cur, offset=(0, 0), min_iou=cfg.MIN_BBOX_IOU, max_cent_dist=cfg.MAX_CENT_DIST_PX)
    assert baseline.loc[baseline["event"] == "one_to_one", "prev_uid"].tolist() == ["A1_1_0"]
    for order in ([1, 0], [0, 1]):
        shuffled = temp.match_frames(
            prev.iloc[order].reset_index(drop=True),
            cur,
            offset=(0, 0),
            min_iou=cfg.MIN_BBOX_IOU,
            max_cent_dist=cfg.MAX_CENT_DIST_PX,
        )
        pd.testing.assert_frame_equal(shuffled, baseline)


def test_match_uses_registration_offset_to_compare_boxes():
    """守符号：偏移是加到下一帧坐标上，-70 能把下移 70 行的实例搬回并得到 IoU=1。"""
    prev = _meta("day0", "A1", {1: (20, 20, 60, 60)})
    cur = _meta("day3", "A1", {1: (90, 20, 130, 60)})  # 内容下移 70 行
    # 偏移 0：既无 bbox 重叠、质心距离 70 > 60，不应成为候选
    none = temp.match_frames(prev, cur, offset=(0, 0), min_iou=cfg.MIN_BBOX_IOU, max_cent_dist=cfg.MAX_CENT_DIST_PX)
    assert none.empty
    # 把下一帧搬回上一帧坐标系（偏移 -70）后完全重合
    hit = temp.match_frames(prev, cur, offset=(-70, 0), min_iou=cfg.MIN_BBOX_IOU, max_cent_dist=cfg.MAX_CENT_DIST_PX)
    assert hit["event"].tolist() == ["one_to_one"]
    assert hit["iou"].iloc[0] == pytest.approx(1.0)
    assert hit["cent_dist"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------- register_frame


def test_register_frame_recovers_known_shift():
    """守符号：内容下移 5 行右移 3 列时返回 (-5, -3)——偏移是加在下一帧坐标上。"""
    shape = (240, 320)
    prev = _blob_mask(shape, [(60, 60), (150, 200), (100, 280)])
    cur = _shift_mask(prev, 5, 3)  # 内容下移 5 行、右移 3 列
    # 返回的偏移是"加到下一帧坐标上"还原到上一帧的位移，所以是 (-5, -3)
    assert temp.register_frame(prev, cur, max_shift=cfg.ALIGN_MAX_SHIFT_PX) == (-5, -3)


def test_register_frame_ignores_intensity_and_works_on_masks():
    """守口径：只用前景二值（labels > 0），标签值 3/7 的差异不影响偏移估计。"""
    # 掩码二值即使被"强度"污染（这里换成标签值 3/7）也应等价：我们用 labels > 0 做输入
    shape = (240, 320)
    prev = _blob_mask(shape, [(80, 80), (160, 220)]).astype(np.uint16)
    cur = _shift_mask(prev > 0, -4, 2).astype(np.uint16) * 7
    # 内容上移 4 行、左移 2 列 → 还原偏移 (4, -2)
    assert temp.register_frame(prev, cur, max_shift=cfg.ALIGN_MAX_SHIFT_PX) == (4, -2)


def test_register_frame_falls_back_when_shift_exceeds_limit():
    """守降级：平移 40 px 超 ±8 px 上限时退回 (0, 0)，并留一条含"超过"的告警。"""
    shape = (240, 320)
    prev = _blob_mask(shape, [(60, 60), (150, 200)])
    cur = _shift_mask(prev, 40, 0)
    warnings_out: list[str] = []
    offset = temp.register_frame(prev, cur, max_shift=cfg.ALIGN_MAX_SHIFT_PX, warnings_out=warnings_out)
    assert offset == (0, 0)
    # 恰好一条告警：既不重复记录，也不许静默吞掉
    assert len(warnings_out) == 1 and "超过" in warnings_out[0]


def test_register_frame_falls_back_when_upstream_raises(monkeypatch):
    """守降级：上游相位相关抛异常时退回 (0, 0) 并留"配准失败"告警，不中断整轮。"""

    def boom(*args, **kwargs):
        """替身：模拟上游 translation_register_correlation 直接抛异常。"""
        raise RuntimeError("上游相位相关爆了")

    monkeypatch.setattr(temp.r2d, "translation_register_correlation", boom)
    warnings_out: list[str] = []
    offset = temp.register_frame(np.ones((8, 8), dtype=bool), np.ones((8, 8), dtype=bool), 8, warnings_out=warnings_out)
    assert offset == (0, 0)
    assert len(warnings_out) == 1 and "配准失败" in warnings_out[0]


def test_register_frame_falls_back_on_empty_mask():
    """守降级：两张空掩码不做估计，退回 (0, 0) 并留"空掩码"告警。"""
    warnings_out: list[str] = []
    offset = temp.register_frame(
        np.zeros((8, 8), dtype=bool), np.zeros((8, 8), dtype=bool), 8, warnings_out=warnings_out
    )
    assert offset == (0, 0)
    assert len(warnings_out) == 1 and "空掩码" in warnings_out[0]


# ---------------------------------------------------------------- build_tracks：偏移复合


def _toy_metas() -> dict[str, pd.DataFrame]:
    """三帧 toy meta：day0 在 (20..60)、day3 与 day5 的内容在 day0 坐标系的同一位置。

    在各自的原始坐标系里，day3/day5 的 bbox 都被下移了 70 行；能否匹配完全取决于
    偏移复合是否正确（见 `test_build_tracks_composes_pair_offset_via_reference_frame`）。
    """
    return {
        "day0/A1": _meta("day0", "A1", {1: (20, 20, 60, 60)}),
        "day3/A1": _meta("day3", "A1", {1: (90, 20, 130, 60)}),
        "day5/A1": _meta("day5", "A1", {1: (90, 20, 130, 60)}),
    }


def test_build_tracks_composes_pair_offset_via_reference_frame(monkeypatch):
    """守口径：1→2 的偏移必须用 O5−O3 复合得 (0, 0)，直接拿 O5 会整体错位 70 行。"""
    metas = _toy_metas()
    masks = {"day0/A1": np.zeros((8, 8)), "day3/A1": np.ones((8, 8)), "day5/A1": np.full((8, 8), 2.0)}
    # 用 stub 直接指定"各帧 → day0 参考系"的偏移（真实 register_frame 会因 ±8 上限退回 0，
    # 这里要单独验证复合公式，不受搜索范围限制）
    to_reference = {id(masks["day3/A1"]): (-70, 0), id(masks["day5/A1"]): (-70, 0)}

    def fake_register(prev, cur, max_shift, warnings_out=None, kinds_out=None):
        """替身：按 cur 掩码的 identity 直接返回预设的"到 day0 参考系"偏移元组。"""
        return to_reference[id(cur)]

    monkeypatch.setattr(temp, "register_frame", fake_register)
    qc: dict = {}
    tracks = temp.build_tracks(metas, masks=masks, qc=qc)
    # 0→1 用 O1=-70：day3 的 bbox 被搬回 day0 位置；1→2 必须用 O5-O3=(0,0)，直接拿 O5 会错位 70 行
    assert qc["fovs"]["A1"]["registration"]["offset_to_reference"] == {"0": [0, 0], "1": [-70, 0], "2": [-70, 0]}
    assert qc["fovs"]["A1"]["registration"]["pair_offsets"] == {"0->1": [-70, 0], "1->2": [0, 0]}
    assert qc["fovs"]["A1"]["pairs"]["1->2"]["n_one_to_one"] == 1
    sizes = tracks.groupby("track_id").size()
    # 复合算对了，三帧才可能串成唯一一条 track
    assert sizes.tolist() == [3]


def test_build_tracks_records_registration_warnings_in_qc(monkeypatch):
    """守告警：配准降级要同时进 QC 顶层 warnings 与逐视野 registration.warnings。"""
    metas = _toy_metas()
    masks = {key: np.zeros((8, 8)) for key in metas}

    def fake_register(prev, cur, max_shift, warnings_out=None, kinds_out=None):
        """替身：每次配准往 warnings_out 追加一条告警串并退回 (0, 0)。"""
        target = warnings_out if warnings_out is not None else temp._REGISTRATION_WARNINGS
        target.append("模拟配准失败：故意退回 (0, 0)")
        return (0, 0)

    monkeypatch.setattr(temp, "register_frame", fake_register)
    qc: dict = {}
    tracks = temp.build_tracks(metas, masks=masks, qc=qc)
    assert "模拟配准失败" in " ".join(qc["warnings"])
    assert len(qc["fovs"]["A1"]["registration"]["warnings"]) == 2  # day3、day5 各一次
    assert tracks["spot_uid"].is_unique


def test_build_tracks_rejects_two_metas_for_the_same_frame():
    """守边界：同一 (day, fov) 出现两份 meta 时报错，不许静默挑一份继续建轨。"""
    metas = {
        "day0/A1": _meta("day0", "A1", {1: (20, 20, 60, 60)}),
        "day0/A1_dup": _meta("day0", "A1", {1: (20, 20, 60, 60)}),
    }
    masks = {key: np.zeros((8, 8)) for key in metas}
    with pytest.raises(RuntimeError, match="day0/A1"):
        temp.build_tracks(metas, masks=masks)


def test_build_tracks_rejects_non_contiguous_frames():
    """守边界：缺中间帧（只有 0 与 2）必须报错，别让 SPOT 按 d_Frame=1 拆成两段。"""
    # 缺 day3（帧 1）：若照旧把它当成 0→2 的帧对，本阶段能建出长度 2 的 track，
    # 但 SPOT 的 d_Frame=1 解析会把它拆成两段，两边结论相反 —— 所以建轨前先报错。
    metas = {
        "day0/A1": _meta("day0", "A1", {1: (20, 20, 60, 60)}),
        "day5/A1": _meta("day5", "A1", {1: (20, 20, 60, 60)}),
    }
    masks = {key: np.zeros((8, 8)) for key in metas}
    with pytest.raises(RuntimeError, match="A1") as excinfo:
        temp.build_tracks(metas, masks=masks)
    message = str(excinfo.value)
    # 错误信息要同时给出实际帧号、缺的帧号，以及为什么要连续（SPOT 的 d_Frame）
    assert "[0, 2]" in message and "缺少帧 [1]" in message and "d_Frame" in message


def test_build_tracks_rejects_frame_axis_not_starting_at_zero():
    """守边界：帧轴从 day3（帧 1）起始、缺帧 0 时同样非法，错误信息要列出缺哪帧。"""
    # 帧轴直接从 day3（帧 1）开始，同样不符合 0/1/2 连续编码
    metas = {
        "day3/A1": _meta("day3", "A1", {1: (20, 20, 60, 60)}),
        "day5/A1": _meta("day5", "A1", {1: (20, 20, 60, 60)}),
    }
    masks = {key: np.zeros((8, 8)) for key in metas}
    # match 里的方括号是正则元字符，必须转义
    with pytest.raises(RuntimeError, match="缺少帧 \\[0\\]"):
        temp.build_tracks(metas, masks=masks)


def test_build_tracks_warns_when_composed_pair_offset_exceeds_limit(monkeypatch):
    """守边界：两个 ±8 内合法的偏移相减可到 16 px，必须逐帧对标记超限并留告警，不许裁剪。"""
    # 两个各自在 ±8 px 内被接受的偏移，相减可以到 16 px：必须留告警 + 逐帧对标记，且不静默裁剪
    metas = _toy_metas()
    masks = {"day0/A1": np.zeros((8, 8)), "day3/A1": np.ones((8, 8)), "day5/A1": np.full((8, 8), 2.0)}
    to_reference = {id(masks["day3/A1"]): (-8, 0), id(masks["day5/A1"]): (8, 0)}

    def fake_register(prev, cur, max_shift, warnings_out=None, kinds_out=None):
        """替身：day3 回参考系 -8、day5 回参考系 +8，复合出 16 px 的超限偏移。"""
        return to_reference[id(cur)]

    monkeypatch.setattr(temp, "register_frame", fake_register)
    qc: dict = {}
    temp.build_tracks(metas, masks=masks, qc=qc)

    registration = qc["fovs"]["A1"]["registration"]
    assert registration["pair_offsets"] == {"0->1": [-8, 0], "1->2": [16, 0]}
    pairs = qc["fovs"]["A1"]["pairs"]
    # 用 is False/True 钉住布尔类型：不许写成 NaN 或 0/1
    assert pairs["0->1"]["offset_exceeds_limit"] is False
    assert pairs["1->2"]["offset_exceeds_limit"] is True
    assert qc["totals"]["n_pairs_offset_exceeding_limit"] == 1
    assert any("1->2" in warning and "超过" in warning for warning in qc["warnings"])
    assert len(registration["warnings"]) == 1


def test_register_frame_tags_warning_kinds_for_each_fallback():
    """每种降级都要带机器可读的类型：报告只把 offset_exceeds_limit 当“被拒的估计”。"""
    shape = (240, 320)
    prev = _blob_mask(shape, [(60, 60), (150, 200)])
    # 1) 偏移超上限 → 被拒（不是“配准失败”）
    kinds: dict[str, int] = {}
    temp.register_frame(prev, _shift_mask(prev, 40, 0), cfg.ALIGN_MAX_SHIFT_PX, kinds_out=kinds)
    assert kinds == {temp.WARNING_KIND_OFFSET_EXCEEDS_LIMIT: 1}

    # 2) 空掩码
    kinds = {}
    temp.register_frame(np.zeros((8, 8), bool), np.zeros((8, 8), bool), 8, kinds_out=kinds)
    assert kinds == {temp.WARNING_KIND_EMPTY_MASK: 1}

    # 3) 上游异常
    kinds = {}

    def boom(*args, **kwargs):
        """替身：模拟上游相位相关抛异常，逼出第三种降级类型。"""
        raise RuntimeError("上游相位相关爆了")

    original = temp.r2d.translation_register_correlation
    temp.r2d.translation_register_correlation = boom
    try:
        temp.register_frame(np.ones((8, 8), bool), np.ones((8, 8), bool), 8, kinds_out=kinds)
    finally:
        temp.r2d.translation_register_correlation = original
    assert kinds == {temp.WARNING_KIND_UPSTREAM_ERROR: 1}


def test_build_tracks_counts_warning_kinds_separately_from_the_warning_list(monkeypatch):
    """告警类型混合时：字符串列表照旧给人读，`warning_kinds` 才是“被拒次数”的唯一来源。"""
    metas = _toy_metas()
    masks = {"day0/A1": np.zeros((8, 8)), "day3/A1": np.ones((8, 8)), "day5/A1": np.full((8, 8), 2.0)}
    # day3 → 参考系：空掩码（不是被拒）；day5 → 参考系：偏移超上限（被拒）。复合偏移再越界一次。
    plan = {
        id(masks["day3/A1"]): ("empty", (0, 0)),
        id(masks["day5/A1"]): ("rejected", (16, 0)),
    }

    def fake_register(prev, cur, max_shift, warnings_out=None, kinds_out=None):
        """替身：day3 走空掩码降级、day5 走偏移被拒，返回 (dy, dx) 元组。"""
        what, offset = plan[id(cur)]
        if what == "empty":
            temp._record_warning(
                warnings_out,
                "空掩码（前景像素 prev=0、cur=0），偏移退回 (0, 0)",
                temp.WARNING_KIND_EMPTY_MASK,
                kinds_out,
            )
        else:
            temp._record_warning(
                warnings_out,
                "配准偏移 (dy=16, dx=0) 超过 ±8 px 上限，偏移退回 (0, 0)",
                temp.WARNING_KIND_OFFSET_EXCEEDS_LIMIT,
                kinds_out,
            )
            return offset
        return (0, 0)

    monkeypatch.setattr(temp, "register_frame", fake_register)
    qc: dict = {}
    temp.build_tracks(metas, masks=masks, qc=qc)

    registration = qc["fovs"]["A1"]["registration"]
    # 三条字符串告警（空掩码 + 被拒 + 复合偏移越界），但只有 1 次是“被拒的相位相关估计”
    assert len(registration["warnings"]) == 3
    assert registration["warning_kinds"] == {
        temp.WARNING_KIND_EMPTY_MASK: 1,
        temp.WARNING_KIND_OFFSET_EXCEEDS_LIMIT: 1,
        temp.WARNING_KIND_COMPOUND_OFFSET_EXCEEDS_LIMIT: 1,
    }
    assert registration["warning_kinds"][temp.WARNING_KIND_OFFSET_EXCEEDS_LIMIT] != len(registration["warnings"])


def test_frame_pair_geometry_details_aligned_and_shifted_pairs():
    """合成掩码：一对本来对齐、一对故意错开 12 px → 只有前者算 `aligned_at_zero`。"""
    shape = (200, 240)
    prev = _blob_mask(shape, [(60, 60), (150, 200)])
    aligned = prev.copy()
    shifted = _shift_mask(prev, 12, 0)

    aligned_entry = temp.check_frame_pair_geometry(prev, aligned, cfg.ALIGN_MAX_SHIFT_PX)
    assert aligned_entry["aligned_at_zero"] is True
    assert aligned_entry["zero_offset_iou"] == pytest.approx(1.0)
    assert aligned_entry["best_offset"] == [0, 0]

    shifted_entry = temp.check_frame_pair_geometry(prev, shifted, cfg.ALIGN_MAX_SHIFT_PX)
    assert shifted_entry["aligned_at_zero"] is False
    assert shifted_entry["zero_offset_iou"] < shifted_entry["best_iou"]
    # 真实平移 12 px 确实超出 ±8 px 的搜索范围：最佳偏移落在窗口边界 -8 px（单位是像素，不是网格步）。
    # 但 IoU 增益已经足以判"未对齐"。
    assert shifted_entry["best_offset"] == [-cfg.ALIGN_MAX_SHIFT_PX, 0]
    assert shifted_entry["iou_gain"] > 0.0


def test_frame_pair_geometry_reports_real_pixel_offsets():
    """已知的真实像素平移必须报成同样多的像素（下采样只许省算力，不许改单位/量级）。"""
    shape = (200, 240)
    prev = _blob_mask(shape, [(60, 60), (150, 200)])
    # 内容下移 6 行、左移 3 列 → 还原到 prev 坐标的偏移是 (-6, +3) px
    entry = temp.check_frame_pair_geometry(prev, _shift_mask(prev, 6, -3), cfg.ALIGN_MAX_SHIFT_PX)

    assert entry["offset_unit"] == "px"
    assert entry["best_offset"] == [-6, 3]
    assert entry["best_iou"] == pytest.approx(1.0)
    assert entry["zero_offset_iou"] < entry["best_iou"]
    assert entry["aligned_at_zero"] is False


def test_registration_geometry_lands_in_qc_and_leaves_masks_untouched(monkeypatch):
    """几何核验进 QC，且不改任何掩码/匹配：它的输入是二值图，输出只多一块 QC。"""
    metas = _toy_metas()
    shape = (120, 160)
    base = _blob_mask(shape, [(60, 80)], radius=20)
    masks = {
        "day0/A1": base.astype(np.uint16),
        "day3/A1": base.astype(np.uint16),
        "day5/A1": _shift_mask(base, 12, 0).astype(np.uint16),
    }
    before = {key: np.array(mask, copy=True) for key, mask in masks.items()}

    def fake_register(prev, cur, max_shift, warnings_out=None, kinds_out=None):
        """替身：配准一律返回 (0, 0)，让几何核验成为本用例唯一的检查对象。"""
        return (0, 0)

    monkeypatch.setattr(temp, "register_frame", fake_register)
    qc: dict = {}
    tracks = temp.build_tracks(metas, masks=masks, qc=qc)

    geometry = qc["registration_geometry"]
    # 落进 QC 的偏移单位必须是像素：`max_shift_px` 就是真实扫描窗口，不能是网格步数
    assert geometry["offset_unit"] == "px"
    assert geometry["max_shift_px"] == cfg.ALIGN_MAX_SHIFT_PX
    assert geometry["n_pairs"] == 2 and geometry["all_aligned_at_zero"] is False
    assert geometry["pairs"]["A1"]["0->1"]["aligned_at_zero"] is True
    assert geometry["pairs"]["A1"]["1->2"]["aligned_at_zero"] is False
    # 故意错开 12 px 的帧对：最佳偏移顶在 ±8 px 的窗口边界（px，不是网格步）
    assert geometry["pairs"]["A1"]["1->2"]["best_offset"] == [-cfg.ALIGN_MAX_SHIFT_PX, 0]
    # 输入掩码一个像素都没变，track 也照常建出来
    for key, mask in masks.items():
        assert np.array_equal(mask, before[key])
    assert len(tracks) == 3


# ---------------------------------------------------------------- run_stage6


def _write_all_instances(root, metas: dict[str, pd.DataFrame]) -> None:
    """把各帧 meta 拼成 instances/all_instances.csv（run_stage6 读的 Stage 1 汇总）。"""
    outdir = root / "instances"
    outdir.mkdir(parents=True, exist_ok=True)
    pd.concat(list(metas.values()), ignore_index=True).to_csv(outdir / "all_instances.csv", index=False)


def _write_stage1_qc(root, offsets: dict[str, tuple[int, int]]) -> None:
    """写 Stage 1 的 stage1_qc.json（run_stage6 从这里读逐 (day, fov) 的对齐偏移）。"""
    outdir = root / "instances"
    outdir.mkdir(parents=True, exist_ok=True)
    qc = {
        key: {"n_instances": 1, "align_dy": int(dy), "align_dx": int(dx), "n_is_small": 0}
        for key, (dy, dx) in offsets.items()
    }
    (outdir / "stage1_qc.json").write_text(json.dumps(qc, ensure_ascii=False), encoding="utf-8")


def _setup_run_stage6(monkeypatch, tmp_path, *, write_stage1_qc: bool = True) -> dict[str, np.ndarray]:
    """造一套三帧×单视野的假数据：两个不动、不变的类器官 → 两条长度 3 的 track。"""
    shape = (240, 320)
    labels: dict[str, np.ndarray] = {}
    metas: dict[str, pd.DataFrame] = {}
    for day in ("day0", "day3", "day5"):
        labels[day] = np.zeros(shape, dtype=np.uint16)
        labels[day][30:70, 40:80] = 1
        labels[day][150:190, 220:260] = 2
        image = np.where(labels[day] > 0, 140, 200).astype(np.uint8)
        metas[f"{day}/A1"] = inst.build_catalog(day, "A1", image=image, labels=labels[day])["meta"]
    _write_all_instances(tmp_path, metas)
    if write_stage1_qc:
        _write_stage1_qc(tmp_path, {key: (0, 0) for key in metas})
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(temp, "_read_binary", lambda day, fov: labels[day] > 0)
    return labels


def test_run_stage6_writes_tracks_and_match_qc(monkeypatch, tmp_path):
    """守契约：tracks.csv / matches.csv / match_qc.json 三件套的列、计数与 QC 字段都要对得上。"""
    _setup_run_stage6(monkeypatch, tmp_path)

    temp.run_stage6()

    tracks = pd.read_csv(tmp_path / "analysis" / "tracks.csv")
    required = {"instance_uid", "spot_uid", "track_id", "fov", "frame_idx", "time_hours", "day"}
    assert required <= set(tracks.columns)
    # 2 个类器官 × 3 帧 = 6 行：实例一个都不许丢
    assert len(tracks) == 6
    assert (tracks.groupby("track_id").size() >= 2).sum() > 0
    assert tracks["spot_uid"].is_unique

    qc = json.loads((tmp_path / "analysis" / "match_qc.json").read_text(encoding="utf-8"))
    fov_qc = qc["fovs"]["A1"]
    assert set(fov_qc["pairs"]) == {"0->1", "1->2"}
    assert [row["n_one_to_one"] for row in fov_qc["pairs"].values()] == [2, 2]
    assert fov_qc["registration"]["offset_to_reference"] == {"0": [0, 0], "1": [0, 0], "2": [0, 0]}
    assert fov_qc["registration"]["align_offsets"] == {"0": [0, 0], "1": [0, 0], "2": [0, 0]}
    # 直方图的键是字符串帧数（写进 JSON 后不能是 int 键）
    assert fov_qc["track_length_histogram"] == {"3": 2}
    assert qc["totals"]["n_tracks_len_ge2"] == 2
    assert qc["totals"]["n_one_to_one"] == 4
    assert qc["config"]["min_bbox_iou"] == cfg.MIN_BBOX_IOU
    assert qc["config"]["align_max_shift_px"] == cfg.ALIGN_MAX_SHIFT_PX
    assert qc["warnings"] == []

    # 逐帧对匹配表必须落盘：合并/split 的归属不能只靠重跑本阶段反推
    matches = pd.read_csv(tmp_path / "analysis" / "matches.csv")
    assert list(matches.columns) == ["fov", "pair", "prev_uid", "cur_uid", "iou", "cent_dist", "event"]
    assert set(matches["pair"]) == {"0->1", "1->2"}
    assert set(matches["fov"]) == {"A1"}
    assert len(matches) == 4
    assert set(matches["event"]) == {"one_to_one"}
    # 每一行都能直接读出"上一帧哪颗 → 下一帧哪颗"
    assert set(
        zip(matches.loc[matches["pair"] == "0->1", "prev_uid"], matches.loc[matches["pair"] == "0->1", "cur_uid"])
    ) == {
        ("A1_1_0", "A1_1_1"),
        ("A1_2_0", "A1_2_1"),
    }
    # QC 里记录匹配表行数（总数 + 逐帧对）
    assert qc["match_table"] == {
        "file": "matches.csv",
        "n_rows": 4,
        "columns": ["fov", "pair", "prev_uid", "cur_uid", "iou", "cent_dist", "event"],
    }
    assert sum(row["n_match_rows"] for row in fov_qc["pairs"].values()) == 4
    assert qc["totals"]["n_match_rows"] == 4


def test_run_stage6_stops_when_all_tracks_are_singletons(monkeypatch, tmp_path):
    """守硬断言：track 全长 1 时报中文错误，且三件套一个都不许落盘。"""
    # 两帧内容相距 180 行：既无 IoU、质心距离也超上限，所有实例都会是单帧 track
    metas = {
        "day0/A1": _meta("day0", "A1", {1: (20, 20, 60, 60)}),
        "day3/A1": _meta("day3", "A1", {1: (200, 20, 240, 60)}),
    }
    _write_all_instances(tmp_path, metas)
    _write_stage1_qc(tmp_path, {key: (0, 0) for key in metas})
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(temp, "_read_binary", lambda day, fov: np.zeros((240, 320), dtype=bool))

    with pytest.raises(RuntimeError, match="长度为 1"):
        temp.run_stage6()
    # 硬断言先于落盘：不许留下"看起来跑成功了"的 tracks.csv / matches.csv
    assert not (tmp_path / "analysis" / "tracks.csv").exists()
    assert not (tmp_path / "analysis" / "matches.csv").exists()
    assert not (tmp_path / "analysis" / "match_qc.json").exists()


def test_run_stage6_reports_missing_stage1_summary(monkeypatch, tmp_path):
    """守边界：缺 all_instances.csv 时立刻报错，错误信息要点出文件名。"""
    monkeypatch.setattr(cfg, "OUTPUT_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="all_instances.csv"):
        temp.run_stage6()


def test_run_stage6_warns_when_stage1_qc_is_missing(monkeypatch, tmp_path):
    """守降级：缺 stage1_qc.json 不硬错，但要在 QC warnings 里写明 align_offset 未知。"""
    # 没有 stage1_qc.json 时不知道对齐偏移：不硬错（告警可免），但必须让 QC 里看得见
    _setup_run_stage6(monkeypatch, tmp_path, write_stage1_qc=False)
    temp.run_stage6()
    qc = json.loads((tmp_path / "analysis" / "match_qc.json").read_text(encoding="utf-8"))
    assert any("stage1_qc.json" in warning and "align_offset" in warning for warning in qc["warnings"])


def test_load_align_offsets_reads_stage1_qc(tmp_path):
    """守口径：从 stage1_qc.json 读出 {(day, fov): (dy, dx)}；文件缺失时给空字典 + 告警串。"""
    path = tmp_path / "stage1_qc.json"
    path.write_text(json.dumps({"day0/A1": {"align_dy": 2, "align_dx": -1}}), encoding="utf-8")
    offsets, warning = temp._load_align_offsets(path)
    assert offsets == {"day0/A1": (2, -1)}
    # 正常读到文件时不能顺带产出一条告警
    assert warning is None

    missing, missing_warning = temp._load_align_offsets(tmp_path / "nope.json")
    assert missing == {}
    assert missing_warning is not None and "stage1_qc.json" in missing_warning


def test_run_stage6_registers_with_stage1_align_offsets(monkeypatch, tmp_path):
    """守口径：配准前先用 Stage 1 align_offset 把掩码搬到 meta 坐标系，否则错位会被当成真实位移。"""
    # day3 的原始标签图内容比对齐图高 2 行、左 1 列（align_offset = (2, 1)），meta 仍在对齐坐标系。
    # 配准若直接拿原始标签图，day3→day0 的偏移会多出 (2, 1)；用对齐后的标签图则应为 (0, 0)。
    shape = (240, 320)
    aligned = np.zeros(shape, dtype=np.uint16)
    aligned[40:80, 60:100] = 1
    raw3 = _shift_mask(aligned > 0, -2, -1).astype(np.uint16)
    image = np.where(aligned > 0, 140, 200).astype(np.uint8)

    meta0 = inst.build_catalog("day0", "A1", image=image, labels=aligned)["meta"]
    catalog3 = inst.build_catalog("day3", "A1", image=image, labels=raw3)
    assert catalog3["align_offset"] == (2, 1)  # 前提：Stage 1 确实找到了这个偏移
    # meta 来自对齐后的标签图，所以两帧的 bbox 完全相同
    assert (
        catalog3["meta"][["r0", "c0", "r1", "c1"]].iloc[0].tolist() == meta0[["r0", "c0", "r1", "c1"]].iloc[0].tolist()
    )

    metas = {"day0/A1": meta0, "day3/A1": catalog3["meta"]}
    masks = {"day0/A1": aligned > 0, "day3/A1": raw3 > 0}

    # 对照：不给对齐偏移，配准会把标签图的整体错位当成真实位移
    raw_qc: dict = {}
    temp.build_tracks(metas, masks=masks, qc=raw_qc)
    assert raw_qc["fovs"]["A1"]["registration"]["pair_offsets"]["0->1"] == [2, 1]

    # 给了对齐偏移：先搬到 meta 坐标系再配准，偏移回到 (0, 0) 且回显进 QC
    qc: dict = {}
    matches: list[pd.DataFrame] = []
    temp.build_tracks(
        metas,
        masks=masks,
        qc=qc,
        align_offsets={"day0/A1": (0, 0), "day3/A1": (2, 1)},
        matches_out=matches,
    )
    registration = qc["fovs"]["A1"]["registration"]
    assert registration["align_offsets"] == {"0": [0, 0], "1": [2, 1]}
    assert registration["pair_offsets"] == {"0->1": [0, 0]}
    assert qc["fovs"]["A1"]["pairs"]["0->1"]["n_one_to_one"] == 1
    assert len(matches) == 1 and list(matches[0].columns) == [
        "fov",
        "pair",
        "prev_uid",
        "cur_uid",
        "iou",
        "cent_dist",
        "event",
    ]
    assert matches[0]["pair"].tolist() == ["0->1"]


def test_read_binary_reads_stage1_label_tif(monkeypatch, tmp_path):
    """守口径：从 dayX/Output/<fov>_多聚焦融合.tif 读标签，转成前景布尔掩码（不保留标签号）。"""
    labels = np.zeros((10, 12), dtype=np.uint16)
    labels[2:5, 3:6] = 4
    outdir = tmp_path / "day0" / "Output"
    outdir.mkdir(parents=True)
    tifffile.imwrite(outdir / f"A1{_NAME_SUFFIX}.tif", labels)
    monkeypatch.setattr(cfg, "DATA_ROOT", tmp_path)
    binary = temp._read_binary("day0", "A1")
    assert binary.dtype == bool
    assert np.array_equal(binary, labels > 0)


# ---------------------------------------------------------------- 合并父实例的选取与轨迹终止


def test_merge_parent_is_the_largest_absorbed_instance():
    """守口径：合并的父 track 取面积最大的被吸收实例，bbox IoU 更高的那颗要让位。"""
    # A 的 bbox IoU 更高，但 B 面积更大：按 Spec §6-Stage6-2"父 track 取面积最大者"，
    # 应该由 B 延续轨迹、A 被吸收（贪心赢家要让位）。
    prev = _meta("day0", "A1", {1: (20, 20, 60, 60), 2: (10, 10, 190, 190)})
    cur = _meta("day3", "A1", {1: (20, 20, 100, 100)})
    table = temp.match_frames(prev, cur, offset=(0, 0), min_iou=cfg.MIN_BBOX_IOU, max_cent_dist=cfg.MAX_CENT_DIST_PX)
    assert table.loc[table["event"] == "one_to_one", "prev_uid"].tolist() == ["A1_2_0"]
    assert table.loc[table["event"] == "merge", "prev_uid"].tolist() == ["A1_1_0"]
    tracks = temp.assign_tracks([table], pd.concat([prev, cur], ignore_index=True))
    by_uid = tracks.set_index("instance_uid")
    assert by_uid.loc["A1_1_0", "track_id"] != by_uid.loc["A1_2_0", "track_id"]
    assert (tracks["track_id"] == by_uid.loc["A1_1_0", "track_id"]).sum() == 1
    assert tracks["spot_uid"].is_unique


def test_absorbed_instance_with_earlier_lineage_ends_its_own_track():
    """守口径：中途被吸收的实例轨迹停在被吸收帧，不并入吸收者轨迹（同帧不许两条同 track）。"""
    # frame1 的 x1 先把 frame0 的 p0 接成一条链，随后在 1→2 被面积更大的 x2 吸收：
    # x1 的轨迹应该停在 frame1（不能并进 x2 的轨迹，否则 frame2 会出现两个同 track 实例）。
    meta0 = _meta("day0", "A1", {1: (20, 20, 60, 60)})
    meta1 = _meta("day3", "A1", {1: (20, 20, 60, 60), 2: (10, 10, 190, 190)})
    meta2 = _meta("day5", "A1", {1: (20, 20, 100, 100)})
    tables = [
        temp.match_frames(a, b, offset=(0, 0), min_iou=cfg.MIN_BBOX_IOU, max_cent_dist=cfg.MAX_CENT_DIST_PX)
        for a, b in ((meta0, meta1), (meta1, meta2))
    ]
    assert (tables[1]["event"] == "merge").sum() == 1
    tracks = temp.assign_tracks(tables, pd.concat([meta0, meta1, meta2], ignore_index=True))
    assert tracks["spot_uid"].is_unique
    by_uid = tracks.set_index("instance_uid")
    old_track = tracks[tracks["track_id"] == by_uid.loc["A1_1_0", "track_id"]]
    assert list(old_track["frame_idx"]) == [0, 1]  # 轨迹停在被吸收的那一帧
    new_track = tracks[tracks["track_id"] == by_uid.loc["A1_2_1", "track_id"]]
    assert list(new_track["frame_idx"]) == [1, 2]


def test_build_tracks_rejects_missing_mask():
    """守边界：masks 少一帧时按缺的 key 报错，不许静默跳过该帧的配准。"""
    metas = _toy_metas()
    masks = {"day0/A1": np.zeros((8, 8))}  # 少了 day3/day5
    with pytest.raises(RuntimeError, match="day3/A1"):
        temp.build_tracks(metas, masks=masks)
