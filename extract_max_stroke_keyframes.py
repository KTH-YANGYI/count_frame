from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Event:
    start: int
    end: int
    peak: int
    height: float
    width: int
    score: float
    event_type: str = ""


def moving_average_nan(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if window <= 1:
        return x.copy()
    y = x.copy()
    idx = np.arange(len(x), dtype=np.float32)
    mask = np.isfinite(y)
    if not np.any(mask):
        return np.zeros_like(y)
    if not np.all(mask):
        y[~mask] = np.interp(idx[~mask], idx[mask], y[mask])
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(y, kernel, mode="same")


def normalize_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if abs(mx - mn) < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def finite_signal_extrema(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return {"min_idx": -1, "min_value": 0.0, "max_idx": -1, "max_value": 0.0}
    valid_idx = np.where(finite)[0]
    valid_values = x[finite]
    min_rel = int(np.argmin(valid_values))
    max_rel = int(np.argmax(valid_values))
    return {
        "min_idx": int(valid_idx[min_rel]),
        "min_value": float(valid_values[min_rel]),
        "max_idx": int(valid_idx[max_rel]),
        "max_value": float(valid_values[max_rel]),
    }


def clipped_similarity_to_value(x: np.ndarray, target: float, scale: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if scale <= 1e-6:
        return np.zeros_like(x)
    out = 1.0 - np.abs(x - float(target)) / float(scale)
    return np.clip(out, 0.0, 1.0)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx}")
    return frame


def select_rois_interactive(video_path: str) -> Dict[str, List[int]]:
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot open video for ROI selection")

    print("Draw workpiece ROI, then press ENTER or SPACE.")
    wp = cv2.selectROI("workpiece_roi", frame, fromCenter=False, showCrosshair=True)
    print("Draw pose ROI, then press ENTER or SPACE.")
    pose = cv2.selectROI("pose_roi", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    def xywh_to_xyxy(rect: Tuple[int, int, int, int]) -> List[int]:
        x, y, w, h = rect
        return [int(x), int(y), int(x + w), int(y + h)]

    return {
        "workpiece_roi": xywh_to_xyxy(wp),
        "pose_roi": xywh_to_xyxy(pose),
    }


def load_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def validate_roi(roi: Sequence[int], frame_width: int, frame_height: int, name: str) -> List[int]:
    if len(roi) != 4:
        raise ValueError(f"{name} must have 4 integers, got: {roi}")
    x1, y1, x2, y2 = map(int, roi)
    x1 = max(0, min(frame_width - 1, x1))
    y1 = max(0, min(frame_height - 1, y1))
    x2 = max(x1 + 1, min(frame_width, x2))
    y2 = max(y1 + 1, min(frame_height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} is empty after validation: {roi}")
    return [x1, y1, x2, y2]


def crop_xyxy(frame: np.ndarray, roi: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, roi)
    return frame[y1:y2, x1:x2]


def copper_centroid_y(
    video_path: str,
    roi: Sequence[int],
    hsv_lo: Sequence[int],
    hsv_hi: Sequence[int],
    min_pixels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    x1, y1, x2, y2 = map(int, roi)
    hsv_lo_arr = np.array(hsv_lo, dtype=np.uint8)
    hsv_hi_arr = np.array(hsv_hi, dtype=np.uint8)

    ysig: List[float] = []
    counts: List[int] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        roi_img = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lo_arr, hsv_hi_arr)
        mask = cv2.medianBlur(mask, 3)
        ys, xs = np.where(mask > 0)
        if len(ys) < min_pixels:
            ysig.append(np.nan)
            counts.append(int(len(ys)))
        else:
            ysig.append(float(ys.mean() + y1))
            counts.append(int(len(ys)))
    cap.release()
    return np.asarray(ysig, dtype=np.float32), np.asarray(counts, dtype=np.float32)


def peak_groups(
    signal: np.ndarray,
    quantile: float,
    min_width: int,
    gap: int,
    boundary_margin: int,
) -> List[Tuple[int, int]]:
    s = np.asarray(signal, dtype=np.float32)
    thr = float(np.nanquantile(s, quantile))
    idx = np.where(np.isfinite(s) & (s >= thr))[0]
    if len(idx) == 0:
        return []

    groups: List[Tuple[int, int]] = []
    st = int(idx[0])
    prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev <= gap:
            prev = i
        else:
            groups.append((st, prev))
            st = prev = i
    groups.append((st, prev))

    filtered: List[Tuple[int, int]] = []
    n = len(s)
    end_cutoff = n - 1 - boundary_margin
    for st, en in groups:
        clipped_en = min(en, end_cutoff)
        if clipped_en < st:
            continue
        if clipped_en - st + 1 < min_width:
            continue
        filtered.append((st, clipped_en))
    return filtered


def threshold_groups(
    signal: np.ndarray,
    threshold: float,
    min_width: int,
    gap: int,
    boundary_margin: int,
) -> List[Tuple[int, int]]:
    s = np.asarray(signal, dtype=np.float32)
    idx = np.where(np.isfinite(s) & (s >= float(threshold)))[0]
    if len(idx) == 0:
        return []

    groups: List[Tuple[int, int]] = []
    st = int(idx[0])
    prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev <= gap:
            prev = i
        else:
            groups.append((st, prev))
            st = prev = i
    groups.append((st, prev))

    filtered: List[Tuple[int, int]] = []
    n = len(s)
    end_cutoff = n - 1 - boundary_margin
    for st, en in groups:
        clipped_en = min(en, end_cutoff)
        if clipped_en < st:
            continue
        if clipped_en - st + 1 < min_width:
            continue
        filtered.append((st, clipped_en))
    return filtered


def groups_to_events(signal: np.ndarray, groups: Iterable[Tuple[int, int]], event_type: str = "") -> List[Event]:
    s = np.asarray(signal, dtype=np.float32)
    events: List[Event] = []
    for st, en in groups:
        seg = s[st : en + 1]
        if len(seg) == 0:
            continue
        peak_rel = int(np.nanargmax(seg))
        peak = int(st + peak_rel)
        height = float(s[peak])
        width = int(en - st + 1)
        score = float(height * width)
        events.append(Event(start=int(st), end=int(en), peak=peak, height=height, width=width, score=score, event_type=event_type))
    return events


def groups_to_events_with_peak_signal(
    group_signal: np.ndarray,
    peak_signal: np.ndarray,
    groups: Iterable[Tuple[int, int]],
    event_type: str = "",
) -> List[Event]:
    group_values = np.asarray(group_signal, dtype=np.float32)
    peak_values = np.asarray(peak_signal, dtype=np.float32)
    events: List[Event] = []
    for st, en in groups:
        group_seg = group_values[st : en + 1]
        peak_seg = peak_values[st : en + 1]
        if len(group_seg) == 0 or len(peak_seg) == 0:
            continue
        peak_rel = int(np.nanargmax(peak_seg))
        peak = int(st + peak_rel)
        height = float(peak_values[peak])
        width = int(en - st + 1)
        score = float(height * width)
        events.append(Event(start=int(st), end=int(en), peak=peak, height=height, width=width, score=score, event_type=event_type))
    return events


def nms_events(events: List[Event], min_separation: int, max_events: int, selection: str) -> List[Event]:
    if selection == "earliest":
        ordered = sorted(events, key=lambda e: e.peak)
    else:
        ordered = sorted(events, key=lambda e: (-e.score, -e.height, e.peak))
    chosen: List[Event] = []
    for e in ordered:
        if all(abs(e.peak - c.peak) > min_separation for c in chosen):
            chosen.append(e)
        if len(chosen) >= max_events:
            break
    return sorted(chosen, key=lambda e: e.peak)


def average_cross_score_gap(
    events: Sequence[Event],
    primary_state_score: np.ndarray,
    secondary_state_score: np.ndarray,
) -> float:
    gaps: List[float] = []
    for e in events:
        if 0 <= e.peak < len(primary_state_score):
            gaps.append(abs(float(primary_state_score[e.peak]) - float(secondary_state_score[e.peak])))
    if not gaps:
        return 0.0
    return float(sum(gaps) / len(gaps))


def build_margin_rule_events(
    dominant_signal: np.ndarray,
    other_signal: np.ndarray,
    event_type: str,
    average_gap: float,
    min_width: int,
    gap: int,
    boundary_margin: int,
    start_frame: int,
    end_frame: int,
) -> List[Event]:
    if average_gap <= 1e-6:
        return []
    margin_signal = np.asarray(dominant_signal, dtype=np.float32) - np.asarray(other_signal, dtype=np.float32)
    groups = threshold_groups(
        margin_signal,
        threshold=float(average_gap * 0.5),
        min_width=min_width,
        gap=gap,
        boundary_margin=boundary_margin,
    )
    events = groups_to_events_with_peak_signal(
        group_signal=margin_signal,
        peak_signal=dominant_signal,
        groups=groups,
        event_type=event_type,
    )
    return [e for e in events if start_frame <= e.peak <= end_frame]


def keep_margin_events_after_first_base(
    margin_events: Sequence[Event],
    base_events: Sequence[Event],
) -> List[Event]:
    if not base_events:
        return list(margin_events)
    first_base_peak = min(e.peak for e in base_events)
    return [e for e in margin_events if e.peak >= first_base_peak]


def collapse_consecutive_same_type_events(events: Sequence[Event]) -> List[Event]:
    ordered = sorted(events, key=lambda e: e.peak)
    if not ordered:
        return []

    collapsed: List[Event] = []
    run: List[Event] = [ordered[0]]

    for event in ordered[1:]:
        if event.event_type == run[-1].event_type:
            run.append(event)
            continue
        collapsed.append(max(run, key=lambda e: (e.score, e.height, -e.peak)))
        run = [event]

    collapsed.append(max(run, key=lambda e: (e.score, e.height, -e.peak)))
    return sorted(collapsed, key=lambda e: e.peak)


def raw_similarity_to_reference(
    video_path: str,
    ref_frame_idx: int,
    roi: Sequence[int],
    resize_hw: Tuple[int, int],
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    ref = read_frame(cap, ref_frame_idx)
    ref_gray = cv2.cvtColor(crop_xyxy(ref, roi), cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.resize(ref_gray, resize_hw).astype(np.float32)
    ref_gray = (ref_gray - ref_gray.mean()) / (ref_gray.std() + 1e-6)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    sims: List[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(crop_xyxy(frame, roi), cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, resize_hw).astype(np.float32)
        gray = (gray - gray.mean()) / (gray.std() + 1e-6)
        sims.append(float(np.mean(gray * ref_gray)))
    cap.release()
    return np.asarray(sims, dtype=np.float32)


def build_state_scores(
    depth_signal: np.ndarray,
    primary_frame_idx: int,
    secondary_frame_idx: int,
    sim_primary_signal: np.ndarray,
    sim_secondary_signal: np.ndarray,
    alpha_depth: float,
    alpha_pose: float,
) -> Dict[str, np.ndarray]:
    primary_depth_value = float(depth_signal[primary_frame_idx]) if np.isfinite(depth_signal[primary_frame_idx]) else float(np.nanmin(depth_signal))
    secondary_depth_value = float(depth_signal[secondary_frame_idx]) if np.isfinite(depth_signal[secondary_frame_idx]) else float(np.nanmax(depth_signal))
    dynamic_range = abs(secondary_depth_value - primary_depth_value)
    if dynamic_range <= 1e-6:
        dynamic_range = float(np.nanmax(depth_signal) - np.nanmin(depth_signal)) if np.isfinite(depth_signal).any() else 1.0
    dynamic_range = max(dynamic_range, 1.0)

    primary_depth_score = clipped_similarity_to_value(depth_signal, primary_depth_value, dynamic_range)
    secondary_depth_score = clipped_similarity_to_value(depth_signal, secondary_depth_value, dynamic_range)
    primary_pose_score = normalize_signal(sim_primary_signal)
    secondary_pose_score = normalize_signal(sim_secondary_signal)

    primary_state_score = alpha_depth * primary_depth_score + alpha_pose * primary_pose_score
    secondary_state_score = alpha_depth * secondary_depth_score + alpha_pose * secondary_pose_score

    return {
        "primary_depth_score": primary_depth_score,
        "secondary_depth_score": secondary_depth_score,
        "primary_pose_score": primary_pose_score,
        "secondary_pose_score": secondary_pose_score,
        "primary_state_score": primary_state_score,
        "secondary_state_score": secondary_state_score,
        "primary_depth_value": np.asarray([primary_depth_value], dtype=np.float32),
        "secondary_depth_value": np.asarray([secondary_depth_value], dtype=np.float32),
        "depth_dynamic_range": np.asarray([dynamic_range], dtype=np.float32),
    }


def choose_reference_peak(depth_signal: np.ndarray, cfg: Dict) -> Event:
    groups = peak_groups(
        depth_signal,
        quantile=float(cfg.get("depth_quantile", 0.88)),
        min_width=int(cfg.get("min_plateau_width", 5)),
        gap=int(cfg.get("merge_gap", 4)),
        boundary_margin=int(cfg.get("boundary_margin", 20)),
    )
    events = groups_to_events(depth_signal, groups)
    if not events:
        peak = int(np.nanargmax(depth_signal))
        return Event(start=peak, end=peak, peak=peak, height=float(depth_signal[peak]), width=1, score=float(depth_signal[peak]))
    return max(events, key=lambda e: (e.score, e.height))


def resolve_reference_event(depth_signal: np.ndarray, cfg: Dict, frame_count: int, cli_reference_frame: Optional[int]) -> Tuple[Event, str]:
    reference_frame_idx = cli_reference_frame
    if reference_frame_idx is None and "reference_frame_idx" in cfg:
        reference_frame_idx = int(cfg["reference_frame_idx"])
    if reference_frame_idx is None and "reference_frame_hint" in cfg:
        reference_frame_idx = int(cfg["reference_frame_hint"])
    if reference_frame_idx is not None:
        reference_frame_idx = int(max(0, min(frame_count - 1, reference_frame_idx)))
        height = float(depth_signal[reference_frame_idx]) if np.isfinite(depth_signal[reference_frame_idx]) else 0.0
        return (
            Event(
                start=reference_frame_idx,
                end=reference_frame_idx,
                peak=reference_frame_idx,
                height=height,
                width=1,
                score=height,
            ),
            "provided_reference_frame",
        )
    return choose_reference_peak(depth_signal, cfg), "auto_depth_peak"


def build_cycle_events(
    primary_events: Sequence[Event],
    secondary_events: Sequence[Event],
    expected_gap_frames: int,
    tolerance_frames: int,
) -> List[Event]:
    if not secondary_events:
        return []
    if not primary_events:
        return list(secondary_events)

    min_gap = max(1, expected_gap_frames - tolerance_frames)
    max_gap = expected_gap_frames + tolerance_frames
    used_primary: set[int] = set()
    cycle_events: List[Event] = []

    for secondary in sorted(secondary_events, key=lambda e: e.peak):
        candidates = []
        for idx, primary in enumerate(primary_events):
            if idx in used_primary:
                continue
            gap = secondary.peak - primary.peak
            if gap <= 0:
                continue
            if min_gap <= gap <= max_gap:
                candidates.append((abs(gap - expected_gap_frames), -primary.score, idx, primary, gap))
        if not candidates:
            for idx, primary in enumerate(primary_events):
                if idx in used_primary:
                    continue
                gap = secondary.peak - primary.peak
                if gap <= 0:
                    continue
                if gap <= max(expected_gap_frames * 2, tolerance_frames * 2):
                    candidates.append((abs(gap - expected_gap_frames) + 1000.0, -primary.score, idx, primary, gap))
        if candidates:
            candidates.sort(key=lambda item: item[:2])
            _, _, chosen_idx, primary, gap = candidates[0]
            used_primary.add(chosen_idx)
            gap_penalty = abs(gap - expected_gap_frames) / max(expected_gap_frames, 1)
            cycle_score = float(secondary.score + 0.5 * primary.score - 0.25 * gap_penalty)
            cycle_events.append(
                Event(
                    start=int(primary.peak),
                    end=int(secondary.peak),
                    peak=int(secondary.peak),
                    height=float(secondary.height),
                    width=int(secondary.width),
                    score=cycle_score,
                )
            )
        else:
            cycle_events.append(secondary)
    return cycle_events


def event_color(event_type: str) -> str:
    if event_type == "reference_plus_2s":
        return "tab:orange"
    if event_type == "reference":
        return "tab:blue"
    return "tab:gray"


def event_label(event_type: str) -> str:
    if event_type == "reference_plus_2s":
        return "+2s"
    if event_type == "reference":
        return "ref"
    return "event"


def draw_selected_events(
    ax,
    events: Sequence[Event],
    fps: float,
    values: Optional[np.ndarray] = None,
    annotate: bool = False,
) -> None:
    y_levels = (0.94, 0.86, 0.78)
    labeled_types: set[str] = set()
    for i, e in enumerate(events, start=1):
        color = event_color(e.event_type)
        legend_label = None
        if e.event_type not in labeled_types:
            legend_label = f"{event_label(e.event_type)} window"
            labeled_types.add(e.event_type)
        ax.axvspan(e.start / fps, e.end / fps, color=color, alpha=0.10, lw=0, label=legend_label)
        ax.axvline(e.peak / fps, linestyle="--", color=color, linewidth=1.1, alpha=0.85)
        if values is not None and 0 <= e.peak < len(values) and np.isfinite(values[e.peak]):
            ax.scatter(
                [e.peak / fps],
                [float(values[e.peak])],
                s=28,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )
        if annotate:
            ax.text(
                e.peak / fps,
                y_levels[(i - 1) % len(y_levels)],
                f"#{i} {event_label(e.event_type)}",
                transform=ax.get_xaxis_transform(),
                fontsize=8,
                ha="left",
                va="top",
                color=color,
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )


def add_reference_guides(
    ax,
    fps: float,
    primary_reference_frame_idx: Optional[int] = None,
    secondary_reference_frame_idx: Optional[int] = None,
    values: Optional[np.ndarray] = None,
) -> None:
    if primary_reference_frame_idx is not None:
        ax.axvline(
            primary_reference_frame_idx / fps,
            color="tab:purple",
            linestyle=":",
            linewidth=1.6,
            alpha=0.9,
            label="reference frame",
        )
        if values is not None and 0 <= primary_reference_frame_idx < len(values) and np.isfinite(values[primary_reference_frame_idx]):
            ax.axhline(float(values[primary_reference_frame_idx]), color="tab:purple", linestyle=":", linewidth=1.0, alpha=0.45)
    if secondary_reference_frame_idx is not None:
        ax.axvline(
            secondary_reference_frame_idx / fps,
            color="tab:orange",
            linestyle=":",
            linewidth=1.6,
            alpha=0.9,
            label="reference + 2s",
        )
        if values is not None and 0 <= secondary_reference_frame_idx < len(values) and np.isfinite(values[secondary_reference_frame_idx]):
            ax.axhline(float(values[secondary_reference_frame_idx]), color="tab:orange", linestyle=":", linewidth=1.0, alpha=0.45)


def finalize_axis(ax, ylabel: str, ylim: Optional[Tuple[float, float]] = None) -> None:
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.18, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_debug(
    out_path: Path,
    depth_signal: np.ndarray,
    events: Sequence[Event],
    fps: float,
    primary_pose_score: np.ndarray,
    secondary_pose_score: np.ndarray,
    primary_state_score: np.ndarray,
    secondary_state_score: np.ndarray,
    primary_reference_frame_idx: Optional[int] = None,
    secondary_reference_frame_idx: Optional[int] = None,
) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(out_path.parent)
    x = np.arange(len(depth_signal)) / max(fps, 1e-6)
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15, 10.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.5, 1.5, 2.0], "hspace": 0.08},
    )
    ax_depth, ax_pose_primary, ax_pose_secondary, ax_state = axes

    ax_depth.plot(x, depth_signal, label="depth signal", color="tab:blue", linewidth=2.0)
    add_reference_guides(
        ax_depth,
        fps=fps,
        primary_reference_frame_idx=primary_reference_frame_idx,
        secondary_reference_frame_idx=secondary_reference_frame_idx,
        values=depth_signal,
    )
    draw_selected_events(ax_depth, events, fps=fps, values=depth_signal, annotate=False)
    finalize_axis(ax_depth, "depth y")
    ax_depth.set_title("Keyframe extraction debug overview")
    ax_depth.legend(loc="upper right", ncol=2, framealpha=0.92)

    ax_pose_primary.plot(x, primary_pose_score, label="reference structure sim", color="tab:green", linewidth=2.0)
    add_reference_guides(
        ax_pose_primary,
        fps=fps,
        primary_reference_frame_idx=primary_reference_frame_idx,
        secondary_reference_frame_idx=secondary_reference_frame_idx,
    )
    finalize_axis(ax_pose_primary, "ref sim", ylim=(-0.02, 1.08))
    ax_pose_primary.legend(loc="upper right", ncol=2, framealpha=0.92)

    ax_pose_secondary.plot(x, secondary_pose_score, label="+2s structure sim", color="tab:red", linewidth=2.0)
    add_reference_guides(
        ax_pose_secondary,
        fps=fps,
        primary_reference_frame_idx=primary_reference_frame_idx,
        secondary_reference_frame_idx=secondary_reference_frame_idx,
    )
    finalize_axis(ax_pose_secondary, "+2s sim", ylim=(-0.02, 1.08))
    ax_pose_secondary.legend(loc="upper right", ncol=2, framealpha=0.92)

    ax_state.plot(x, primary_state_score, label="reference fused score", color="tab:purple", linewidth=2.0)
    ax_state.plot(x, secondary_state_score, label="+2s fused score", color="tab:brown", linewidth=2.0)
    add_reference_guides(
        ax_state,
        fps=fps,
        primary_reference_frame_idx=primary_reference_frame_idx,
        secondary_reference_frame_idx=secondary_reference_frame_idx,
    )
    draw_selected_events(ax_state, events, fps=fps, annotate=True)
    finalize_axis(ax_state, "fused score", ylim=(-0.02, 1.08))
    ax_state.set_xlabel("time (s)")
    ax_state.legend(loc="upper right", ncol=2, framealpha=0.92)

    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)


def plot_event_windows(
    out_path: Path,
    depth_signal: np.ndarray,
    events: Sequence[Event],
    fps: float,
    primary_state_score: np.ndarray,
    secondary_state_score: np.ndarray,
    window_seconds: float = 1.6,
) -> None:
    import matplotlib.pyplot as plt

    if not events:
        return

    ensure_dir(out_path.parent)
    window_frames = max(1, int(round(window_seconds * max(fps, 1e-6))))
    ncols = 2
    nrows = int(np.ceil(len(events) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, 3.2 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for ax in axes.ravel():
        ax.set_visible(False)

    for i, e in enumerate(events):
        ax = axes.ravel()[i]
        ax.set_visible(True)
        st = max(0, int(e.peak - window_frames))
        en = min(len(depth_signal) - 1, int(e.peak + window_frames))
        idx = np.arange(st, en + 1)
        x = idx / max(fps, 1e-6)
        window_depth = np.asarray(depth_signal[st : en + 1], dtype=np.float32)
        score_signal = primary_state_score if e.event_type != "reference_plus_2s" else secondary_state_score
        window_score = np.asarray(score_signal[st : en + 1], dtype=np.float32)
        color = event_color(e.event_type)

        ax.plot(x, window_depth, color="tab:blue", linewidth=1.8, label="depth signal")
        ax.axvspan(e.start / fps, e.end / fps, color=color, alpha=0.14, lw=0)
        ax.axvline(e.peak / fps, color=color, linestyle="--", linewidth=1.2)
        if 0 <= e.peak < len(depth_signal) and np.isfinite(depth_signal[e.peak]):
            ax.scatter(
                [e.peak / fps],
                [float(depth_signal[e.peak])],
                s=26,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )
        finalize_axis(ax, "depth y")

        ax_score = ax.twinx()
        ax_score.plot(x, window_score, color=color, linewidth=1.8, alpha=0.95, label=f"{event_label(e.event_type)} state")
        ax_score.set_ylim(-0.02, 1.08)
        ax_score.set_ylabel("state", color=color)
        ax_score.tick_params(axis="y", colors=color)
        ax_score.spines["top"].set_visible(False)

        handles_left, labels_left = ax.get_legend_handles_labels()
        handles_right, labels_right = ax_score.get_legend_handles_labels()
        ax.legend(handles_left + handles_right, labels_left + labels_right, loc="upper left", fontsize=8, framealpha=0.9)
        ax.set_title(f"#{i + 1} {event_label(e.event_type)}  f{e.peak}  t={e.peak / fps:.3f}s", fontsize=10)
        ax.set_xlabel("time (s)")

    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)


def save_keyframes(
    video_path: str,
    events: Sequence[Event],
    out_dir: Path,
    prefix: str,
) -> None:
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    for i, e in enumerate(events, start=1):
        frame = read_frame(cap, e.peak)
        t = e.peak / fps if fps > 0 else 0.0
        type_part = f"_{e.event_type}" if e.event_type else ""
        out_path = out_dir / f"{prefix}_{i:02d}{type_part}_f{e.peak}_t{t:.3f}s.jpg"
        cv2.imwrite(str(out_path), frame)
    cap.release()


def save_events_csv(out_path: Path, events: Sequence[Event], fps: float) -> None:
    ensure_dir(out_path.parent)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "event_type", "frame", "time_seconds", "plateau_start", "plateau_end", "score", "height", "width"])
        for i, e in enumerate(events, start=1):
            writer.writerow([i, e.event_type, e.peak, e.peak / fps if fps > 0 else 0.0, e.start, e.end, e.score, e.height, e.width])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract bottom-most / max-stroke keyframes from a machine video.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--config", default=None, help="JSON config path")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--max-events", type=int, default=None, help="Override max events")
    parser.add_argument("--fused-quantile", type=float, default=None, help="Override fused score quantile threshold")
    parser.add_argument("--min-plateau-width", type=int, default=None, help="Override minimum plateau width")
    parser.add_argument("--start-frame", type=int, default=0, help="Optional frame-range start for post filtering")
    parser.add_argument("--end-frame", type=int, default=-1, help="Optional frame-range end for post filtering")
    parser.add_argument("--selection", choices=["score", "earliest"], default="score", help="How to choose when more events exist than requested")
    parser.add_argument("--select-roi", action="store_true", help="Interactively select workpiece ROI and pose ROI from frame 0")
    parser.add_argument("--save-config", default=None, help="Optional JSON path to save the merged config after ROI selection")
    parser.add_argument("--reference-frame", type=int, default=None, help="Optional explicit reference frame index for pose similarity")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.select_roi:
        cfg.update(select_rois_interactive(args.video))
        if args.save_config:
            save_json(Path(args.save_config), cfg)
            print(f"Saved config to {args.save_config}")

    required = ["workpiece_roi", "pose_roi"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SystemExit(f"Missing config keys: {missing}")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    hsv_lo = cfg.get("hsv_lo", [5, 50, 50])
    hsv_hi = cfg.get("hsv_hi", [30, 255, 255])
    min_pixels = int(cfg.get("min_pixels", 120))
    smooth_window = int(cfg.get("smooth_window", 7))
    depth_quantile = float(cfg.get("depth_quantile", 0.88))
    fused_quantile = float(args.fused_quantile if args.fused_quantile is not None else cfg.get("fused_quantile", 0.88))
    min_plateau_width = int(args.min_plateau_width if args.min_plateau_width is not None else cfg.get("min_plateau_width", 5))
    merge_gap = int(cfg.get("merge_gap", 4))
    boundary_margin = int(cfg.get("boundary_margin", 20))
    min_separation = int(cfg.get("min_separation", 30))
    max_events = int(args.max_events if args.max_events is not None else cfg.get("max_events", 2))
    resize_hw = tuple(cfg.get("similarity_resize_hw", [32, 32]))
    alpha_depth = float(cfg.get("alpha_depth", 0.6))
    alpha_pose = float(cfg.get("alpha_pose", 0.4))

    cap = cv2.VideoCapture(args.video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if frame_count <= 0 or frame_width <= 0 or frame_height <= 0:
        raise SystemExit(f"Cannot read video metadata: {args.video}")

    cfg["workpiece_roi"] = validate_roi(cfg["workpiece_roi"], frame_width, frame_height, "workpiece_roi")
    cfg["pose_roi"] = validate_roi(cfg["pose_roi"], frame_width, frame_height, "pose_roi")

    depth_raw, mask_count = copper_centroid_y(
        video_path=args.video,
        roi=cfg["workpiece_roi"],
        hsv_lo=hsv_lo,
        hsv_hi=hsv_hi,
        min_pixels=min_pixels,
    )
    depth_signal = moving_average_nan(depth_raw, smooth_window)
    depth_extrema = finite_signal_extrema(depth_signal)

    secondary_reference_frame_idx = None
    if "secondary_reference_frame_idx" in cfg:
        secondary_reference_frame_idx = int(max(0, min(frame_count - 1, int(cfg["secondary_reference_frame_idx"]))))
    elif "secondary_reference_used_time_sec" in cfg and fps > 0:
        secondary_reference_frame_idx = int(max(0, min(frame_count - 1, round(float(cfg["secondary_reference_used_time_sec"]) * fps))))
    secondary_reference_time_seconds = None
    if secondary_reference_frame_idx is not None:
        secondary_reference_time_seconds = secondary_reference_frame_idx / fps if fps > 0 else 0.0

    ref_event, reference_source = resolve_reference_event(depth_signal, cfg, frame_count=frame_count, cli_reference_frame=args.reference_frame)
    primary_reference_frame_idx = int(ref_event.peak)
    if secondary_reference_frame_idx is None:
        secondary_reference_frame_idx = int(depth_extrema["max_idx"]) if depth_extrema["max_idx"] >= 0 else primary_reference_frame_idx
        secondary_reference_time_seconds = secondary_reference_frame_idx / fps if fps > 0 else 0.0

    sim_primary_raw = raw_similarity_to_reference(
        video_path=args.video,
        ref_frame_idx=primary_reference_frame_idx,
        roi=cfg["pose_roi"],
        resize_hw=resize_hw,
    )
    sim_secondary_raw = raw_similarity_to_reference(
        video_path=args.video,
        ref_frame_idx=secondary_reference_frame_idx,
        roi=cfg["pose_roi"],
        resize_hw=resize_hw,
    )
    sim_primary_signal = moving_average_nan(sim_primary_raw, smooth_window)
    sim_secondary_signal = moving_average_nan(sim_secondary_raw, smooth_window)

    state_scores = build_state_scores(
        depth_signal=depth_signal,
        primary_frame_idx=primary_reference_frame_idx,
        secondary_frame_idx=secondary_reference_frame_idx,
        sim_primary_signal=sim_primary_signal,
        sim_secondary_signal=sim_secondary_signal,
        alpha_depth=alpha_depth,
        alpha_pose=alpha_pose,
    )
    primary_state_score = state_scores["primary_state_score"]
    secondary_state_score = state_scores["secondary_state_score"]

    primary_groups = peak_groups(
        primary_state_score,
        quantile=fused_quantile,
        min_width=min_plateau_width,
        gap=merge_gap,
        boundary_margin=boundary_margin,
    )
    secondary_groups = peak_groups(
        secondary_state_score,
        quantile=fused_quantile,
        min_width=min_plateau_width,
        gap=merge_gap,
        boundary_margin=boundary_margin,
    )
    primary_events = groups_to_events(primary_state_score, primary_groups, event_type="reference")
    secondary_events = groups_to_events(secondary_state_score, secondary_groups, event_type="reference_plus_2s")

    # Optional frame range filter.
    start_frame = int(max(0, args.start_frame))
    end_frame = int(frame_count - 1 if args.end_frame < 0 else min(frame_count - 1, args.end_frame))
    primary_events = [e for e in primary_events if start_frame <= e.peak <= end_frame]
    secondary_events = [e for e in secondary_events if start_frame <= e.peak <= end_frame]

    if not primary_events:
        peak = int(np.nanargmax(primary_state_score[start_frame : end_frame + 1])) + start_frame
        primary_events = [Event(start=peak, end=peak, peak=peak, height=float(primary_state_score[peak]), width=1, score=float(primary_state_score[peak]), event_type="reference")]
    if not secondary_events:
        peak = int(np.nanargmax(secondary_state_score[start_frame : end_frame + 1])) + start_frame
        secondary_events = [Event(start=peak, end=peak, peak=peak, height=float(secondary_state_score[peak]), width=1, score=float(secondary_state_score[peak]), event_type="reference_plus_2s")]

    base_selected_primary = nms_events(
        primary_events,
        min_separation=min_separation,
        max_events=max_events,
        selection=args.selection,
    )
    base_selected_secondary = nms_events(
        secondary_events,
        min_separation=min_separation,
        max_events=max_events,
        selection=args.selection,
    )

    reference_average_gap = average_cross_score_gap(
        base_selected_primary,
        primary_state_score=primary_state_score,
        secondary_state_score=secondary_state_score,
    )
    secondary_average_gap = average_cross_score_gap(
        base_selected_secondary,
        primary_state_score=primary_state_score,
        secondary_state_score=secondary_state_score,
    )

    margin_rule_primary_events = build_margin_rule_events(
        dominant_signal=primary_state_score,
        other_signal=secondary_state_score,
        event_type="reference",
        average_gap=reference_average_gap,
        min_width=min_plateau_width,
        gap=merge_gap,
        boundary_margin=boundary_margin,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    margin_rule_secondary_events = build_margin_rule_events(
        dominant_signal=secondary_state_score,
        other_signal=primary_state_score,
        event_type="reference_plus_2s",
        average_gap=secondary_average_gap,
        min_width=min_plateau_width,
        gap=merge_gap,
        boundary_margin=boundary_margin,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    margin_rule_primary_events = keep_margin_events_after_first_base(
        margin_rule_primary_events,
        base_selected_primary,
    )
    margin_rule_secondary_events = keep_margin_events_after_first_base(
        margin_rule_secondary_events,
        base_selected_secondary,
    )

    selected_primary = nms_events(
        primary_events + margin_rule_primary_events,
        min_separation=min_separation,
        max_events=max_events,
        selection=args.selection,
    )
    selected_secondary = nms_events(
        secondary_events + margin_rule_secondary_events,
        min_separation=min_separation,
        max_events=max_events,
        selection=args.selection,
    )
    selected = sorted(selected_primary + selected_secondary, key=lambda e: e.peak)
    selected = collapse_consecutive_same_type_events(selected)
    selected_primary = [e for e in selected if e.event_type == "reference"]
    selected_secondary = [e for e in selected if e.event_type == "reference_plus_2s"]

    save_keyframes(args.video, selected, out_dir, prefix=Path(args.video).stem)
    save_events_csv(out_dir / "events.csv", selected, fps)
    plot_debug(
        out_dir / "debug_signals.png",
        depth_signal,
        selected,
        fps,
        primary_pose_score=state_scores["primary_pose_score"],
        secondary_pose_score=state_scores["secondary_pose_score"],
        primary_state_score=primary_state_score,
        secondary_state_score=secondary_state_score,
        primary_reference_frame_idx=primary_reference_frame_idx,
        secondary_reference_frame_idx=secondary_reference_frame_idx,
    )

    save_json(
        out_dir / "run_summary.json",
        {
            "video": str(args.video),
            "fps": fps,
            "frame_count": frame_count,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "reference_peak_frame": ref_event.peak,
            "reference_peak_time_seconds": ref_event.peak / fps if fps > 0 else 0.0,
            "reference_source": reference_source,
            "secondary_reference_frame_idx": secondary_reference_frame_idx,
            "secondary_reference_time_seconds": secondary_reference_time_seconds,
            "depth_signal_min_frame": int(depth_extrema["min_idx"]),
            "depth_signal_min_time_seconds": depth_extrema["min_idx"] / fps if fps > 0 and depth_extrema["min_idx"] >= 0 else None,
            "depth_signal_min_value": float(depth_extrema["min_value"]),
            "depth_signal_max_frame": int(depth_extrema["max_idx"]),
            "depth_signal_max_time_seconds": depth_extrema["max_idx"] / fps if fps > 0 and depth_extrema["max_idx"] >= 0 else None,
            "depth_signal_max_value": float(depth_extrema["max_value"]),
            "primary_reference_depth_value": float(depth_signal[primary_reference_frame_idx]) if np.isfinite(depth_signal[primary_reference_frame_idx]) else None,
            "secondary_reference_depth_value": float(depth_signal[secondary_reference_frame_idx]) if np.isfinite(depth_signal[secondary_reference_frame_idx]) else None,
            "config": cfg,
            "mask_count_summary": {
                "mean": float(np.nanmean(mask_count)) if len(mask_count) else 0.0,
                "max": float(np.nanmax(mask_count)) if len(mask_count) else 0.0,
                "min": float(np.nanmin(mask_count)) if len(mask_count) else 0.0,
            },
            "margin_rule": {
                "enabled": True,
                "ratio": 0.5,
                "reference_average_gap": float(reference_average_gap),
                "reference_threshold": float(reference_average_gap * 0.5),
                "reference_extra_events": [
                    {
                        "frame": e.peak,
                        "time_seconds": e.peak / fps if fps > 0 else 0.0,
                        "start": e.start,
                        "end": e.end,
                        "score": e.score,
                    }
                    for e in margin_rule_primary_events
                ],
                "reference_plus_2s_average_gap": float(secondary_average_gap),
                "reference_plus_2s_threshold": float(secondary_average_gap * 0.5),
                "reference_plus_2s_extra_events": [
                    {
                        "frame": e.peak,
                        "time_seconds": e.peak / fps if fps > 0 else 0.0,
                        "start": e.start,
                        "end": e.end,
                        "score": e.score,
                    }
                    for e in margin_rule_secondary_events
                ],
            },
            "reference_top_events": [
                {
                    "event_type": e.event_type,
                    "frame": e.peak,
                    "time_seconds": e.peak / fps if fps > 0 else 0.0,
                    "start": e.start,
                    "end": e.end,
                    "score": e.score,
                }
                for e in primary_events[: min(8, len(primary_events))]
            ],
            "reference_plus_2s_top_events": [
                {
                    "event_type": e.event_type,
                    "frame": e.peak,
                    "time_seconds": e.peak / fps if fps > 0 else 0.0,
                    "start": e.start,
                    "end": e.end,
                    "score": e.score,
                }
                for e in secondary_events[: min(8, len(secondary_events))]
            ],
            "selected_reference_events": [
                {
                    "event_type": e.event_type,
                    "frame": e.peak,
                    "time_seconds": e.peak / fps if fps > 0 else 0.0,
                    "start": e.start,
                    "end": e.end,
                    "score": e.score,
                    "height": e.height,
                    "width": e.width,
                }
                for e in selected_primary
            ],
            "selected_reference_plus_2s_events": [
                {
                    "event_type": e.event_type,
                    "frame": e.peak,
                    "time_seconds": e.peak / fps if fps > 0 else 0.0,
                    "start": e.start,
                    "end": e.end,
                    "score": e.score,
                    "height": e.height,
                    "width": e.width,
                }
                for e in selected_secondary
            ],
            "selected_events": [
                {
                    "event_type": e.event_type,
                    "frame": e.peak,
                    "time_seconds": e.peak / fps if fps > 0 else 0.0,
                    "start": e.start,
                    "end": e.end,
                    "score": e.score,
                    "height": e.height,
                    "width": e.width,
                }
                for e in selected
            ],
        },
    )

    print(f"Saved {len(selected)} keyframes to {out_dir}")
    for i, e in enumerate(selected, start=1):
        t = e.peak / fps if fps > 0 else 0.0
        print(f"#{i}: frame={e.peak}, time={t:.3f}s, width={e.width}, score={e.score:.4f}")


if __name__ == "__main__":
    main()
