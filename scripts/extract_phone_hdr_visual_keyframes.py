from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
COUNT_SRC_DIR = Path("/Users/yangyi/Desktop/masterthesis/count/src")

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from extract_max_stroke_keyframes import (  # noqa: E402
    Event,
    average_cross_score_gap,
    build_margin_rule_events,
    build_state_scores,
    collapse_consecutive_same_type_events,
    finite_signal_extrema,
    groups_to_events,
    keep_margin_events_after_first_base,
    moving_average_nan,
    nms_events,
    peak_groups,
    validate_roi,
)


@dataclass(frozen=True)
class SegmentSpec:
    segment_id: str
    video_name: str
    reference_time_sec: float
    start_time_sec: float
    end_time_sec: Optional[float]
    workpiece_roi: Tuple[int, int, int, int]
    pose_roi: Tuple[int, int, int, int]


DEFAULT_SEGMENTS: Tuple[SegmentSpec, ...] = (
    SegmentSpec(
        segment_id="IMG_0238",
        video_name="IMG_0238.MOV",
        reference_time_sec=18.5,
        start_time_sec=0.0,
        end_time_sec=None,
        workpiece_roi=(775, 80, 910, 805),
        pose_roi=(610, 340, 990, 780),
    ),
    SegmentSpec(
        segment_id="IMG_0239",
        video_name="IMG_0239.MOV",
        reference_time_sec=8.0,
        start_time_sec=0.0,
        end_time_sec=None,
        workpiece_roi=(705, 0, 875, 805),
        pose_roi=(565, 340, 970, 805),
    ),
    SegmentSpec(
        segment_id="IMG_0240_before_2min",
        video_name="IMG_0240.MOV",
        reference_time_sec=39.0,
        start_time_sec=0.0,
        end_time_sec=120.0,
        workpiece_roi=(825, 0, 970, 805),
        pose_roi=(720, 350, 1045, 845),
    ),
    SegmentSpec(
        segment_id="IMG_0240_after_2min",
        video_name="IMG_0240.MOV",
        reference_time_sec=163.0,
        start_time_sec=120.0,
        end_time_sec=None,
        workpiece_roi=(705, 0, 875, 805),
        pose_roi=(565, 340, 970, 805),
    ),
    SegmentSpec(
        segment_id="IMG_0241",
        video_name="IMG_0241.MOV",
        reference_time_sec=12.0,
        start_time_sec=0.0,
        end_time_sec=None,
        workpiece_roi=(725, 0, 905, 805),
        pose_roi=(585, 340, 990, 805),
    ),
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx}")
    return frame


def crop_xyxy(frame: np.ndarray, roi: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, roi)
    return frame[y1:y2, x1:x2]


def normalized_pose_gray(frame: np.ndarray, roi: Sequence[int], resize_hw: Tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(crop_xyxy(frame, roi), cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, resize_hw).astype(np.float32)
    return (gray - gray.mean()) / (gray.std() + 1e-6)


def pose_similarity(frame: np.ndarray, roi: Sequence[int], resize_hw: Tuple[int, int], reference_gray: np.ndarray) -> float:
    gray = normalized_pose_gray(frame, roi, resize_hw)
    return float(np.mean(gray * reference_gray))


def event_to_absolute(event: Event, frame_offset: int) -> Event:
    return Event(
        start=event.start + frame_offset,
        end=event.end + frame_offset,
        peak=event.peak + frame_offset,
        height=event.height,
        width=event.width,
        score=event.score,
        event_type=event.event_type,
    )


def write_events_csv(path: Path, events: Sequence[Event], fps: float) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "event_type",
                "frame",
                "time_seconds",
                "plateau_start",
                "plateau_end",
                "score",
                "height",
                "width",
            ]
        )
        for rank, event in enumerate(events, start=1):
            writer.writerow(
                [
                    rank,
                    event.event_type,
                    event.peak,
                    event.peak / fps if fps > 0 else 0.0,
                    event.start,
                    event.end,
                    event.score,
                    event.height,
                    event.width,
                ]
            )


def plot_debug(
    out_path: Path,
    frame_numbers: np.ndarray,
    fps: float,
    depth_signal: np.ndarray,
    primary_state_score: np.ndarray,
    secondary_state_score: np.ndarray,
    events: Sequence[Event],
    frame_offset: int,
    reference_frame_idx: int,
    secondary_reference_frame_idx: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(out_path.parent)
    x = frame_numbers / max(fps, 1e-6)
    fig, axes = plt.subplots(3, 1, figsize=(15, 8.5), sharex=True, constrained_layout=True)
    ax_depth, ax_primary, ax_secondary = axes

    ax_depth.plot(x, depth_signal, color="tab:blue", linewidth=1.5, label="workpiece depth y")
    ax_primary.plot(x, primary_state_score, color="tab:purple", linewidth=1.4, label="reference fused score")
    ax_secondary.plot(x, secondary_state_score, color="tab:orange", linewidth=1.4, label="reference +2s fused score")

    for ax in axes:
        ax.axvline(reference_frame_idx / fps, color="tab:purple", linestyle=":", linewidth=1.2, alpha=0.8)
        ax.axvline(secondary_reference_frame_idx / fps, color="tab:orange", linestyle=":", linewidth=1.2, alpha=0.8)
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", framealpha=0.9)

    for event in events:
        local_peak = event.peak - frame_offset
        if 0 <= local_peak < len(frame_numbers):
            color = "tab:orange" if event.event_type == "reference_plus_2s" else "tab:purple"
            ax_primary.axvline(event.peak / fps, color=color, linestyle="--", linewidth=0.7, alpha=0.55)
            ax_secondary.axvline(event.peak / fps, color=color, linestyle="--", linewidth=0.7, alpha=0.55)

    ax_depth.set_ylabel("depth y")
    ax_primary.set_ylabel("ref score")
    ax_secondary.set_ylabel("+2s score")
    ax_secondary.set_xlabel("time (s)")
    fig.savefig(str(out_path), dpi=170)
    plt.close(fig)


def contact_sheet(
    manifest_rows: Sequence[Dict[str, str]],
    event_rows: Sequence[Dict[str, str]],
    out_path: Path,
    columns: int = 8,
    thumb_width: int = 220,
) -> None:
    if not manifest_rows:
        return
    event_by_rank = {str(row["rank"]): row for row in event_rows}
    tiles: List[np.ndarray] = []
    for row in manifest_rows:
        image_path = row.get("output_path", "")
        image = cv2.imread(image_path)
        if image is None:
            continue
        h, w = image.shape[:2]
        thumb_height = max(1, int(round(h * thumb_width / max(w, 1))))
        thumb = cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        label_pad = np.full((54, thumb_width, 3), 255, dtype=np.uint8)
        event = event_by_rank.get(str(row.get("event_index", "")), {})
        label_1 = f"#{row.get('event_index', '')} {event.get('event_type', '')}"
        label_2 = f"f{event.get('frame', '')} t={event.get('time_seconds', '')[:7]}s"
        cv2.putText(label_pad, label_1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(label_pad, label_2, (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 80, 80), 1, cv2.LINE_AA)
        tiles.append(np.vstack([label_pad, thumb]))

    if not tiles:
        return
    cell_h = max(tile.shape[0] for tile in tiles)
    cell_w = max(tile.shape[1] for tile in tiles)
    rows: List[np.ndarray] = []
    for start in range(0, len(tiles), columns):
        row_tiles = tiles[start : start + columns]
        while len(row_tiles) < columns:
            row_tiles.append(np.full((cell_h, cell_w, 3), 245, dtype=np.uint8))
        padded = []
        for tile in row_tiles:
            canvas = np.full((cell_h, cell_w, 3), 245, dtype=np.uint8)
            canvas[: tile.shape[0], : tile.shape[1]] = tile
            padded.append(canvas)
        rows.append(np.hstack(padded))
    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), np.vstack(rows))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def import_avfoundation_exporter():
    if COUNT_SRC_DIR.exists() and str(COUNT_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(COUNT_SRC_DIR))
    try:
        from bend_counter.avfoundation_export import AVFrameRequest, export_avfoundation_frames
    except ImportError as exc:
        raise RuntimeError(
            "AVFoundation export requires the Bend Counter helper at "
            f"{COUNT_SRC_DIR}. Use the default OpenCV export, or add the count/src path."
        ) from exc
    return AVFrameRequest, export_avfoundation_frames


def export_opencv_keyframes(
    video_path: Path,
    events: Sequence[Event],
    fps: float,
    output_dir: Path,
    output_format: str,
) -> List[Dict[str, str]]:
    ensure_dir(output_dir)
    fmt = output_format.lower().lstrip(".")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for keyframe export: {video_path}")

    manifest_rows: List[Dict[str, str]] = []
    for rank, event in enumerate(events, start=1):
        requested_time_sec = event.peak / fps if fps > 0 else 0.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(event.peak))
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_MSEC, requested_time_sec * 1000.0)
            ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Unable to decode event {rank} at frame {event.peak}")

        actual_frame = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
        actual_frame = max(0, actual_frame)
        actual_time_sec = actual_frame / fps if fps > 0 else requested_time_sec
        output_path = output_dir / f"event_{rank:04d}.{fmt}"
        if not cv2.imwrite(str(output_path), frame):
            cap.release()
            raise RuntimeError(f"Unable to write keyframe: {output_path}")
        manifest_rows.append(
            {
                "event_index": str(rank),
                "requested_time_sec": f"{requested_time_sec:.6f}",
                "actual_time_sec": f"{actual_time_sec:.6f}",
                "frame_index": str(actual_frame),
                "output_path": str(output_path),
            }
        )

    cap.release()
    manifest_path = output_dir / "opencv_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["event_index", "requested_time_sec", "actual_time_sec", "frame_index", "output_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_rows


def export_keyframes(
    video_path: Path,
    events: Sequence[Event],
    fps: float,
    segment_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Path, List[Dict[str, str]], str, Path]:
    if args.frame_export_mode == "avfoundation":
        AVFrameRequest, export_avfoundation_frames = import_avfoundation_exporter()
        keyframes_dir = segment_dir / "keyframes_avfoundation_srgb"
        requests = [
            AVFrameRequest(event_index=rank, timestamp_sec=event.peak / fps if fps > 0 else 0.0)
            for rank, event in enumerate(events, start=1)
        ]
        manifest_rows = export_avfoundation_frames(
            video_path=video_path,
            requests=requests,
            output_dir=keyframes_dir,
            output_format=args.avfoundation_output_format,
            preserve_orientation=True,
            tone_map_hdr=True,
            time_tolerance_sec=args.avfoundation_time_tolerance_sec,
            clang_cache_dir=args.avfoundation_clang_cache_dir,
        )
        manifest_path = keyframes_dir / "avfoundation_srgb_manifest.csv"
        return keyframes_dir, manifest_rows, "avfoundation_srgb", manifest_path

    keyframes_dir = segment_dir / f"keyframes_opencv_{args.opencv_output_format.lower().lstrip('.')}"
    manifest_rows = export_opencv_keyframes(
        video_path=video_path,
        events=events,
        fps=fps,
        output_dir=keyframes_dir,
        output_format=args.opencv_output_format,
    )
    manifest_path = keyframes_dir / "opencv_manifest.csv"
    return keyframes_dir, manifest_rows, f"opencv_{args.opencv_output_format.lower().lstrip('.')}", manifest_path


def analyze_segment(
    spec: SegmentSpec,
    video_dir: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    video_path = video_dir / spec.video_name
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    segment_dir = ensure_dir(output_root / spec.segment_id)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_count <= 0 or frame_width <= 0 or frame_height <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video metadata: {video_path}")

    start_frame = int(max(0, min(frame_count - 1, round(spec.start_time_sec * fps))))
    end_frame = frame_count - 1 if spec.end_time_sec is None else int(min(frame_count - 1, round(spec.end_time_sec * fps)))
    if end_frame <= start_frame:
        cap.release()
        raise RuntimeError(f"Empty segment range for {spec.segment_id}")

    reference_frame_idx = int(max(start_frame, min(end_frame, round(spec.reference_time_sec * fps))))
    secondary_reference_frame_idx = int(max(start_frame, min(end_frame, round((spec.reference_time_sec + args.secondary_offset_sec) * fps))))
    workpiece_roi = validate_roi(spec.workpiece_roi, frame_width, frame_height, "workpiece_roi")
    pose_roi = validate_roi(spec.pose_roi, frame_width, frame_height, "pose_roi")
    resize_hw = tuple(map(int, args.similarity_resize_hw))

    primary_ref = read_frame_at(cap, reference_frame_idx)
    secondary_ref = read_frame_at(cap, secondary_reference_frame_idx)
    primary_ref_gray = normalized_pose_gray(primary_ref, pose_roi, resize_hw)
    secondary_ref_gray = normalized_pose_gray(secondary_ref, pose_roi, resize_hw)

    hsv_lo = np.array(args.hsv_lo, dtype=np.uint8)
    hsv_hi = np.array(args.hsv_hi, dtype=np.uint8)
    x1, y1, x2, y2 = map(int, workpiece_roi)

    depth_raw: List[float] = []
    mask_count: List[int] = []
    sim_primary_raw: List[float] = []
    sim_secondary_raw: List[float] = []
    frame_numbers: List[int] = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_idx in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        roi_img = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
        mask = cv2.medianBlur(mask, 3)
        ys, _ = np.where(mask > 0)
        count = int(len(ys))
        if count < args.min_pixels:
            depth_raw.append(np.nan)
        else:
            depth_raw.append(float(ys.mean() + y1))
        mask_count.append(count)
        sim_primary_raw.append(pose_similarity(frame, pose_roi, resize_hw, primary_ref_gray))
        sim_secondary_raw.append(pose_similarity(frame, pose_roi, resize_hw, secondary_ref_gray))
        frame_numbers.append(frame_idx)
    cap.release()

    if not frame_numbers:
        raise RuntimeError(f"No frames decoded for {spec.segment_id}")

    frame_numbers_arr = np.asarray(frame_numbers, dtype=np.int32)
    depth_raw_arr = np.asarray(depth_raw, dtype=np.float32)
    mask_count_arr = np.asarray(mask_count, dtype=np.float32)
    depth_signal = moving_average_nan(depth_raw_arr, args.smooth_window)
    sim_primary_signal = moving_average_nan(np.asarray(sim_primary_raw, dtype=np.float32), args.smooth_window)
    sim_secondary_signal = moving_average_nan(np.asarray(sim_secondary_raw, dtype=np.float32), args.smooth_window)

    reference_local_idx = reference_frame_idx - start_frame
    secondary_local_idx = secondary_reference_frame_idx - start_frame
    state_scores = build_state_scores(
        depth_signal=depth_signal,
        primary_frame_idx=reference_local_idx,
        secondary_frame_idx=secondary_local_idx,
        sim_primary_signal=sim_primary_signal,
        sim_secondary_signal=sim_secondary_signal,
        alpha_depth=args.alpha_depth,
        alpha_pose=args.alpha_pose,
    )
    primary_state_score = state_scores["primary_state_score"]
    secondary_state_score = state_scores["secondary_state_score"]

    primary_groups = peak_groups(
        primary_state_score,
        quantile=args.fused_quantile,
        min_width=args.min_plateau_width,
        gap=args.merge_gap,
        boundary_margin=args.boundary_margin,
    )
    secondary_groups = peak_groups(
        secondary_state_score,
        quantile=args.fused_quantile,
        min_width=args.min_plateau_width,
        gap=args.merge_gap,
        boundary_margin=args.boundary_margin,
    )
    primary_events = groups_to_events(primary_state_score, primary_groups, event_type="reference")
    secondary_events = groups_to_events(secondary_state_score, secondary_groups, event_type="reference_plus_2s")

    local_end = len(frame_numbers_arr) - 1
    if not primary_events:
        peak = int(np.nanargmax(primary_state_score))
        primary_events = [Event(peak, peak, peak, float(primary_state_score[peak]), 1, float(primary_state_score[peak]), "reference")]
    if not secondary_events:
        peak = int(np.nanargmax(secondary_state_score))
        secondary_events = [
            Event(peak, peak, peak, float(secondary_state_score[peak]), 1, float(secondary_state_score[peak]), "reference_plus_2s")
        ]

    base_selected_primary = nms_events(primary_events, args.min_separation, args.max_events_per_type, args.selection)
    base_selected_secondary = nms_events(secondary_events, args.min_separation, args.max_events_per_type, args.selection)
    reference_average_gap = average_cross_score_gap(base_selected_primary, primary_state_score, secondary_state_score)
    secondary_average_gap = average_cross_score_gap(base_selected_secondary, primary_state_score, secondary_state_score)

    margin_rule_primary_events = build_margin_rule_events(
        dominant_signal=primary_state_score,
        other_signal=secondary_state_score,
        event_type="reference",
        average_gap=reference_average_gap,
        min_width=args.min_plateau_width,
        gap=args.merge_gap,
        boundary_margin=args.boundary_margin,
        start_frame=0,
        end_frame=local_end,
    )
    margin_rule_secondary_events = build_margin_rule_events(
        dominant_signal=secondary_state_score,
        other_signal=primary_state_score,
        event_type="reference_plus_2s",
        average_gap=secondary_average_gap,
        min_width=args.min_plateau_width,
        gap=args.merge_gap,
        boundary_margin=args.boundary_margin,
        start_frame=0,
        end_frame=local_end,
    )
    margin_rule_primary_events = keep_margin_events_after_first_base(margin_rule_primary_events, base_selected_primary)
    margin_rule_secondary_events = keep_margin_events_after_first_base(margin_rule_secondary_events, base_selected_secondary)

    selected_primary = nms_events(
        primary_events + margin_rule_primary_events,
        args.min_separation,
        args.max_events_per_type,
        args.selection,
    )
    selected_secondary = nms_events(
        secondary_events + margin_rule_secondary_events,
        args.min_separation,
        args.max_events_per_type,
        args.selection,
    )
    selected_local = collapse_consecutive_same_type_events(sorted(selected_primary + selected_secondary, key=lambda event: event.peak))
    selected = [event_to_absolute(event, start_frame) for event in selected_local]

    events_csv = segment_dir / "events.csv"
    write_events_csv(events_csv, selected, fps)

    keyframes_dir, manifest_rows, frame_export_label, keyframes_manifest = export_keyframes(video_path, selected, fps, segment_dir, args)

    event_rows = read_csv_rows(events_csv)
    contact_sheet(manifest_rows, event_rows, segment_dir / "contact_sheet.png", columns=args.contact_sheet_columns)

    depth_extrema = finite_signal_extrema(depth_signal)
    plot_debug(
        out_path=segment_dir / "debug_signals.png",
        frame_numbers=frame_numbers_arr,
        fps=fps,
        depth_signal=depth_signal,
        primary_state_score=primary_state_score,
        secondary_state_score=secondary_state_score,
        events=selected,
        frame_offset=start_frame,
        reference_frame_idx=reference_frame_idx,
        secondary_reference_frame_idx=secondary_reference_frame_idx,
    )

    segment_config = {
        "segment_id": spec.segment_id,
        "video": str(video_path),
        "reference_time_sec": spec.reference_time_sec,
        "secondary_reference_time_sec": spec.reference_time_sec + args.secondary_offset_sec,
        "start_time_sec": spec.start_time_sec,
        "end_time_sec": spec.end_time_sec,
        "workpiece_roi": workpiece_roi,
        "pose_roi": pose_roi,
        "hsv_lo": list(map(int, args.hsv_lo)),
        "hsv_hi": list(map(int, args.hsv_hi)),
        "min_pixels": args.min_pixels,
        "smooth_window": args.smooth_window,
        "fused_quantile": args.fused_quantile,
        "min_plateau_width": args.min_plateau_width,
        "merge_gap": args.merge_gap,
        "boundary_margin": args.boundary_margin,
        "min_separation": args.min_separation,
        "max_events_per_type": args.max_events_per_type,
        "similarity_resize_hw": list(resize_hw),
        "alpha_depth": args.alpha_depth,
        "alpha_pose": args.alpha_pose,
        "frame_export": frame_export_label,
    }
    (segment_dir / "roi_config.json").write_text(json.dumps(segment_config, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "segment_id": spec.segment_id,
        "video_name": spec.video_name,
        "video_path": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "segment_start_frame": start_frame,
        "segment_end_frame": int(frame_numbers_arr[-1]),
        "segment_start_time_sec": start_frame / fps if fps > 0 else 0.0,
        "segment_end_time_sec": int(frame_numbers_arr[-1]) / fps if fps > 0 else 0.0,
        "reference_frame_idx": reference_frame_idx,
        "reference_time_sec": reference_frame_idx / fps if fps > 0 else 0.0,
        "secondary_reference_frame_idx": secondary_reference_frame_idx,
        "secondary_reference_time_sec": secondary_reference_frame_idx / fps if fps > 0 else 0.0,
        "selected_event_count": len(selected),
        "reference_event_count": sum(1 for event in selected if event.event_type == "reference"),
        "reference_plus_2s_event_count": sum(1 for event in selected if event.event_type == "reference_plus_2s"),
        "primary_candidate_count": len(primary_events),
        "secondary_candidate_count": len(secondary_events),
        "margin_rule_reference_extra_count": len(margin_rule_primary_events),
        "margin_rule_reference_plus_2s_extra_count": len(margin_rule_secondary_events),
        "depth_signal_min_frame": int(depth_extrema["min_idx"] + start_frame),
        "depth_signal_max_frame": int(depth_extrema["max_idx"] + start_frame),
        "mask_count_mean": float(np.nanmean(mask_count_arr)) if len(mask_count_arr) else 0.0,
        "mask_count_min": float(np.nanmin(mask_count_arr)) if len(mask_count_arr) else 0.0,
        "mask_count_max": float(np.nanmax(mask_count_arr)) if len(mask_count_arr) else 0.0,
        "events_csv": str(events_csv),
        "frame_export": frame_export_label,
        "keyframes_dir": str(keyframes_dir),
        "keyframes_manifest": str(keyframes_manifest),
        "contact_sheet": str(segment_dir / "contact_sheet.png"),
        "debug_signals": str(segment_dir / "debug_signals.png"),
        "roi_config": str(segment_dir / "roi_config.json"),
    }
    (segment_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def write_batch_summary(output_root: Path, summaries: Sequence[Dict[str, object]]) -> None:
    json_path = output_root / "batch_summary.json"
    json_path.write_text(json.dumps(list(summaries), indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_root / "batch_summary.csv"
    fieldnames = [
        "segment_id",
        "video_name",
        "selected_event_count",
        "reference_event_count",
        "reference_plus_2s_event_count",
        "segment_start_time_sec",
        "segment_end_time_sec",
        "reference_time_sec",
        "secondary_reference_time_sec",
        "events_csv",
        "frame_export",
        "keyframes_dir",
        "keyframes_manifest",
        "contact_sheet",
        "debug_signals",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract visual-fusion keyframes from iPhone videos.")
    parser.add_argument("--video-dir", required=True, help="Directory containing IMG_0238.MOV ... IMG_0241.MOV")
    parser.add_argument("--output-root", required=True, help="Output directory for all artifacts")
    parser.add_argument(
        "--frame-export-mode",
        choices=["opencv", "avfoundation"],
        default="opencv",
        help="Frame image exporter. Default is OpenCV; use avfoundation explicitly for iPhone HDR tone mapping.",
    )
    parser.add_argument("--opencv-output-format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--hsv-lo", nargs=3, type=int, default=[0, 45, 40], help="HSV lower bound for copper mask")
    parser.add_argument("--hsv-hi", nargs=3, type=int, default=[30, 255, 255], help="HSV upper bound for copper mask")
    parser.add_argument("--min-pixels", type=int, default=40)
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--fused-quantile", type=float, default=0.88)
    parser.add_argument("--min-plateau-width", type=int, default=5)
    parser.add_argument("--merge-gap", type=int, default=4)
    parser.add_argument("--boundary-margin", type=int, default=20)
    parser.add_argument("--min-separation", type=int, default=35)
    parser.add_argument("--max-events-per-type", type=int, default=240)
    parser.add_argument("--selection", choices=["score", "earliest"], default="score")
    parser.add_argument("--similarity-resize-hw", nargs=2, type=int, default=[32, 32])
    parser.add_argument("--alpha-depth", type=float, default=0.6)
    parser.add_argument("--alpha-pose", type=float, default=0.4)
    parser.add_argument("--secondary-offset-sec", type=float, default=2.0)
    parser.add_argument("--contact-sheet-columns", type=int, default=8)
    parser.add_argument("--avfoundation-output-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--avfoundation-time-tolerance-sec", type=float, default=0.05)
    parser.add_argument("--avfoundation-clang-cache-dir", default="/private/tmp/codex-clang-cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir)
    output_root = ensure_dir(Path(args.output_root))

    summaries = []
    failures = []
    for spec in DEFAULT_SEGMENTS:
        print(f"[RUN ] {spec.segment_id} from {spec.video_name}")
        try:
            summary = analyze_segment(spec, video_dir, output_root, args)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {spec.segment_id}: {exc}")
            failures.append({"segment_id": spec.segment_id, "error": str(exc)})
            continue
        summaries.append(summary)
        print(f"[DONE] {spec.segment_id}: {summary['selected_event_count']} keyframes")

    write_batch_summary(output_root, summaries)
    if failures:
        (output_root / "batch_failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
        raise SystemExit(1)
    print(f"Batch summary: {output_root / 'batch_summary.csv'}")


if __name__ == "__main__":
    main()
