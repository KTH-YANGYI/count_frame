from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_event_rows(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame


def crop_xyxy(frame: np.ndarray, roi: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, roi)
    return frame[y1:y2, x1:x2]


def fit_width(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        return np.zeros((width, width, 3), dtype=np.uint8)
    scale = width / float(w)
    return cv2.resize(image, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def make_panel(
    frame: np.ndarray,
    workpiece_roi: Sequence[int],
    pose_roi: Sequence[int],
    hsv_lo: Sequence[int],
    hsv_hi: Sequence[int],
    title: str,
) -> np.ndarray:
    x1, y1, x2, y2 = map(int, workpiece_roi)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
    px1, py1, px2, py2 = map(int, pose_roi)
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (0, 255, 0), 3)
    cv2.putText(overlay, title, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, "workpiece_roi", (x1, max(32, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, "pose_roi", (px1, max(32, py1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    crop = crop_xyxy(frame, workpiece_roi)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Hue is in [0,179] in OpenCV. Scale to [0,255] for color mapping.
    h_vis = cv2.applyColorMap(cv2.convertScaleAbs(h, alpha=255.0 / 179.0), cv2.COLORMAP_HSV)
    s_vis = cv2.applyColorMap(s, cv2.COLORMAP_TURBO)
    v_vis = cv2.applyColorMap(v, cv2.COLORMAP_BONE)

    mask = cv2.inRange(hsv, np.array(hsv_lo, dtype=np.uint8), np.array(hsv_hi, dtype=np.uint8))
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay_mask = crop.copy()
    overlay_mask[mask > 0] = (0.35 * overlay_mask[mask > 0] + 0.65 * np.array([0, 255, 255])).astype(np.uint8)

    mask_ratio = float((mask > 0).mean())
    lines = [
        title,
        f"workpiece ROI = [{x1}, {y1}, {x2}, {y2}]",
        f"pose ROI = [{px1}, {py1}, {px2}, {py2}]",
        f"hsv_lo = {list(map(int, hsv_lo))}",
        f"hsv_hi = {list(map(int, hsv_hi))}",
        f"mask pixels = {int(mask.sum() / 255)} ({mask_ratio * 100:.1f}%)",
        f"H mean/std = {float(h.mean()):.1f} / {float(h.std()):.1f}",
        f"S mean/std = {float(s.mean()):.1f} / {float(s.std()):.1f}",
        f"V mean/std = {float(v.mean()):.1f} / {float(v.std()):.1f}",
    ]
    cell_w = 520
    top_row = [
        fit_width(overlay, cell_w),
        fit_width(crop, cell_w),
        fit_width(h_vis, cell_w),
    ]
    bottom_row = [
        fit_width(s_vis, cell_w),
        fit_width(v_vis, cell_w),
        fit_width(overlay_mask, cell_w),
    ]

    def add_caption(image: np.ndarray, caption: str) -> np.ndarray:
        pad = np.full((50, image.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(pad, caption, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (35, 35, 35), 2, cv2.LINE_AA)
        return np.vstack([pad, image])

    top_row = [
        add_caption(top_row[0], "Full frame + ROIs"),
        add_caption(top_row[1], "Workpiece crop (BGR)"),
        add_caption(top_row[2], "Hue (H)"),
    ]
    bottom_row = [
        add_caption(bottom_row[0], "Saturation (S)"),
        add_caption(bottom_row[1], "Value (V)"),
        add_caption(bottom_row[2], "HSV threshold mask overlay"),
    ]

    # Equalize row heights.
    top_h = max(img.shape[0] for img in top_row)
    bottom_h = max(img.shape[0] for img in bottom_row)

    def pad_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
        if image.shape[0] == target_h:
            return image
        pad = np.full((target_h - image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
        return np.vstack([image, pad])

    top_row = [pad_to_height(img, top_h) for img in top_row]
    bottom_row = [pad_to_height(img, bottom_h) for img in bottom_row]

    row1 = np.hstack(top_row)
    row2 = np.hstack(bottom_row)
    stats = np.full((280, row1.shape[1], 3), 248, dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(stats, line, (18, 36 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (40, 40, 40), 2, cv2.LINE_AA)

    panel = np.vstack([row1, row2, stats])
    return panel


def make_compact_panel(
    frame: np.ndarray,
    workpiece_roi: Sequence[int],
    hsv_lo: Sequence[int],
    hsv_hi: Sequence[int],
    title: str,
) -> np.ndarray:
    crop = crop_xyxy(frame, workpiece_roi)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    h_vis = cv2.applyColorMap(cv2.convertScaleAbs(h, alpha=255.0 / 179.0), cv2.COLORMAP_HSV)
    s_vis = cv2.applyColorMap(s, cv2.COLORMAP_TURBO)
    v_vis = cv2.applyColorMap(v, cv2.COLORMAP_BONE)
    mask = cv2.inRange(hsv, np.array(hsv_lo, dtype=np.uint8), np.array(hsv_hi, dtype=np.uint8))
    overlay_mask = crop.copy()
    overlay_mask[mask > 0] = (0.35 * overlay_mask[mask > 0] + 0.65 * np.array([0, 255, 255])).astype(np.uint8)

    cell_w = 640
    imgs = [
        ("Original crop", fit_width(crop, cell_w)),
        ("Hue (H)", fit_width(h_vis, cell_w)),
        ("Saturation (S)", fit_width(s_vis, cell_w)),
        ("Value (V)", fit_width(v_vis, cell_w)),
        ("Final extracted region", fit_width(overlay_mask, cell_w)),
    ]

    def add_caption(image: np.ndarray, caption: str) -> np.ndarray:
        pad = np.full((58, image.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(pad, caption, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (35, 35, 35), 2, cv2.LINE_AA)
        return np.vstack([pad, image])

    imgs = [(caption, add_caption(img, caption)) for caption, img in imgs]
    top_h = max(img.shape[0] for _, img in imgs)

    padded = []
    for _, img in imgs:
        if img.shape[0] < top_h:
            pad = np.full((top_h - img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)

    row = np.hstack(padded)
    footer = np.full((110, row.shape[1], 3), 248, dtype=np.uint8)
    x1, y1, x2, y2 = map(int, workpiece_roi)
    cv2.putText(footer, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(footer, f"workpiece ROI = [{x1}, {y1}, {x2}, {y2}]", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (60, 60, 60), 2, cv2.LINE_AA)
    cv2.putText(
        footer,
        f"hsv_lo = {list(map(int, hsv_lo))}    hsv_hi = {list(map(int, hsv_hi))}    mask pixels = {int(mask.sum() / 255)}",
        (20, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (75, 75, 75),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([row, footer])


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize workpiece HSV panels for selected event frames.")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--events-csv", required=True, help="Path to events.csv")
    parser.add_argument("--out-dir", required=True, help="Directory for output panels")
    parser.add_argument("--event-indices", nargs="*", type=int, default=[1, 2, 4, 8], help="1-based event indices to visualize")
    parser.add_argument("--compact", action="store_true", help="Export 4 clean panels: crop, H, mask, final result")
    args = parser.parse_args()

    video_path = Path(args.video)
    cfg = load_config(Path(args.config))
    event_rows = load_event_rows(Path(args.events_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_indices = [idx for idx in args.event_indices if 1 <= idx <= len(event_rows)]
    manifest_lines = ["# Video 4 workpiece HSV panels", ""]
    for idx in selected_indices:
        row = event_rows[idx - 1]
        frame_idx = int(row["frame"])
        time_sec = float(row["time_seconds"])
        event_type = row["event_type"]
        frame = read_frame(video_path, frame_idx)
        title = f"Event #{idx} | {event_type} | f{frame_idx} | t={time_sec:.3f}s"
        if args.compact:
            panel = make_compact_panel(
                frame=frame,
                workpiece_roi=cfg["workpiece_roi"],
                hsv_lo=cfg["hsv_lo"],
                hsv_hi=cfg["hsv_hi"],
                title=title,
            )
            out_path = out_dir / f"event_{idx:02d}_f{frame_idx}_t{time_sec:.3f}s_hsv_compact.png"
        else:
            panel = make_panel(
                frame=frame,
                workpiece_roi=cfg["workpiece_roi"],
                pose_roi=cfg["pose_roi"],
                hsv_lo=cfg["hsv_lo"],
                hsv_hi=cfg["hsv_hi"],
                title=title,
            )
            out_path = out_dir / f"event_{idx:02d}_f{frame_idx}_t{time_sec:.3f}s_hsv_panel.png"
        cv2.imwrite(str(out_path), panel)
        manifest_lines.append(f"- {out_path.name}")

    (out_dir / "README.md").write_text("\n".join(manifest_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
