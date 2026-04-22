from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np


def numeric_sort_key(path: Path) -> tuple:
    return (not path.stem.isdigit(), int(path.stem) if path.stem.isdigit() else path.stem)


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_frame(video_path: Path, frame_idx: int) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame, fps


def clamp_xyxy(roi: Sequence[int], frame_width: int, frame_height: int) -> List[int]:
    x1, y1, x2, y2 = map(int, roi)
    x1 = max(0, min(frame_width - 1, x1))
    y1 = max(0, min(frame_height - 1, y1))
    x2 = max(x1 + 1, min(frame_width, x2))
    y2 = max(y1 + 1, min(frame_height, y2))
    return [x1, y1, x2, y2]


def crop_xyxy(frame: np.ndarray, roi: Sequence[int]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, roi)
    return frame[y1:y2, x1:x2]


def fit_width(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((width, width, 3), 255, dtype=np.uint8)
    scale = width / float(w)
    return cv2.resize(image, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def add_caption(image: np.ndarray, caption: str) -> np.ndarray:
    pad = np.full((56, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(pad, caption, (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (40, 40, 40), 2, cv2.LINE_AA)
    return np.vstack([pad, image])


def pad_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image
    pad = np.full((target_height - image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
    return np.vstack([image, pad])


def build_mask_preview(
    frame: np.ndarray,
    workpiece_roi: Sequence[int],
    hsv_lo: Sequence[int],
    hsv_hi: Sequence[int],
    title: str,
) -> tuple[np.ndarray, int, float]:
    x1, y1, x2, y2 = map(int, workpiece_roi)

    full_frame = frame.copy()
    cv2.rectangle(full_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(full_frame, title, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(full_frame, "workpiece_roi", (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 255), 2, cv2.LINE_AA)

    crop = crop_xyxy(frame, workpiece_roi)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo, dtype=np.uint8), np.array(hsv_hi, dtype=np.uint8))
    mask_pixels = int(mask.sum() / 255)
    mask_ratio = float((mask > 0).mean()) if mask.size else 0.0

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = crop.copy()
    overlay[mask > 0] = (0.35 * overlay[mask > 0] + 0.65 * np.array([0, 255, 255])).astype(np.uint8)

    cell_width = 430
    panels = [
        add_caption(fit_width(full_frame, cell_width), "Full frame + ROI"),
        add_caption(fit_width(crop, cell_width), "Workpiece crop"),
        add_caption(fit_width(mask_bgr, cell_width), "Binary mask"),
        add_caption(fit_width(overlay, cell_width), "Mask overlay"),
    ]
    row_height = max(panel.shape[0] for panel in panels)
    row = np.hstack([pad_to_height(panel, row_height) for panel in panels])

    footer = np.full((132, row.shape[1], 3), 248, dtype=np.uint8)
    lines = [
        title,
        f"workpiece ROI = [{x1}, {y1}, {x2}, {y2}]",
        f"hsv_lo = {list(map(int, hsv_lo))}    hsv_hi = {list(map(int, hsv_hi))}",
        f"mask pixels = {mask_pixels}    mask ratio = {mask_ratio * 100:.2f}%",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(footer, line, (18, 30 + idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (45, 45, 45), 2, cv2.LINE_AA)

    return np.vstack([row, footer]), mask_pixels, mask_ratio


def write_html(rows: List[Dict[str, str]], out_path: Path) -> None:
    lines = [
        "<!doctype html>",
        '<html><head><meta charset="utf-8"><title>Workpiece Masks</title>',
        '<style>body{font-family:Segoe UI,sans-serif;margin:24px;background:#f5f5f5;color:#222}h1{margin-bottom:8px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(540px,1fr));gap:20px}.card{background:#fff;border:1px solid #ddd;border-radius:12px;padding:14px;box-shadow:0 1px 4px rgba(0,0,0,.04)}img{max-width:100%;height:auto;border-radius:8px;border:1px solid #d4d4d4}.meta{font-size:14px;line-height:1.5;margin-bottom:10px}</style></head><body>',
        "<h1>Workpiece Mask Previews</h1>",
        "<p>Each preview uses the configured reference frame and the same HSV thresholding used by the current depth-signal pipeline.</p>",
        '<div class="grid">',
    ]
    for row in rows:
        preview_name = Path(row["preview_path"]).name
        lines.extend(
            [
                '<div class="card">',
                (
                    f'<div class="meta"><strong>Video {row["video_id"]}</strong><br>'
                    f'frame={row["frame_idx"]} time={row["time_seconds"]}s<br>'
                    f'mask pixels={row["mask_pixels"]} ratio={row["mask_ratio_percent"]}%<br>'
                    f'roi={row["workpiece_roi"]}<br>'
                    f'hsv_lo={row["hsv_lo"]} hsv_hi={row["hsv_hi"]}</div>'
                ),
                f'<img src="previews/{preview_name}" alt="Video {row["video_id"]} workpiece mask preview">',
                "</div>",
            ]
        )
    lines.extend(["</div>", "</body></html>"])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-video workpiece mask previews from the configured reference frame.")
    parser.add_argument("--video-dir", required=True, help="Directory containing input videos")
    parser.add_argument("--config-dir", required=True, help="Directory containing per-video JSON configs")
    parser.add_argument("--out-dir", required=True, help="Directory for output previews and manifest")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    config_dir = Path(args.config_dir)
    out_dir = Path(args.out_dir)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []

    for config_path in sorted(config_dir.glob("*.json"), key=numeric_sort_key):
        if config_path.stem == "template":
            continue

        video_path = video_dir / f"{config_path.stem}.mp4"
        if not video_path.exists():
            print(f"[SKIP] {config_path.name}: missing video {video_path.name}")
            continue

        cfg = load_config(config_path)
        frame, fps = read_frame(video_path, int(cfg.get("reference_frame_idx", 0)))
        frame_height, frame_width = frame.shape[:2]
        workpiece_roi = clamp_xyxy(cfg["workpiece_roi"], frame_width=frame_width, frame_height=frame_height)
        hsv_lo = cfg.get("hsv_lo", [5, 80, 50])
        hsv_hi = cfg.get("hsv_hi", [22, 255, 255])
        frame_idx = int(cfg.get("reference_frame_idx", 0))
        time_seconds = frame_idx / fps if fps > 0 else 0.0
        title = f"Video {config_path.stem} | ref frame {frame_idx} | t={time_seconds:.3f}s"

        preview, mask_pixels, mask_ratio = build_mask_preview(
            frame=frame,
            workpiece_roi=workpiece_roi,
            hsv_lo=hsv_lo,
            hsv_hi=hsv_hi,
            title=title,
        )
        preview_path = preview_dir / f"{config_path.stem}_refmask_f{frame_idx}_t{time_seconds:.3f}s.png"
        cv2.imwrite(str(preview_path), preview)

        manifest_rows.append(
            {
                "video_id": config_path.stem,
                "video_path": str(video_path),
                "config_path": str(config_path),
                "frame_idx": str(frame_idx),
                "time_seconds": f"{time_seconds:.3f}",
                "mask_pixels": str(mask_pixels),
                "mask_ratio_percent": f"{mask_ratio * 100:.2f}",
                "workpiece_roi": ",".join(map(str, workpiece_roi)),
                "hsv_lo": ",".join(map(str, hsv_lo)),
                "hsv_hi": ",".join(map(str, hsv_hi)),
                "preview_path": str(preview_path),
            }
        )
        print(f"[OK  ] {config_path.stem}: {preview_path.name}")

    manifest_path = out_dir / "workpiece_mask_manifest.csv"
    with open(manifest_path, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "video_id",
            "video_path",
            "config_path",
            "frame_idx",
            "time_seconds",
            "mask_pixels",
            "mask_ratio_percent",
            "workpiece_roi",
            "hsv_lo",
            "hsv_hi",
            "preview_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    html_path = out_dir / "index.html"
    write_html(manifest_rows, html_path)

    print(f"Exported {len(manifest_rows)} previews to {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
