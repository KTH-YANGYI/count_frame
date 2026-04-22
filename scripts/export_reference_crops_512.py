from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2

import _path  # noqa: F401
from frame_count.config_utils import BUNDLED_REFERENCE_CSV, load_reference_rows, parse_roi_rect


def numeric_video_sort_key(path: Path) -> tuple:
    return (not path.stem.isdigit(), int(path.stem) if path.stem.isdigit() else path.stem)


def read_video_frame(video_path: Path, frame_idx: int) -> Tuple[cv2.typing.MatLike, float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame, fps


def square_crop_xyxy_from_xywh(
    rect_xywh: Sequence[int],
    frame_width: int,
    frame_height: int,
) -> List[int]:
    x, y, w, h = map(int, rect_xywh)
    side = max(1, min(max(w, h), frame_width, frame_height))
    cx = x + w / 2.0
    cy = y + h / 2.0

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = x1 + side
    y2 = y1 + side

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > frame_width:
        shift = x2 - frame_width
        x1 -= shift
        x2 = frame_width
    if y2 > frame_height:
        shift = y2 - frame_height
        y1 -= shift
        y2 = frame_height

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width, x2)
    y2 = min(frame_height, y2)
    return [int(x1), int(y1), int(x2), int(y2)]


def crop_xyxy(frame, roi_xyxy: Sequence[int]):
    x1, y1, x2, y2 = map(int, roi_xyxy)
    return frame[y1:y2, x1:x2]


def write_manifest(rows: List[Dict[str, str]], out_path: Path) -> None:
    fieldnames = [
        "index",
        "video_id",
        "video_name",
        "source_frame_idx",
        "source_time_seconds",
        "fps",
        "roi_rect_xywh",
        "crop_xyxy",
        "frame_file",
        "crop_file",
    ]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    default_reference_csv = str(BUNDLED_REFERENCE_CSV) if BUNDLED_REFERENCE_CSV.exists() else None
    parser = argparse.ArgumentParser(description="Export one reference frame and one 512x512 crop per video.")
    parser.add_argument(
        "--reference-csv",
        default=default_reference_csv,
        help="Path to reference_frames.csv; defaults to data/reference/reference_frames.csv when bundled",
    )
    parser.add_argument("--video-dir", required=True, help="Directory containing input videos")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--size", type=int, default=512, help="Final crop size in pixels")
    args = parser.parse_args()

    if not args.reference_csv:
        raise SystemExit("reference CSV not provided and bundled CSV is missing")

    reference_csv = Path(args.reference_csv)
    if not reference_csv.exists():
        raise SystemExit(f"reference CSV not found: {reference_csv}")

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    crops_dir = out_dir / "crops"
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    rows = load_reference_rows(reference_csv)
    manifest_rows: List[Dict[str, str]] = []

    export_index = 1
    for video_path in sorted(video_dir.glob("*.mp4"), key=numeric_video_sort_key):
        row = rows.get(video_path.stem)
        if row is None:
            continue

        frame_idx = int(float(row["reference_frame_idx"]))
        roi_rect_xywh = parse_roi_rect(row["roi_rect"])
        frame, fps = read_video_frame(video_path, frame_idx)
        frame_height, frame_width = frame.shape[:2]
        crop_roi_xyxy = square_crop_xyxy_from_xywh(
            rect_xywh=roi_rect_xywh,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        crop = crop_xyxy(frame, crop_roi_xyxy)
        crop_resized = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_AREA)

        frame_name = f"{export_index}_frame.jpg"
        crop_name = f"{export_index}_crop.jpg"
        frame_path = frames_dir / frame_name
        crop_path = crops_dir / crop_name

        cv2.imwrite(str(frame_path), frame)
        cv2.imwrite(str(crop_path), crop_resized)

        manifest_rows.append(
            {
                "index": str(export_index),
                "video_id": video_path.stem,
                "video_name": video_path.name,
                "source_frame_idx": str(frame_idx),
                "source_time_seconds": f"{(frame_idx / fps) if fps > 0 else 0.0:.6f}",
                "fps": f"{fps:.6f}",
                "roi_rect_xywh": ",".join(map(str, roi_rect_xywh)),
                "crop_xyxy": ",".join(map(str, crop_roi_xyxy)),
                "frame_file": frame_name,
                "crop_file": crop_name,
            }
        )
        print(f"[OK  ] #{export_index} video={video_path.stem} frame={frame_idx} -> {frame_name}, {crop_name}")
        export_index += 1

    manifest_path = out_dir / "manifest.csv"
    write_manifest(manifest_rows, manifest_path)
    print(f"Exported {len(manifest_rows)} frame/crop pairs to {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
