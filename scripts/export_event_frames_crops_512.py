from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2


def numeric_sort_key_from_name(name: str) -> tuple:
    return (not name.isdigit(), int(name) if name.isdigit() else name)


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_event_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
        "global_index",
        "video_id",
        "video_name",
        "event_rank",
        "event_type",
        "source_frame_idx",
        "source_time_seconds",
        "fps",
        "roi_rect_xywh",
        "crop_xyxy",
        "frame_file",
        "crop_file",
        "frame_relpath",
        "crop_relpath",
    ]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one original frame and one 512x512 crop for every event in a batch result.")
    parser.add_argument("--batch-output-dir", required=True, help="Batch result directory containing per-video events.csv")
    parser.add_argument("--video-dir", required=True, help="Directory containing source videos")
    parser.add_argument("--config-dir", required=True, help="Directory containing per-video JSON configs")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--size", type=int, default=512, help="Final crop size in pixels")
    args = parser.parse_args()

    batch_output_dir = Path(args.batch_output_dir)
    video_dir = Path(args.video_dir)
    config_dir = Path(args.config_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_exported = 0

    for video_output_dir in sorted(
        [p for p in batch_output_dir.iterdir() if p.is_dir()],
        key=lambda p: numeric_sort_key_from_name(p.name),
    ):
        video_id = video_output_dir.name
        events_path = video_output_dir / "events.csv"
        if not events_path.exists():
            continue

        config_path = config_dir / f"{video_id}.json"
        video_path = video_dir / f"{video_id}.mp4"
        if not config_path.exists() or not video_path.exists():
            continue

        cfg = load_config(config_path)
        event_rows = load_event_rows(events_path)
        roi_rect_xywh = cfg.get("_meta", {}).get("roi_rect_xywh")
        if not roi_rect_xywh:
            continue

        video_export_dir = out_dir / video_id
        frames_dir = video_export_dir / "frames"
        crops_dir = video_export_dir / "crops"
        frames_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows: List[Dict[str, str]] = []

        for row in event_rows:
            event_rank = int(row["rank"])
            event_type = row["event_type"]
            frame_idx = int(row["frame"])
            time_seconds = float(row["time_seconds"])

            frame, fps = read_video_frame(video_path, frame_idx)
            frame_height, frame_width = frame.shape[:2]
            crop_roi_xyxy = square_crop_xyxy_from_xywh(
                rect_xywh=roi_rect_xywh,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            crop = crop_xyxy(frame, crop_roi_xyxy)
            crop_resized = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_AREA)

            global_index = total_exported + 1
            frame_name = f"{global_index}_frame.jpg"
            crop_name = f"{global_index}_crop.jpg"
            frame_path = frames_dir / frame_name
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(frame_path), frame)
            cv2.imwrite(str(crop_path), crop_resized)

            manifest_rows.append(
                {
                    "global_index": str(global_index),
                    "video_id": video_id,
                    "video_name": video_path.name,
                    "event_rank": str(event_rank),
                    "event_type": event_type,
                    "source_frame_idx": str(frame_idx),
                    "source_time_seconds": f"{time_seconds:.6f}",
                    "fps": f"{fps:.6f}",
                    "roi_rect_xywh": ",".join(map(str, roi_rect_xywh)),
                    "crop_xyxy": ",".join(map(str, crop_roi_xyxy)),
                    "frame_file": frame_name,
                    "crop_file": crop_name,
                    "frame_relpath": f"{video_id}/frames/{frame_name}",
                    "crop_relpath": f"{video_id}/crops/{crop_name}",
                }
            )
            print(
                f"[OK  ] video={video_id} global_index={global_index} event={event_rank} "
                f"type={event_type} frame={frame_idx} -> {frame_name}, {crop_name}"
            )
            total_exported += 1

        manifest_path = video_export_dir / "manifest.csv"
        write_manifest(manifest_rows, manifest_path)
        print(f"[DONE] video={video_id} exported {len(manifest_rows)} pairs -> {video_export_dir}")

    print(f"Exported {total_exported} event frame/crop pairs to {out_dir}")


if __name__ == "__main__":
    main()
