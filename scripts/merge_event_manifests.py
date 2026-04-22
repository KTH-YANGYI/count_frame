from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def numeric_sort_key_from_name(name: str) -> tuple:
    return (not name.isdigit(), int(name) if name.isdigit() else name)


def load_manifest_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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
        "video_dir",
        "video_local_index",
        "frame_relpath",
        "crop_relpath",
    ]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-video event manifests into one global manifest.")
    parser.add_argument("--root-dir", required=True, help="Root directory containing per-video export folders")
    parser.add_argument("--out-file", default=None, help="Optional output CSV path; defaults to <root-dir>/manifest_all.csv")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_file = Path(args.out_file) if args.out_file else root_dir / "manifest_all.csv"

    merged_rows: List[Dict[str, str]] = []
    global_index = 1

    for video_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()], key=lambda p: numeric_sort_key_from_name(p.name)):
        manifest_path = video_dir / "manifest.csv"
        if not manifest_path.exists():
            continue

        for row in load_manifest_rows(manifest_path):
            frame_file = row["frame_file"]
            crop_file = row["crop_file"]
            video_local_index = row.get("video_local_index", row.get("index", row.get("global_index", "")))
            merged_rows.append(
                {
                    "global_index": row.get("global_index", str(global_index)),
                    "video_id": row["video_id"],
                    "video_name": row["video_name"],
                    "event_rank": row["event_rank"],
                    "event_type": row["event_type"],
                    "source_frame_idx": row["source_frame_idx"],
                    "source_time_seconds": row["source_time_seconds"],
                    "fps": row["fps"],
                    "roi_rect_xywh": row["roi_rect_xywh"],
                    "crop_xyxy": row["crop_xyxy"],
                    "frame_file": frame_file,
                    "crop_file": crop_file,
                    "video_dir": video_dir.name,
                    "video_local_index": video_local_index,
                    "frame_relpath": row.get("frame_relpath", f"{video_dir.name}/frames/{frame_file}"),
                    "crop_relpath": row.get("crop_relpath", f"{video_dir.name}/crops/{crop_file}"),
                }
            )
            global_index += 1

    write_manifest(merged_rows, out_file)
    print(f"Merged {len(merged_rows)} rows into {out_file}")


if __name__ == "__main__":
    main()
