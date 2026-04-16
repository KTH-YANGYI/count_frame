from __future__ import annotations

import argparse
from pathlib import Path

from config_utils import build_config_from_reference_row, load_reference_rows, save_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-video JSON configs from reference_frames.csv.")
    parser.add_argument("--reference-csv", required=True, help="Path to reference_frames.csv")
    parser.add_argument("--video-dir", required=True, help="Directory containing input videos")
    parser.add_argument("--out-dir", required=True, help="Directory to write generated JSON configs")
    args = parser.parse_args()

    reference_csv = Path(args.reference_csv)
    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)

    rows = load_reference_rows(reference_csv)
    generated = 0
    skipped = 0
    for video_path in sorted(video_dir.glob("*.mp4"), key=lambda p: (not p.stem.isdigit(), p.stem)):
        row = rows.get(video_path.stem)
        if row is None:
            skipped += 1
            continue
        row = dict(row)
        row["source_csv"] = str(reference_csv)
        cfg = build_config_from_reference_row(
            video_path=video_path,
            row=row,
            use_reference_frame=True,
        )
        save_config(out_dir / f"{video_path.stem}.json", cfg)
        generated += 1

    print(f"Generated {generated} config files into {out_dir}")
    print(f"Skipped {skipped} videos without a matching CSV row")


if __name__ == "__main__":
    main()
