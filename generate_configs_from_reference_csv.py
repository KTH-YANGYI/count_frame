from __future__ import annotations

import argparse
from pathlib import Path

from config_utils import BUNDLED_REFERENCE_CSV, build_config_from_reference_row, load_reference_rows, save_config


def numeric_video_sort_key(path: Path) -> tuple:
    return (not path.stem.isdigit(), int(path.stem) if path.stem.isdigit() else path.stem)


def main() -> None:
    default_reference_csv = str(BUNDLED_REFERENCE_CSV) if BUNDLED_REFERENCE_CSV.exists() else None
    parser = argparse.ArgumentParser(description="Generate per-video JSON configs from reference_frames.csv.")
    parser.add_argument(
        "--reference-csv",
        default=default_reference_csv,
        help="Path to reference_frames.csv; defaults to data/reference/reference_frames.csv when bundled",
    )
    parser.add_argument("--video-dir", required=True, help="Directory containing input videos")
    parser.add_argument("--out-dir", required=True, help="Directory to write generated JSON configs")
    args = parser.parse_args()

    if not args.reference_csv:
        raise SystemExit("reference CSV not provided and bundled CSV is missing")

    reference_csv = Path(args.reference_csv)
    if not reference_csv.exists():
        raise SystemExit(f"reference CSV not found: {reference_csv}")
    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)

    rows = load_reference_rows(reference_csv)
    generated = 0
    skipped = 0
    for video_path in sorted(video_dir.glob("*.mp4"), key=numeric_video_sort_key):
        row = rows.get(video_path.stem)
        if row is None:
            skipped += 1
            continue
        row = dict(row)
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
