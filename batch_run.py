from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from config_utils import build_config_from_reference_row, load_reference_rows, save_config


def numeric_video_sort_key(path: Path) -> tuple:
    return (not path.stem.isdigit(), int(path.stem) if path.stem.isdigit() else path.stem)


def maybe_generate_config(
    video_path: Path,
    config_dir: Path,
    reference_rows: Optional[Dict[str, Dict[str, str]]],
    generated_config_dir: Optional[Path],
    use_reference_frame: bool,
) -> Optional[Path]:
    config_path = config_dir / f"{video_path.stem}.json"
    if config_path.exists():
        return config_path
    if reference_rows is None or generated_config_dir is None:
        return None

    row = reference_rows.get(video_path.stem)
    if row is None:
        return None

    row = dict(row)
    cfg = build_config_from_reference_row(
        video_path=video_path,
        row=row,
        use_reference_frame=use_reference_frame,
    )
    generated_config_path = generated_config_dir / f"{video_path.stem}.json"
    save_config(generated_config_path, cfg)
    return generated_config_path


def read_event_count(video_output_dir: Path) -> int:
    events_csv = video_output_dir / "events.csv"
    if not events_csv.exists():
        return 0
    with open(events_csv, "r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def read_reference_source(video_output_dir: Path) -> str:
    summary_path = video_output_dir / "run_summary.json"
    if not summary_path.exists():
        return ""
    with open(summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return str(payload.get("reference_source", ""))


def write_batch_summary(out_path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "video_id",
        "video_name",
        "status",
        "config_path",
        "output_dir",
        "event_count",
        "reference_source",
        "selection",
        "max_events_override",
        "fused_quantile_override",
        "min_plateau_width_override",
        "note",
    ]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run max-stroke keyframe extraction over a directory of videos.")
    parser.add_argument("--input-dir", required=True, help="Directory containing input videos")
    parser.add_argument("--config-dir", required=True, help="Directory containing per-video JSON configs")
    parser.add_argument("--output-dir", required=True, help="Directory containing output folders")
    parser.add_argument("--reference-csv", default=None, help="Optional reference_frames.csv used to auto-generate missing configs")
    parser.add_argument("--generated-config-dir", default=None, help="Where auto-generated configs should be written; defaults to <config-dir>/generated")
    parser.add_argument("--selection", choices=["score", "earliest"], default="score", help="Selection mode passed to the extractor")
    parser.add_argument("--max-events", type=int, default=None, help="Optional override passed to the extractor")
    parser.add_argument("--fused-quantile", type=float, default=None, help="Optional fused quantile override passed to the extractor")
    parser.add_argument("--min-plateau-width", type=int, default=None, help="Optional minimum plateau width override passed to the extractor")
    parser.add_argument("--start-frame", type=int, default=0, help="Optional override passed to the extractor")
    parser.add_argument("--end-frame", type=int, default=-1, help="Optional override passed to the extractor")
    parser.add_argument("--skip-existing", action="store_true", help="Skip videos whose output directory already exists and is non-empty")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed video")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_rows = None
    generated_config_dir = None
    if args.reference_csv:
        reference_rows = load_reference_rows(Path(args.reference_csv))
        generated_config_dir = Path(args.generated_config_dir) if args.generated_config_dir else config_dir / "generated"
        generated_config_dir.mkdir(parents=True, exist_ok=True)

    extractor_path = Path(__file__).with_name("extract_max_stroke_keyframes.py")
    failures = []
    processed = 0
    skipped = 0
    summary_rows: List[Dict[str, str]] = []

    for video_path in sorted(input_dir.glob("*.mp4"), key=numeric_video_sort_key):
        config_path = maybe_generate_config(
            video_path=video_path,
            config_dir=config_dir,
            reference_rows=reference_rows,
            generated_config_dir=generated_config_dir,
            use_reference_frame=True,
        )
        if config_path is None:
            print(f"[SKIP] {video_path.name}: no config found")
            skipped += 1
            summary_rows.append(
                {
                    "video_id": video_path.stem,
                    "video_name": video_path.name,
                    "status": "skipped",
                    "config_path": "",
                    "output_dir": str(output_dir / video_path.stem),
                    "event_count": "",
                    "reference_source": "",
                    "selection": args.selection,
                    "max_events_override": "" if args.max_events is None else str(args.max_events),
                    "fused_quantile_override": "" if args.fused_quantile is None else str(args.fused_quantile),
                    "min_plateau_width_override": "" if args.min_plateau_width is None else str(args.min_plateau_width),
                    "note": "no config found",
                }
            )
            continue

        video_output_dir = output_dir / video_path.stem
        if args.skip_existing and video_output_dir.exists() and any(video_output_dir.iterdir()):
            print(f"[SKIP] {video_path.name}: output already exists")
            skipped += 1
            summary_rows.append(
                {
                    "video_id": video_path.stem,
                    "video_name": video_path.name,
                    "status": "skipped",
                    "config_path": str(config_path),
                    "output_dir": str(video_output_dir),
                    "event_count": str(read_event_count(video_output_dir)),
                    "reference_source": read_reference_source(video_output_dir),
                    "selection": args.selection,
                    "max_events_override": "" if args.max_events is None else str(args.max_events),
                    "fused_quantile_override": "" if args.fused_quantile is None else str(args.fused_quantile),
                    "min_plateau_width_override": "" if args.min_plateau_width is None else str(args.min_plateau_width),
                    "note": "output already exists",
                }
            )
            continue

        cmd = [
            sys.executable,
            str(extractor_path),
            "--video",
            str(video_path),
            "--config",
            str(config_path),
            "--out-dir",
            str(video_output_dir),
            "--selection",
            args.selection,
            "--start-frame",
            str(args.start_frame),
            "--end-frame",
            str(args.end_frame),
        ]
        if args.fused_quantile is not None:
            cmd.extend(["--fused-quantile", str(args.fused_quantile)])
        if args.min_plateau_width is not None:
            cmd.extend(["--min-plateau-width", str(args.min_plateau_width)])
        if args.max_events is not None:
            cmd.extend(["--max-events", str(args.max_events)])

        print(f"[RUN ] {video_path.name} with {config_path.name}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures.append(video_path.name)
            print(f"[FAIL] {video_path.name} (exit={result.returncode})")
            summary_rows.append(
                {
                    "video_id": video_path.stem,
                    "video_name": video_path.name,
                    "status": "failed",
                    "config_path": str(config_path),
                    "output_dir": str(video_output_dir),
                    "event_count": "",
                    "reference_source": "",
                    "selection": args.selection,
                    "max_events_override": "" if args.max_events is None else str(args.max_events),
                    "fused_quantile_override": "" if args.fused_quantile is None else str(args.fused_quantile),
                    "min_plateau_width_override": "" if args.min_plateau_width is None else str(args.min_plateau_width),
                    "note": f"extractor exit={result.returncode}",
                }
            )
            if args.fail_fast:
                break
        else:
            processed += 1
            summary_rows.append(
                {
                    "video_id": video_path.stem,
                    "video_name": video_path.name,
                    "status": "processed",
                    "config_path": str(config_path),
                    "output_dir": str(video_output_dir),
                    "event_count": str(read_event_count(video_output_dir)),
                    "reference_source": read_reference_source(video_output_dir),
                    "selection": args.selection,
                    "max_events_override": "" if args.max_events is None else str(args.max_events),
                    "fused_quantile_override": "" if args.fused_quantile is None else str(args.fused_quantile),
                    "min_plateau_width_override": "" if args.min_plateau_width is None else str(args.min_plateau_width),
                    "note": "",
                }
            )

    summary_path = output_dir / "batch_summary.csv"
    write_batch_summary(summary_path, summary_rows)

    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {len(failures)}")
    print(f"Batch summary: {summary_path}")
    if failures:
        print("Failure list:")
        for name in failures:
            print(f"  - {name}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
