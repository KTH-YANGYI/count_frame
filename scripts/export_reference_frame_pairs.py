from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable

import cv2

import _path  # noqa: F401
from frame_count.config_utils import BUNDLED_REFERENCE_CSV, build_config_from_reference_row, load_reference_rows, read_video_metadata


def numeric_video_sort_key(path: Path) -> tuple:
    return (not path.stem.isdigit(), int(path.stem) if path.stem.isdigit() else path.stem)


def draw_rois(frame, pose_roi, workpiece_roi, label: str):
    overlay = frame.copy()
    cv2.rectangle(overlay, (pose_roi[0], pose_roi[1]), (pose_roi[2], pose_roi[3]), (0, 255, 0), 3)
    cv2.rectangle(overlay, (workpiece_roi[0], workpiece_roi[1]), (workpiece_roi[2], workpiece_roi[3]), (0, 0, 255), 3)
    cv2.putText(overlay, label, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, "pose_roi", (pose_roi[0], max(30, pose_roi[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay, "workpiece_roi", (workpiece_roi[0], max(30, workpiece_roi[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return overlay


def save_frame(
    cap: cv2.VideoCapture,
    out_frames_dir: Path,
    out_overlay_dir: Path,
    video_id: str,
    label: str,
    frame_idx: int,
    time_sec: float,
    cfg: Dict,
) -> Dict[str, str]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from video {video_id}")

    frame_name = f"{video_id}_{label}_f{frame_idx}_t{time_sec:.3f}s.jpg"
    overlay_name = f"{video_id}_{label}_f{frame_idx}_t{time_sec:.3f}s_overlay.jpg"
    frame_path = out_frames_dir / frame_name
    overlay_path = out_overlay_dir / overlay_name

    cv2.imwrite(str(frame_path), frame)
    overlay = draw_rois(
        frame=frame,
        pose_roi=cfg["pose_roi"],
        workpiece_roi=cfg["workpiece_roi"],
        label=f"{video_id} {label}",
    )
    cv2.imwrite(str(overlay_path), overlay)

    return {
        "frame_path": str(frame_path),
        "overlay_path": str(overlay_path),
        "frame_idx": str(frame_idx),
        "time_sec": f"{time_sec:.6f}",
    }


def write_html(manifest_rows: Iterable[Dict[str, str]], out_path: Path) -> None:
    lines = [
        "<!doctype html>",
        '<html><head><meta charset="utf-8"><title>Reference Frame Pairs</title>',
        '<style>body{font-family:Segoe UI,sans-serif;margin:24px;} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:20px;} .card{border:1px solid #ddd;border-radius:10px;padding:12px;} .pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px;} img{max-width:100%;height:auto;border-radius:6px;border:1px solid #ccc;} .meta{font-size:14px;line-height:1.5;margin-bottom:8px;} .title{font-weight:600;margin-bottom:6px;}</style></head><body>',
        "<h1>Reference Frames And +2s Frames</h1>",
        '<div class="grid">',
    ]
    for row in manifest_rows:
        primary_frame_name = Path(row["primary_frame_path"]).name
        primary_overlay_name = Path(row["primary_overlay_path"]).name
        secondary_frame_name = Path(row["secondary_frame_path"]).name
        secondary_overlay_name = Path(row["secondary_overlay_path"]).name
        lines.extend(
            [
                '<div class="card">',
                f'<div class="meta"><strong>Video {row["video_id"]}</strong><br>ref={row["primary_reference_time_sec"]}s / f{row["primary_reference_frame_idx"]}<br>+2s={row["secondary_reference_used_time_sec"]}s / f{row["secondary_reference_frame_idx"]}<br>clamped_by_video_end={row["secondary_reference_clamped_by_video_end"]}<br>pose_roi={row["pose_roi"]}<br>workpiece_roi={row["workpiece_roi"]}</div>',
                '<div class="pair">',
                f'<div><div class="title">Primary Frame</div><img src="frames/{primary_frame_name}" alt="primary frame"><div style="height:8px"></div><img src="frames_with_rois/{primary_overlay_name}" alt="primary overlay"></div>',
                f'<div><div class="title">+2s Frame</div><img src="frames/{secondary_frame_name}" alt="secondary frame"><div style="height:8px"></div><img src="frames_with_rois/{secondary_overlay_name}" alt="secondary overlay"></div>',
                "</div>",
                "</div>",
            ]
        )
    lines.extend(["</div>", "</body></html>"])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    default_reference_csv = str(BUNDLED_REFERENCE_CSV) if BUNDLED_REFERENCE_CSV.exists() else None
    parser = argparse.ArgumentParser(description="Export primary reference frames and +2s frames for all videos.")
    parser.add_argument(
        "--reference-csv",
        default=default_reference_csv,
        help="Path to reference_frames.csv; defaults to data/reference/reference_frames.csv when bundled",
    )
    parser.add_argument("--video-dir", required=True, help="Directory containing input videos")
    parser.add_argument("--out-dir", required=True, help="Output directory for extracted frames")
    args = parser.parse_args()

    if not args.reference_csv:
        raise SystemExit("reference CSV not provided and bundled CSV is missing")

    reference_csv = Path(args.reference_csv)
    if not reference_csv.exists():
        raise SystemExit(f"reference CSV not found: {reference_csv}")
    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    out_frames_dir = out_dir / "frames"
    out_overlay_dir = out_dir / "frames_with_rois"
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    out_overlay_dir.mkdir(parents=True, exist_ok=True)

    rows = load_reference_rows(reference_csv)
    manifest_rows = []

    for video_path in sorted(video_dir.glob("*.mp4"), key=numeric_video_sort_key):
        row = rows.get(video_path.stem)
        if row is None:
            continue
        row = dict(row)
        cfg = build_config_from_reference_row(video_path=video_path, row=row, use_reference_frame=True)
        meta = read_video_metadata(video_path)

        primary_frame_idx = int(cfg["reference_frame_idx"])
        primary_time_sec = primary_frame_idx / meta["fps"] if meta["fps"] > 0 else float(cfg.get("reference_time_sec", 0.0))
        secondary_frame_idx = int(cfg["secondary_reference_frame_idx"])
        secondary_time_sec = float(cfg["secondary_reference_used_time_sec"])

        cap = cv2.VideoCapture(str(video_path))
        primary_info = save_frame(
            cap=cap,
            out_frames_dir=out_frames_dir,
            out_overlay_dir=out_overlay_dir,
            video_id=video_path.stem,
            label="ref",
            frame_idx=primary_frame_idx,
            time_sec=primary_time_sec,
            cfg=cfg,
        )
        secondary_info = save_frame(
            cap=cap,
            out_frames_dir=out_frames_dir,
            out_overlay_dir=out_overlay_dir,
            video_id=video_path.stem,
            label="plus2",
            frame_idx=secondary_frame_idx,
            time_sec=secondary_time_sec,
            cfg=cfg,
        )
        cap.release()

        manifest_rows.append(
            {
                "video_id": video_path.stem,
                "video_path": str(video_path),
                "duration_sec": f"{meta['frame_count'] / meta['fps'] if meta['fps'] > 0 else 0.0:.6f}",
                "fps": f"{meta['fps']:.6f}",
                "primary_reference_time_sec": f"{float(cfg.get('reference_time_sec', primary_time_sec)):.6f}",
                "primary_reference_frame_idx": str(primary_frame_idx),
                "secondary_reference_target_time_sec": f"{float(cfg['secondary_reference_time_sec']):.6f}",
                "secondary_reference_used_time_sec": f"{secondary_time_sec:.6f}",
                "secondary_reference_frame_idx": str(secondary_frame_idx),
                "secondary_reference_clamped_by_video_end": str(bool(cfg.get("secondary_reference_clamped_by_video_end", False))),
                "primary_frame_path": primary_info["frame_path"],
                "primary_overlay_path": primary_info["overlay_path"],
                "secondary_frame_path": secondary_info["frame_path"],
                "secondary_overlay_path": secondary_info["overlay_path"],
                "pose_roi": ",".join(map(str, cfg["pose_roi"])),
                "workpiece_roi": ",".join(map(str, cfg["workpiece_roi"])),
            }
        )

    manifest_path = out_dir / "reference_frame_pairs.csv"
    with open(manifest_path, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "video_id",
            "video_path",
            "duration_sec",
            "fps",
            "primary_reference_time_sec",
            "primary_reference_frame_idx",
            "secondary_reference_target_time_sec",
            "secondary_reference_used_time_sec",
            "secondary_reference_frame_idx",
            "secondary_reference_clamped_by_video_end",
            "primary_frame_path",
            "primary_overlay_path",
            "secondary_frame_path",
            "secondary_overlay_path",
            "pose_roi",
            "workpiece_roi",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    html_path = out_dir / "index.html"
    write_html(manifest_rows, html_path)

    print(f"Exported {len(manifest_rows)} reference-frame pairs into {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
