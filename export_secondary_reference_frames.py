from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import cv2

from config_utils import build_config_from_reference_row, load_reference_rows, read_video_metadata


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


def save_secondary_frame(
    video_path: Path,
    out_frames_dir: Path,
    out_overlay_dir: Path,
    cfg: Dict,
) -> Dict[str, str]:
    meta = read_video_metadata(video_path)
    secondary_frame_idx = int(cfg["secondary_reference_frame_idx"])
    secondary_time_sec = float(cfg["secondary_reference_used_time_sec"])

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, secondary_frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {secondary_frame_idx} from {video_path}")

    frame_name = f"{video_path.stem}_plus2_f{secondary_frame_idx}_t{secondary_time_sec:.3f}s.jpg"
    overlay_name = f"{video_path.stem}_plus2_f{secondary_frame_idx}_t{secondary_time_sec:.3f}s_overlay.jpg"
    frame_path = out_frames_dir / frame_name
    overlay_path = out_overlay_dir / overlay_name
    cv2.imwrite(str(frame_path), frame)
    overlay = draw_rois(
        frame=frame,
        pose_roi=cfg["pose_roi"],
        workpiece_roi=cfg["workpiece_roi"],
        label=f"{video_path.stem} +2s",
    )
    cv2.imwrite(str(overlay_path), overlay)

    return {
        "video_id": video_path.stem,
        "video_path": str(video_path),
        "duration_sec": f"{meta['frame_count'] / meta['fps'] if meta['fps'] > 0 else 0.0:.6f}",
        "fps": f"{meta['fps']:.6f}",
        "primary_reference_time_sec": f"{float(cfg.get('reference_time_sec', 0.0)):.6f}",
        "primary_reference_frame_idx": str(int(cfg["reference_frame_idx"])),
        "secondary_reference_target_time_sec": f"{float(cfg['secondary_reference_time_sec']):.6f}",
        "secondary_reference_used_time_sec": f"{float(cfg['secondary_reference_used_time_sec']):.6f}",
        "secondary_reference_frame_idx": str(secondary_frame_idx),
        "secondary_reference_clamped_by_video_end": str(bool(cfg.get("secondary_reference_clamped_by_video_end", False))),
        "frame_path": str(frame_path),
        "overlay_path": str(overlay_path),
        "pose_roi": ",".join(map(str, cfg["pose_roi"])),
        "workpiece_roi": ",".join(map(str, cfg["workpiece_roi"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the secondary reference frame (reference + 2s, capped at 8s) for every video.")
    parser.add_argument("--reference-csv", required=True, help="Path to reference_frames.csv")
    parser.add_argument("--video-dir", required=True, help="Directory containing input videos")
    parser.add_argument("--out-dir", required=True, help="Output directory for extracted frames")
    args = parser.parse_args()

    reference_csv = Path(args.reference_csv)
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
        row["source_csv"] = str(reference_csv)
        cfg = build_config_from_reference_row(video_path=video_path, row=row, use_reference_frame=True)
        manifest_rows.append(
            save_secondary_frame(
                video_path=video_path,
                out_frames_dir=out_frames_dir,
                out_overlay_dir=out_overlay_dir,
                cfg=cfg,
            )
        )

    manifest_path = out_dir / "secondary_reference_frames.csv"
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
            "frame_path",
            "overlay_path",
            "pose_roi",
            "workpiece_roi",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    html_lines = [
        "<!doctype html>",
        '<html><head><meta charset="utf-8"><title>Secondary Reference Frames</title>',
        '<style>body{font-family:Segoe UI,sans-serif;margin:24px;} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:20px;} .card{border:1px solid #ddd;border-radius:10px;padding:12px;} img{max-width:100%;height:auto;border-radius:6px;border:1px solid #ccc;} .meta{font-size:14px;line-height:1.5;margin-bottom:8px;}</style></head><body>',
        "<h1>Secondary Reference Frames (+2s)</h1>",
        '<div class="grid">',
    ]
    for row in manifest_rows:
        frame_name = Path(row["frame_path"]).name
        overlay_name = Path(row["overlay_path"]).name
        html_lines.extend(
            [
                '<div class="card">',
                f'<div class="meta"><strong>Video {row["video_id"]}</strong><br>ref={row["primary_reference_time_sec"]}s / f{row["primary_reference_frame_idx"]}<br>+2s target={row["secondary_reference_target_time_sec"]}s used={row["secondary_reference_used_time_sec"]}s / f{row["secondary_reference_frame_idx"]}<br>clamped_by_video_end={row["secondary_reference_clamped_by_video_end"]}<br>pose_roi={row["pose_roi"]}<br>workpiece_roi={row["workpiece_roi"]}</div>',
                f'<div><img src="frames/{frame_name}" alt="frame"></div>',
                f'<div style="margin-top:8px;"><img src="frames_with_rois/{overlay_name}" alt="overlay"></div>',
                "</div>",
            ]
        )
    html_lines.extend(["</div>", "</body></html>"])
    html_path = out_dir / "index.html"
    html_path.write_text("\n".join(html_lines), encoding="utf-8")

    print(f"Exported {len(manifest_rows)} secondary reference frames into {out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"HTML: {html_path}")


if __name__ == "__main__":
    main()
