from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_REFERENCE_CSV = PROJECT_ROOT / "data" / "reference" / "reference_frames.csv"


DEFAULT_CONFIG: Dict = {
    "hsv_lo": [5, 80, 50],
    "hsv_hi": [22, 255, 255],
    "min_pixels": 40,
    "smooth_window": 7,
    "depth_quantile": 0.88,
    "fused_quantile": 0.88,
    "min_plateau_width": 5,
    "merge_gap": 4,
    "boundary_margin": 20,
    "min_separation": 35,
    "max_events": 10,
    "similarity_resize_hw": [32, 32],
    "alpha_depth": 0.6,
    "alpha_pose": 0.4,
}

SECONDARY_REFERENCE_OFFSET_SEC = 2.0


def project_relative_path_str(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_roi_rect(text: str) -> Tuple[int, int, int, int]:
    parts = [int(float(p.strip())) for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"roi_rect must have 4 numbers, got: {text}")
    return parts[0], parts[1], parts[2], parts[3]


def xywh_to_xyxy(rect: Sequence[int]) -> List[int]:
    x, y, w, h = map(int, rect)
    return [x, y, x + w, y + h]


def clamp_xyxy(roi: Sequence[int], frame_width: int, frame_height: int) -> List[int]:
    x1, y1, x2, y2 = map(int, roi)
    x1 = max(0, min(frame_width - 1, x1))
    y1 = max(0, min(frame_height - 1, y1))
    x2 = max(x1 + 1, min(frame_width, x2))
    y2 = max(y1 + 1, min(frame_height, y2))
    return [x1, y1, x2, y2]


def read_video_metadata(video_path: Path) -> Dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if frame_count <= 0 or frame_width <= 0 or frame_height <= 0:
        raise RuntimeError(f"Cannot read video metadata: {video_path}")
    return {
        "fps": fps,
        "frame_count": frame_count,
        "frame_width": frame_width,
        "frame_height": frame_height,
    }


def read_video_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame


def infer_orientation(roi_source: str, frame_width: int, frame_height: int) -> str:
    roi_source = roi_source.lower()
    if "landscape" in roi_source:
        return "landscape"
    if "portrait" in roi_source:
        return "portrait"
    return "landscape" if frame_width >= frame_height else "portrait"


def copper_mask_bgr(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([5, 60, 40], np.uint8), np.array([28, 255, 255], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def detect_seeded_workpiece_component(frame: np.ndarray, pose_seed_xyxy: Sequence[int]) -> Dict[str, float]:
    frame_height, frame_width = frame.shape[:2]
    seed_x1, seed_y1, seed_x2, seed_y2 = map(int, pose_seed_xyxy)
    seed_w = seed_x2 - seed_x1
    seed_h = seed_y2 - seed_y1
    seed_cx = (seed_x1 + seed_x2) / 2.0

    def collect_candidates(search_xyxy: Sequence[int], overlap_req: float, center_ratio: float):
        sx1, sy1, sx2, sy2 = map(int, search_xyxy)
        crop = frame[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            return []
        mask = copper_mask_bgr(crop)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for i in range(1, num):
            bx, by, bw, bh, area = stats[i]
            if area < 50:
                continue
            aspect = bh / max(bw, 1)
            if aspect < 1.8:
                continue
            gx1, gy1 = sx1 + int(bx), sy1 + int(by)
            gx2, gy2 = gx1 + int(bw), gy1 + int(bh)
            cx = float(centroids[i][0] + sx1)
            cy = float(centroids[i][1] + sy1)
            overlap_y = max(0, min(gy2, seed_y2) - max(gy1, seed_y1)) / max(bh, 1)
            if overlap_y < overlap_req:
                continue
            if abs(cx - seed_cx) > max(seed_w * center_ratio, 42):
                continue
            center_score = 1.0 - min(abs(cx - seed_cx) / max(seed_w * center_ratio, 1), 1.0)
            score = min(aspect / 3.0, 2.0) * 2.0 + overlap_y * 1.8 + center_score * 2.0 + min(area / 1800.0, 2.0) * 0.8
            candidates.append(
                {
                    "score": float(score),
                    "cx": float(cx),
                    "cy": float(cy),
                    "x1": int(gx1),
                    "y1": int(gy1),
                    "x2": int(gx2),
                    "y2": int(gy2),
                    "bw": int(bw),
                    "bh": int(bh),
                    "area": int(area),
                    "aspect": float(aspect),
                    "overlap_y": float(overlap_y),
                    "search_xyxy": [int(sx1), int(sy1), int(sx2), int(sy2)],
                }
            )
        return sorted(candidates, key=lambda item: item["score"], reverse=True)

    expanded_search_xyxy = clamp_xyxy(
        [
            seed_x1 - int(seed_w * 0.18),
            seed_y1 - int(seed_h * 1.25),
            seed_x2 + int(seed_w * 0.18),
            seed_y2 + int(seed_h * 0.20),
        ],
        frame_width=frame_width,
        frame_height=frame_height,
    )
    candidates = collect_candidates(expanded_search_xyxy, overlap_req=0.18, center_ratio=0.40)
    detection_mode = "expanded_search"
    if not candidates:
        candidates = collect_candidates(pose_seed_xyxy, overlap_req=0.0, center_ratio=0.55)
        detection_mode = "seed_only"
    if not candidates:
        return {
            "score": 0.0,
            "cx": float(seed_cx),
            "cy": float((seed_y1 + seed_y2) / 2.0),
            "x1": int(seed_x1),
            "y1": int(seed_y1),
            "x2": int(seed_x2),
            "y2": int(seed_y2),
            "bw": int(seed_w),
            "bh": int(seed_h),
            "area": 0,
            "aspect": 0.0,
            "overlap_y": 0.0,
            "search_xyxy": list(map(int, pose_seed_xyxy)),
            "mode": "seed_center_fallback",
        }

    best = dict(candidates[0])
    best["mode"] = detection_mode
    return best


def derive_workpiece_roi(pose_rect_xywh: Sequence[int], orientation: str, frame_width: int, frame_height: int) -> List[int]:
    x, y, w, h = map(int, pose_rect_xywh)
    if orientation == "landscape":
        roi = [
            x + int(round(w * 0.23)),
            y - int(round(h * 1.75)),
            x + int(round(w * 0.90)),
            y + int(round(h * 1.40)),
        ]
    else:
        roi = [
            x + int(round(w * 0.32)),
            y + int(round(h * 0.48)),
            x + int(round(w * 0.68)),
            y + int(round(h * 1.02)),
        ]

    roi = clamp_xyxy(roi, frame_width=frame_width, frame_height=frame_height)

    # Keep the derived ROI narrow and tall enough for the centroid-based depth signal.
    if roi[2] - roi[0] < 40:
        cx = (roi[0] + roi[2]) // 2
        roi[0] = max(0, cx - 20)
        roi[2] = min(frame_width, cx + 20)
    if roi[3] - roi[1] < 80:
        roi[3] = min(frame_height, roi[1] + 80)
    return clamp_xyxy(roi, frame_width=frame_width, frame_height=frame_height)


def derive_rois_from_reference_frame(
    frame: np.ndarray,
    pose_seed_xyxy: Sequence[int],
    orientation: str,
) -> Dict[str, object]:
    frame_height, frame_width = frame.shape[:2]
    seed_x1, seed_y1, seed_x2, seed_y2 = map(int, pose_seed_xyxy)
    seed_w = seed_x2 - seed_x1
    seed_h = seed_y2 - seed_y1
    component = detect_seeded_workpiece_component(frame, pose_seed_xyxy)
    cx = float(component["cx"])

    if orientation == "landscape":
        workpiece_half_w = max(34, int(seed_w * 0.17))
        workpiece_roi = clamp_xyxy(
            [
                int(round(cx - workpiece_half_w)),
                seed_y1 - int(seed_h * 1.05),
                int(round(cx + workpiece_half_w)),
                seed_y2 + int(seed_h * 0.10),
            ],
            frame_width=frame_width,
            frame_height=frame_height,
        )
        pose_half_w = max(90, int(seed_w * 0.52))
        pose_half_h = max(90, int(seed_h * 0.52))
        pose_cy = seed_y1 + int(seed_h * 0.48)
    else:
        workpiece_half_w = max(30, int(seed_w * 0.16))
        workpiece_roi = clamp_xyxy(
            [
                int(round(cx - workpiece_half_w)),
                seed_y1 - int(seed_h * 0.55),
                int(round(cx + workpiece_half_w)),
                seed_y2 + int(seed_h * 0.12),
            ],
            frame_width=frame_width,
            frame_height=frame_height,
        )
        pose_half_w = max(86, int(seed_w * 0.38))
        pose_half_h = max(92, int(min(seed_h * 0.32, 115)))
        pose_cy = seed_y1 + int(seed_h * 0.44)

    pose_roi = clamp_xyxy(
        [
            int(round(cx - pose_half_w)),
            int(round(pose_cy - pose_half_h)),
            int(round(cx + pose_half_w)),
            int(round(pose_cy + pose_half_h)),
        ],
        frame_width=frame_width,
        frame_height=frame_height,
    )

    return {
        "pose_seed_xyxy": list(map(int, pose_seed_xyxy)),
        "component": component,
        "workpiece_roi": workpiece_roi,
        "pose_roi": pose_roi,
    }


def load_reference_rows(csv_path: Path) -> Dict[str, Dict[str, str]]:
    csv_path = csv_path.resolve()
    rows: Dict[str, Dict[str, str]] = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row["source_csv"] = project_relative_path_str(csv_path)
            video_id = str(row["video_id"]).strip()
            if video_id:
                rows[video_id] = row
    return rows


def parse_reference_time_sec(row: Dict[str, str]) -> float:
    for key in ("reference_time_used_sec", "reference_time_requested_sec"):
        value = str(row.get(key, "")).strip()
        if value:
            return float(value)
    return 0.0


def frame_idx_from_time_sec(time_sec: float, fps: float, frame_count: int) -> int:
    if fps <= 0 or frame_count <= 0:
        return 0
    return int(max(0, min(frame_count - 1, round(time_sec * fps))))


def compute_secondary_reference(
    reference_time_sec: float,
    fps: float,
    frame_count: int,
    offset_sec: float = SECONDARY_REFERENCE_OFFSET_SEC,
) -> Dict[str, float]:
    target_time_sec = reference_time_sec + offset_sec
    raw_frame_idx = int(round(target_time_sec * fps)) if fps > 0 else 0
    frame_idx = frame_idx_from_time_sec(target_time_sec, fps=fps, frame_count=frame_count)
    used_time_sec = frame_idx / fps if fps > 0 else target_time_sec
    return {
        "offset_sec": float(offset_sec),
        "target_time_sec": float(target_time_sec),
        "frame_idx": int(frame_idx),
        "used_time_sec": float(used_time_sec),
        "clamped_by_video_end": bool(frame_count > 0 and raw_frame_idx > frame_count - 1),
    }


def build_config_from_reference_row(
    video_path: Path,
    row: Dict[str, str],
    use_reference_frame: bool = False,
    base_config: Optional[Dict] = None,
) -> Dict:
    meta = read_video_metadata(video_path)
    primary_reference_time_sec = parse_reference_time_sec(row)
    secondary_reference = compute_secondary_reference(
        reference_time_sec=primary_reference_time_sec,
        fps=meta["fps"],
        frame_count=meta["frame_count"],
    )
    pose_rect_xywh = parse_roi_rect(row["roi_rect"])
    pose_seed_xyxy = clamp_xyxy(
        xywh_to_xyxy(pose_rect_xywh),
        frame_width=meta["frame_width"],
        frame_height=meta["frame_height"],
    )
    orientation = infer_orientation(
        row.get("roi_source", ""),
        frame_width=meta["frame_width"],
        frame_height=meta["frame_height"],
    )
    reference_frame = read_video_frame(video_path, int(float(row["reference_frame_idx"])))
    derived_rois = derive_rois_from_reference_frame(
        frame=reference_frame,
        pose_seed_xyxy=pose_seed_xyxy,
        orientation=orientation,
    )
    pose_roi = derived_rois["pose_roi"]
    workpiece_roi = derived_rois["workpiece_roi"]

    cfg = dict(DEFAULT_CONFIG)
    if base_config:
        cfg.update(base_config)
    cfg.update(
        {
            "workpiece_roi": workpiece_roi,
            "reference_time_sec": float(primary_reference_time_sec),
            "reference_frame_idx": int(float(row["reference_frame_idx"])),
            "secondary_reference_offset_sec": float(secondary_reference["offset_sec"]),
            "secondary_reference_time_sec": float(secondary_reference["target_time_sec"]),
            "secondary_reference_used_time_sec": float(secondary_reference["used_time_sec"]),
            "secondary_reference_frame_idx": int(secondary_reference["frame_idx"]),
            "secondary_reference_clamped_by_video_end": bool(secondary_reference["clamped_by_video_end"]),
            "pose_roi": pose_roi,
            "reference_frame_hint": int(float(row["reference_frame_idx"])),
            "_meta": {
                "source_csv": str(row.get("source_csv", "")),
                "video_id": str(row.get("video_id", "")),
                "roi_rect_xywh": list(map(int, pose_rect_xywh)),
                "pose_seed_xyxy": list(map(int, pose_seed_xyxy)),
                "roi_source": row.get("roi_source", ""),
                "orientation": orientation,
                "reference_time_requested_sec": row.get("reference_time_requested_sec", ""),
                "reference_time_used_sec": row.get("reference_time_used_sec", ""),
                "reference_frame_idx": int(float(row["reference_frame_idx"])),
                "reference_roi_component": {
                    "mode": derived_rois["component"]["mode"],
                    "score": float(derived_rois["component"]["score"]),
                    "cx": float(derived_rois["component"]["cx"]),
                    "cy": float(derived_rois["component"]["cy"]),
                    "x1": int(derived_rois["component"]["x1"]),
                    "y1": int(derived_rois["component"]["y1"]),
                    "x2": int(derived_rois["component"]["x2"]),
                    "y2": int(derived_rois["component"]["y2"]),
                    "bw": int(derived_rois["component"]["bw"]),
                    "bh": int(derived_rois["component"]["bh"]),
                    "area": int(derived_rois["component"]["area"]),
                    "aspect": float(derived_rois["component"]["aspect"]),
                    "overlap_y": float(derived_rois["component"]["overlap_y"]),
                    "search_xyxy": list(map(int, derived_rois["component"]["search_xyxy"])),
                },
                "secondary_reference_time_sec": float(secondary_reference["target_time_sec"]),
                "secondary_reference_used_time_sec": float(secondary_reference["used_time_sec"]),
                "secondary_reference_frame_idx": int(secondary_reference["frame_idx"]),
                "secondary_reference_offset_sec": float(secondary_reference["offset_sec"]),
                "secondary_reference_clamped_by_video_end": bool(secondary_reference["clamped_by_video_end"]),
            },
        }
    )
    if use_reference_frame:
        cfg["reference_frame_idx"] = int(float(row["reference_frame_idx"]))
    return cfg


def save_config(path: Path, cfg: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
