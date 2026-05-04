# Frame & Count

Tools for extracting two types of keyframes from machine videos:

- `reference`: frames that match the primary reference pose.
- `reference_plus_2s`: frames that match the state two seconds after the primary reference.

The repository contains the reference metadata needed to generate per-video configs:

- `data/reference/reference_frames.csv`

Raw videos and generated outputs are intentionally not stored in Git. Pass the video directory when running the scripts.

## Repository Layout

```text
frame_count/                 Shared Python helpers used by the command-line tools
scripts/                     Runnable extraction, export, and visualization commands
configs/                     Per-video JSON configs plus a template
data/reference/              Bundled reference-frame metadata
docs/                        Architecture and workflow notes
outputs/                     Generated results, ignored by Git
```

The main extraction logic lives in `scripts/extract_max_stroke_keyframes.py`. Config generation and reference-frame helpers use `frame_count/config_utils.py`.

## Setup

```bash
python -m pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Typical Workflow

### 1. Generate configs from the bundled reference CSV

```bash
python scripts/generate_configs_from_reference_csv.py ^
  --video-dir "D:\path\to\videos" ^
  --out-dir configs\generated
```

To use a different reference CSV:

```bash
python scripts/generate_configs_from_reference_csv.py ^
  --reference-csv "D:\path\to\reference_frames.csv" ^
  --video-dir "D:\path\to\videos" ^
  --out-dir configs\generated
```

### 2. Run one video

```bash
python scripts/extract_max_stroke_keyframes.py ^
  --video "D:\path\to\videos\13.mp4" ^
  --config configs\13.json ^
  --out-dir outputs\13
```

### 3. Run a batch

```bash
python scripts/batch_run.py ^
  --input-dir "D:\path\to\videos" ^
  --config-dir configs ^
  --output-dir outputs\batch_demo
```

If a config is missing and `data/reference/reference_frames.csv` contains a matching `video_id`, the batch runner can auto-generate the missing config into `configs/generated/`.

## iPhone / HDR Visual Keyframe Extraction

`scripts/extract_phone_hdr_visual_keyframes.py` is the visual-only extractor for the four iPhone videos in `手机(1)`. It reuses this project's visual keyframe logic and the manually tuned ROIs; it does not use the audio/event-selection logic from `count`.

By default, selected frames are exported with OpenCV:

```bash
conda run -n frame-count python scripts/extract_phone_hdr_visual_keyframes.py \
  --video-dir "/Users/yangyi/Desktop/masterthesis/手机(1)" \
  --output-root "/Users/yangyi/Desktop/masterthesis/手机(1)/keyframe_extraction_20260504/visual_fusion_opencv"
```

Use AVFoundation only when explicitly requested, for example when you want macOS/CoreImage HDR tone mapping for iPhone HDR `.MOV` files:

```bash
conda run -n frame-count python scripts/extract_phone_hdr_visual_keyframes.py \
  --video-dir "/Users/yangyi/Desktop/masterthesis/手机(1)" \
  --output-root "/Users/yangyi/Desktop/masterthesis/手机(1)/keyframe_extraction_20260504/visual_fusion_avfoundation" \
  --frame-export-mode avfoundation
```

AVFoundation export is imported only in `--frame-export-mode avfoundation` mode. The default OpenCV mode writes `keyframes_opencv_jpg/event_0001.jpg` style files; AVFoundation mode writes `keyframes_avfoundation_srgb/event_0001.png` style files. Each segment also writes an image manifest (`opencv_manifest.csv` or `avfoundation_srgb_manifest.csv`) plus `events.csv`, so the exported image can be mapped back to the source video, event rank, frame, and timestamp.

## Review And Export Helpers

Export reference and `reference + 2s` frame pairs:

```bash
python scripts/export_reference_frame_pairs.py ^
  --video-dir "D:\path\to\videos" ^
  --out-dir outputs\reference_pairs
```

Export only `reference + 2s` frames:

```bash
python scripts/export_secondary_reference_frames.py ^
  --video-dir "D:\path\to\videos" ^
  --out-dir outputs\secondary_reference_frames
```

Preview configured workpiece masks:

```bash
python scripts/export_workpiece_masks.py ^
  --video-dir "D:\path\to\videos" ^
  --config-dir configs ^
  --out-dir outputs\workpiece_masks
```

Export 512x512 crops from reference frames:

```bash
python scripts/export_reference_crops_512.py ^
  --video-dir "D:\path\to\videos" ^
  --out-dir outputs\reference_crops_512
```

Export 512x512 crops for detected events:

```bash
python scripts/export_event_frames_crops_512.py ^
  --batch-output-dir outputs\batch_demo ^
  --video-dir "D:\path\to\videos" ^
  --config-dir configs ^
  --out-dir outputs\event_crops_512
```

Merge per-video event crop manifests:

```bash
python scripts/merge_event_manifests.py ^
  --root-dir outputs\event_crops_512
```

## Outputs

Each extracted video output directory usually contains:

- keyframe images
- keyframe image manifest
- `events.csv`
- `debug_signals.png`
- `run_summary.json`

Batch runs also create `batch_summary.csv` in the batch output directory.
