# Frame & Count

本项目用于从视频中提取两类关键帧：

- `reference`
- `reference + 2s`

仓库内已包含参考帧表：

- `data/reference/reference_frames.csv`

原始视频不包含在仓库中，运行时需要你自己指定视频文件或视频目录。

## 安装

```bash
pip install -r requirements.txt
```

## 1. 生成配置

如果你要针对一批新视频生成配置：

```bash
python generate_configs_from_reference_csv.py ^
  --video-dir "D:\\path\\to\\videos" ^
  --out-dir configs\\generated
```

如果你想显式指定别的参考帧 CSV：

```bash
python generate_configs_from_reference_csv.py ^
  --reference-csv "D:\\path\\to\\reference_frames.csv" ^
  --video-dir "D:\\path\\to\\videos" ^
  --out-dir configs\\generated
```

## 2. 运行单个视频

```bash
python extract_max_stroke_keyframes.py ^
  --video "D:\\path\\to\\videos\\13.mp4" ^
  --config configs\\13.json ^
  --out-dir outputs\\13
```

## 3. 批量运行

```bash
python batch_run.py ^
  --input-dir "D:\\path\\to\\videos" ^
  --config-dir configs ^
  --output-dir outputs\\batch_demo
```

如果 `configs/` 中缺少某些视频的配置，批处理默认会使用仓库内置的 `data/reference/reference_frames.csv` 自动补生成。

## 4. 导出参考帧对照

导出 `reference` 与 `reference + 2s` 对照图：

```bash
python export_reference_frame_pairs.py ^
  --video-dir "D:\\path\\to\\videos" ^
  --out-dir outputs\\reference_pairs
```

只导出 `reference + 2s` 参考帧：

```bash
python export_secondary_reference_frames.py ^
  --video-dir "D:\\path\\to\\videos" ^
  --out-dir outputs\\secondary_reference_frames
```

导出工件 HSV / mask 预览：

```bash
python export_workpiece_masks.py ^
  --video-dir "D:\\path\\to\\videos" ^
  --config-dir configs ^
  --out-dir outputs\\workpiece_masks
```

## 输出内容

每个视频的输出目录下通常包含：

- 关键帧图片
- `events.csv`
- `debug_signals.png`
- `run_summary.json`
