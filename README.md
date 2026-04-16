# Frame & Count

基于 `CODEX_BUILD.md` 搭建的本地工程，用于从视频中提取工件达到“最大行程 / 到底”位置的关键帧。

## 当前目录结构

```text
frame&count/
├─ extract_max_stroke_keyframes.py
├─ batch_run.py
├─ generate_configs_from_reference_csv.py
├─ config_utils.py
├─ requirements.txt
├─ configs/
│  └─ template.json
└─ outputs/
```

视频素材默认不复制到工程内，直接从现有目录读取：

- 输入视频目录：`C:\Users\18046\Desktop\master\masterthesis\rawdata\rawdata`
- 参考帧 CSV：`C:\Users\18046\Desktop\master\masterthesis\frameextraction\debug_inspect\reference_frames_from_config\reference_frames.csv`

## 安装

```bash
pip install -r requirements.txt
```

## 方式 1：先根据参考帧 CSV 批量生成配置

```bash
python generate_configs_from_reference_csv.py ^
  --reference-csv "C:\Users\18046\Desktop\master\masterthesis\frameextraction\debug_inspect\reference_frames_from_config\reference_frames.csv" ^
  --video-dir "C:\Users\18046\Desktop\master\masterthesis\rawdata\rawdata" ^
  --out-dir configs
```

生成后的配置默认就会直接使用 CSV 里的 `reference_frame_idx` 作为姿态参考帧。

## 方式 2：单视频运行

以 `4.mp4` 为例：

```bash
python extract_max_stroke_keyframes.py ^
  --video "C:\Users\18046\Desktop\master\masterthesis\rawdata\rawdata\4.mp4" ^
  --config configs\4.json ^
  --out-dir outputs\4
```

## 方式 3：批量运行

如果 `configs/<视频名>.json` 已经存在：

```bash
python batch_run.py ^
  --input-dir "C:\Users\18046\Desktop\master\masterthesis\rawdata\rawdata" ^
  --config-dir configs ^
  --output-dir outputs
```

如果缺少配置，但你有参考帧 CSV，也可以让批处理时自动生成：

```bash
python batch_run.py ^
  --input-dir "C:\Users\18046\Desktop\master\masterthesis\rawdata\rawdata" ^
  --config-dir configs ^
  --output-dir outputs ^
  --reference-csv "C:\Users\18046\Desktop\master\masterthesis\frameextraction\debug_inspect\reference_frames_from_config\reference_frames.csv"
```

## 交互式标注 ROI

如果某个视频需要手工微调：

```bash
python extract_max_stroke_keyframes.py ^
  --video "C:\path\to\video.mp4" ^
  --out-dir outputs\debug ^
  --select-roi ^
  --save-config configs\debug.json
```

## 输出内容

每个视频的输出目录下会生成：

- 关键帧图片
- `events.csv`
- `debug_signals.png`
- `run_summary.json`

## 说明

- 生成配置时，会把 CSV 里的 `roi_rect` 作为 `pose_roi`。
- 同时会根据 ROI 的方向和尺寸自动派生一个更窄的 `workpiece_roi`，用于深度信号计算。
- 默认直接使用你提供的参考帧；只有配置里没有参考帧信息时，才会退回自动选择。
