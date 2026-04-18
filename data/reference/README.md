# Reference CSV

这个目录保存仓库内置的参考帧表：

- [reference_frames.csv](/C:/Users/18046/Desktop/master/masterthesis/frame&count/data/reference/reference_frames.csv:1)

## 这份 CSV 用来做什么

代码真正依赖的是这些字段：

- `video_id`
- `reference_time_requested_sec`
- `reference_time_used_sec`
- `reference_frame_idx`
- `roi_rect`
- `roi_source`

它们用于：

- 为每个视频定位主参考帧
- 派生 `reference + 2s` 的次参考帧
- 初始化 `pose_roi`
- 进一步推导 `workpiece_roi`

## 为什么还保留图像路径列

CSV 里还有这些字段：

- `full_frame`
- `full_frame_with_roi`
- `crop`

这些列来自原始参考帧检查流程，主要用于人工回看和追溯，不是当前提取代码的必需输入。

本仓库没有把对应的大批量参考图片一起纳入版本控制，以避免仓库体积膨胀；因此这些列在 GitHub 展示版里可以视为说明性元数据。
