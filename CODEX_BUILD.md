# 工件“到底最大行程”关键帧提取工程说明

## 1. 目标

构建一个**本地可运行**的小工程，从视频中稳定提取工件到达“最大行程 / 到底”位置的关键帧，要求：

1. 不要提取一堆相邻重复帧。
2. 尽量跟踪**工件本体**，不要被操作者手部运动、机身震动、光照波动带偏。
3. 允许视频里出现多次相似动作，但最终只输出 `K` 个最可信的“到底帧”。
4. 支持两种模式：
   - `score`：按“到底程度 + 姿态一致性”得分最高的事件输出。
   - `earliest`：按时间顺序输出最早出现的 `K` 个事件。

---

## 2. 我最终建议的方法

我试过三类思路：

### 方法 A：全帧光流 / 帧差
优点：不需要显式建模。

缺点：
- 手套、手臂、机器其它零件一动，响应也很大。
- 只能知道“发生了运动”，不容易知道“是不是工件真正到底”。
- 容易提取出很多错误峰值。

结论：**不推荐作为最终方案。**

### 方法 B：只看机器某个运动零件的姿态
优点：容易找到重复节拍。

缺点：
- 机器某个摆臂到位，不一定等价于**工件真正到底**。
- 某些视频里可能出现多个相似姿态，但工件深度不完全一样。

结论：**可作为辅助信号，不建议单独用。**

### 方法 C：工件位置信号 + 机器姿态相似度融合（最终采用）
核心思想：

1. 在一个很窄的 `workpiece_roi` 里，只保留工件本体。
2. 用 HSV 颜色分割把铜色工件提出来。
3. 建一个一维“深度信号” `y(t)`：
   - 对每一帧，计算工件像素纵坐标的质心。
   - 工件越往下，`y(t)` 越大。
4. 先根据 `y(t)` 粗找一批“疑似到底平台”。
5. 选一个最稳定的平台峰值，取该帧作为参考帧。
6. 在更大的 `pose_roi` 中，把每一帧和参考帧做归一化相似度比较，得到 `s(t)`。
7. 融合：

   \[
   f(t) = \alpha \cdot \widetilde{y}(t) + (1-\alpha) \cdot \widetilde{s}(t)
   \]

8. 对 `f(t)` 做平台检测，只保留宽度足够、彼此间隔足够远的事件。
9. 每个平台只取**1 帧**，默认取峰值帧。

这个方法的优点是：
- `y(t)` 保证是在看工件本体；
- `s(t)` 保证姿态确实像“到底时的机器状态”；
- 两者融合后，误检明显少于单独用任何一个信号。

---

## 3. 工程内的关键公式

### 3.1 工件深度信号
设第 `t` 帧里，工件分割掩膜为 `M_t`，其中像素纵坐标为 `v`，则：

\[
 y_t = \frac{1}{|M_t|} \sum_{(u,v) \in M_t} v
\]

解释：
- `y_t` 越大，说明工件整体越靠下；
- 如果你的视频是反方向运动，只需要把“取最大值”改成“取最小值”。

### 3.2 参考姿态相似度
把 `pose_roi` 灰度化、缩放、零均值归一化后，和参考帧做点积相似度：

\[
 s_t = \frac{1}{N} \sum_{i=1}^{N} \hat{I}_{t,i} \cdot \hat{I}_{ref,i}
\]

这里本质上就是归一化相关思想，和模板匹配里常用的 NCC 思路一致。

### 3.3 融合得分
\[
 f_t = \alpha \cdot \widetilde{y_t} + \beta \cdot \widetilde{s_t}
\]

默认取：
- `alpha_depth = 0.6`
- `alpha_pose = 0.4`

原因：我在你的视频上试下来，**工件本体深度**应该比**机器姿态相似度**权重大一点。

---

## 4. 本地工程文件

### 已生成文件

- `extract_max_stroke_keyframes.py`
- `configs/video1_demo.json`
- `configs/video2_demo.json`
- `results_video1/`
- `results_video2/`
- `requirements.txt`

### 目录建议

```text
project/
├─ extract_max_stroke_keyframes.py
├─ requirements.txt
├─ configs/
│  ├─ video1_demo.json
│  └─ video2_demo.json
├─ inputs/
│  ├─ 4.mp4
│  └─ 3f576401-2e87-42d7-9e64-acf5f2ccf296.mp4
└─ outputs/
   ├─ video1/
   └─ video2/
```

---

## 5. 如何运行

### 5.1 安装依赖

```bash
pip install -r requirements.txt
```

### 5.2 跑第 1 段视频

```bash
python extract_max_stroke_keyframes.py \
  --video inputs/4.mp4 \
  --config configs/video1_demo.json \
  --out-dir outputs/video1
```

### 5.3 跑第 2 段视频

```bash
python extract_max_stroke_keyframes.py \
  --video inputs/3f576401-2e87-42d7-9e64-acf5f2ccf296.mp4 \
  --config configs/video2_demo.json \
  --out-dir outputs/video2
```

### 5.4 如果你确认“只要最早出现的两个到底事件”

```bash
python extract_max_stroke_keyframes.py \
  --video inputs/xxx.mp4 \
  --config configs/xxx.json \
  --out-dir outputs/xxx \
  --selection earliest \
  --max-events 2
```

### 5.5 如果你只想截取某一个时间段内的到底帧

```bash
python extract_max_stroke_keyframes.py \
  --video inputs/xxx.mp4 \
  --config configs/xxx.json \
  --out-dir outputs/xxx \
  --start-frame 0 \
  --end-frame 260 \
  --max-events 2
```

---

## 6. 参数怎么调

### `workpiece_roi`
只框住工件主运动路径，越窄越好，但要保证工件在整个行程中都在里面。

### `pose_roi`
框住“到底姿态”有辨识度的机械结构，通常比 `workpiece_roi` 大。

### `hsv_lo / hsv_hi`
如果工件不是铜色，或者现场光照变化很大，需要重新调 HSV 阈值。

### `fused_quantile`
越大越严格，提取出来的事件越少。

### `min_plateau_width`
越大越能抑制单帧噪声峰值。

### `min_separation`
同一段到底平台里不会重复出很多帧，建议按帧率设置：
- 25 fps 视频：`30 ~ 40`
- 30 fps 视频：`35 ~ 50`

---

## 7. 我在你这两段视频上的实验结果

### 视频 1：`4.mp4`
工程默认参数输出的两个关键帧：

1. `frame = 84`, `time = 2.800 s`
2. `frame = 709`, `time = 23.633 s`

说明：
- 这两个平台都很稳定，姿态一致度也高；
- 在这个视频里我还观察到一个较弱候选，大约在 `frame ≈ 202`；
- 如果你的业务定义里“只有两个最大行程”，那当前输出是合理的。

### 视频 2：`3f576401-2e87-42d7-9e64-acf5f2ccf296.mp4`
工程默认参数在 `max_events = 2` 时输出：

1. `frame = 286`, `time = 11.440 s`
2. `frame = 385`, `time = 15.400 s`

说明：
- 第 2 段视频里，信号上能看到**不止两个**很像“到底平台”的段；
- 如果你的业务上就是只要两个，那么当前脚本会按得分最高的两个返回；
- 如果你真正想要“最早出现的两个”，改成 `--selection earliest`；
- 如果你真正想要“某一段工序中的两个”，请加 `--start-frame / --end-frame` 限定时间段。

---

## 8. 为什么这个方案比较稳

1. **只看工件本体**，不会被整机大范围运动直接带偏。
2. **加了姿态相似度**，所以不是“工件颜色一出现就算到底”。
3. **平台合并 + 最小间隔约束**，不会输出一大串相邻重复帧。
4. **支持强制只取 2 个事件**，适合你的当前需求。

---

## 9. 我建议 Codex 继续怎么扩展

### 第一优先级：交互式 ROI 标注
给脚本增加：
- 第一帧可视化；
- 鼠标框 `workpiece_roi`；
- 鼠标框 `pose_roi`；
- 自动保存成 JSON 配置。

### 第二优先级：批处理模式
支持一个目录下所有视频批量跑：

```bash
python batch_run.py --input-dir inputs --config-dir configs --output-dir outputs
```

### 第三优先级：可视化报告
每段视频自动生成：
- 原视频关键帧缩略图；
- 深度曲线；
- 融合得分曲线；
- 最终 `events.csv`。

### 第四优先级：完全不依赖颜色
如果后面工件不是铜色，建议加一个备选方案：
- 先人工框出工件模板；
- 用局部模板跟踪 / 小目标跟踪替代 HSV 颜色分割；
- 再复用同样的平台检测逻辑。

---

## 10. 参考思路

和姿态相似度相关的部分，属于归一化相关 / 模板匹配思想。实现时可以参考：

- J. P. Lewis, **Fast Normalized Cross-Correlation**, 1995.
- OpenCV 官方 `matchTemplate` 文档。

---

## 11. 一句话结论

**最实用、最稳的方案不是“全帧找运动峰值”，而是“工件 ROI 的一维深度信号 + 机器到底姿态相似度 + 平台级事件合并”。**

对于你这类视频，这个方案比单纯帧差、单纯光流、单纯机器姿态判断都更靠谱。
