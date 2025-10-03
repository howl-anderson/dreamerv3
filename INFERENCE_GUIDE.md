# DreamerV3 推理和可视化指南

## 🎯 目标
运行训练好的 DreamerV3 模型进行推理，并录制视频展示智能体的行为。

## 📋 方法对比

### 方法 1: 使用 `eval_only` 脚本 (官方)
```bash
# 使用训练好的 checkpoint 进行评估
python dreamerv3/main.py \
  --configs atari \
  --task atari_pong \
  --script eval_only \
  --run.from_checkpoint ~/logdir/dreamer/checkpoint.pkl \
  --run.steps 10000 \
  --run.envs 1
```

**特点:**
- ✅ 官方支持，稳定
- ✅ 自动记录图像序列到 logdir
- ❌ 不直接生成视频文件
- ❌ 需要手动转换图像为视频

### 方法 2: 使用自定义推理脚本 (推荐)
```bash
# 使用我创建的 inference_video.py
python inference_video.py \
  --checkpoint ~/logdir/dreamer/checkpoint.pkl \
  --task atari_pong \
  --episodes 5 \
  --output ./videos/
```

**特点:**
- ✅ 直接生成 MP4 视频
- ✅ 可自定义录制参数
- ✅ 清晰的进度显示
- ✅ 每个 episode 单独保存

### 方法 3: 查看训练时生成的视频 (最简单)
DreamerV3 训练时会自动生成 "open loop" 预测视频！

```bash
# 1. 训练时已经生成了视频，查看 logdir
ls ~/logdir/dreamer/{timestamp}/

# 2. 使用 Scope 查看器
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
# 然后打开浏览器访问 http://localhost:8000
```

## 🎬 视频生成详解

### 训练时自动生成的视频

在 `agent.py` 的 `report()` 方法中 (line 289-307)，会生成：

```python
# Video preds
for key in self.dec.imgkeys:
    # 生成预测视频 (真实 | 预测 | 误差)
    video = jnp.concatenate([true, pred, error], 2)
    metrics[f'openloop/{key}'] = grid
```

这些视频会保存在:
- 格式: NumPy 数组
- 位置: `{logdir}/openloop/{key}`
- 内容: 真实观察 | 模型预测 | 预测误差

### 自定义视频录制

使用 `inference_video.py`:

```python
# 核心推理循环
for step in range(max_steps):
    # 智能体决策
    carry, action, _ = agent.policy(carry, obs, mode='eval')

    # 环境步进
    obs = env.step(action)

    # 录制帧
    video_logger.add_frame(obs['image'])

    if obs['is_last']:
        break

# 保存为 MP4
video_logger.save('episode_001.mp4')
```

## 📦 依赖安装

```bash
# 安装视频处理库
uv add imageio imageio-ffmpeg

# 或使用 pip
pip install imageio imageio-ffmpeg
```

## 🚀 完整使用流程

### 步骤 1: 训练模型
```bash
# 训练 DreamerV3 (会自动保存 checkpoint 和视频)
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer_pong \
  --configs atari \
  --task atari_pong \
  --run.steps 1000000
```

### 步骤 2: 查看训练视频
```bash
# 方法 A: 使用 Scope 查看器 (推荐)
pip install -U scope
python -m scope.viewer --basedir ~/logdir/dreamer_pong --port 8000

# 方法 B: 直接读取保存的图像
import numpy as np
video = np.load('~/logdir/dreamer_pong/openloop/image.npy')
```

### 步骤 3: 运行推理并录制视频
```bash
# 使用自定义脚本
python inference_video.py \
  --checkpoint ~/logdir/dreamer_pong/checkpoint.pkl \
  --task atari_pong \
  --episodes 10 \
  --output ./pong_videos/
```

### 步骤 4: 查看视频
```bash
# 视频保存在 output 目录
ls ./pong_videos/
# episode_000_score_21.0.mp4
# episode_001_score_19.0.mp4
# ...

# 播放视频
open ./pong_videos/episode_000_score_21.0.mp4
```

## 🎨 可视化选项

### 1. 实时渲染 (Gym 环境)
```python
env = gym.make('Pong-v0', render_mode='human')
# 会打开窗口实时显示
```

### 2. 离线视频录制 (推荐)
```python
# 使用 imageio
import imageio
writer = imageio.get_writer('output.mp4', fps=30)
for frame in frames:
    writer.append_data(frame)
writer.close()
```

### 3. 使用 OpenCV
```python
import cv2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (width, height))
for frame in frames:
    out.write(frame)
out.release()
```

## 📊 Checkpoint 位置

训练时 checkpoint 自动保存在:
```
~/logdir/dreamer_{task}/{timestamp}/
├── checkpoint.pkl          # 最新的 checkpoint
├── config.yaml            # 训练配置
├── metrics.jsonl          # 指标日志
├── scores.jsonl           # 分数日志
└── openloop/              # 预测视频
    └── image/
```

## 🔧 高级用法

### 并行评估多个 checkpoint
```bash
for ckpt in ~/logdir/dreamer_*/checkpoint.pkl; do
    python inference_video.py \
        --checkpoint "$ckpt" \
        --task atari_pong \
        --episodes 5 \
        --output "./videos/$(basename $(dirname $ckpt))"
done
```

### 对比不同任务
```bash
for task in atari_pong atari_breakout dmc_walker_walk; do
    python inference_video.py \
        --checkpoint ~/logdir/dreamer_${task}/checkpoint.pkl \
        --task ${task} \
        --episodes 3 \
        --output ./videos/${task}/
done
```

### 生成 GIF (适合展示)
```python
import imageio

# 读取视频
reader = imageio.get_reader('episode_001.mp4')
frames = [frame for frame in reader]

# 保存为 GIF (降采样以减小文件)
imageio.mimsave('episode_001.gif', frames[::2], fps=15)
```

## 🐛 常见问题

### Q1: 如何找到最新的 checkpoint?
```bash
# 找到最新的 logdir
ls -lt ~/logdir/dreamer_* | head -1

# 或使用命令
latest=$(ls -td ~/logdir/dreamer_*/ | head -1)
echo "${latest}checkpoint.pkl"
```

### Q2: 视频是黑屏怎么办?
检查:
1. 环境是否返回 'image' 观察
2. 图像格式是否正确 (uint8, shape [H,W,3])
3. 是否启用了渲染: `render=True`

### Q3: 没有训练好的模型怎么办?
```bash
# 方法 1: 使用随机智能体测试
python inference_video.py \
  --checkpoint none \
  --task atari_pong \
  --episodes 2 \
  --use-random-agent

# 方法 2: 下载预训练模型
# (查看官方仓库是否有预训练权重)
```

## 📚 参考资料

- 官方文档: [DreamerV3 GitHub](https://github.com/danijar/dreamerv3)
- Scope 查看器: [Scope](https://github.com/danijar/scope)
- 论文: [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)

## 🎯 快速开始

```bash
# 1. 安装依赖
uv add imageio imageio-ffmpeg

# 2. 运行推理 (使用训练好的 checkpoint)
python inference_video.py \
  --checkpoint ~/logdir/dreamer_pong/checkpoint.pkl \
  --task atari_pong \
  --episodes 5 \
  --output ./videos/

# 3. 查看视频
open ./videos/episode_000_*.mp4
```

祝你玩得开心！🎮
