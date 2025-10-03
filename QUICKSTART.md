# DreamerV3 快速开始指南

## 📦 安装

```bash
# 方法 1: 使用 UV (推荐)
uv add git+https://github.com/danijar/dreamerv3

# 方法 2: 本地安装
git clone https://github.com/danijar/dreamerv3
cd dreamerv3
uv sync

# 如需 CUDA 支持 (Linux/Windows)
uv sync --extra cuda12
```

## 🚀 训练模型

```bash
# 训练 Atari Pong
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer_pong \
  --configs atari \
  --task atari_pong \
  --run.steps 1000000

# 训练 DMC Walker Walk
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer_walker \
  --configs dmc \
  --task dmc_walker_walk \
  --run.steps 1000000
```

## 🎬 推理和可视化

### 方法 1: 录制视频 (推荐)

```bash
# 安装视频依赖
uv add imageio imageio-ffmpeg

# 运行推理并录制视频
python inference_video.py \
  --checkpoint ~/logdir/dreamer_pong/checkpoint.pkl \
  --task atari_pong \
  --episodes 5 \
  --output ./videos/

# 查看视频
open ./videos/episode_000_score_21.0.mp4
```

### 方法 2: 使用 Scope 查看器

```bash
# 安装 Scope
pip install -U scope

# 启动查看器
python -m scope.viewer --basedir ~/logdir --port 8000

# 浏览器访问 http://localhost:8000
```

## 📁 项目结构

```
dreamerv3/
├── dreamerv3/              # 主包
│   ├── main.py            # 训练入口
│   ├── agent.py           # DreamerV3 智能体
│   ├── rssm.py            # 状态空间模型
│   └── configs.yaml       # 配置文件
├── embodied/              # 工具库
│   ├── core/              # 核心功能
│   ├── envs/              # 环境包装器
│   ├── jax/               # JAX 工具
│   └── run/               # 训练脚本
├── inference_video.py     # 推理视频录制
├── pyproject.toml         # UV 项目配置
└── README.md              # 主文档
```

## 🛠️ 常用命令

```bash
# 构建包
uv build

# 安装开发依赖
uv sync --group dev

# 运行测试
uv run pytest

# 查看包信息
uv tree
```

## 📚 更多文档

- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) - 详细的推理和可视化指南
- [MIGRATION_TO_UV.md](MIGRATION_TO_UV.md) - UV 迁移分析
- [README.md](README.md) - 完整的项目文档

## 🎯 典型工作流

1. **训练模型**
   ```bash
   python dreamerv3/main.py --configs atari --task atari_pong
   ```

2. **监控训练**
   ```bash
   python -m scope.viewer --basedir ~/logdir --port 8000
   ```

3. **推理可视化**
   ```bash
   python inference_video.py \
     --checkpoint ~/logdir/dreamer_pong/checkpoint.pkl \
     --task atari_pong \
     --output ./videos/
   ```

4. **查看结果**
   ```bash
   open ./videos/episode_*.mp4
   ```

## 🐛 故障排除

### CUDA 错误
```bash
# 使用 CPU
python dreamerv3/main.py --jax.platform cpu ...

# 减小 batch size
python dreamerv3/main.py --batch_size 1 ...
```

### 内存不足
```bash
# 使用较小的模型
python dreamerv3/main.py --configs atari size12m ...
```

### 找不到 checkpoint
```bash
# 查找最新的 checkpoint
ls -lt ~/logdir/dreamer_*/checkpoint.pkl | head -1
```

## 🚀 开始你的第一个实验！

```bash
# 完整流程示例
git clone https://github.com/danijar/dreamerv3
cd dreamerv3
uv sync
uv add imageio imageio-ffmpeg

# 训练 (可以先用 debug 配置快速测试)
python dreamerv3/main.py \
  --logdir ~/logdir/test \
  --configs debug atari \
  --task atari_pong \
  --run.steps 10000

# 推理
python inference_video.py \
  --checkpoint ~/logdir/test/checkpoint.pkl \
  --task atari_pong \
  --episodes 3 \
  --output ./test_videos/

# 查看
open ./test_videos/
```

祝你好运！🎮
