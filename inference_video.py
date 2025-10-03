#!/usr/bin/env python3
"""
DreamerV3 推理和视频录制脚本

用法:
  python inference_video.py \
    --checkpoint ~/logdir/dreamer/xxx/checkpoint.pkl \
    --task atari_pong \
    --episodes 5 \
    --output ./videos/
"""

import argparse
import pathlib
import sys
from functools import partial as bind

import elements
import embodied
import numpy as np
import imageio

# 添加项目路径
folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder))

from dreamerv3.agent import Agent
from dreamerv3 import main as dreamerv3_main


def make_video_logger(output_dir, episode_num):
    """创建视频记录器"""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    class VideoLogger:
        def add_frame(self, frame):
            """添加帧到视频"""
            if frame.dtype == np.uint8:
                frames.append(frame)

        def save(self, name=None):
            """保存视频"""
            if not frames:
                print("⚠️  没有帧可以保存")
                return

            filename = output_dir / (name or f'episode_{episode_num:03d}.mp4')
            print(f"💾 保存视频: {filename} ({len(frames)} 帧)")

            # 使用 imageio 保存视频
            imageio.mimsave(
                filename,
                frames,
                fps=30,
                codec='libx264',
                quality=8
            )
            frames.clear()

    return VideoLogger()


def inference_with_video(
    checkpoint_path,
    task='atari_pong',
    num_episodes=5,
    output_dir='./videos',
    render=True,
    max_steps=10000,
):
    """运行推理并录制视频"""

    print("=" * 80)
    print("DreamerV3 推理和视频录制")
    print("=" * 80)
    print(f"📦 Checkpoint: {checkpoint_path}")
    print(f"🎮 Task: {task}")
    print(f"📹 Episodes: {num_episodes}")
    print(f"💾 Output: {output_dir}")
    print("=" * 80)

    # 加载配置
    import ruamel.yaml as yaml
    config_file = folder / 'dreamerv3' / 'configs.yaml'
    configs = yaml.YAML(typ='safe').load(config_file.read_text())

    # 创建基础配置
    config = elements.Config(configs['defaults'])

    # 更新任务配置
    suite = task.split('_')[0]
    if suite in configs:
        config = config.update(configs[suite])

    config = config.update(
        task=task,
        logdir=f'./eval_logs/{task}',
        seed=0,
    )

    # 创建环境
    print("\n🌍 创建环境...")
    env = dreamerv3_main.make_env(config, 0)

    # 创建智能体
    print("🤖 创建智能体...")
    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}

    agent = Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
    ))

    # 加载 checkpoint
    print(f"📥 加载 checkpoint: {checkpoint_path}")
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(checkpoint_path, keys=['agent'])

    # 推理循环
    print(f"\n▶️  开始推理 ({num_episodes} episodes)...\n")

    for ep in range(num_episodes):
        print(f"Episode {ep + 1}/{num_episodes}")
        video_logger = make_video_logger(output_dir, ep)

        # 重置环境和智能体
        obs = env.reset()
        carry = agent.init_policy(batch_size=1)

        episode_reward = 0
        episode_length = 0

        for step in range(max_steps):
            # 将观察转换为批次格式
            obs_batch = {k: v[None, ...] for k, v in obs.items()}

            # 智能体决策
            carry, action, _ = agent.policy(carry, obs_batch, mode='eval')

            # 提取动作 (移除批次维度)
            action = {k: v[0] for k, v in action.items()}

            # 环境步进
            obs = env.step(action)

            # 记录视频帧
            if 'image' in obs and render:
                video_logger.add_frame(obs['image'])

            episode_reward += obs.get('reward', 0)
            episode_length += 1

            # 检查是否结束
            if obs.get('is_last', False):
                break

        # 保存视频
        video_logger.save(f'episode_{ep:03d}_score_{episode_reward:.1f}.mp4')

        print(f"  ✅ Score: {episode_reward:.2f}, Length: {episode_length}")

    env.close()
    print(f"\n✨ 完成! 视频已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='DreamerV3 推理和视频录制')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint 文件路径')
    parser.add_argument('--task', type=str, default='atari_pong',
                        help='任务名称 (例如: atari_pong, dmc_walker_walk)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='录制的 episode 数量')
    parser.add_argument('--output', type=str, default='./videos',
                        help='视频输出目录')
    parser.add_argument('--max-steps', type=int, default=10000,
                        help='每个 episode 的最大步数')
    parser.add_argument('--no-render', action='store_true',
                        help='不录制视频')

    args = parser.parse_args()

    inference_with_video(
        checkpoint_path=args.checkpoint,
        task=args.task,
        num_episodes=args.episodes,
        output_dir=args.output,
        render=not args.no_render,
        max_steps=args.max_steps,
    )


if __name__ == '__main__':
    main()
