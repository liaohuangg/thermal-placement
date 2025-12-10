import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os

# 抑制一些不重要的警告
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
# 抑制 ALSA 音频警告（WSL 环境中没有音频设备是正常的）
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
# 重定向 ALSA 错误到 /dev/null
import sys
if sys.platform == 'linux':
    os.environ['SDL_AUDIODRIVER'] = 'dummy'

# 生成环境。此处的环境是CartPole游戏程序。
# render_mode 选择：
# - "human": 实时显示窗口（需要 X11 转发）
# - "rgb_array": 获取图像数据用于保存

def test_x11_connection():
    """测试 X11 连接是否可用"""
    display = os.getenv('DISPLAY')
    if not display:
        return False
    # 简单检查：如果 DISPLAY 设置了，就尝试使用
    # 如果失败，会在运行时捕获异常并切换模式
    return True

# 检查 X11 连接
# 为了确保能看到动画，我们使用 rgb_array 模式并保存 GIF
# 如果用户想要实时显示，可以配置好 X11 后手动改为 'human'
use_human_mode = False  # 设置为 True 以尝试实时显示（需要 VcXsrv 运行）

if use_human_mode and os.getenv('DISPLAY') and test_x11_connection():
    # 如果配置了 X11 转发且连接可用，使用实时显示
    render_mode = 'human'
    print("使用实时显示模式 (render_mode='human')")
    print(f"DISPLAY={os.getenv('DISPLAY')}")
else:
    # 使用 rgb_array 模式保存图像（更可靠）
    render_mode = 'rgb_array'
    print("使用图像保存模式 (render_mode='rgb_array')")
    print("程序将自动保存 GIF 动画文件")
    if not os.getenv('DISPLAY'):
        print("提示: 配置 X11 转发可以实时显示动画窗口")
        print("  1. 在 Windows 上安装并运行 VcXsrv")
        print("  2. 运行: source quick_setup_x11.sh")
        print("  3. 在代码中设置 use_human_mode = True")

# 使用 CartPole-v1 而不是 v0，避免弃用警告
env = gym.make('CartPole-v1', render_mode=render_mode)
# 重置环境,让小车回到起点。并输出初始状态。
state, info = env.reset()

# 存储每一帧的图像（仅在 rgb_array 模式下）
frames = []

for t in range(100):
    # 渲染环境
    if render_mode == 'rgb_array':
        # 获取当前帧的图像数据
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    else:
        # human 模式下直接显示，不需要获取帧数据
        try:
            env.render()
        except Exception as e:
            # 如果显示失败，切换到 rgb_array 模式
            print(f"\n警告: 实时显示失败 ({e})")
            print("自动切换到图像保存模式...")
            render_mode = 'rgb_array'
            env.close()
            env = gym.make('CartPole-v1', render_mode='rgb_array')
            state, info = env.reset()
            frames = []
            frame = env.render()
            if frame is not None:
                frames.append(frame)
    print(state)
    # 方便起见,此处均匀抽样生成一个动作。在实际应用中,应当依据状态,用策略函数生成动作。
    action = env.action_space.sample()
    # 智能体真正执行动作。然后环境更新状态,并反馈一个奖励。
    state, reward, terminated, truncated, info = env.step(action)
    # done等于True意味着游戏结束; done等于False意味着游戏继续。
    done = terminated or truncated
    if done:
        print('Finished')
        break

env.close()

# 如果有保存的帧，显示最后一帧或保存为GIF（仅在 rgb_array 模式下）
if render_mode == 'rgb_array' and frames:
    print(f"\n共捕获了 {len(frames)} 帧图像")
    # 显示最后一帧
    plt.figure(figsize=(8, 6))
    plt.imshow(frames[-1])
    plt.title('CartPole - Last Frame', fontsize=14)
    plt.axis('off')
    plt.savefig('cartpole_last_frame.png', dpi=100, bbox_inches='tight')
    print("✅ 已保存最后一帧图像到: cartpole_last_frame.png")
    
    # 保存为GIF动画
    try:
        import imageio
        gif_path = 'cartpole_animation.gif'
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"✅ 已保存动画到: {gif_path}")
        print(f"   你可以在 Windows 文件管理器中打开这个文件查看动画！")
    except ImportError:
        print("❌ 无法保存 GIF: imageio 未安装")
        print("   运行: conda run -n dl pip install imageio")