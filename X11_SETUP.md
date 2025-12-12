# X11 转发配置指南

本指南将帮助你在 WSL 中配置 X11 转发，以便实时显示图形界面。

## 步骤 1: 在 Windows 上安装 X 服务器

### 选项 A: VcXsrv (推荐)
1. 下载 VcXsrv: https://sourceforge.net/projects/vcxsrv/
2. 安装并运行 VcXsrv Windows X Server
3. 启动时选择：
   - Display settings: Multiple windows
   - Client startup: Start no client
   - Extra settings: ✅ 勾选 "Disable access control" (重要！)
   - 点击 "Finish"

### 选项 B: Xming
1. 下载 Xming: https://sourceforge.net/projects/xming/
2. 安装并运行 Xming
3. 在 Xming 设置中允许来自网络的连接

## 步骤 2: 配置 WSL

### 方法 1: 临时设置（当前会话有效）
```bash
source setup_x11.sh
```

### 方法 2: 永久设置（推荐）
将以下内容添加到 `~/.bashrc` 文件末尾：

```bash
# X11 转发配置
export DISPLAY=$(ip route show | grep -i default | awk '{ print $3}'):0.0
```

然后运行：
```bash
source ~/.bashrc
```

## 步骤 3: 测试配置

运行测试命令：
```bash
conda run -n dl python -c "import matplotlib.pyplot as plt; import numpy as np; x = np.linspace(0, 10, 100); plt.plot(x, np.sin(x)); plt.title('X11 Test'); plt.show()"
```

如果能看到图形窗口弹出，说明配置成功！

## 步骤 4: 运行 CartPole 程序

使用实时显示模式运行：
```bash
conda run -n dl python src/rl_learning.py
```

## 故障排除

### 问题 1: 无法显示窗口
- 确保 X 服务器正在运行
- 检查防火墙设置
- 确认 DISPLAY 环境变量已设置：`echo $DISPLAY`

### 问题 2: 连接被拒绝
- 在 X 服务器设置中启用 "Disable access control"
- 检查 Windows 防火墙是否阻止了连接

### 问题 3: 窗口显示但无法交互
- 这是正常的，CartPole 环境会自动更新显示

## 注意事项

- X 服务器需要在每次使用前启动
- 如果 Windows IP 地址改变，需要更新 DISPLAY 变量
- 某些程序可能仍然无法显示（取决于具体实现）

