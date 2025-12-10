# X11 转发快速开始

## ✅ 已完成的配置

1. ✅ 代码已更新，支持自动检测 X11 配置
2. ✅ 创建了配置脚本
3. ✅ DISPLAY 环境变量已设置

## 📋 接下来需要做的

### 步骤 1: 在 Windows 上安装并运行 X 服务器

**推荐使用 VcXsrv：**

1. 下载：https://sourceforge.net/projects/vcxsrv/
2. 安装后运行 "XLaunch"
3. 配置选项：
   - Display settings: 选择 "Multiple windows"
   - Client startup: 选择 "Start no client"
   - Extra settings: ✅ **必须勾选 "Disable access control"**
   - 点击 "Finish"

### 步骤 2: 配置 WSL（每次新终端会话）

在 WSL 终端中运行：
```bash
cd /home/huangl/new_workspace/placement/thermal-placement
source quick_setup_x11.sh
```

### 步骤 3: 永久配置（可选，推荐）

将以下命令添加到 `~/.bashrc`：
```bash
echo 'export DISPLAY=$(ip route show | grep -i default | awk "{ print \$3}"):0.0' >> ~/.bashrc
source ~/.bashrc
```

### 步骤 4: 运行程序

```bash
cd /home/huangl/new_workspace/placement/thermal-placement
conda run -n dl python src/rl_learning.py
```

## 🎯 程序行为

- **如果检测到 DISPLAY 环境变量**：使用 `render_mode='human'`，会弹出实时动画窗口
- **如果没有 DISPLAY**：使用 `render_mode='rgb_array'`，保存图像和 GIF 文件

## 🔧 测试 X11 是否工作

运行简单测试：
```bash
conda run -n dl python -c "import matplotlib.pyplot as plt; import numpy as np; plt.plot([1,2,3]); plt.show()"
```

如果能看到图形窗口，说明配置成功！

## ❓ 故障排除

**问题：窗口没有弹出**
- 确保 VcXsrv 正在运行
- 检查是否勾选了 "Disable access control"
- 运行 `echo $DISPLAY` 确认环境变量已设置

**问题：连接被拒绝**
- 检查 Windows 防火墙设置
- 确认 VcXsrv 允许网络连接

