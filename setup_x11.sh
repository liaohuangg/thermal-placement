#!/bin/bash
# X11 转发配置脚本
# 用于在 WSL 中显示图形界面

# 获取 Windows 主机 IP 地址
WINDOWS_IP=$(ip route show | grep -i default | awk '{ print $3}')

# 设置 DISPLAY 环境变量
export DISPLAY=$WINDOWS_IP:0.0

echo "X11 转发已配置"
echo "DISPLAY=$DISPLAY"
echo ""
echo "请确保："
echo "1. 已在 Windows 上安装并运行 X 服务器（如 VcXsrv）"
echo "2. X 服务器允许来自网络的连接"
echo "3. 防火墙允许 X 服务器的连接"
echo ""
echo "要永久设置，请将以下内容添加到 ~/.bashrc:"
echo "export DISPLAY=\$(ip route show | grep -i default | awk '{ print \$3}'):0.0"

