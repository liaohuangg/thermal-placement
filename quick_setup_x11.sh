#!/bin/bash
# 快速配置 X11 转发

WINDOWS_IP=$(ip route show | grep -i default | awk '{ print $3}')
export DISPLAY=$WINDOWS_IP:0.0

echo "=========================================="
echo "X11 转发配置完成！"
echo "=========================================="
echo "DISPLAY=$DISPLAY"
echo ""
echo "请确保："
echo "1. 已在 Windows 上安装并运行 X 服务器（VcXsrv 或 Xming）"
echo "2. X 服务器已启用 'Disable access control'"
echo ""
echo "测试命令："
echo "  conda run -n dl python src/rl_learning.py"
echo ""
echo "要永久设置，运行："
echo "  echo 'export DISPLAY=\$(ip route show | grep -i default | awk \"{ print \\\$3}\"):0.0' >> ~/.bashrc"
echo "  source ~/.bashrc"

