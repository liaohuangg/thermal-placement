#!/bin/bash
# X11 连接诊断脚本

echo "=========================================="
echo "X11 连接诊断"
echo "=========================================="

# 检查 DISPLAY 环境变量
echo "1. 检查 DISPLAY 环境变量:"
if [ -z "$DISPLAY" ]; then
    echo "   ❌ DISPLAY 未设置"
    echo "   运行: source quick_setup_x11.sh"
else
    echo "   ✅ DISPLAY=$DISPLAY"
fi

# 检查 Windows IP 可达性
WINDOWS_IP=$(echo $DISPLAY | cut -d: -f1)
echo ""
echo "2. 检查 Windows IP ($WINDOWS_IP) 可达性:"
if ping -c 1 -W 1 $WINDOWS_IP > /dev/null 2>&1; then
    echo "   ✅ IP 地址可达"
else
    echo "   ❌ IP 地址不可达"
fi

# 尝试测试 X11 连接
echo ""
echo "3. 测试 X11 连接:"
if command -v xset > /dev/null 2>&1; then
    if timeout 2 xset q > /dev/null 2>&1; then
        echo "   ✅ X11 连接正常"
    else
        echo "   ❌ X11 连接失败"
        echo "   请确保 VcXsrv 正在运行"
    fi
else
    echo "   ⚠️  xset 命令不可用，无法测试"
fi

echo ""
echo "=========================================="
echo "建议:"
echo "1. 确保 VcXsrv 正在 Windows 上运行"
echo "2. 在 VcXsrv 启动时勾选 'Disable access control'"
echo "3. 检查 Windows 防火墙设置"
echo "=========================================="


