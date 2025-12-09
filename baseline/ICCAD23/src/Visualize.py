"""
布局可视化工具

从 layout.json 读取芯片布局，生成可视化PNG图片。
芯片使用半透明矩形显示，便于识别重叠等问题。
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import random
from typing import List, Dict, Tuple

# 配置字体
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_layout_from_json(json_file: str) -> List[Dict]:
    """
    从JSON文件加载布局
    
    Args:
        json_file: JSON文件路径
        
    Returns:
        芯片列表，每个芯片是一个字典
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('chiplets', [])


def generate_color(chip_id: str) -> Tuple[float, float, float]:
    """
    为每个芯片生成一个固定的颜色
    
    使用芯片ID的哈希值来生成确定性的颜色
    
    Args:
        chip_id: 芯片ID
        
    Returns:
        RGB颜色元组 (r, g, b)，范围 [0, 1]
    """
    # 使用哈希生成确定性的随机种子
    random.seed(hash(chip_id))
    
    # 生成较亮的颜色（避免太暗）
    r = random.uniform(0.3, 0.9)
    g = random.uniform(0.3, 0.9)
    b = random.uniform(0.3, 0.9)
    
    return (r, g, b)


def visualize_layout(chiplets: List[Dict], output_file: str = 'layout_visualization.png'):
    """
    可视化芯片布局并保存为PNG图片
    
    Args:
        chiplets: 芯片列表
        output_file: 输出PNG文件路径
    """
    if not chiplets:
        print("错误：没有芯片数据")
        return
    
    # 计算布局边界
    x_min = min(chip['x'] for chip in chiplets)
    y_min = min(chip['y'] for chip in chiplets)
    x_max = max(chip['x'] + chip['width'] for chip in chiplets)
    y_max = max(chip['y'] + chip['height'] for chip in chiplets)
    
    # 添加边距
    margin = max((x_max - x_min), (y_max - y_min)) * 0.1
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 绘制每个芯片
    for chip in chiplets:
        x, y = chip['x'], chip['y']
        width, height = chip['width'], chip['height']
        chip_id = chip['id']
        
        # 生成颜色
        color = generate_color(chip_id)
        
        # 统一样式
        edge_color = 'black'
        edge_width = 2
        alpha = 0.6
        
        # 绘制矩形
        rect = Rectangle(
            (x, y), width, height,
            linewidth=edge_width,
            edgecolor=edge_color,
            facecolor=color,
            alpha=alpha,
            label=chip_id
        )
        ax.add_patch(rect)
        
        # 添加芯片ID标签（在中心）
        center_x = x + width / 2
        center_y = y + height / 2
        ax.text(
            center_x, center_y, chip_id,
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
        
        # 添加尺寸标注
        size_text = f"{width}x{height}"
        ax.text(
            center_x, y - 1,
            size_text,
            ha='center', va='top',
            fontsize=9,
            color='darkblue'
        )
        
        # 添加坐标标注
        coord_text = f"({x:.1f}, {y:.1f})"
        ax.text(
            x, y + height + 0.5,
            coord_text,
            ha='left', va='bottom',
            fontsize=8,
            color='darkgreen'
        )
    
    # 设置坐标轴
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    
    # 标题
    title = f'Chiplet Layout Visualization ({len(chiplets)} chiplets)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 添加图例说明
    legend_text = 'Legend:\n'
    legend_text += '• Black border = Chiplet boundary\n'
    legend_text += '• Semi-transparent fill = Chiplet area'
    
    ax.text(
        0.02, 0.98, legend_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ 布局可视化已保存到: {output_file}")
    print(f"  - 芯片数量: {len(chiplets)}")
    print(f"  - 布局尺寸: {x_max - x_min:.1f} x {y_max - y_min:.1f}")
    print(f"  - 布局面积: {(x_max - x_min) * (y_max - y_min):.1f}")
    
    plt.close()


if __name__ == "__main__":
    import sys
    
    # 默认使用 layout.json
    json_file = '../output/layout.json'
    output_file = '../output/layout_visualization.png'
    
    # 支持命令行参数
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print("芯片布局可视化工具")
    print("=" * 60)
    print(f"输入文件: {json_file}")
    print(f"输出文件: {output_file}")
    print("=" * 60)
    
    try:
        # 加载布局
        chiplets = load_layout_from_json(json_file)
        
        if not chiplets:
            print("\n错误: JSON文件中没有芯片数据")
            print("请确保JSON文件包含 'chiplets' 字段")
            sys.exit(1)
        
        print(f"\n成功加载 {len(chiplets)} 个芯片:")
        for chip in chiplets:
            print(f"  - {chip['id']}: {chip['width']}x{chip['height']} @ ({chip['x']}, {chip['y']})")
        
        # 可视化
        visualize_layout(chiplets, output_file)
        
    except FileNotFoundError:
        print(f"\n错误: 找不到文件 '{json_file}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"\n错误: JSON格式错误 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
