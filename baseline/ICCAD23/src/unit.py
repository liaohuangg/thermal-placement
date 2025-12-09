"""
实用工具函数
提供JSON文件读写、problem和layout的转换等接口
"""

import json
from typing import Dict, List, Tuple, Optional
from chiplet_model import LayoutProblem, Chiplet
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch


def calculate_wirelength(layout: Dict[str, Chiplet], problem: LayoutProblem) -> float:
    """
    计算布局的总线长
    
    对于每一对有连接关系的芯片，计算它们中心点之间的欧几里得距离（曼哈顿距离）
    总线长 = 所有连接的距离之和
    
    Args:
        layout: 布局字典 {chip_id: Chiplet}
        problem: 布局问题，包含连接关系
        
    Returns:
        总线长（所有连接的中心点距离之和）
    """
    if not layout or not problem.connection_graph.edges():
        return 0.0
    
    total_wirelength = 0.0
    
    # 遍历所有连接
    for chip1_id, chip2_id in problem.connection_graph.edges():
        # 获取两个芯片
        chip1 = layout.get(chip1_id)
        chip2 = layout.get(chip2_id)
        
        if chip1 is None or chip2 is None:
            continue
        
        # 计算中心点坐标
        center1_x = chip1.x + chip1.width / 2
        center1_y = chip1.y + chip1.height / 2
        center2_x = chip2.x + chip2.width / 2
        center2_y = chip2.y + chip2.height / 2
        
        # 计算欧几里得距离（也可以使用曼哈顿距离）
        distance = ((center2_x - center1_x) ** 2 + (center2_y - center1_y) ** 2) ** 0.5
        
        # 累加到总线长
        total_wirelength += distance
    
    return total_wirelength


def calculate_manhattan_wirelength(layout: Dict[str, Chiplet], problem: LayoutProblem) -> float:
    """
    计算布局的总线长（使用曼哈顿距离）
    
    对于每一对有连接关系的芯片，计算它们中心点之间的曼哈顿距离
    曼哈顿距离 = |x1 - x2| + |y1 - y2|
    
    Args:
        layout: 布局字典 {chip_id: Chiplet}
        problem: 布局问题，包含连接关系
        
    Returns:
        总线长（所有连接的曼哈顿距离之和）
    """
    if not layout or not problem.connection_graph.edges():
        return 0.0
    
    total_wirelength = 0.0
    
    # 遍历所有连接
    for chip1_id, chip2_id in problem.connection_graph.edges():
        # 获取两个芯片
        chip1 = layout.get(chip1_id)
        chip2 = layout.get(chip2_id)
        
        if chip1 is None or chip2 is None:
            continue
        
        # 计算中心点坐标
        center1_x = chip1.x + chip1.width / 2
        center1_y = chip1.y + chip1.height / 2
        center2_x = chip2.x + chip2.width / 2
        center2_y = chip2.y + chip2.height / 2
        
        # 计算曼哈顿距离
        distance = abs(center2_x - center1_x) + abs(center2_y - center1_y)
        
        # 累加到总线长
        total_wirelength += distance
    
    return total_wirelength


def calculate_layout_utilization(layout: Dict[str, Chiplet]) -> Tuple[float, float, float, float]:
    """
    计算布局的面积利用率
    
    Args:
        layout: 布局字典 {chip_id: Chiplet}
        
    Returns:
        (utilization, bbox_area, chip_total_area, bbox_width, bbox_height)
        - utilization: 面积利用率 (%)
        - bbox_area: 边界框面积
        - chip_total_area: 芯片总面积
        - bbox_width: 边界框宽度
        - bbox_height: 边界框高度
    """
    if not layout:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    # 计算边界框
    chiplets = list(layout.values())
    x_min = min(chip.x for chip in chiplets)
    y_min = min(chip.y for chip in chiplets)
    x_max = max(chip.x + chip.width for chip in chiplets)
    y_max = max(chip.y + chip.height for chip in chiplets)
    
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    bbox_area = bbox_width * bbox_height
    
    # 计算芯片总面积
    chip_total_area = sum(chip.width * chip.height for chip in chiplets)
    
    # 计算利用率
    utilization = (chip_total_area / bbox_area * 100) if bbox_area > 0 else 0.0
    
    return utilization, bbox_area, chip_total_area, bbox_width, bbox_height


def best_utilization(tcgs: List, layouts: List[Dict[str, Chiplet]]) -> Tuple[int, Dict[str, Chiplet], float]:
    """
    从SA_1返回的解集合中,选择面积利用率最高的解
    
    面积利用率 = 芯片总面积 / 边界框面积
    
    Args:
        tcgs: TCG列表 (SA_1返回的第一个元素)
        layouts: 布局列表 (SA_1返回的第二个元素)
        
    Returns:
        (best_index, best_layout, best_utilization)
        - best_index: 最佳解的索引
        - best_layout: 最佳布局
        - best_utilization: 最佳利用率 (%)
    """
    if not layouts:
        return -1, {}, 0.0
    
    best_index = -1
    best_utilization_value = -1.0
    best_layout = None
    
    for i, layout in enumerate(layouts):
        utilization, bbox_area, chip_area, w, h = calculate_layout_utilization(layout)
        
        if utilization > best_utilization_value:
            best_utilization_value = utilization
            best_index = i
            best_layout = layout
    
    return best_index, best_layout, best_utilization_value


def load_problem_from_json(json_path: str) -> LayoutProblem:
    """
    从JSON文件加载problem
    
    JSON格式:
    {
        "chiplets": [
            {"name": "A", "width": 10.0, "height": 8.0},
            ...
        ],
        "connections": [
            ["A", "B"],
            ...
        ]
    }
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        LayoutProblem对象
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    problem = LayoutProblem()
    
    # 添加芯片
    for chip_data in data["chiplets"]:
        chip = Chiplet(
            chip_id=chip_data["name"],
            width=chip_data["width"],
            height=chip_data["height"]
        )
        problem.add_chiplet(chip)
    
    # 添加连接
    for conn in data["connections"]:
        problem.add_connection(conn[0], conn[1])
    
    print(f"✓ Loaded from {json_path}: {len(data['chiplets'])}chiplets, {len(data['connections'])}connections")
    
    return problem


def save_problem_to_json(problem: LayoutProblem, json_path: str):
    """
    保存problem到JSON文件
    
    Args:
        problem: LayoutProblem对象
        json_path: 输出JSON文件路径
    """
    data = {
        "chiplets": [],
        "connections": []
    }
    
    # 保存芯片信息
    for name, chip in problem.chiplets.items():
        data["chiplets"].append({
            "name": chip.id,
            "width": chip.width,
            "height": chip.height
        })
    
    # 保存连接信息
    for edge in problem.connection_graph.edges():
        data["connections"].append([edge[0], edge[1]])
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved to {json_path}: {len(data['chiplets'])}chiplets, {len(data['connections'])}connections")


def save_layout_to_json(layout: Dict[str, Chiplet], json_path: str):
    """
    保存layout到JSON文件
    
    Args:
        layout: 布局字典 {chip_name: Chiplet对象}
        json_path: 输出JSON文件路径
    """
    data = {
        "chiplets": []
    }
    
    for name, chip in layout.items():
        data["chiplets"].append({
            "id": chip.id,
            "width": chip.width,
            "height": chip.height,
            "x": chip.x,
            "y": chip.y
        })
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved layout to {json_path}: {len(layout)}chiplets")


def load_layout_from_json(json_path: str) -> Dict[str, Chiplet]:
    """
    从JSON文件加载layout
    
    JSON格式:
    {
        "chiplets": [
            {"id": "A", "width": 10, "height": 8, "x": 0.0, "y": 0.0},
            ...
        ]
    }
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        布局字典 {chip_name: Chiplet对象}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    layout = {}
    
    for chip_data in data["chiplets"]:
        chip = Chiplet(
            chip_id=chip_data["id"],
            width=chip_data["width"],
            height=chip_data["height"]
        )
        chip.x = chip_data["x"]
        chip.y = chip_data["y"]
        layout[chip_data["id"]] = chip
    
    print(f"✓ Loaded from {json_path} 加载布局: {len(layout)}chiplets")
    
    return layout


def print_layout_summary(layout: Dict[str, Chiplet], problem: LayoutProblem = None):
    """
    打印布局摘要信息
    
    Args:
        layout: 布局字典
        problem: 布局问题（可选），如果提供则显示线长信息
    """
    print("\nLayout Summary:")
    print("=" * 60)
    
    # 计算边界框
    x_coords = [chip.x for chip in layout.values()]
    y_coords = [chip.y for chip in layout.values()]
    x_max = max(chip.x + chip.width for chip in layout.values())
    y_max = max(chip.y + chip.height for chip in layout.values())
    
    bbox_width = x_max - min(x_coords)
    bbox_height = y_max - min(y_coords)
    
    print(f"Number of chiplets: {len(layout)}")
    print(f"Bounding box: ({min(x_coords):.1f}, {min(y_coords):.1f}) → ({x_max:.1f}, {y_max:.1f})")
    print(f"Total width: {bbox_width:.1f}")
    print(f"Total height: {bbox_height:.1f}")
    print(f"Bounding box area: {bbox_width * bbox_height:.1f}")
    
    # 计算芯片总面积
    total_chip_area = sum(chip.width * chip.height for chip in layout.values())
    utilization = (total_chip_area / (bbox_width * bbox_height)) * 100 if bbox_width * bbox_height > 0 else 0
    
    print(f"芯片总面积: {total_chip_area:.1f}")
    print(f"面积利用率: {utilization:.1f}%")
    
    # 如果提供了problem，计算并显示线长
    if problem is not None:
        euclidean_wl = calculate_wirelength(layout, problem)
        manhattan_wl = calculate_manhattan_wirelength(layout, problem)
        num_connections = problem.connection_graph.number_of_edges()
        
        print(f"\nWirelength Information:")
        print(f"Number of connections: {num_connections}")
        print(f"Total wirelength (Euclidean): {euclidean_wl:.2f}")
        print(f"Total wirelength (Manhattan): {manhattan_wl:.2f}")
        if num_connections > 0:
            print(f"Average wirelength (Euclidean): {euclidean_wl / num_connections:.2f}")
            print(f"Average wirelength (Manhattan): {manhattan_wl / num_connections:.2f}")
    
    print("\n各芯片位置:")
    print("-" * 60)
    for name in sorted(layout.keys()):
        chip = layout[name]
        print(f"  {name:6s}: ({chip.x:6.1f}, {chip.y:6.1f}) | 尺寸: {chip.width:4.1f} × {chip.height:4.1f}")
    print("=" * 60)


def create_example_problem() -> LayoutProblem:
    """
    创建一个示例problem用于测试
    
    Returns:
        示例LayoutProblem对象
    """
    problem = LayoutProblem()
    
    # 创建5chiplets
    chips = [
        Chiplet(chip_id="A", width=8, height=8),
        Chiplet(chip_id="B", width=10, height=10),
        Chiplet(chip_id="C", width=12, height=8),
        Chiplet(chip_id="D", width=10, height=12),
        Chiplet(chip_id="E", width=8, height=10),
    ]
    
    for chip in chips:
        problem.add_chiplet(chip)
    
    # 添加连接
    connections = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),
        ("A", "E"),
        ("B", "D"),
    ]
    
    for conn in connections:
        problem.add_connection(conn[0], conn[1])
    
    return problem


def generate_color(chip_id: str):
    """
    为每chiplets生成一个固定的颜色
    
    使用芯片ID的哈希值来生成确定性的颜色
    
    Args:
        chip_id: 芯片ID
        
    Returns:
        RGB颜色元组 (r, g, b)，范围 [0, 1]
    """
    import random
    # 使用哈希生成确定性的随机种子
    random.seed(hash(chip_id))
    
    # 生成较亮的颜色（避免太暗）
    r = random.uniform(0.3, 0.9)
    g = random.uniform(0.3, 0.9)
    b = random.uniform(0.3, 0.9)
    
    return (r, g, b)


def visualize_layout_with_bridges(layout: Dict[str, Chiplet], 
                                   problem: LayoutProblem,
                                   output_file: str = 'layout_with_bridges.png',
                                   show_bridges: bool = True,
                                   show_coordinates: bool = True):
    """
    可视化芯片布局和硅桥连接，保存为PNG图片
    
    Args:
        layout: 芯片布局字典 {chip_id: Chiplet}
        problem: 布局问题，包含连接要求
        output_file: 输出PNG文件路径
        show_bridges: 是否显示硅桥（默认True）
        show_coordinates: 是否显示坐标标注（默认True）
    """
    from Bridge_Overlap_Adjustment import generate_silicon_bridges
    
    # 配置字体
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    if not layout:
        print("错误：没有芯片数据")
        return
    
    # 计算布局边界
    chiplets = list(layout.values())
    x_min = min(chip.x for chip in chiplets)
    y_min = min(chip.y for chip in chiplets)
    x_max = max(chip.x + chip.width for chip in chiplets)
    y_max = max(chip.y + chip.height for chip in chiplets)
    
    # 添加边距
    margin = max((x_max - x_min), (y_max - y_min)) * 0.1
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 绘制每chiplets
    for chip_id, chip in layout.items():
        x, y = chip.x, chip.y
        width, height = chip.width, chip.height
        
        # 生成颜色
        color = generate_color(chip_id)
        
        # 绘制矩形
        rect = Rectangle(
            (x, y), width, height,
            linewidth=2,
            edgecolor='black',
            facecolor=color,
            alpha=0.6,
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
        if show_coordinates:
            coord_text = f"({x:.1f}, {y:.1f})"
            ax.text(
                x, y + height + 0.5,
                coord_text,
                ha='left', va='bottom',
                fontsize=8,
                color='darkgreen'
            )
    
    # 绘制硅桥
    if show_bridges:
        try:
            bridges = generate_silicon_bridges(layout, problem)
            
            for bridge in bridges:
                bbox = bridge.get_bounding_box()
                x_min_b, y_min_b, x_max_b, y_max_b = bbox
                width_b = x_max_b - x_min_b
                height_b = y_max_b - y_min_b
                
                # 绘制硅桥矩形（红色边框，黄色半透明填充）
                bridge_rect = Rectangle(
                    (x_min_b, y_min_b), width_b, height_b,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='yellow',
                    alpha=0.5,
                    linestyle='--'
                )
                ax.add_patch(bridge_rect)
                
                # 添加硅桥标签
                center_x_b = (x_min_b + x_max_b) / 2
                center_y_b = (y_min_b + y_max_b) / 2
                bridge_label = f"{bridge.chip1_id}-{bridge.chip2_id}"
                ax.text(
                    center_x_b, center_y_b, bridge_label,
                    ha='center', va='center',
                    fontsize=8,
                    color='red',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='red')
                )
        except Exception as e:
            print(f"警告：无法绘制硅桥 - {e}")
    
    # 设置坐标轴
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    
    # 标题
    title = f'Chiplet Layout Visualization ({len(layout)} chiplets'
    if show_bridges:
        title += f', {problem.connection_graph.number_of_edges()} connections'
    title += ')'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 添加图例说明
    legend_text = 'Legend:\n'
    legend_text += '• Black border = Chiplet boundary\n'
    legend_text += '• Semi-transparent fill = Chiplet area'
    if show_bridges:
        legend_text += '\n• Red dashed box = Silicon bridge area\n'
        legend_text += '• Yellow semi-transparent = Bridge occupancy'
    
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
    print(f"\n✓ Layout visualization saved to: {output_file}")
    print(f"  - Number of chiplets: {len(layout)}")
    print(f"  - Number of connections: {problem.connection_graph.number_of_edges()}")
    print(f"  - Layout dimensions: {x_max - x_min:.1f} x {y_max - y_min:.1f}")
    print(f"  - Layout area: {(x_max - x_min) * (y_max - y_min):.1f}")
    
    plt.close()


if __name__ == "__main__":
    # 测试示例
    print("单元测试 - 实用工具函数")
    print("=" * 70)
    
    # 测试1: 加载JSON文件到problem
    print("\n测试1: 加载problem从JSON")
    print("-" * 70)
    problem = load_problem_from_json("../test_input/12core.json")
    print(f"芯片列表: {list(problem.chiplets.keys())}")
    print(f"连接数: {len(list(problem.connection_graph.edges()))}")
    
    # 测试2: 保存problem到JSON
    print("\n测试2: 保存problem到JSON")
    print("-" * 70)
    save_problem_to_json(problem, "test_output_problem.json")
    
    # 测试3: 加载layout从JSON
    print("\n测试3: 加载layout从JSON")
    print("-" * 70)
    from TCG import generate_layout_from_tcg  
    from Generate_initial_TCG import generate_initial_TCG

    layout = generate_layout_from_tcg(generate_initial_TCG(problem), problem)
    print_layout_summary(layout, problem)
    
    # 测试3.1: 计算线长
    print("\n测试3.1: 计算线长")
    print("-" * 70)
    euclidean_wl = calculate_wirelength(layout, problem)
    manhattan_wl = calculate_manhattan_wirelength(layout, problem)
    print(f"Euclidean wirelength: {euclidean_wl:.2f}")
    print(f"Manhattan wirelength: {manhattan_wl:.2f}")
    
    # 测试4: 保存layout
    print("\n测试4: 保存layout到JSON")
    print("-" * 70)
    save_layout_to_json(layout, "test_output_layout.json")
    
    # 测试5: 可视化布局和硅桥
    print("\n测试5: 可视化布局和硅桥")
    print("-" * 70)
    visualize_layout_with_bridges(
        layout, 
        problem, 
        output_file='../output/layout_with_bridges.png',
        show_bridges=True,
        show_coordinates=True
    )
    
    print("\n✓ 所有测试完成")



