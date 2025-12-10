"""
Utility functions for chiplet placement experiments.

这里包含：
- 从 JSON 输入中读取 chiplet 信息；
- 构建随机连接图；
- 绘制方框图（chiplet + phys 点 + 连接箭头）。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from input_process import build_chiplet_table, load_chiplets_json


# ---------------------------------------------------------------------------
# 数据结构与基础读入
# ---------------------------------------------------------------------------


@dataclass
class ChipletNode:
    """A simple wrapper for a chiplet entry."""

    name: str
    dimensions: Dict
    phys: List[Dict]
    power: float


def load_chiplet_nodes(max_nodes: Optional[int] = None) -> List[ChipletNode]:
    """
    Load chiplets from JSON and convert them into :class:`ChipletNode` objects.
    
    Parameters
    ----------
    max_nodes:
        如果指定，只返回前 max_nodes 个 chiplet。默认返回前4个。
    """

    raw = load_chiplets_json()
    table = build_chiplet_table(raw)

    nodes: List[ChipletNode] = []
    limit = max_nodes if max_nodes is not None else 4  # 默认只取前4个
    for row in table[:limit]:
        nodes.append(
            ChipletNode(
                name=row["name"],
                dimensions=row["dimensions"],
                phys=row["phys"],
                power=row["power"],
            )
        )
    return nodes


def generate_random_links(
    node_names: List[str],
    edge_prob: float = 0.2,
    allow_self_loop: bool = False,
    undirected: bool = True,
    fixed_num_edges: int = 10,
) -> List[Tuple[str, str]]:
    """
    生成固定数量的链接信息（每次调用都返回相同的链接）。
    
    使用固定的随机种子，确保对于相同的节点列表，每次生成的链接都是相同的。
    """

    import random

    # 设置固定的随机种子，确保每次生成相同的链接
    random.seed(42)
    
    # 生成所有可能的边对（排除自环）
    all_possible_edges: List[Tuple[str, str]] = []
    n = len(node_names)
    for i in range(n):
        for j in range(n):
            # 明确排除自环（自己链接自己）
            if i == j:
                continue
            if undirected and j <= i:
                # 对于无向图，只保留 i < j 的边
                continue

            all_possible_edges.append((node_names[i], node_names[j]))

    # 如果可能的边数少于固定数量，返回所有边
    if len(all_possible_edges) <= fixed_num_edges:
        # 再次确保没有自环（双重保险）
        edges = [(a, b) for a, b in all_possible_edges if a != b]
        random.seed()
        return edges

    # 随机选择固定数量的边
    edges = random.sample(all_possible_edges, fixed_num_edges)
    
    # 最终过滤：确保没有任何自环（双重保险）
    edges = [(a, b) for a, b in edges if a != b]
    
    # 如果过滤后边数不足，重新选择
    while len(edges) < fixed_num_edges and len(all_possible_edges) > len(edges):
        remaining = [e for e in all_possible_edges if e not in edges]
        if not remaining:
            break
        needed = fixed_num_edges - len(edges)
        additional = random.sample(remaining, min(needed, len(remaining)))
        edges.extend(additional)
        edges = [(a, b) for a, b in edges if a != b]  # 再次过滤
    
    # 重置随机种子，避免影响其他使用随机数的代码
    random.seed()

    return edges


def generate_typed_edges(
    node_names: List[str],
    num_silicon_bridge_edges: int = 5,
    num_normal_edges: int = 5,
    seed: Optional[int] = 42,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """
    生成两种类型的链接边：硅桥互联边和普通链接边。
    
    Parameters
    ----------
    node_names:
        节点名称列表
    num_silicon_bridge_edges:
        硅桥互联边的数量
    num_normal_edges:
        普通链接边的数量
    seed:
        随机种子，用于可重复生成
    
    Returns
    -------
    Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        返回两个列表：
        - 第一个列表：硅桥互联边，格式为 (src, dst, "silicon_bridge")
        - 第二个列表：普通链接边，格式为 (src, dst, "normal")
    """
    if seed is not None:
        random.seed(seed)
    
    # 生成所有可能的边对（排除自环）
    all_possible_edges: List[Tuple[str, str]] = []
    n = len(node_names)
    for i in range(n):
        for j in range(i + 1, n):  # 无向图，只保留 i < j 的边
            all_possible_edges.append((node_names[i], node_names[j]))
    
    # 确保有足够的边可以生成
    total_needed = num_silicon_bridge_edges + num_normal_edges
    if len(all_possible_edges) < total_needed:
        print(f"警告: 可能的边数({len(all_possible_edges)})少于需要的边数({total_needed})")
        # 调整数量
        num_silicon_bridge_edges = min(num_silicon_bridge_edges, len(all_possible_edges) // 2)
        num_normal_edges = min(num_normal_edges, len(all_possible_edges) - num_silicon_bridge_edges)
        total_needed = num_silicon_bridge_edges + num_normal_edges
    
    # 随机选择边
    selected_edges = random.sample(all_possible_edges, total_needed)
    
    # 分配边类型
    silicon_bridge_edges = [
        (src, dst, "silicon_bridge") 
        for src, dst in selected_edges[:num_silicon_bridge_edges]
    ]
    
    normal_edges = [
        (src, dst, "normal") 
        for src, dst in selected_edges[num_silicon_bridge_edges:]
    ]
    
    # 重置随机种子
    if seed is not None:
        random.seed()
    
    return silicon_bridge_edges, normal_edges


def build_random_chiplet_graph(
    edge_prob: float = 0.2,
    max_nodes: Optional[int] = None,
    fixed_num_edges: int = 4,
    num_silicon_bridge_edges: Optional[int] = None,
    num_normal_edges: Optional[int] = None,
    seed: Optional[int] = 42,
) -> Tuple[List[ChipletNode], List[Tuple[str, str]]]:
    """
    Convenience helper: load chiplets and generate a random connectivity graph.
    
    Parameters
    ----------
    edge_prob:
        边的概率（已废弃，现在使用 fixed_num_edges 或 num_silicon_bridge_edges/num_normal_edges）
    max_nodes:
        如果指定，只加载前 max_nodes 个 chiplet。默认只取前4个。
    fixed_num_edges:
        生成的固定边数（当未指定 num_silicon_bridge_edges 和 num_normal_edges 时使用）。默认4条。
    num_silicon_bridge_edges:
        硅桥互联边的数量（如果指定，将使用类型化边生成）
    num_normal_edges:
        普通链接边的数量（如果指定，将使用类型化边生成）
    seed:
        随机种子，用于可重复生成（仅在指定 num_silicon_bridge_edges 或 num_normal_edges 时使用）
    
    Returns
    -------
    Tuple[List[ChipletNode], List[Tuple[str, str]]]:
        返回节点列表和边列表（旧格式，向后兼容）
    """

    nodes = load_chiplet_nodes(max_nodes=max_nodes)
    names = [n.name for n in nodes]
    
    # 如果指定了硅桥互联边或普通互联边的数量，使用类型化边生成
    if num_silicon_bridge_edges is not None or num_normal_edges is not None:
        # 设置默认值
        if num_silicon_bridge_edges is None:
            num_silicon_bridge_edges = fixed_num_edges // 2 if fixed_num_edges > 0 else 0
        if num_normal_edges is None:
            num_normal_edges = fixed_num_edges - num_silicon_bridge_edges if fixed_num_edges > 0 else 0
        
        # 生成类型化边
        silicon_bridge_edges, normal_edges = generate_typed_edges(
            node_names=names,
            num_silicon_bridge_edges=num_silicon_bridge_edges,
            num_normal_edges=num_normal_edges,
            seed=seed
        )
        
        # 合并为旧格式（不带类型标签）
        edges = [(src, dst) for src, dst, _ in silicon_bridge_edges + normal_edges]
    else:
        # 使用旧的生成方法（向后兼容）
        edges = generate_random_links(names, edge_prob=edge_prob, fixed_num_edges=fixed_num_edges)
    
    return nodes, edges


# ---------------------------------------------------------------------------
# 绘图相关
# ---------------------------------------------------------------------------


def default_grid_layout(nodes: List[ChipletNode]) -> Dict[str, Tuple[float, float]]:
    """
    为每个 chiplet 决定一个在大画布上的偏移 (origin_x, origin_y)。

    - 每个 chiplet 内部仍然以左下角为 (0, 0) 局部坐标；
    - 返回一个 dict: name -> (origin_x, origin_y)。
    """

    if not nodes:
        return {}

    max_w = max(n.dimensions.get("x", 0) for n in nodes)
    max_h = max(n.dimensions.get("y", 0) for n in nodes)
    margin = max(max_w, max_h) * 0.3

    n = len(nodes)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    layout: Dict[str, Tuple[float, float]] = {}
    for idx, node in enumerate(nodes):
        r = idx // cols
        c = idx % cols
        origin_x = c * (max_w + margin)
        origin_y = r * (max_h + margin)
        layout[node.name] = (origin_x, origin_y)

    return layout


def draw_chiplet_diagram(
    nodes: List[ChipletNode],
    edges: List[Tuple[str, str]] | List[Tuple[str, str, str]],
    save_path: Optional[str] = None,
    layout: Optional[Dict[str, Tuple[float, float]]] = None,
    edge_types: Optional[Dict[Tuple[str, str], str]] = None,
    fixed_chiplet_names: Optional[set] = None,  # 固定的chiplet名称集合，这些chiplet将用粉红色绘制
):
    """
    画出 chiplet 方框图。

    参数
    ----
    nodes:
        Chiplet 列表。
    edges:
        连接边列表，可以是：
        - 旧格式：形如 (src_name, dst_name)
        - 新格式：形如 (src_name, dst_name, edge_type)，其中 edge_type 为 "silicon_bridge" 或 "normal"
    save_path:
        若给定，则保存到该路径；否则直接 ``plt.show()``。
    layout:
        可选，自定义布局 dict: name -> (origin_x, origin_y)。如果为 None，
        则使用 :func:`default_grid_layout`。
    edge_types:
        可选的边类型映射，格式为 {(src, dst): "silicon_bridge" | "normal"}。
        如果 edges 是新格式或提供了此参数，将根据类型使用不同颜色：
        - 硅桥互联边：绿色
        - 普通链接边：灰色
    fixed_chiplet_names:
        固定的chiplet名称集合。如果提供，这些chiplet将用粉红色绘制，其他chiplet用淡蓝色。
    """

    if not nodes:
        raise ValueError("No chiplet nodes to draw.")

    if layout is None:
        layout = default_grid_layout(nodes)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 记录每个 chiplet 用于连边的锚点坐标（统一使用中心点，与接口无关）
    anchor: Dict[str, Tuple[float, float]] = {}

    # 调试：检查layout和nodes
    print(f"[DEBUG 绘图] nodes数量: {len(nodes)}, layout中的chiplet数量: {len(layout)}")
    missing_in_layout = [n.name for n in nodes if n.name not in layout]
    if missing_in_layout:
        print(f"[DEBUG 绘图] 以下chiplet不在layout中: {missing_in_layout}")

    # 1) 画 chiplet 方框和 phys 锚点
    drawn_count = 0
    for node in nodes:
        if node.name not in layout:
            print(f"[警告] chiplet {node.name} 不在layout中，跳过绘制")
            continue

        origin_x, origin_y = layout[node.name]
        drawn_count += 1
        print(f"[DEBUG 绘图] 绘制 {node.name}: 位置=({origin_x:.2f}, {origin_y:.2f})")
        w = float(node.dimensions.get("x", 0.0))
        h = float(node.dimensions.get("y", 0.0))

        # 判断是否为固定chiplet，固定chiplet使用粉红色，其他使用淡蓝色
        if fixed_chiplet_names is not None and node.name in fixed_chiplet_names:
            facecolor = "pink"  # 粉红色
        else:
            facecolor = "#cce6ff"  # 淡蓝色
        
        rect = Rectangle(
            (origin_x, origin_y),
            w,
            h,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=1.0,
        )
        ax.add_patch(rect)

        # 在左下角写名字
        ax.text(
            origin_x + 0.1,
            origin_y + h + 0.1,
            node.name,
            fontsize=8,
            ha="left",
            va="bottom",
        )

        # phys 点：红色小方块（仅用于显示，不用于连接）
        if node.phys:
            for p in node.phys:
                px = origin_x + float(p.get("x", 0.0))
                py = origin_y + float(p.get("y", 0.0))

                anchor_size = min(w, h) * 0.05
                ax.add_patch(
                    Rectangle(
                        (px - anchor_size / 2.0, py - anchor_size / 2.0),
                        anchor_size,
                        anchor_size,
                        facecolor="red",
                        edgecolor="none",
                    )
                )

        # 所有连接都从chiplet的中心位置出发（与接口无关）
        anchor[node.name] = (origin_x + w / 2.0, origin_y + h / 2.0)

    # 2) 画有向边（箭头）
    # 构建边类型映射
    edge_type_map: Dict[Tuple[str, str], str] = {}
    if edge_types:
        edge_type_map.update(edge_types)
    
    # 从edges中提取类型信息（如果是新格式）
    for edge in edges:
        if len(edge) == 3:
            src, dst, edge_type = edge
            edge_type_map[(src, dst)] = edge_type
        elif len(edge) == 2:
            src, dst = edge
            # 如果没有提供类型信息，默认为普通链接边
            if (src, dst) not in edge_type_map:
                edge_type_map[(src, dst)] = "normal"
    
    for edge in edges:
        # 处理不同格式的边
        if len(edge) == 3:
            src, dst, _ = edge
        elif len(edge) == 2:
            src, dst = edge
        else:
            continue
            
        if src not in anchor or dst not in anchor:
            continue
        sx, sy = anchor[src]
        dx, dy = anchor[dst]
        
        # 根据边类型选择颜色
        edge_type = edge_type_map.get((src, dst), "normal")
        if edge_type == "silicon_bridge":
            edge_color = "green"  # 硅桥互联边：绿色
            linewidth = 2.0  # 稍微粗一点以突出显示
        else:
            edge_color = "gray"  # 普通链接边：灰色
            linewidth = 1.0

        arrow = FancyArrowPatch(
            (sx, sy),
            (dx, dy),
            arrowstyle="->",
            mutation_scale=10,
            linewidth=linewidth,
            color=edge_color,
            alpha=0.8,
        )
        ax.add_patch(arrow)

    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    
    # 调试输出
    print(f"[DEBUG 绘图] 实际绘制了 {drawn_count} 个chiplet（共 {len(nodes)} 个）")

    # 调整视图范围（考虑所有模块的完整范围，包括宽度和高度）
    all_x_min = []
    all_x_max = []
    all_y_min = []
    all_y_max = []
    
    for node in nodes:
        if node.name not in layout:
            continue
        ox, oy = layout[node.name]
        w = float(node.dimensions.get("x", 0.0))
        h = float(node.dimensions.get("y", 0.0))
        # 记录左下角和右上角坐标
        all_x_min.append(ox)
        all_x_max.append(ox + w)
        all_y_min.append(oy)
        all_y_max.append(oy + h)
    
    if all_x_min and all_y_min:
        # 计算所有模块的最小和最大坐标
        x_min = min(all_x_min)
        x_max = max(all_x_max)
        y_min = min(all_y_min)
        y_max = max(all_y_max)
        
        # 添加边距（10%的额外空间）
        x_range = x_max - x_min
        y_range = y_max - y_min
        margin_x = max(x_range * 0.1, 1.0)  # 至少1.0的边距
        margin_y = max(y_range * 0.1, 1.0)
        
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    # 简单测试：随机生成连接并画默认布局
    nodes, edges = build_random_chiplet_graph(edge_prob=0.3)
    # 使用相对路径，输出到项目根目录
    from pathlib import Path
    out_path = Path(__file__).parent.parent / "output" / "chiplet_diagram_from_tool.png"
    draw_chiplet_diagram(nodes, edges, save_path=str(out_path))
    print(f"Diagram saved to: {out_path}")
    
    # 测试生成两种类型的边
    print("\n=== 测试生成两种类型的链接边 ===")
    node_names = [node.name for node in nodes]
    silicon_bridge_edges, normal_edges = generate_typed_edges(
        node_names=node_names,
        num_silicon_bridge_edges=5,
        num_normal_edges=5,
        seed=42
    )
    
    print(f"\n硅桥互联边 ({len(silicon_bridge_edges)} 条):")
    for src, dst, edge_type in silicon_bridge_edges:
        print(f"  {src} <-> {dst} (类型: {edge_type})")
    
    print(f"\n普通链接边 ({len(normal_edges)} 条):")
    for src, dst, edge_type in normal_edges:
        print(f"  {src} <-> {dst} (类型: {edge_type})")


