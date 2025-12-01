"""
Utility functions for chiplet placement experiments.

这里包含：
- 从 JSON 输入中读取 chiplet 信息；
- 构建随机连接图；
- 绘制方框图（chiplet + phys 点 + 连接箭头）。
"""

from __future__ import annotations

import math
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


def build_random_chiplet_graph(
    edge_prob: float = 0.2,
    max_nodes: Optional[int] = None,
    fixed_num_edges: int = 4,
) -> Tuple[List[ChipletNode], List[Tuple[str, str]]]:
    """
    Convenience helper: load chiplets and generate a random connectivity graph.
    
    Parameters
    ----------
    edge_prob:
        边的概率（已废弃，现在使用 fixed_num_edges）
    max_nodes:
        如果指定，只加载前 max_nodes 个 chiplet。默认只取前4个。
    fixed_num_edges:
        生成的固定边数。默认4条。
    """

    nodes = load_chiplet_nodes(max_nodes=max_nodes)
    names = [n.name for n in nodes]
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
    edges: List[Tuple[str, str]],
    save_path: Optional[str] = None,
    layout: Optional[Dict[str, Tuple[float, float]]] = None,
):
    """
    画出 chiplet 方框图。

    参数
    ----
    nodes:
        Chiplet 列表。
    edges:
        连接边列表，形如 (src_name, dst_name)。
    save_path:
        若给定，则保存到该路径；否则直接 ``plt.show()``。
    layout:
        可选，自定义布局 dict: name -> (origin_x, origin_y)。如果为 None，
        则使用 :func:`default_grid_layout`。
    """

    if not nodes:
        raise ValueError("No chiplet nodes to draw.")

    if layout is None:
        layout = default_grid_layout(nodes)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 记录每个 chiplet 用于连边的锚点坐标（统一使用中心点，与接口无关）
    anchor: Dict[str, Tuple[float, float]] = {}

    # 1) 画 chiplet 方框和 phys 锚点
    for node in nodes:
        if node.name not in layout:
            continue

        origin_x, origin_y = layout[node.name]
        w = float(node.dimensions.get("x", 0.0))
        h = float(node.dimensions.get("y", 0.0))

        # 淡蓝色方框
        rect = Rectangle(
            (origin_x, origin_y),
            w,
            h,
            facecolor="#cce6ff",
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
    for src, dst in edges:
        if src not in anchor or dst not in anchor:
            continue
        sx, sy = anchor[src]
        dx, dy = anchor[dst]

        arrow = FancyArrowPatch(
            (sx, sy),
            (dx, dy),
            arrowstyle="->",
            mutation_scale=10,
            linewidth=1.0,
            color="gray",
            alpha=0.8,
        )
        ax.add_patch(arrow)

    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

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
    out_path = Path(__file__).parent.parent / "chiplet_diagram_from_tool.png"
    draw_chiplet_diagram(nodes, edges, save_path=str(out_path))
    print(f"Diagram saved to: {out_path}")


