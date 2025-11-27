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
from typing import Dict, List, Tuple

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


def load_chiplet_nodes() -> List[ChipletNode]:
    """
    Load chiplets from JSON and convert them into :class:`ChipletNode` objects.
    """

    raw = load_chiplets_json()
    table = build_chiplet_table(raw)

    nodes: List[ChipletNode] = []
    for row in table:
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
) -> List[Tuple[str, str]]:
    """
    Randomly generate link information between chiplets.
    """

    import random

    edges: List[Tuple[str, str]] = []

    n = len(node_names)
    for i in range(n):
        for j in range(n):
            if not allow_self_loop and i == j:
                continue
            if undirected and j <= i:
                continue

            if random.random() < edge_prob:
                src = node_names[i]
                dst = node_names[j]
                edges.append((src, dst))

    return edges


def build_random_chiplet_graph(
    edge_prob: float = 0.2,
) -> Tuple[List[ChipletNode], List[Tuple[str, str]]]:
    """
    Convenience helper: load chiplets and generate a random connectivity graph.
    """

    nodes = load_chiplet_nodes()
    names = [n.name for n in nodes]
    edges = generate_random_links(names, edge_prob=edge_prob)
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
    save_path: str | None = None,
    layout: Dict[str, Tuple[float, float]] | None = None,
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

    # 记录每个 chiplet 用于连边的锚点坐标（取第一个 phys，如果没有则用中心点）
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

        # phys 点：红色小方块
        if node.phys:
            for idx, p in enumerate(node.phys):
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

                if idx == 0:
                    anchor[node.name] = (px, py)
        else:
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

    # 调整视图范围
    all_x = []
    all_y = []
    for name, (ox, oy) in layout.items():
        all_x.append(ox)
        all_y.append(oy)
    if all_x and all_y:
        max_w = max(n.dimensions.get("x", 0) for n in nodes)
        max_h = max(n.dimensions.get("y", 0) for n in nodes)
        margin = max(max_w, max_h)
        ax.set_xlim(min(all_x) - margin * 0.3, max(all_x) + max_w + margin * 0.3)
        ax.set_ylim(min(all_y) - margin * 0.3, max(all_y) + max_h + margin * 0.3)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    # 简单测试：随机生成连接并画默认布局
    nodes, edges = build_random_chiplet_graph(edge_prob=0.3)
    out_path = "/root/placement/thermal-placement/chiplet_diagram_from_tool.png"
    draw_chiplet_diagram(nodes, edges, save_path=out_path)
    print(f"Diagram saved to: {out_path}")


