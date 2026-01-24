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

try:
    import pulp
except ImportError:
    pulp = None

try:
    import gurobipy as gp
except ImportError:
    gp = None

from input_process import build_chiplet_table, load_chiplets_json

# 导入ILP相关的类型（如果可用）
try:
    from ilp_method import ILPModelContext, ILPPlacementResult
except ImportError:
    try:
        from ilp_method_gurobi import ILPModelContext, ILPPlacementResult
    except ImportError:
        # 如果ilp_method不可用，定义占位类型
        ILPModelContext = None
        ILPPlacementResult = None


def get_var_value(var):
    """
    统一获取变量值的辅助函数，支持 PuLP 和 Gurobi 变量。
    
    参数:
        var: PuLP 或 Gurobi 变量对象，或 None
    
    返回:
        变量的值，如果变量为 None 则返回 None
    """
    if var is None:
        return None
    
    # 检查是否是 Gurobi 变量
    if gp is not None and isinstance(var, gp.Var):
        try:
            return var.X
        except AttributeError:
            return None
    
    # 检查是否是 PuLP 变量
    if pulp is not None:
        try:
            # 检查是否有 value 方法（PuLP 变量）
            if hasattr(var, 'value'):
                return pulp.value(var)
            # 或者直接调用 value() 方法
            elif callable(getattr(var, 'value', None)):
                return var.value()
        except (AttributeError, TypeError):
            return None
    
    return None


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
    grid_size: float = 1.0,  # 网格大小，用于将网格坐标转换为实际坐标
    rotations: Optional[Dict[str, bool]] = None,  # 旋转信息：name -> 是否旋转
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
        可选，自定义布局 dict: name -> (x_grid, y_grid)，其中坐标是网格坐标（需要乘以 grid_size 得到实际坐标）。
        如果为 None，则使用 :func:`default_grid_layout`。
    edge_types:
        可选的边类型映射，格式为 {(src, dst): "silicon_bridge" | "normal"}。
        如果 edges 是新格式或提供了此参数，将根据类型使用不同颜色：
        - 硅桥互联边：绿色
        - 普通链接边：灰色
    fixed_chiplet_names:
        固定的chiplet名称集合。如果提供，这些chiplet将用粉红色绘制，其他chiplet用淡蓝色。
    grid_size:
        网格大小，用于将 layout 中的网格坐标转换为实际坐标。默认值为 1.0。
    rotations:
        旋转信息字典：name -> 是否旋转。如果提供，会根据旋转状态交换 chiplet 的长宽。
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

        # 获取网格坐标并转换为实际坐标
        x_grid, y_grid = layout[node.name]
        origin_x = float(x_grid) * grid_size
        origin_y = float(y_grid) * grid_size
        
        drawn_count += 1
        print(f"[DEBUG 绘图] 绘制 {node.name}: 网格坐标=({x_grid:.2f}, {y_grid:.2f}), 实际坐标=({origin_x:.2f}, {origin_y:.2f})")
        
        # 获取原始尺寸
        orig_w = float(node.dimensions.get("x", 0.0))
        orig_h = float(node.dimensions.get("y", 0.0))
        
        # 检查是否旋转
        is_rotated = False
        if rotations is not None and node.name in rotations:
            is_rotated = rotations[node.name]
        
        # 如果旋转，交换长宽
        if is_rotated:
            w = orig_h
            h = orig_w
        else:
            w = orig_w
            h = orig_h

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

        # 在chiplet块中心写名字
        center_x = origin_x + w / 2.0
        center_y = origin_y + h / 2.0
        ax.text(
            center_x,
            center_y,
            node.name,
            fontsize=10,
            ha="center",
            va="center",
            weight="bold",
            color="black",
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
    # edges 可能是 (src, dst, conn_type) 格式，其中 conn_type 是整数：1=silicon_bridge, 0=standard
    for edge in edges:
        if len(edge) == 3:
            src, dst, conn_type = edge
            # 将整数 conn_type 转换为字符串类型
            if isinstance(conn_type, int):
                if conn_type == 1:
                    edge_type_map[(src, dst)] = "silicon_bridge"
                    edge_type_map[(dst, src)] = "silicon_bridge"  # 双向
                else:
                    edge_type_map[(src, dst)] = "normal"
                    edge_type_map[(dst, src)] = "normal"  # 双向
            elif isinstance(conn_type, str):
                # 如果已经是字符串，直接使用
                edge_type_map[(src, dst)] = conn_type
                edge_type_map[(dst, src)] = conn_type  # 双向
        elif len(edge) == 2:
            src, dst = edge
            # 如果没有提供类型信息，默认为普通链接边
            if (src, dst) not in edge_type_map:
                edge_type_map[(src, dst)] = "normal"
                edge_type_map[(dst, src)] = "normal"  # 双向
    
    # 调试输出：打印边类型映射
    print(f"[DEBUG 绘图] 边类型映射:")
    for (src, dst), etype in edge_type_map.items():
        print(f"  ({src}, {dst}): {etype}")
    
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
        print(f"[DEBUG 绘图] 绘制边 ({src}, {dst}): 类型={edge_type}")
        if edge_type == "silicon_bridge":
            edge_color = "green"  # 硅桥互联边：绿色
            linewidth = 3.0  # 更粗一点以突出显示
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
        x_grid, y_grid = layout[node.name]
        ox = float(x_grid) * grid_size
        oy = float(y_grid) * grid_size
        
        # 获取原始尺寸
        orig_w = float(node.dimensions.get("x", 0.0))
        orig_h = float(node.dimensions.get("y", 0.0))
        
        # 检查是否旋转
        is_rotated = False
        if rotations is not None and node.name in rotations:
            is_rotated = rotations[node.name]
        
        # 如果旋转，交换长宽
        if is_rotated:
            w = orig_h
            h = orig_w
        else:
            w = orig_w
            h = orig_h
        
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


# ---------------------------------------------------------------------------
# 约束打印功能（用于调试ILP约束）
# ---------------------------------------------------------------------------

if pulp is not None:
    # 约束方向映射表
    SENSE_MAP = {
        pulp.LpConstraintLE: "<=",
        pulp.LpConstraintGE: ">=",
        pulp.LpConstraintEQ: "=",
    }

    def print_constraint_formal(constraint) -> None:
        """
        打印约束的形式化数学表达。
        
        参数:
            constraint: Pulp约束对象或Gurobi约束对象
        """
        # 检查是否是Gurobi约束
        if gp is not None and isinstance(constraint, gp.Constr):
            # Gurobi约束
            try:
                constraint_name = constraint.ConstrName
            except (AttributeError, Exception):
                # 如果约束还没有名称（例如刚添加但模型未更新），跳过打印
                return
            # Gurobi约束的字符串表示
            constraint_str = str(constraint)
            # 打印约束（可以修改为输出到日志文件）
            # print(f"[ADD CONSTRAINT] {constraint_name}: {constraint_str}")
            return
        
        # PuLP约束
        if pulp is not None and isinstance(constraint, pulp.LpConstraint):
            # 处理左侧表达式：移除冗余的 *1.0，美化输出
            lhs = str(constraint.expr).replace("*1.0", "").replace(" + ", " + ").strip()
            
            # 处理右侧常数：Pulp内部存储为 expr + constant <= 0，所以需要取负号
            rhs = round(-constraint.constant, 4)
            
            # 获取约束方向字符串
            sense_str = SENSE_MAP.get(constraint.sense, "?")
            
            # 构建形式化表达式
            formal_expr = f"[{constraint.name}] {lhs} {sense_str} {rhs}"
            
            # 打印约束（可以修改为输出到日志文件）
            # print(f"[ADD CONSTRAINT] {formal_expr}")
            return
        
        # 如果都不匹配，静默忽略
        pass
else:
    # 如果pulp未安装，提供占位函数
    def print_constraint_formal(*args, **kwargs):
        # 检查是否是Gurobi约束
        if gp is not None and len(args) > 0:
            constraint = args[0]
            if isinstance(constraint, gp.Constr):
                # Gurobi约束，静默处理
                return
        # 其他情况抛出错误
        raise ImportError("pulp库未安装，无法使用约束打印功能")


# ---------------------------------------------------------------------------
# ILP求解结果打印函数
# ---------------------------------------------------------------------------

def print_pair_distances_only(
    ctx,
    result,
    solution_idx: int,
    prev_pair_distances_list: Optional[List[Dict[Tuple[int, int], float]]] = None,
    min_pair_dist_diff: float = 1.0,
) -> None:
    """
    简化输出：只打印每对chiplet的相对距离，以及当前解与之前解的距离比较。
    
    参数:
        ctx: ILP模型上下文
        result: 求解结果
        solution_idx: 解的索引（从0开始）
        prev_pair_distances_list: 可选，之前所有解的chiplet对距离列表
        min_pair_dist_diff: 判断距离是否相同的最小差异阈值
    """
    if result.status != "Optimal":
        return
    
    nodes = ctx.nodes
    n = len(nodes)
    
    # 获取当前解的坐标
    x_curr = {}
    y_curr = {}
    for k in range(n):
        x_val = get_var_value(ctx.x_grid_var[k])
        y_val = get_var_value(ctx.y_grid_var[k])
        if x_val is not None and y_val is not None:
            x_curr[k] = float(x_val)
            y_curr[k] = float(y_val)
        else:
            return  # 如果无法获取坐标，直接返回
    
    # 计算当前解的每对chiplet的相对距离（x轴和y轴的绝对值差）
    curr_pair_distances = {}
    chiplet_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for i, j in chiplet_pairs:
        if i in x_curr and j in x_curr and i in y_curr and j in y_curr:
            # 计算x轴和y轴的相对距离（绝对值差）
            x_dist = abs(x_curr[i] - x_curr[j])
            y_dist = abs(y_curr[i] - y_curr[j])
            curr_pair_distances[(i, j)] = (x_dist, y_dist)
    
    # 输出当前解的相对距离
    print(f"\n=== 解 {solution_idx + 1} ===")
    print(f"\n每对chiplet的相对距离（|x[i]-x[j]|, |y[i]-y[j]|）:")
    for i, j in sorted(chiplet_pairs):
        if (i, j) in curr_pair_distances:
            x_dist, y_dist = curr_pair_distances[(i, j)]
            name_i = nodes[i].name if hasattr(nodes[i], 'name') else f"Chiplet_{i}"
            name_j = nodes[j].name if hasattr(nodes[j], 'name') else f"Chiplet_{j}"
            # 计算曼哈顿距离（用于显示）
            manhattan_dist = x_dist + y_dist
            print(f"  ({i},{j}) [{name_i}, {name_j}]: x距离={x_dist:.3f}, y距离={y_dist:.3f}, 曼哈顿距离={manhattan_dist:.3f}")
    
    # 与之前解比较
    if prev_pair_distances_list and len(prev_pair_distances_list) > 0:
        # 获取grid_size，用于将grid坐标距离转换为实际坐标距离
        grid_size_val = ctx.grid_size if hasattr(ctx, 'grid_size') and ctx.grid_size is not None else 1.0
        
        print(f"\n与之前解的距离比较（阈值={min_pair_dist_diff:.3f}）:")
        for prev_idx, prev_distances in enumerate(prev_pair_distances_list):
            print(f"\n  与解 {prev_idx + 1} 比较:")
            same_pairs = []
            diff_pairs_with_info = []  # 存储不同对及其距离差信息
            
            for i, j in sorted(chiplet_pairs):
                if (i, j) in curr_pair_distances and (i, j) in prev_distances:
                    curr_x_dist, curr_y_dist = curr_pair_distances[(i, j)]
                    # prev_distances中存储的是grid坐标的曼哈顿距离，需要转换为实际坐标距离
                    prev_dist_grid = prev_distances[(i, j)]  # grid坐标的曼哈顿距离
                    prev_dist = prev_dist_grid * grid_size_val  # 转换为实际坐标距离
                    
                    # 计算当前解的曼哈顿距离（实际坐标）
                    curr_dist = curr_x_dist + curr_y_dist
                    
                    # 计算距离差（绝对值）
                    dist_diff = abs(curr_dist - prev_dist)
                    
                    if dist_diff < min_pair_dist_diff:
                        same_pairs.append((i, j))
                    else:
                        # 只有当距离差 >= min_pair_dist_diff 时，才认为不同
                        # 在else分支中，dist_diff >= min_pair_dist_diff 总是成立
                        diff_pairs_with_info.append((i, j, curr_dist, prev_dist, dist_diff))
            
            if same_pairs:
                print(f"    相同的chiplet对（距离差 < 阈值 {min_pair_dist_diff:.3f}）:")
                for i, j in same_pairs:
                    if (i, j) in curr_pair_distances and (i, j) in prev_distances:
                        curr_x_dist, curr_y_dist = curr_pair_distances[(i, j)]
                        curr_dist = curr_x_dist + curr_y_dist
                        prev_dist_grid = prev_distances[(i, j)]  # grid坐标距离
                        prev_dist = prev_dist_grid * grid_size_val  # 转换为实际坐标距离
                        dist_diff = abs(curr_dist - prev_dist)
                        name_i = nodes[i].name if hasattr(nodes[i], 'name') else f"Chiplet_{i}"
                        name_j = nodes[j].name if hasattr(nodes[j], 'name') else f"Chiplet_{j}"
                        print(f"      ({i},{j}) [{name_i}, {name_j}]: 当前距离={curr_dist:.3f}, 之前距离={prev_dist:.3f}, 距离差={dist_diff:.3f} (< {min_pair_dist_diff:.3f})")
            if diff_pairs_with_info:
                print(f"    不同的chiplet对（距离差 >= 阈值 {min_pair_dist_diff:.3f}）:")
                for i, j, curr_dist, prev_dist, dist_diff in diff_pairs_with_info:
                    name_i = nodes[i].name if hasattr(nodes[i], 'name') else f"Chiplet_{i}"
                    name_j = nodes[j].name if hasattr(nodes[j], 'name') else f"Chiplet_{j}"
                    print(f"      ({i},{j}) [{name_i}, {name_j}]: 当前距离={curr_dist:.3f}, 之前距离={prev_dist:.3f}, 距离差={dist_diff:.3f} (>= {min_pair_dist_diff:.3f}, 满足阈值)")
            if not same_pairs and not diff_pairs_with_info:
                print(f"    (无数据)")
    else:
        print(f"\n(第一个解，无历史解可比较)")


def print_all_variables(
    ctx: ILPModelContext, 
    result: ILPPlacementResult,
    prev_pair_distances_list: Optional[List[Dict[Tuple[int, int], float]]] = None
) -> None:
    """
    打印所有变量的值，包括排除约束相关的变量。
    
    参数:
        ctx: ILP模型上下文
        result: 求解结果
        prev_pair_distances_list: 可选，之前所有解的chiplet对距离列表，用于显示对比信息
    """
    if result.status != "Optimal":
        return
    
    nodes = ctx.nodes
    n = len(nodes)
    
    print("\n" + "=" * 80)
    print("变量值详情")
    print("=" * 80)
    
    # 1. 坐标变量 (x, y)
    print("\n【坐标变量】")
    for k in range(n):
        x_val = get_var_value(ctx.x[k])
        y_val = get_var_value(ctx.y[k])
        node_name = nodes[k].name if hasattr(nodes[k], 'name') else f"Chiplet_{k}"
        print(f"  x[{k}] ({node_name}): {x_val}")
        print(f"  y[{k}] ({node_name}): {y_val}")
    
    # 2. 网格坐标变量 (x_grid, y_grid)
    print("\n【网格坐标变量】")
    for k in range(n):
        # 兼容 PuLP 和 Gurobi 的变量获取方式
        if hasattr(ctx.prob, 'variablesDict'):
            # PuLP
            x_grid_var = ctx.prob.variablesDict().get(f"x_grid_{k}")
            y_grid_var = ctx.prob.variablesDict().get(f"y_grid_{k}")
        elif hasattr(ctx.prob, 'getVarByName'):
            # Gurobi
            x_grid_var = ctx.prob.getVarByName(f"x_grid_{k}")
            y_grid_var = ctx.prob.getVarByName(f"y_grid_{k}")
        else:
            x_grid_var = None
            y_grid_var = None
        x_grid_val = get_var_value(x_grid_var)
        y_grid_val = get_var_value(y_grid_var)
        node_name = nodes[k].name if hasattr(nodes[k], 'name') else f"Chiplet_{k}"
        print(f"  x_grid[{k}] ({node_name}): {x_grid_val}")
        print(f"  y_grid[{k}] ({node_name}): {y_grid_val}")
    
    # 3. 旋转变量 (r)
    print("\n【旋转变量】")
    for k in range(n):
        r_val = get_var_value(ctx.r[k])
        rotated_str = "是" if (r_val is not None and r_val > 0.5) else "否"
        node_name = nodes[k].name if hasattr(nodes[k], 'name') else f"Chiplet_{k}"
        print(f"  r[{k}] ({node_name}): {r_val} (旋转: {rotated_str})")
    
    # 4. 宽度和高度变量 (w, h)
    print("\n【尺寸变量】")
    for k in range(n):
        # 兼容 PuLP 和 Gurobi 的变量获取方式
        if hasattr(ctx.prob, 'variablesDict'):
            # PuLP
            w_var = ctx.prob.variablesDict().get(f"w_{k}")
            h_var = ctx.prob.variablesDict().get(f"h_{k}")
        elif hasattr(ctx.prob, 'getVarByName'):
            # Gurobi
            w_var = ctx.prob.getVarByName(f"w_{k}")
            h_var = ctx.prob.getVarByName(f"h_{k}")
        else:
            w_var = None
            h_var = None
        w_val = get_var_value(w_var)
        h_val = get_var_value(h_var)
        node_name = nodes[k].name if hasattr(nodes[k], 'name') else f"Chiplet_{k}"
        print(f"  w[{k}] ({node_name}): {w_val}")
        print(f"  h[{k}] ({node_name}): {h_val}")
    
    # 5. 中心坐标变量 (cx, cy)
    if hasattr(ctx, 'cx') and ctx.cx is not None:
        print("\n【中心坐标变量】")
        for k in range(n):
            cx_val = get_var_value(ctx.cx[k])
            cy_val = get_var_value(ctx.cy[k])
            node_name = nodes[k].name if hasattr(nodes[k], 'name') else f"Chiplet_{k}"
            print(f"  cx[{k}] ({node_name}): {cx_val}")
            print(f"  cy[{k}] ({node_name}): {cy_val}")
    
    # 6. 相邻方式变量 (z1, z2, z1L, z1R, z2D, z2U)
    # 兼容新旧格式：如果有 connected_pairs 使用它，否则使用 silicon_bridge_pairs
    connected_pairs = getattr(ctx, 'connected_pairs', None)
    if connected_pairs is None:
        # 新格式：使用 silicon_bridge_pairs（只有它们有相邻约束变量）
        connected_pairs = getattr(ctx, 'silicon_bridge_pairs', [])
    
    if len(connected_pairs) > 0:
        print("\n【相邻方式变量】")
        for i, j in connected_pairs:
            name_i = nodes[i].name if hasattr(nodes[i], 'name') else f"Chiplet_{i}"
            name_j = nodes[j].name if hasattr(nodes[j], 'name') else f"Chiplet_{j}"
            z1_val = get_var_value(ctx.z1.get((i, j))) if (i, j) in ctx.z1 else None
            z2_val = get_var_value(ctx.z2.get((i, j))) if (i, j) in ctx.z2 else None
            z1L_val = get_var_value(ctx.z1L.get((i, j))) if (i, j) in ctx.z1L else None
            z1R_val = get_var_value(ctx.z1R.get((i, j))) if (i, j) in ctx.z1R else None
            z2D_val = get_var_value(ctx.z2D.get((i, j))) if (i, j) in ctx.z2D else None
            z2U_val = get_var_value(ctx.z2U.get((i, j))) if (i, j) in ctx.z2U else None
            print(f"  模块对 ({name_i}, {name_j}):")
            print(f"    z1[{i},{j}] (水平相邻): {z1_val}")
            print(f"    z2[{i},{j}] (垂直相邻): {z2_val}")
            if z1_val is not None and z1_val > 0.5:
                print(f"      z1R[{i},{j}] (i在右): {z1R_val}")
            if z2_val is not None and z2_val > 0.5:
                print(f"      z2D[{i},{j}] (i在下): {z2D_val}")
                print(f"      z2U[{i},{j}] (i在上): {z2U_val}")
    
    # 7. 非重叠约束变量 (p_left, p_right, p_down, p_up)
    print("\n【非重叠约束变量】")
    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append((i, j))
    
    for i, j in all_pairs:
        name_i = nodes[i].name if hasattr(nodes[i], 'name') else f"Chiplet_{i}"
        name_j = nodes[j].name if hasattr(nodes[j], 'name') else f"Chiplet_{j}"
        # 兼容 PuLP 和 Gurobi 的变量获取方式
        if hasattr(ctx.prob, 'variablesDict'):
            # PuLP
            p_left_var = ctx.prob.variablesDict().get(f"p_left_{i}_{j}")
            p_right_var = ctx.prob.variablesDict().get(f"p_right_{i}_{j}")
            p_down_var = ctx.prob.variablesDict().get(f"p_down_{i}_{j}")
            p_up_var = ctx.prob.variablesDict().get(f"p_up_{i}_{j}")
        elif hasattr(ctx.prob, 'getVarByName'):
            # Gurobi
            p_left_var = ctx.prob.getVarByName(f"p_left_{i}_{j}")
            p_right_var = ctx.prob.getVarByName(f"p_right_{i}_{j}")
            p_down_var = ctx.prob.getVarByName(f"p_down_{i}_{j}")
            p_up_var = ctx.prob.getVarByName(f"p_up_{i}_{j}")
        else:
            p_left_var = p_right_var = p_down_var = p_up_var = None
        
        p_left_val = get_var_value(p_left_var)
        p_right_val = get_var_value(p_right_var)
        p_down_val = get_var_value(p_down_var)
        p_up_val = get_var_value(p_up_var)
        
        print(f"  模块对 ({name_i}, {name_j}):")
        print(f"    p_left[{i},{j}]: {p_left_val}")
        print(f"    p_right[{i},{j}]: {p_right_val}")
        print(f"    p_down[{i},{j}]: {p_down_val}")
        print(f"    p_up[{i},{j}]: {p_up_val}")
    
    # 8. 边界框变量
    print("\n【边界框变量】")
    bbox_w_val = get_var_value(ctx.bbox_w)
    bbox_h_val = get_var_value(ctx.bbox_h)
    print(f"  bbox_w: {bbox_w_val}")
    print(f"  bbox_h: {bbox_h_val}")
    
    # 9. 其他辅助变量（shared_x, shared_y, dx_abs, dy_abs, bbox_min/max等）
    print("\n【其他辅助变量】")
    other_vars = []
    # 兼容 PuLP 和 Gurobi 的变量获取方式
    if hasattr(ctx.prob, 'variablesDict'):
        # PuLP
        var_dict = ctx.prob.variablesDict()
    elif hasattr(ctx.prob, 'getVars'):
        # Gurobi
        var_dict = {var.VarName: var for var in ctx.prob.getVars()}
    else:
        var_dict = {}
    
    for var_name, var in var_dict.items():
        if var_name.startswith("shared_") or var_name.startswith("dx_abs_") or \
           var_name.startswith("dy_abs_") or var_name.startswith("bbox_") or \
           var_name.startswith("bbox_area_proxy"):
            # 排除排除约束相关的变量（这些会在后面单独打印）
            if not (var_name.startswith("dx_abs_pair_") or var_name.startswith("dy_abs_pair_") or \
                    var_name.startswith("dx_grid_abs_pair_") or var_name.startswith("dy_grid_abs_pair_")):
                val = get_var_value(var)
                if val is not None:
                    other_vars.append((var_name, val))
    
    if other_vars:
        for var_name, val in sorted(other_vars):
            print(f"  {var_name}: {val}")
    else:
        print("  (无)")
    
    # 10. 排除解约束相关变量和约束（仅在第二次及以后的求解中打印）
    exclude_vars = []
    # 收集所有排除解约束相关的变量，包括所有可能的变量名模式
    # 兼容 PuLP 和 Gurobi 的变量获取方式
    if hasattr(ctx.prob, 'variablesDict'):
        # PuLP
        var_dict = ctx.prob.variablesDict()
    elif hasattr(ctx.prob, 'getVars'):
        # Gurobi
        var_dict = {var.VarName: var for var in ctx.prob.getVars()}
    else:
        var_dict = {}
    
    for var_name, var in var_dict.items():
        # 检查是否是排除解约束相关的变量
        is_exclude_var = (
            var_name.startswith("dx_abs_pair_") or 
            var_name.startswith("dy_abs_pair_") or 
            var_name.startswith("dx_grid_abs_pair_") or 
            var_name.startswith("dy_grid_abs_pair_") or 
            var_name.startswith("dist_curr_pair_") or 
            var_name.startswith("dist_diff_pair_") or 
            var_name.startswith("dist_diff_abs_pair_") or 
            var_name.startswith("diff_dist_pair_") or 
            var_name.startswith("same_dist_pair_")
        )
        if is_exclude_var:
            val = get_var_value(var)
            # 即使值为None也记录，以便调试
            exclude_vars.append((var_name, val))
    
    if exclude_vars:
        print("\n" + "=" * 80)
        print("排除解约束相关变量和约束")
        print("=" * 80)
        
        # 10.1 打印排除约束相关的变量
        print("\n【排除约束变量】")
        
        # 按变量类型分组
        dx_abs_pair_vars = []
        dy_abs_pair_vars = []
        dx_grid_abs_pair_vars = []
        dy_grid_abs_pair_vars = []
        dist_curr_pair_vars = []
        dist_diff_pair_vars = []
        dist_diff_abs_pair_vars = []
        diff_dist_pair_vars = []
        same_dist_pair_vars = []
        
        for var_name, val in exclude_vars:
            if var_name.startswith("dx_grid_abs_pair_"):
                dx_grid_abs_pair_vars.append((var_name, val))
            elif var_name.startswith("dy_grid_abs_pair_"):
                dy_grid_abs_pair_vars.append((var_name, val))
            elif var_name.startswith("dist_curr_pair_"):
                dist_curr_pair_vars.append((var_name, val))
            elif var_name.startswith("dx_abs_pair_"):
                dx_abs_pair_vars.append((var_name, val))
            elif var_name.startswith("dy_abs_pair_"):
                dy_abs_pair_vars.append((var_name, val))
            elif var_name.startswith("dist_diff_pair_") and not var_name.startswith("dist_diff_abs_pair_"):
                dist_diff_pair_vars.append((var_name, val))
            elif var_name.startswith("dist_diff_abs_pair_"):
                dist_diff_abs_pair_vars.append((var_name, val))
            elif var_name.startswith("diff_dist_pair_"):
                diff_dist_pair_vars.append((var_name, val))
            elif var_name.startswith("same_dist_pair_"):
                same_dist_pair_vars.append((var_name, val))
        
        if dx_grid_abs_pair_vars:
            print("\n  dx_grid_abs_pair (chiplet对的x方向grid坐标距离绝对值):")
            for var_name, val in sorted(dx_grid_abs_pair_vars):
                print(f"    {var_name}: {val}")
        
        if dy_grid_abs_pair_vars:
            print("\n  dy_grid_abs_pair (chiplet对的y方向grid坐标距离绝对值):")
            for var_name, val in sorted(dy_grid_abs_pair_vars):
                print(f"    {var_name}: {val}")
        
        # 按chiplet对组织显示，使输出更清晰
        import re
        pair_info = {}  # key: (i, j), value: dict with all related vars
        
        # 解析所有变量，按chiplet对分组
        unmatched_vars = []  # 记录无法匹配的变量
        for var_name, val in exclude_vars:
            # 匹配模式：{prefix}_{suffix}_{i}_{j} 或 {prefix}_{suffix}_{i}_{j}_prev{prev_idx}
            # 注意：变量名可能是 dist_diff_abs_pair_{suffix}_{i}_{j}_prev{prev_idx}
            match = re.search(r'([^_]+(?:_[^_]+)*)_[^_]+_(\d+)_(\d+)(?:_prev(\d+))?', var_name)
            if match:
                prefix = match.group(1)
                i_val = int(match.group(2))
                j_val = int(match.group(3))
                prev_idx = match.group(4)
                pair_key = (i_val, j_val)
                
                if pair_key not in pair_info:
                    pair_info[pair_key] = {
                        'dx_grid_abs': None,
                        'dy_grid_abs': None,
                        'dist_curr': None,
                        'dist_diff': {},
                        'dist_diff_abs': {},
                        'diff_dist': None,
                        'same_dist': {}
                    }
                
                # 处理各种变量前缀
                if prefix == 'dx_grid_abs_pair':
                    pair_info[pair_key]['dx_grid_abs'] = val
                elif prefix == 'dy_grid_abs_pair':
                    pair_info[pair_key]['dy_grid_abs'] = val
                elif prefix == 'dist_curr_pair':
                    pair_info[pair_key]['dist_curr'] = val
                elif prefix == 'dist_diff_pair' and prev_idx:
                    pair_info[pair_key]['dist_diff'][int(prev_idx)] = val
                elif prefix == 'dist_diff_abs_pair' and prev_idx:
                    pair_info[pair_key]['dist_diff_abs'][int(prev_idx)] = val
                elif prefix == 'diff_dist_pair':
                    pair_info[pair_key]['diff_dist'] = val
                elif prefix == 'same_dist_pair' and prev_idx:
                    pair_info[pair_key]['same_dist'][int(prev_idx)] = val
                else:
                    # 无法匹配的变量，记录到unmatched_vars
                    unmatched_vars.append((var_name, val))
            else:
                # 无法解析的变量，记录到unmatched_vars
                unmatched_vars.append((var_name, val))
        
        # 按chiplet对显示详细信息
        if pair_info:
            print("\n  【按chiplet对分组显示】")
            for (i, j) in sorted(pair_info.keys()):
                info = pair_info[(i, j)]
                name_i = nodes[i].name if hasattr(nodes[i], 'name') and i < len(nodes) else f"Chiplet_{i}"
                name_j = nodes[j].name if hasattr(nodes[j], 'name') and j < len(nodes) else f"Chiplet_{j}"
                
                print(f"\n    模块对 ({name_i}, {name_j}) [索引: ({i}, {j})]:")
                
                if info['dx_grid_abs'] is not None:
                    print(f"      dx_grid_abs (x方向grid距离): {info['dx_grid_abs']:.2f}")
                if info['dy_grid_abs'] is not None:
                    print(f"      dy_grid_abs (y方向grid距离): {info['dy_grid_abs']:.2f}")
                if info['dist_curr'] is not None:
                    print(f"      dist_curr (当前距离，grid单位): {info['dist_curr']:.2f}")
                    print(f"        验证: dx_grid_abs + dy_grid_abs = {info['dx_grid_abs']:.2f} + {info['dy_grid_abs']:.2f} = {info['dx_grid_abs'] + info['dy_grid_abs']:.2f}")
                
                if info['dist_diff'] or info['dist_diff_abs']:
                    print(f"      与之前解的距离比较:")
                    for prev_idx in sorted(set(list(info['dist_diff'].keys()) + list(info['dist_diff_abs'].keys()))):
                        dist_diff = info['dist_diff'].get(prev_idx, None)
                        dist_diff_abs = info['dist_diff_abs'].get(prev_idx, None)
                        same_dist = info['same_dist'].get(prev_idx, None)
                        
                        # 显示之前解的距离（如果可用）
                        prev_dist = None
                        if prev_pair_distances_list and prev_idx < len(prev_pair_distances_list):
                            prev_dist = prev_pair_distances_list[prev_idx].get((i, j), None)
                        
                        print(f"        解 {prev_idx}:")
                        if prev_dist is not None:
                            print(f"          之前解的距离: {prev_dist:.2f} (grid单位)")
                        if info['dist_curr'] is not None:
                            print(f"          当前解的距离: {info['dist_curr']:.2f} (grid单位)")
                        if dist_diff is not None:
                            print(f"          距离差 (dist_diff): {dist_diff:.2f}")
                            if prev_dist is not None and info['dist_curr'] is not None:
                                print(f"            验证: {info['dist_curr']:.2f} - {prev_dist:.2f} = {dist_diff:.2f}")
                        if dist_diff_abs is not None:
                            print(f"          距离差绝对值 (dist_diff_abs): {dist_diff_abs:.2f}")
                        if same_dist is not None:
                            same_str = "是" if same_dist > 0.5 else "否"
                            print(f"          是否相同 (same_dist_pair): {same_dist} ({same_str})")
                            if dist_diff_abs is not None:
                                if same_dist > 0.5:
                                    print(f"            → 距离差 {dist_diff_abs:.2f} < 阈值，标记为相同")
                                else:
                                    print(f"            → 距离差 {dist_diff_abs:.2f} >= 阈值，标记为不同")
                
                if info['diff_dist'] is not None:
                    diff_str = "是" if info['diff_dist'] > 0.5 else "否"
                    print(f"      diff_dist_pair (与所有之前解都不同): {info['diff_dist']} ({diff_str})")
                    if info['diff_dist'] > 0.5:
                        print(f"        → 该chiplet对的距离与所有之前解都不同，满足排除约束")
        
        # 保留原有的详细变量列表输出（作为补充）
        if dist_curr_pair_vars:
            print("\n  【详细变量列表 - dist_curr_pair】")
            for var_name, val in sorted(dist_curr_pair_vars):
                print(f"    {var_name}: {val:.2f}")
        
        if dx_abs_pair_vars:
            print("\n  【详细变量列表 - dx_abs_pair (旧版本)】")
            for var_name, val in sorted(dx_abs_pair_vars):
                print(f"    {var_name}: {val:.2f}")
        
        if dy_abs_pair_vars:
            print("\n  【详细变量列表 - dy_abs_pair (旧版本)】")
            for var_name, val in sorted(dy_abs_pair_vars):
                print(f"    {var_name}: {val:.2f}")
        
        if dist_diff_pair_vars:
            print("\n  【详细变量列表 - dist_diff_pair】")
            for var_name, val in sorted(dist_diff_pair_vars):
                print(f"    {var_name}: {val:.2f}")
        
        if dist_diff_abs_pair_vars:
            print("\n  【详细变量列表 - dist_diff_abs_pair】")
            for var_name, val in sorted(dist_diff_abs_pair_vars):
                if val is not None:
                    print(f"    {var_name}: {val:.2f}")
                else:
                    print(f"    {var_name}: None (未求解)")
        
        if diff_dist_pair_vars:
            print("\n  【详细变量列表 - diff_dist_pair (二进制)】")
            for var_name, val in sorted(diff_dist_pair_vars):
                if val is not None:
                    diff_str = "是" if val > 0.5 else "否"
                    print(f"    {var_name}: {val} ({diff_str})")
                else:
                    print(f"    {var_name}: None (未求解)")
        
        # 打印所有其他排除解约束相关的变量（包括无法匹配的）
        if unmatched_vars:
            print("\n  【其他排除解约束相关变量（未在分组中显示）】")
            for var_name, val in sorted(unmatched_vars):
                if val is not None:
                    print(f"    {var_name}: {val}")
                else:
                    print(f"    {var_name}: None (未求解)")
        
        # 打印所有排除解约束相关变量的完整列表（用于调试）
        print("\n  【完整变量列表（所有排除解约束相关变量）】")
        for var_name, val in sorted(exclude_vars):
            if val is not None:
                # 根据变量类型格式化输出
                if var_name.startswith("diff_dist_pair_") or var_name.startswith("same_dist_pair_"):
                    # 二进制变量
                    binary_str = "是" if val > 0.5 else "否"
                    print(f"    {var_name}: {val} ({binary_str})")
                elif isinstance(val, (int, float)):
                    # 数值变量
                    print(f"    {var_name}: {val:.4f}")
                else:
                    print(f"    {var_name}: {val}")
            else:
                print(f"    {var_name}: None (未求解)")
        
        if same_dist_pair_vars:
            print("\n  same_dist_pair (chiplet对的距离是否与某个之前解相同，二进制变量):")
            # 按chiplet对和之前解索引分组显示
            same_dist_by_pair = {}
            import re
            for var_name, val in same_dist_pair_vars:
                # 解析变量名：same_dist_pair_{suffix}_{i}_{j}_prev{prev_idx}
                # 使用正则表达式匹配：same_dist_pair_*_数字_数字_prev数字
                match = re.search(r'same_dist_pair_[^_]+_(\d+)_(\d+)_prev(\d+)', var_name)
                if match:
                    i_val = int(match.group(1))
                    j_val = int(match.group(2))
                    prev_idx = int(match.group(3))
                    pair_key = (i_val, j_val, prev_idx)
                    if pair_key not in same_dist_by_pair:
                        same_dist_by_pair[pair_key] = []
                    same_dist_by_pair[pair_key].append((var_name, val))
                else:
                    # 如果正则匹配失败，直接显示变量名
                    if "unknown" not in same_dist_by_pair:
                        same_dist_by_pair["unknown"] = []
                    same_dist_by_pair["unknown"].append((var_name, val))
            
            # 按chiplet对和之前解索引排序显示
            for key, vars_list in sorted(same_dist_by_pair.items()):
                if key == "unknown":
                    print("    无法解析的变量:")
                    for var_name, val in sorted(vars_list):
                        print(f"      {var_name}: {val}")
                else:
                    i, j, prev_idx = key
                    name_i = nodes[i].name if hasattr(nodes[i], 'name') and i < len(nodes) else f"Chiplet_{i}"
                    name_j = nodes[j].name if hasattr(nodes[j], 'name') and j < len(nodes) else f"Chiplet_{j}"
                    print(f"    模块对 ({name_i}, {name_j}) 与解 {prev_idx}:")
                    for var_name, val in sorted(vars_list):
                        print(f"      {var_name}: {val}")
        
        # 10.2 打印排除约束相关的约束
        print("\n【排除约束】")
        exclude_constraints = []
        for constraint_name, constraint in ctx.prob.constraints.items():
            if constraint_name.startswith("dx_abs_pair_") or constraint_name.startswith("dy_abs_pair_") or \
               constraint_name.startswith("dx_grid_abs_pair_") or constraint_name.startswith("dy_grid_abs_pair_") or \
               constraint_name.startswith("dist_curr_pair_") or \
               constraint_name.startswith("dist_diff_pair_") or constraint_name.startswith("dist_diff_abs_pair_") or \
               constraint_name.startswith("exclude_solution_dist_pair_") or \
               constraint_name.startswith("same_dist_pair_") or constraint_name.startswith("diff_dist_pair_implies_") or \
               constraint_name.startswith("not_same_implies_") or constraint_name.startswith("all_not_same_implies_"):
                exclude_constraints.append(constraint_name)
        
        if exclude_constraints:
            print(f"  共找到 {len(exclude_constraints)} 个排除约束:")
            for constraint_name in sorted(exclude_constraints):
                constraint = ctx.prob.constraints[constraint_name]
                print(f"    {constraint_name}: {constraint}")
        else:
            print("  (未找到排除约束)")
    else:
        print("\n【排除解约束】")
        print("  (第一次求解，无排除约束)")
