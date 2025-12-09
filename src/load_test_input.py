"""
从测试输入JSON文件加载chiplet和连接信息，转换为ChipletNode对象和边列表。

参考格式：
- 输入JSON包含 `chiplets` 和 `connections` 字段
- `chiplets`: 数组，每个元素包含 `name`, `width`, `height`
- `connections`: 数组，每个元素是 `[src, dst]` 或 `[src, dst, {weight: ...}]`
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from tool import ChipletNode


def load_test_input_json(json_path: str) -> Dict[str, Any]:
    """
    加载测试输入JSON文件。
    
    参数
    ----
    json_path:
        JSON文件路径
        
    返回
    ----
    dict
        包含 `chiplets` 和 `connections` 字段的字典
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_chiplet_nodes_from_json(json_data: Dict[str, Any]) -> List[ChipletNode]:
    """
    从JSON数据构建ChipletNode对象列表。
    
    参数
    ----
    json_data:
        包含 `chiplets` 字段的字典
        
    返回
    ----
    List[ChipletNode]
        ChipletNode对象列表
    """
    nodes = []
    
    for chiplet_info in json_data.get("chiplets", []):
        name = chiplet_info.get("name", "")
        width = chiplet_info.get("width", 0.0)
        height = chiplet_info.get("height", 0.0)
        
        # 构建dimensions字典（使用x和y对应width和height）
        dimensions = {
            "x": width,
            "y": height
        }
        
        # 构建ChipletNode对象
        node = ChipletNode(
            name=name,
            dimensions=dimensions,
            phys=[],  # 测试输入中没有phys信息，使用空列表
            power=0.0  # 测试输入中没有power信息，使用0.0
        )
        nodes.append(node)
    
    return nodes


def build_edges_from_json(json_data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    从JSON数据构建边列表。
    
    参数
    ----
    json_data:
        包含 `connections` 字段的字典
        
    返回
    ----
    List[Tuple[str, str]]
        边列表，每个元素是 (src, dst) 元组
    """
    edges = []
    
    for conn in json_data.get("connections", []):
        if len(conn) >= 2:
            src = conn[0]
            dst = conn[1]
            edges.append((src, dst))
    
    return edges


def load_test_case(json_path: str) -> Tuple[List[ChipletNode], List[Tuple[str, str]]]:
    """
    从测试输入JSON文件加载chiplet和连接信息。
    
    参数
    ----
    json_path:
        JSON文件路径
        
    返回
    ----
    Tuple[List[ChipletNode], List[Tuple[str, str]]]
        (chiplet节点列表, 边列表)
    """
    json_data = load_test_input_json(json_path)
    nodes = build_chiplet_nodes_from_json(json_data)
    edges = build_edges_from_json(json_data)
    
    return nodes, edges


if __name__ == "__main__":
    # 测试：加载5core.json
    test_file = Path(__file__).parent.parent / "baseline" / "ICCAD23" / "test_input" / "5core.json"
    nodes, edges = load_test_case(str(test_file))
    
    print(f"加载了 {len(nodes)} 个chiplet:")
    for node in nodes:
        print(f"  {node.name}: {node.dimensions.get('x', 0)} x {node.dimensions.get('y', 0)}")
    
    print(f"\n加载了 {len(edges)} 条边:")
    for src, dst in edges:
        print(f"  {src} -> {dst}")

