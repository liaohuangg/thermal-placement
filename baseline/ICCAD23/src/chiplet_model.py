"""
芯片布局问题建模模块

这个模块定义了芯片布局问题的核心数据结构:
- Chiplet: 表示单个矩形芯片
- LayoutProblem: 表示完整的布局问题，包含所有芯片和它们的连接关系
"""

import json
import networkx as nx
from typing import Dict, List, Tuple, Optional

# ==================== 全局配置变量 ====================
# 最小重叠长度：相邻芯片共享边的最小长度
MIN_OVERLAP = 1

# 浮点数比较的容差，用于处理浮点数精度问题
EPSILON = 1e-9


class Chiplet:
    """
    表示一个矩形芯片块
    
    Attributes:
        id (str): 芯片的唯一标识符
        width (float): 芯片的宽度
        height (float): 芯片的高度
        x (float): 芯片左下角的x坐标
        y (float): 芯片左下角的y坐标
    """
    
    def __init__(self, chip_id: str, width: float, height: float, 
                 x: float = 0.0, y: float = 0.0, thermal_power: float = 0.0):
        """
        初始化一个Chiplet实例
        
        Args:
            chip_id: 芯片的唯一ID
            width: 芯片宽度
            height: 芯片高度
            x: 左下角x坐标，默认为0
            y: 左下角y坐标，默认为0
        """
        self.id = chip_id
        self.width = width
        self.height = height
        self.x = x
        self.y = y
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        获取芯片的边界框
        
        Returns:
            (x_min, y_min, x_max, y_max): 芯片的边界坐标
        """
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def __repr__(self) -> str:
        """返回芯片的字符串表示"""
        return (f"Chiplet(id='{self.id}', width={self.width}, height={self.height}, "
                f"x={self.x}, y={self.y})")


class LayoutProblem:
    """
    表示完整的芯片布局问题
    
    Attributes:
        chiplets (Dict[str, Chiplet]): 所有芯片的字典，键为芯片ID
        connection_graph (nx.Graph): 表示芯片之间连接关系的无向图
    """
    
    def __init__(self):
        """初始化一个空的布局问题"""
        self.chiplets: Dict[str, Chiplet] = {}
        self.connection_graph: nx.Graph = nx.Graph()
    
    def add_chiplet(self, chiplet: Chiplet) -> None:
        """
        添加一个芯片到问题中
        
        Args:
            chiplet: 要添加的Chiplet对象
        """
        self.chiplets[chiplet.id] = chiplet
        self.connection_graph.add_node(chiplet.id)
    
    def add_connection(self, chip_id1: str, chip_id2: str, weight: float = 1.0) -> None:
        """
        在两个芯片之间添加连接关系
        
        Args:
            chip_id1: 第一个芯片的ID
            chip_id2: 第二个芯片的ID
            weight: 连接的权重，默认为1.0
        """
        if chip_id1 in self.chiplets and chip_id2 in self.chiplets:
            self.connection_graph.add_edge(chip_id1, chip_id2, weight=weight)
        else:
            raise ValueError(f"芯片 {chip_id1} 或 {chip_id2} 不存在")
    
    def get_chiplet(self, chip_id: str) -> Optional[Chiplet]:
        """
        根据ID获取芯片
        
        Args:
            chip_id: 芯片的ID
            
        Returns:
            对应的Chiplet对象，如果不存在则返回None
        """
        return self.chiplets.get(chip_id)
    
    def get_neighbors(self, chip_id: str) -> List[str]:
        """
        获取与指定芯片相连的所有芯片ID
        
        Args:
            chip_id: 芯片的ID
            
        Returns:
            相邻芯片的ID列表
        """
        if chip_id in self.connection_graph:
            return list(self.connection_graph.neighbors(chip_id))
        return []
    
    def get_connection_weight(self, chip_id1: str, chip_id2: str) -> Optional[float]:
        """
        获取两个芯片之间的连接权重
        
        Args:
            chip_id1: 第一个芯片的ID
            chip_id2: 第二个芯片的ID
            
        Returns:
            连接的权重，如果不存在连接则返回None
        """
        if self.connection_graph.has_edge(chip_id1, chip_id2):
            return self.connection_graph[chip_id1][chip_id2].get('weight', 1.0)
        return None
    
    def __repr__(self) -> str:
        """返回布局问题的字符串表示"""
        return (f"LayoutProblem(chiplets={len(self.chiplets)}, "
                f"connections={self.connection_graph.number_of_edges()})")
    def is_cyclic(self) -> bool:
        """
        检查连接图是否包含环
        
        Returns:
            如果连接图包含环则返回True，否则返回False
        """
        try:
            cycles = nx.find_cycle(self.connection_graph)
            return True
        except nx.exception.NetworkXNoCycle:
            return False


def load_problem_from_json(file_path: str) -> LayoutProblem:
    """
    从JSON文件加载芯片布局问题
    
    JSON文件格式应为:
    {
        "dies": [
            {"id": "chip1", "width": 10, "height": 20},
            {"id": "chip2", "width": 15, "height": 25},
            ...
        ],
        "connections": [
            ["chip1", "chip2"],
            ["chip2", "chip3"],
            ...
        ]
    }
    
    Args:
        file_path: JSON文件的路径
        
    Returns:
        加载好数据的LayoutProblem实例
        
    Raises:
        FileNotFoundError: 如果文件不存在
        json.JSONDecodeError: 如果JSON格式不正确
        KeyError: 如果JSON缺少必要的字段
    """
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建LayoutProblem实例
    problem = LayoutProblem()
    
    # 添加所有芯片
    if 'dies' not in data:
        raise KeyError("JSON文件必须包含 'dies' 字段")
    
    for die_data in data['dies']:
        # 验证必要字段
        if 'id' not in die_data or 'width' not in die_data or 'height' not in die_data:
            raise KeyError("每个die必须包含 'id', 'width', 和 'height' 字段")
        
        # 创建Chiplet对象
        chiplet = Chiplet(
            chip_id=die_data['id'],
            width=float(die_data['width']),
            height=float(die_data['height']),
            x=float(die_data.get('x', 0.0)),
            y=float(die_data.get('y', 0.0))
        )
        problem.add_chiplet(chiplet)
    
    # 添加所有连接
    if 'connections' in data:
        for connection in data['connections']:
            # 支持两种格式：[chip1, chip2] 或 [chip1, chip2, weight]
            if len(connection) == 2:
                problem.add_connection(connection[0], connection[1])
            elif len(connection) == 3:
                problem.add_connection(connection[0], connection[1], float(connection[2]))
            else:
                raise ValueError("每个连接必须是包含2个或3个元素的列表: [chip1, chip2] 或 [chip1, chip2, weight]")
    
    return problem


# ==================== 布局验证函数 ====================

def has_overlap(chip1: Chiplet, chip2: Chiplet) -> bool:
    """
    检查两个芯片是否存在重叠
    
    两个矩形重叠当且仅当它们在X轴和Y轴上都有重叠。
    使用EPSILON来处理浮点数精度问题。
    
    Args:
        chip1: 第一个芯片
        chip2: 第二个芯片
        
    Returns:
        如果两个芯片有重叠则返回True，否则返回False
    """
    x1_min, y1_min, x1_max, y1_max = chip1.get_bounds()
    x2_min, y2_min, x2_max, y2_max = chip2.get_bounds()
    
    # 检查X轴重叠：两个区间 [x1_min, x1_max] 和 [x2_min, x2_max] 是否重叠
    # 不重叠的条件是：x1_max <= x2_min 或 x2_max <= x1_min
    # 重叠的条件是上述条件的否定
    x_overlap = not (x1_max <= x2_min + EPSILON or x2_max <= x1_min + EPSILON)
    
    # 检查Y轴重叠
    y_overlap = not (y1_max <= y2_min + EPSILON or y2_max <= y1_min + EPSILON)
    
    # 只有当X轴和Y轴都重叠时，两个矩形才真正重叠
    return x_overlap and y_overlap


def get_adjacency_info(chip1: Chiplet, chip2: Chiplet) -> Tuple[bool, float, str]:
    """
    检查两个芯片是否邻接，并计算它们共享边的长度
    
    两个芯片邻接的定义：
    1. 它们的边在X轴或Y轴上精确接触（边对齐）
    2. 它们在该轴上有足够的重叠（共享边长度 >= MIN_OVERLAP）
    3. 它们在垂直于接触边的方向上没有间隙
    
    Args:
        chip1: 第一个芯片
        chip2: 第二个芯片
        
    Returns:
        (is_adjacent, overlap_length, direction): 
        - is_adjacent: 是否邻接
        - overlap_length: 共享边的长度
        - direction: 邻接方向 ('left', 'right', 'top', 'bottom', 'none')
    """
    x1_min, y1_min, x1_max, y1_max = chip1.get_bounds()
    x2_min, y2_min, x2_max, y2_max = chip2.get_bounds()
    
    # 检查四个可能的邻接方向
    
    # 1. chip1 在 chip2 左边 (chip1的右边 == chip2的左边)
    if abs(x1_max - x2_min) < EPSILON:
        # 计算Y轴上的重叠长度
        y_overlap_start = max(y1_min, y2_min)
        y_overlap_end = min(y1_max, y2_max)
        overlap_length = y_overlap_end - y_overlap_start
        if overlap_length >= MIN_OVERLAP - EPSILON:
            return True, overlap_length, 'right'
    
    # 2. chip1 在 chip2 右边 (chip1的左边 == chip2的右边)
    if abs(x1_min - x2_max) < EPSILON:
        y_overlap_start = max(y1_min, y2_min)
        y_overlap_end = min(y1_max, y2_max)
        overlap_length = y_overlap_end - y_overlap_start
        if overlap_length >= MIN_OVERLAP - EPSILON:
            return True, overlap_length, 'left'
    
    # 3. chip1 在 chip2 下面 (chip1的上边 == chip2的下边)
    if abs(y1_max - y2_min) < EPSILON:
        # 计算X轴上的重叠长度
        x_overlap_start = max(x1_min, x2_min)
        x_overlap_end = min(x1_max, x2_max)
        overlap_length = x_overlap_end - x_overlap_start
        if overlap_length >= MIN_OVERLAP - EPSILON:
            return True, overlap_length, 'top'
    
    # 4. chip1 在 chip2 上面 (chip1的下边 == chip2的上边)
    if abs(y1_min - y2_max) < EPSILON:
        x_overlap_start = max(x1_min, x2_min)
        x_overlap_end = min(x1_max, x2_max)
        overlap_length = x_overlap_end - x_overlap_start
        if overlap_length >= MIN_OVERLAP - EPSILON:
            return True, overlap_length, 'bottom'
    
    # 没有邻接
    return False, 0.0, 'none'


def is_layout_valid(layout: Dict[str, Chiplet], problem: LayoutProblem, 
                    verbose: bool = False) -> bool:
    """
    验证给定的布局方案是否满足所有物理规则
    
    检查规则：
    1. 无重叠规则：任何两个芯片的矩形区域都不能有任何重叠
    2. 连接规则：对于连接图中的每条边，对应的两个芯片必须：
       a) 物理上邻接（边缘精确接触）
       b) 共享边的长度 >= MIN_OVERLAP
    
    Args:
        layout: 芯片布局字典，格式为 {chip_id: chip_object}
        problem: 布局问题实例，包含连接图等规则信息
        verbose: 是否输出详细的验证信息（用于调试）
        
    Returns:
        如果布局满足所有规则返回True，否则返回False
    """
    chip_list = list(layout.values())
    n = len(chip_list)
    
    if verbose:
        print(f"开始验证布局，共 {n} 个芯片")
        print("=" * 60)
    
    # ===== 规则1: 检查无重叠规则 =====
    if verbose:
        print("\n检查规则1: 无重叠规则")
        print("-" * 60)
    
    for i in range(n):
        for j in range(i + 1, n):
            chip1 = chip_list[i]
            chip2 = chip_list[j]
            
            if has_overlap(chip1, chip2):
                if verbose:
                    print(f"✗ 发现重叠: {chip1.id} 和 {chip2.id}")
                    print(f"  {chip1.id} 边界: {chip1.get_bounds()}")
                    print(f"  {chip2.id} 边界: {chip2.get_bounds()}")
                return False
    
    if verbose:
        print(f"✓ 通过: 所有 {n*(n-1)//2} 对芯片都无重叠")
    
    # ===== 规则2: 检查连接规则 =====
    if verbose:
        print("\n检查规则2: 连接规则")
        print("-" * 60)
    
    connection_count = 0
    for edge in problem.connection_graph.edges():
        chip_id1, chip_id2 = edge
        
        # 确保两个芯片都在布局中
        if chip_id1 not in layout or chip_id2 not in layout:
            if verbose:
                print(f"✗ 错误: 连接的芯片不在布局中: {chip_id1} - {chip_id2}")
            return False
        
        chip1 = layout[chip_id1]
        chip2 = layout[chip_id2]
        
        # 检查是否邻接以及共享边长度
        is_adjacent, overlap_length, direction = get_adjacency_info(chip1, chip2)
        
        connection_count += 1
        
        if not is_adjacent:
            if verbose:
                print(f"✗ 连接 {chip_id1} - {chip_id2} 失败: 芯片不邻接")
                print(f"  {chip_id1} 边界: {chip1.get_bounds()}")
                print(f"  {chip_id2} 边界: {chip2.get_bounds()}")
            return False
        
        if overlap_length < MIN_OVERLAP - EPSILON:
            if verbose:
                print(f"✗ 连接 {chip_id1} - {chip_id2} 失败: 共享边长度不足")
                print(f"  共享边长度: {overlap_length:.6f}, 最小要求: {MIN_OVERLAP}")
            return False
        
        if verbose:
            print(f"✓ 连接 {chip_id1} - {chip_id2}: 方向={direction}, "
                  f"共享长度={overlap_length:.2f}")
    
    if verbose:
        print(f"\n✓ 通过: 所有 {connection_count} 个连接都满足要求")
        print("=" * 60)
        print("✓ 布局验证通过!")
    
    return True


if __name__ == "__main__":
    # 简单测试
    print("芯片布局问题建模模块")
    print("=" * 50)
    
    # 创建一个简单的示例
    problem = LayoutProblem()
    
    # 添加芯片
    chip1 = Chiplet("A", 10, 20)
    chip2 = Chiplet("B", 15, 25)
    chip3 = Chiplet("C", 12, 18)
    
    
    problem.add_chiplet(chip1)
    problem.add_chiplet(chip2)
    problem.add_chiplet(chip3)
    print(problem.get_chiplet(chip_id="B"))
    
    # 添加连接
    problem.add_connection("A", "B")
    problem.add_connection("B", "C")
    
    print(f"问题: {problem}")
    print(f"\n芯片列表:")
    for chip_id, chiplet in problem.chiplets.items():
        print(f"  {chiplet}")
    
    print(f"\n连接关系:")
    for chip_id in problem.chiplets:
        neighbors = problem.get_neighbors(chip_id)
        print(f"  {chip_id} 连接到: {neighbors}")