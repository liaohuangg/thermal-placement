
"""
硅桥合法性检测模块

检测芯片布局中硅桥的物理占用区域是否存在冲突。
硅桥是连接两个相邻chiplet的物理结构,占用空间为 MIN_OVERLAP × 硅桥长度。
硅桥重叠是指不同硅桥的矩形区域在空间中发生重叠。
"""

from typing import Dict, List, Tuple, Optional
from chiplet_model import Chiplet, LayoutProblem, get_adjacency_info, MIN_OVERLAP

# 硅桥的固定长度(沿着芯片边界方向)
SILICONBRIDGE_LENGTH = 1


class SiliconBridge:
    """
    硅桥数据结构
    
    表示连接两个相邻chiplet的物理硅桥结构。
    硅桥是一个矩形,尺寸为 bridge_width × SILICONBRIDGE_LENGTH。
    - bridge_width: 硅桥宽度(垂直于芯片边界方向),默认为MIN_OVERLAP,可调整
    - SILICONBRIDGE_LENGTH: 硅桥长度(跨越芯片边界方向),固定值
    
    硅桥默认位于两个chiplet重叠边的正中间,但可以沿着重叠边滑动,
    只要保证硅桥完全在重叠边范围内即可。
    """
    
    def __init__(self, chip1_id: str, chip2_id: str, chip1: Chiplet, chip2: Chiplet, 
                 bridge_width: Optional[float] = None):
        """
        初始化硅桥
        
        Args:
            chip1_id: 第一个芯片ID
            chip2_id: 第二个芯片ID
            chip1: 第一个芯片对象
            chip2: 第二个芯片对象
            bridge_width: 硅桥宽度(可选),默认为MIN_OVERLAP
        """
        self.chip1_id = chip1_id
        self.chip2_id = chip2_id
        
        # 检查两个芯片是否邻接
        is_adj, overlap_len, direction = get_adjacency_info(chip1, chip2)
        
        if not is_adj:
            raise ValueError(f"Chiplets {chip1_id} and {chip2_id} are not adjacent")
        
        # 设置硅桥宽度(默认为MIN_OVERLAP)
        self.bridge_width = bridge_width if bridge_width is not None else MIN_OVERLAP
        
        # 检查重叠边长度是否足够放置硅桥
        # 硅桥的宽度(沿重叠边方向)需要小于等于重叠边长度
        if overlap_len < self.bridge_width:
            raise ValueError(f"Overlap length {overlap_len:.2f} is too short for silicon bridge width {self.bridge_width:.2f}")
        
        self.direction = direction  # 从chip1的视角看chip2的方向
        self.bridge_length = SILICONBRIDGE_LENGTH  # 硅桥长度(固定值)
        
        # 计算重叠边的范围
        if direction in ['left', 'right']:
            # 垂直重叠边
            self.overlap_start = max(chip1.y, chip2.y)
            self.overlap_end = min(chip1.y + chip1.height, chip2.y + chip2.height)
        else:  # 'top' or 'bottom'
            # 水平重叠边
            self.overlap_start = max(chip1.x, chip2.x)
            self.overlap_end = min(chip1.x + chip1.width, chip2.x + chip2.width)
        
        # 硅桥中心位置(默认在重叠边正中间,可以调整)
        self.bridge_center = (self.overlap_start + self.overlap_end) / 2.0
        
        # 计算硅桥的矩形边界框(x_min, y_min, x_max, y_max)
        self._compute_bounding_box(chip1, chip2)
    
    def _compute_bounding_box(self, chip1: Chiplet, chip2: Chiplet):
        """计算硅桥的矩形边界框
        
        硅桥跨越两个芯片之间的间隙:
        - 垂直于边界方向: 硅桥宽度为bridge_width,居中放置
        - 沿着边界方向: 硅桥长度为SILICONBRIDGE_LENGTH,居中放置在重叠边中心
        """
        # 硅桥沿重叠边方向的范围(居中放置)
        bridge_half_length = self.bridge_length / 2.0
        bridge_start = self.bridge_center - bridge_half_length
        bridge_end = self.bridge_center + bridge_half_length
        
        # 硅桥垂直于边界方向的宽度(均匀分布在两芯片之间)
        half_width = self.bridge_width / 2.0
        
        if self.direction == 'right':
            # chip1在左,chip2在右 (水平相邻)
            # 硅桥在x方向(水平方向)上长度为SILICONBRIDGE_LENGTH,居中在边界处
            # 硅桥在y方向(垂直方向)上宽度为MIN_OVERLAP,居中在重叠边中心
            boundary = chip1.x + chip1.width
            self.x_min = boundary - bridge_half_length
            self.x_max = boundary + bridge_half_length
            self.y_min = self.bridge_center - half_width
            self.y_max = self.bridge_center + half_width
            
        elif self.direction == 'left':
            # chip1在右,chip2在左 (水平相邻)
            boundary = chip1.x
            self.x_min = boundary - bridge_half_length
            self.x_max = boundary + bridge_half_length
            self.y_min = self.bridge_center - half_width
            self.y_max = self.bridge_center + half_width
            
        elif self.direction == 'top':
            # chip1在下,chip2在上 (垂直相邻)
            # 硅桥在y方向(垂直方向)上长度为SILICONBRIDGE_LENGTH,居中在边界处
            # 硅桥在x方向(水平方向)上宽度为MIN_OVERLAP,居中在重叠边中心
            boundary = chip1.y + chip1.height
            self.y_min = boundary - bridge_half_length
            self.y_max = boundary + bridge_half_length
            self.x_min = self.bridge_center - half_width
            self.x_max = self.bridge_center + half_width
            
        elif self.direction == 'bottom':
            # chip1在上,chip2在下 (垂直相邻)
            boundary = chip1.y
            self.y_min = boundary - bridge_half_length
            self.y_max = boundary + bridge_half_length
            self.x_min = self.bridge_center - half_width
            self.x_max = self.bridge_center + half_width
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """
        获取硅桥的矩形边界框
        
        Returns:
            (x_min, y_min, x_max, y_max)
        """
        return (self.x_min, self.y_min, self.x_max, self.y_max)
    
    def __repr__(self) -> str:
        return (f"SiliconBridge({self.chip1_id}-{self.chip2_id}, "
                f"bbox=({self.x_min:.1f},{self.y_min:.1f})-({self.x_max:.1f},{self.y_max:.1f}))")


def rectangles_overlap(rect1: Tuple[float, float, float, float], 
                       rect2: Tuple[float, float, float, float]) -> bool:
    """
    检查两个矩形是否重叠
    
    Args:
        rect1: 第一个矩形 (x_min, y_min, x_max, y_max)
        rect2: 第二个矩形 (x_min, y_min, x_max, y_max)
        
    Returns:
        如果重叠返回True
    """
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2
    
    # 检查是否有重叠
    overlap_x = not (x1_max <= x2_min or x2_max <= x1_min)
    overlap_y = not (y1_max <= y2_min or y2_max <= y1_min)
    
    return overlap_x and overlap_y


def generate_silicon_bridges(layout: Dict[str, Chiplet], 
                             problem: LayoutProblem) -> List[SiliconBridge]:
    """
    从布局和连接要求生成所有硅桥
    
    Args:
        layout: 芯片布局
        problem: 布局问题（包含连接要求）
        
    Returns:
        硅桥列表
    """
    bridges = []
    
    for chip1_id, chip2_id in problem.connection_graph.edges():
        chip1 = layout[chip1_id]
        chip2 = layout[chip2_id]
        
        # 检查是否邻接
        is_adj, _, _ = get_adjacency_info(chip1, chip2)
        
        if not is_adj:
            # 不邻接，无法创建硅桥
            continue
        
        try:
            bridge = SiliconBridge(chip1_id, chip2_id, chip1, chip2)
            bridges.append(bridge)
        except ValueError:
            # 芯片不邻接,跳过
            continue
    
    return bridges


def SiliconBridge_is_legal(layout: Dict[str, Chiplet], 
                           problem: LayoutProblem, 
                           verbose: bool = False) -> bool:
    """
    检测硅桥布局的合法性
    
    检查所有硅桥的矩形区域是否在空间中发生重叠。
    每个硅桥是一个 MIN_OVERLAP × bridge_length 的矩形。
    
    Args:
        layout: 芯片布局字典 {chip_id: Chiplet}
        problem: 布局问题，包含连接要求
        verbose: 是否打印详细信息
        
    Returns:
        如果所有硅桥都合法（无重叠）返回True，否则返回False
    """
    if verbose:
        print("\n" + "=" * 60)
        print("硅桥合法性检测")
        print("=" * 60)
    
    # 生成所有硅桥
    bridges = generate_silicon_bridges(layout, problem)
    
    if verbose:
        print(f"\n生成了 {len(bridges)} 个硅桥:")
        for i, bridge in enumerate(bridges, 1):
            bbox = bridge.get_bounding_box()
            print(f"  [{i}] {bridge.chip1_id}-{bridge.chip2_id}: "
                  f"矩形=({bbox[0]:.1f},{bbox[1]:.1f})-({bbox[2]:.1f},{bbox[3]:.1f}), "
                  f"宽度={bridge.bridge_width:.2f}, 长度={bridge.bridge_length:.2f}, "
                  f"面积={bridge.bridge_width * bridge.bridge_length:.2f}")
    
    # 检测任意两个硅桥之间是否重叠
    all_legal = True
    conflict_count = 0
    
    for i in range(len(bridges)):
        for j in range(i + 1, len(bridges)):
            bridge1 = bridges[i]
            bridge2 = bridges[j]
            
            bbox1 = bridge1.get_bounding_box()
            bbox2 = bridge2.get_bounding_box()
            
            if rectangles_overlap(bbox1, bbox2):
                all_legal = False
                conflict_count += 1
                
                if verbose:
                    print(f"\n✗ 冲突检测到！硅桥重叠:")
                    print(f"  硅桥1: {bridge1.chip1_id}-{bridge1.chip2_id}")
                    print(f"    矩形区域: ({bbox1[0]:.1f}, {bbox1[1]:.1f}) 到 ({bbox1[2]:.1f}, {bbox1[3]:.1f})")
                    print(f"  硅桥2: {bridge2.chip1_id}-{bridge2.chip2_id}")
                    print(f"    矩形区域: ({bbox2[0]:.1f}, {bbox2[1]:.1f}) 到 ({bbox2[2]:.1f}, {bbox2[3]:.1f})")
                    print(f"  两个硅桥的物理区域发生重叠!")
    
    if verbose:
        print("\n" + "=" * 60)
        if all_legal:
            print("✓ 硅桥布局合法：所有硅桥的矩形区域互不重叠")
        else:
            print(f"✗ 硅桥布局非法：发现 {conflict_count} 处硅桥重叠")
            print("  建议：调整芯片布局或硅桥位置,避免硅桥的物理区域重叠")
        print("=" * 60)
    
    return all_legal


# 示例和测试
if __name__ == "__main__":
    from chiplet_model import Chiplet, LayoutProblem
    
    print("硅桥合法性检测 - 示例")
    print("=" * 60)
    
    # 示例1: 创建一个会导致硅桥冲突的布局
    print("\n示例1: 硅桥重叠的情况")
    print("-" * 60)
    
    problem1 = LayoutProblem()
    
    # 添加芯片 - 创建一个T形布局
    #   [C]
    # [A][B]
    chips1 = [
        Chiplet("A", 10, 10, x=0, y=0),
        Chiplet("B", 10, 10, x=10, y=0),
        Chiplet("C", 10, 10, x=10, y=10),
    ]
    
    for chip in chips1:
        problem1.add_chiplet(chip)
    
    # 添加连接: A-B (水平) 和 B-C (垂直)
    # 如果两个硅桥位置不当,可能在B的边界处重叠
    problem1.add_connection("A", "B")
    problem1.add_connection("B", "C")
    
    layout1 = {chip.id: chip for chip in chips1}
    
    is_legal1 = SiliconBridge_is_legal(layout1, problem1, verbose=True)
    print(f"\n示例1结果: {'合法' if is_legal1 else '非法'}")
    
    # 示例2: 合法的硅桥布局
    print("\n" + "=" * 60)
    print("示例2: 合法的硅桥布局")
    print("-" * 60)
    
    problem2 = LayoutProblem()
    
    # 添加芯片 - 创建一个一字形布局
    # [A][B][C]
    chips2 = [
        Chiplet("A", 10, 10, x=0, y=0),
        Chiplet("B", 10, 10, x=10, y=0),
        Chiplet("C", 10, 10, x=20, y=0),
    ]
    
    for chip in chips2:
        problem2.add_chiplet(chip)
    
    # 添加连接: A-B 和 B-C (都是水平的,不会重叠)
    problem2.add_connection("A", "B")
    problem2.add_connection("B", "C")
    
    layout2 = {chip.id: chip for chip in chips2}
    
    is_legal2 = SiliconBridge_is_legal(layout2, problem2, verbose=True)
    print(f"\n示例2结果: {'合法' if is_legal2 else '非法'}")


