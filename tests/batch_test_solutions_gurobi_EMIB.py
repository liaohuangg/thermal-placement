#!/usr/bin/env python3
"""
批量测试脚本：对 test_input 目录下的所有例子进行解的搜索（Gurobi版本）。

用法：
    python3 batch_test_solutions_gurobi_EMIB.py [--min-pair-dist-diff DIFF] [--files FILE1 FILE2 ...]
    
例如：
    # 处理所有文件
    python3 batch_test_solutions_gurobi.py
    
    # 指定参数处理所有文件
    python3 batch_test_solutions_gurobi_EMIB.py --min-pair-dist-diff 3.0
    
    python batch_test_solutions_gurobi_EMIB.py --min-pair-dist-diff 1.0 --files 5core.json

    python batch_test_solutions_gurobi_EMIB.py --min-pair-dist-diff 10.0 --files 2core.json
    # 只处理指定的文件
    python3 batch_test_solutions_gurobi_EMIB.py --files 5core.json 6core.json
    
    # 指定文件（可以不带.json扩展名）
    python3 batch_test_solutions_gurobi_EMIB.py --files 5core 6core 8core
"""

import sys
import argparse
import shutil
from pathlib import Path
import logging
from datetime import datetime
import time
from typing import Optional, List

# 确保可以导入 ilp_reduce_EMIB_search_gurobi
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ilp_reduce_EMIB_search_gurobi import search_multiple_solutions as solve_emib_reduce
from ilp_method_gurobi import ILPPlacementResult
import json
try:
    from tool import ChipletNode
except ImportError:
    # 如果导入失败，定义一个简单的类
    class ChipletNode:
        def __init__(self, name, dimensions, phys=None, power=0.0):
            self.name = name
            self.dimensions = dimensions
            self.phys = phys or []
            self.power = power


# 默认参数配置（EMIB版本不做“多解搜索”，每个case最多返回1个解）
DEFAULT_MIN_SHARED_LENGTH = 0.1
DEFAULT_FIXED_CHIPLET_IDX = 0
DEFAULT_MIN_PAIR_DIST_DIFF = 7.0


def extract_core_name(json_file: Path) -> str:
    """
    从JSON文件名提取核心名称，用于创建输出子目录。
    
    例如：
        "5core.json" -> "5_core"
        "10core.json" -> "10_core"
        "13.json" -> "13_core"
    """
    name = json_file.stem  # 去掉扩展名
    
    # 如果文件名以 "core" 结尾，去掉 "core"
    if name.endswith("core"):
        name = name[:-4]  # 去掉 "core"
    
    # 添加 "_core" 后缀
    return f"{name}_core"


class TeeOutput:
    """
    一个类，用于将输出同时发送到多个流（如stdout和文件）。
    """
    def __init__(self, *streams):
        self.streams = streams
    
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
    
    def flush(self):
        for stream in self.streams:
            stream.flush()


def print_solution_coordinates_and_distances(
    solution: ILPPlacementResult,
    solution_idx: int,
    nodes: List,
) -> None:
    """
    打印解的chiplet坐标位置和相对距离。
    
    参数:
        solution: ILP求解结果
        solution_idx: 解的索引（从0开始）
        nodes: 节点列表，用于获取chiplet名称
    """
    if solution.status != "Optimal":
        return
    
    print(f"\n=== 解 {solution_idx + 1} ===")
    print(f"\n求解状态: {solution.status}")
    print(f"求解时间: {solution.solve_time:.2f} 秒")
    print(f"目标函数值: {solution.objective_value:.2f}")
    
    # 获取坐标
    layout = solution.layout
    
    # 建立名称到索引的映射（基于nodes列表的顺序）
    name_to_idx = {}
    idx_to_name = {}
    for idx, node in enumerate(nodes):
        node_name = node.name if hasattr(node, 'name') else f"Chiplet_{idx}"
        name_to_idx[node_name] = idx
        idx_to_name[idx] = node_name
    
    # 如果layout中的名称不在nodes中，需要添加
    # 但为了保持一致性，我们只处理nodes中存在的chiplet
    n = len(nodes)
    
    # 获取每个chiplet的坐标（按索引）
    x_coords = {}
    y_coords = {}
    for idx in range(n):
        node_name = idx_to_name[idx]
        if node_name in layout:
            x_coords[idx], y_coords[idx] = layout[node_name]
        else:
            # 如果layout中没有该节点，尝试从其他名称匹配
            # 或者设置为0
            x_coords[idx] = 0.0
            y_coords[idx] = 0.0
    
    # 输出每个chiplet的坐标位置
    print(f"\n每个chiplet的坐标位置:")
    for idx in range(n):
        node_name = idx_to_name[idx]
        if idx in x_coords and idx in y_coords:
            x_val = x_coords[idx]
            y_val = y_coords[idx]
            print(f"  [{idx}] {node_name}: x={x_val:.3f}, y={y_val:.3f}")
    
    # 计算并输出每对chiplet的相对距离
    print(f"\n每对chiplet的相对距离（|x[i]-x[j]|, |y[i]-y[j]|）:")
    chiplet_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    for i, j in sorted(chiplet_pairs):
        if i in x_coords and j in x_coords and i in y_coords and j in y_coords:
            x_dist = abs(x_coords[i] - x_coords[j])
            y_dist = abs(y_coords[i] - y_coords[j])
            manhattan_dist = x_dist + y_dist
            name_i = idx_to_name[i]
            name_j = idx_to_name[j]
            print(f"  ({i},{j}) [{name_i}, {name_j}]: x距离={x_dist:.3f}, y距离={y_dist:.3f}, 曼哈顿距离={manhattan_dist:.3f}")


def setup_logging(log_dir: Path, core_name: str):
    """
    设置日志记录，将日志保存到指定目录。
    同时设置stdout重定向，使print输出也保存到日志文件。
    
    Args:
        log_dir: 日志文件保存目录
        core_name: 核心名称
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{core_name}_gurobi.log"
    
    # 打开日志文件用于写入
    log_file_handle = open(log_file, 'w', encoding='utf-8')
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 保存原始的stdout
    original_stdout = sys.stdout
    
    # 创建Tee输出，同时输出到stdout和日志文件
    tee = TeeOutput(original_stdout, log_file_handle)
    
    # 重定向stdout到Tee输出
    sys.stdout = tee
    
    return log_file, log_file_handle, original_stdout


def run_batch_tests(
    min_shared_length: float = DEFAULT_MIN_SHARED_LENGTH,
    fixed_chiplet_idx: int = DEFAULT_FIXED_CHIPLET_IDX,
    min_pair_dist_diff: float = DEFAULT_MIN_PAIR_DIST_DIFF,
    time_limit: int = 600,  # 求解时间限制（秒），默认10分钟
    test_input_dir: Optional[Path] = None,
    output_base_dir: Optional[Path] = None,
    json_files: Optional[List[str]] = None,
):
    """
    批量运行测试用例（Gurobi版本）。
    
    参数:
        （EMIB版本不做多解搜索，每个case最多返回1个解）
        min_shared_length: 相邻chiplet之间的最小共享边长
        fixed_chiplet_idx: 固定位置的chiplet索引
        min_pair_dist_diff: chiplet对之间距离差异的最小阈值
        test_input_dir: 测试输入目录，如果为None则使用默认路径
        output_base_dir: 输出基础目录，如果为None则使用默认路径
        json_files: 要处理的JSON文件列表（文件名或完整路径），如果为None则处理所有文件
    """
    # 确定项目根目录
    project_root = Path(__file__).parent.parent
    
    # 确定测试输入目录
    if test_input_dir is None:
        test_input_dir = project_root / "benchmark" / "test_input"
    
    if not test_input_dir.exists():
        print(f"错误：测试输入目录不存在: {test_input_dir}")
        sys.exit(1)
    
    # 确定输出基础目录
    if output_base_dir is None:
        output_base_dir = project_root / "output_gurobi"
    
    # 创建输出基础目录（如果不存在）
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找JSON文件
    if json_files is not None and len(json_files) > 0:
        # 如果指定了文件列表，只处理这些文件
        selected_files = []
        for file_spec in json_files:
            # 尝试作为完整路径
            file_path = Path(file_spec)
            if file_path.is_absolute() and file_path.exists():
                selected_files.append(file_path)
            else:
                # 尝试作为相对于test_input_dir的文件名
                file_path = test_input_dir / file_spec
                if file_path.exists():
                    selected_files.append(file_path)
                else:
                    # 尝试添加.json扩展名
                    if not file_spec.endswith('.json'):
                        file_path = test_input_dir / f"{file_spec}.json"
                        if file_path.exists():
                            selected_files.append(file_path)
                        else:
                            print(f"警告：找不到文件: {file_spec}，跳过")
                    else:
                        print(f"警告：找不到文件: {file_spec}，跳过")
        
        json_files_list = sorted(selected_files)
        
        if not json_files_list:
            print(f"错误：没有找到任何有效的JSON文件")
            sys.exit(1)
    else:
        # 如果没有指定文件列表，处理所有JSON文件
        json_files_list = sorted(test_input_dir.glob("*.json"))
    
    if not json_files_list:
        print(f"警告：在 {test_input_dir} 目录下没有找到JSON文件")
        return
    
    print(f"\n{'='*80}")
    time_start = time.time()
    print(f"批量测试开始 (Gurobi版本)")
    print(f"{'='*80}")
    print(f"测试输入目录: {test_input_dir}")
    print(f"输出基础目录: {output_base_dir}")
    print(f"找到 {len(json_files_list)} 个测试文件")
    if json_files is not None and len(json_files) > 0:
        print(f"指定处理的文件: {', '.join([f.name for f in json_files_list])}")
    print(f"\n参数配置:")
    print(f"  - min_shared_length: {min_shared_length}")
    print(f"  - fixed_chiplet_idx: {fixed_chiplet_idx}")
    print(f"  - min_pair_dist_diff: {min_pair_dist_diff}")
    print(f"  - time_limit: {time_limit} 秒 ({time_limit/60:.1f} 分钟)")
    
    print(f"{'='*80}\n")
    
    # 统计信息
    success_count = 0
    fail_count = 0
    total_solutions_found = 0  # 总共找到的解的数量
    total_solutions_expected = 0  # 期望找到的解的数量（EMIB每个case最多1个）
    results_summary = []
    
    # 遍历每个JSON文件
    for idx, json_file in enumerate(json_files_list, 1):
        core_name = extract_core_name(json_file)
        
        # 创建分类输出目录
        log_dir = output_base_dir / "log" / core_name
        lp_dir = output_base_dir / "lp" / core_name
        fig_dir = output_base_dir / "fig" / core_name
        
        print(f"\n[{idx}/{len(json_files_list)}] 处理文件: {json_file.name}")
        print(f"  核心名称: {core_name}")
        print(f"  Log目录: {log_dir}")
        print(f"  LP目录: {lp_dir}")
        print(f"  Fig目录: {fig_dir}")
        
        # 删除旧的输出目录（如果存在）
        for old_dir in [log_dir, lp_dir, fig_dir]:
            if old_dir.exists():
                print(f"  删除旧的输出目录: {old_dir}")
                shutil.rmtree(old_dir)
        
        # 创建输出目录
        log_dir.mkdir(parents=True, exist_ok=True)
        lp_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志（同时重定向stdout到日志文件）
        log_file, log_file_handle, original_stdout = setup_logging(log_dir, core_name)
        logger = logging.getLogger()
        
        logger.info(f"{'='*80}")
        logger.info(f"开始处理: {json_file.name} (Gurobi版本)")
        logger.info(f"核心名称: {core_name}")
        logger.info(f"Log目录: {log_dir}")
        logger.info(f"LP目录: {lp_dir}")
        logger.info(f"Fig目录: {fig_dir}")
        logger.info(f"参数配置:")
        logger.info(f"  - min_shared_length: {min_shared_length}")
        logger.info(f"  - fixed_chiplet_idx: {fixed_chiplet_idx}")
        logger.info(f"  - min_pair_dist_diff: {min_pair_dist_diff}")
        logger.info(f"  - time_limit: {time_limit} 秒 ({time_limit/60:.1f} 分钟)")
        logger.info(f"{'='*80}")
        
        try:
            # 加载节点信息（用于输出坐标和相对距离）
            nodes = []
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if "chiplets" in data and isinstance(data["chiplets"], list):
                    # 格式1: ICCAD23格式
                    for chiplet_info in data["chiplets"]:
                        name = chiplet_info.get("name", "")
                        width = chiplet_info.get("width", 0.0)
                        height = chiplet_info.get("height", 0.0)
                        nodes.append(
                            ChipletNode(
                                name=name,
                                dimensions={"x": width, "y": height},
                                phys=[],
                                power=chiplet_info.get("power", 0.0),
                            )
                        )
                else:
                    # 格式2: 字典格式
                    for chiplet_name, chiplet_data in data.items():
                        if isinstance(chiplet_data, dict) and "dimensions" in chiplet_data:
                            dims = chiplet_data["dimensions"]
                            width = dims.get("x", 0.0) if isinstance(dims, dict) else 0.0
                            height = dims.get("y", 0.0) if isinstance(dims, dict) else 0.0
                            nodes.append(
                                ChipletNode(
                                    name=chiplet_name,
                                    dimensions={"x": width, "y": height},
                                    phys=chiplet_data.get("phys", []),
                                    power=chiplet_data.get("power", 0.0),
                                )
                            )
            except Exception as e:
                logger.warning(f"加载节点信息失败: {e}，将使用解的layout信息推断节点")
            
            # 运行 EMIB 降级求解（使用Gurobi版本）
            logger.info(f"调用 EMIB reduce solver (Gurobi版本)...")
            
            # 转换为相对路径（相对于项目根目录）
            project_root = Path(__file__).parent.parent
            try:
                lp_dir_relative = lp_dir.relative_to(project_root)
                fig_dir_relative = fig_dir.relative_to(project_root)
            except ValueError:
                # 如果无法转换为相对路径，使用绝对路径
                lp_dir_relative = lp_dir
                fig_dir_relative = fig_dir
            
            # 在求解之前输出参数值，确认参数传入成功
            logger.info(f"{'='*80}")
            logger.info(f"求解参数确认（在调用 EMIB reduce solver 之前）:")
            logger.info(f"  - min_shared_length: {min_shared_length} (类型: {type(min_shared_length).__name__})")
            logger.info(f"  - fixed_chiplet_idx: {fixed_chiplet_idx} (类型: {type(fixed_chiplet_idx).__name__})")
            logger.info(f"  - min_pair_dist_diff: {min_pair_dist_diff} (类型: {type(min_pair_dist_diff).__name__})")
            logger.info(f"  - time_limit: {time_limit} (类型: {type(time_limit).__name__})")
            logger.info(f"  - input_json_path: {str(json_file.absolute())}")
            logger.info(f"  - output_dir: {str(lp_dir_relative)}")
            logger.info(f"  - image_output_dir: {str(fig_dir_relative)}")
            logger.info(f"{'='*80}")
            logger.info(f"开始调用 EMIB reduce solver...")
            
            sols = solve_emib_reduce(
                num_solutions=1,
                min_shared_length=min_shared_length,
                input_json_path=str(json_file.absolute()),
                fixed_chiplet_idx=fixed_chiplet_idx,
                min_pair_dist_diff=min_pair_dist_diff,
                time_limit=time_limit,
                output_dir=str(lp_dir_relative),  # .lp文件保存到lp目录（相对路径）
                image_output_dir=str(fig_dir_relative)  # 图片保存到fig目录（相对路径）
            )
            time_end = time.time()
            logger.info(f"\nEMIB求解返回 {len(sols)} 个解（最多1个）。")
            print(f"  ✓ EMIB求解：返回 {len(sols)} 个解")
            
            # 更新解的数量统计
            total_solutions_found += len(sols)
            total_solutions_expected += 1
            
            # 如果没有从JSON加载到节点，尝试从第一个解的layout推断
            if len(nodes) == 0 and len(sols) > 0 and sols[0].status == "Optimal":
                layout = sols[0].layout
                for idx, (name, (x, y)) in enumerate(sorted(layout.items())):
                    nodes.append(
                        ChipletNode(
                            name=name,
                            dimensions={"x": 0.0, "y": 0.0},  # 尺寸未知
                            phys=[],
                            power=0.0,
                        )
                    )
            
            # 输出每个解的坐标位置和相对距离
            if len(nodes) > 0:
                for idx, sol in enumerate(sols):
                    print_solution_coordinates_and_distances(sol, idx, nodes)
                    logger.info(f"\n解 {idx + 1} 的坐标和相对距离已输出")
            
            success_count += 1
            results_summary.append({
                'file': json_file.name,
                'core_name': core_name,
                'status': 'success',
                'num_solutions': len(sols),
                'log_dir': str(log_dir),
                'lp_dir': str(lp_dir),
                'fig_dir': str(fig_dir)
            })
            
        except Exception as e:
            logger.error(f"\n错误：求解过程中出现异常: {e}", exc_info=True)
            print(f"  ✗ 失败：{e}")
            
            fail_count += 1
            results_summary.append({
                'file': json_file.name,
                'core_name': core_name,
                'status': 'failed',
                'error': str(e),
                'log_dir': str(log_dir),
                'lp_dir': str(lp_dir),
                'fig_dir': str(fig_dir)
            })
        finally:
            # 恢复原始的stdout
            sys.stdout = original_stdout
            # 关闭日志文件句柄
            if log_file_handle:
                log_file_handle.close()
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"批量测试完成 (Gurobi版本)")
    time_end = time.time()
    print(f"  共花费时间: {time_end - time_start:.2f} 秒")
    print(f"{'='*80}")
    print(f"文件处理统计:")
    print(f"  成功: {success_count}/{len(json_files_list)} 个文件")
    print(f"  失败: {fail_count}/{len(json_files_list)} 个文件")
    print(f"解的数量统计:")
    print(f"  找到: {total_solutions_found}/{total_solutions_expected} 个解")
    print(f"\n详细结果:")
    for result in results_summary:
        if result['status'] == 'success':
            print(f"  ✓ {result['file']} -> {result['core_name']}: {result['num_solutions']} 个解")
            print(f"    Log目录: {result.get('log_dir', 'N/A')}")
            print(f"    LP目录: {result.get('lp_dir', 'N/A')}")
            print(f"    Fig目录: {result.get('fig_dir', 'N/A')}")
        else:
            print(f"  ✗ {result['file']} -> {result['core_name']}: 失败")
            print(f"    错误: {result.get('error', 'Unknown error')}")
            print(f"    Log目录: {result.get('log_dir', 'N/A')}")
            print(f"    LP目录: {result.get('lp_dir', 'N/A')}")
            print(f"    Fig目录: {result.get('fig_dir', 'N/A')}")
    print(f"{'='*80}\n")


def main():
    """主函数，解析命令行参数并运行批量测试。"""
    parser = argparse.ArgumentParser(
        description='批量测试脚本：对 test_input 目录下的所有例子进行 EMIB(硅桥)降级求解（Gurobi版本）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认参数
  python3 batch_test_solutions_gurobi_EMIB.py
  
  # 指定参数
  python3 batch_test_solutions_gurobi_EMIB.py --min-pair-dist-diff 3.0
  
  # 处理指定文件
  python3 batch_test_solutions_gurobi_EMIB.py --files 5core.json 6core.json
        """
    )
    
    parser.add_argument(
        '--min-shared-length',
        type=float,
        default=DEFAULT_MIN_SHARED_LENGTH,
        dest='min_shared_length',  # 显式指定目标属性名
        help=f'相邻chiplet之间的最小共享边长（默认: {DEFAULT_MIN_SHARED_LENGTH}）'
    )
    
    parser.add_argument(
        '--fixed-chiplet-idx',
        type=int,
        default=DEFAULT_FIXED_CHIPLET_IDX,
        help=f'固定位置的chiplet索引（默认: {DEFAULT_FIXED_CHIPLET_IDX}）'
    )
    
    parser.add_argument(
        '--min-pair-dist-diff',
        type=float,
        default=DEFAULT_MIN_PAIR_DIST_DIFF,
        help=f'chiplet对之间距离差异的最小阈值（默认: {DEFAULT_MIN_PAIR_DIST_DIFF}）'
    )
    
    parser.add_argument(
        '--time-limit',
        type=int,
        default=600,
        help='求解时间限制（秒），默认: 600 (10分钟)'
    )
    
    parser.add_argument(
        '--test-input-dir',
        type=str,
        default=None,
        help='测试输入目录（默认: benchmark/test_input）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出基础目录（默认: output_gurobi）'
    )
    
    parser.add_argument(
        '--files',
        type=str,
        nargs='+',
        default=None,
        metavar='FILE',
        help='指定要处理的JSON文件列表（文件名或完整路径），可以指定多个文件。例如: --files 5core.json 6core.json 或 --files 5core 6core'
    )
    
    # 在解析之前输出原始命令行参数
    print(f"\n{'='*80}")
    print(f"原始命令行参数:")
    print(f"  sys.argv: {sys.argv}")
    print(f"{'='*80}\n")
    
    args = parser.parse_args()
    
    # 在解析参数后立即输出参数值，确认参数传入成功
    print(f"\n{'='*80}")
    print(f"命令行参数解析结果:")
    print(f"  - min_shared_length: {args.min_shared_length} (类型: {type(args.min_shared_length).__name__})")
    print(f"  - fixed_chiplet_idx: {args.fixed_chiplet_idx} (类型: {type(args.fixed_chiplet_idx).__name__})")
    print(f"  - min_pair_dist_diff: {args.min_pair_dist_diff} (类型: {type(args.min_pair_dist_diff).__name__})")
    print(f"  - time_limit: {args.time_limit} (类型: {type(args.time_limit).__name__})")
    print(f"  - files: {args.files}")
    print(f"  - test_input_dir: {args.test_input_dir}")
    print(f"  - output_dir: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # 检查是否有未识别的参数
    if hasattr(args, '__dict__'):
        print(f"所有解析后的参数:")
        for key, value in vars(args).items():
            print(f"  {key}: {value} (类型: {type(value).__name__})")
        print(f"{'='*80}\n")
    
    # 转换路径
    test_input_dir = Path(args.test_input_dir) if args.test_input_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # 运行批量测试
    run_batch_tests(
        min_shared_length=args.min_shared_length,
        fixed_chiplet_idx=args.fixed_chiplet_idx,
        min_pair_dist_diff=args.min_pair_dist_diff,
        time_limit=args.time_limit,
        test_input_dir=test_input_dir,
        output_base_dir=output_dir,
        json_files=args.files,
    )


if __name__ == "__main__":
    main()

