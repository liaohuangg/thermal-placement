#!/usr/bin/env python3
"""
批量测试脚本：对 test_input 目录下的所有例子进行解的搜索。

用法：
    python3 batch_test_solutions.py [--num-solutions N] [--grid-size SIZE] [--min-pair-dist-diff DIFF] [--files FILE1 FILE2 ...]
    
例如：
    # 处理所有文件
    python3 batch_test_solutions.py
    
    # 指定参数处理所有文件
    python3 batch_test_solutions.py --num-solutions 4 --grid-size 0.5 --min-pair-dist-diff 3.0
    
    python batch_test_solutions.py --num-solutions 30 --grid-size 0.5 --min-pair-dist-diff 10.0 --files 5core.json

    python batch_test_solutions.py --num-solutions 30 --grid-size 0.5 --min-pair-dist-diff 10.0 --files 2core.json
    # 只处理指定的文件
    python3 batch_test_solutions.py --files 5core.json 6core.json
    
    # 指定文件（可以不带.json扩展名）
    python3 batch_test_solutions.py --files 5core 6core 8core
"""

import sys
import argparse
import shutil
from pathlib import Path
import logging
from datetime import datetime
from typing import Optional, List

# 确保可以导入 ilp_sub_solution_search
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ilp_sub_solution_search import search_multiple_solutions


# 默认参数配置
DEFAULT_NUM_SOLUTIONS = 4
DEFAULT_MIN_SHARED_LENGTH = 0.1
DEFAULT_GRID_SIZE = 0.5
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


def setup_logging(output_dir: Path, core_name: str):
    """
    设置日志记录，将日志保存到输出目录。
    同时设置stdout重定向，使print输出也保存到日志文件。
    """
    log_file = output_dir / f"{core_name}.log"
    
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
    num_solutions: int = DEFAULT_NUM_SOLUTIONS,
    min_shared_length: float = DEFAULT_MIN_SHARED_LENGTH,
    grid_size: float = DEFAULT_GRID_SIZE,
    fixed_chiplet_idx: int = DEFAULT_FIXED_CHIPLET_IDX,
    min_pair_dist_diff: float = DEFAULT_MIN_PAIR_DIST_DIFF,
    test_input_dir: Optional[Path] = None,
    output_base_dir: Optional[Path] = None,
    json_files: Optional[List[str]] = None,
):
    """
    批量运行测试用例。
    
    参数:
        num_solutions: 需要搜索的解的数量
        min_shared_length: 相邻chiplet之间的最小共享边长
        grid_size: 网格大小
        fixed_chiplet_idx: 固定位置的chiplet索引
        min_pair_dist_diff: chiplet对之间距离差异的最小阈值，如果为None则使用grid_size或默认值1.0
        test_input_dir: 测试输入目录，如果为None则使用默认路径
        output_base_dir: 输出基础目录，如果为None则使用默认路径
        json_files: 要处理的JSON文件列表（文件名或完整路径），如果为None则处理所有文件
    """
    # 确定项目根目录
    project_root = Path(__file__).parent.parent
    
    # 确定测试输入目录
    if test_input_dir is None:
        test_input_dir = project_root / "baseline" / "ICCAD23" / "test_input"
    
    if not test_input_dir.exists():
        print(f"错误：测试输入目录不存在: {test_input_dir}")
        sys.exit(1)
    
    # 确定输出基础目录
    if output_base_dir is None:
        output_base_dir = project_root / "output"
    
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
    print(f"批量测试开始")
    print(f"{'='*80}")
    print(f"测试输入目录: {test_input_dir}")
    print(f"输出基础目录: {output_base_dir}")
    print(f"找到 {len(json_files_list)} 个测试文件")
    if json_files is not None and len(json_files) > 0:
        print(f"指定处理的文件: {', '.join([f.name for f in json_files_list])}")
    print(f"\n参数配置:")
    print(f"  - num_solutions: {num_solutions}")
    print(f"  - min_shared_length: {min_shared_length}")
    print(f"  - grid_size: {grid_size}")
    print(f"  - fixed_chiplet_idx: {fixed_chiplet_idx}")
    print(f"  - min_pair_dist_diff: {min_pair_dist_diff}")
    
    print(f"{'='*80}\n")
    
    # 统计信息
    success_count = 0
    fail_count = 0
    results_summary = []
    
    # 遍历每个JSON文件
    for idx, json_file in enumerate(json_files_list, 1):
        core_name = extract_core_name(json_file)
        output_dir = output_base_dir / core_name
        
        print(f"\n[{idx}/{len(json_files_list)}] 处理文件: {json_file.name}")
        print(f"  核心名称: {core_name}")
        print(f"  输出目录: {output_dir}")
        
        # 删除旧的输出目录（如果存在）
        if output_dir.exists():
            print(f"  删除旧的输出目录: {output_dir}")
            shutil.rmtree(output_dir)
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志（同时重定向stdout到日志文件）
        log_file, log_file_handle, original_stdout = setup_logging(output_dir, core_name)
        logger = logging.getLogger()
        
        logger.info(f"{'='*80}")
        logger.info(f"开始处理: {json_file.name}")
        logger.info(f"核心名称: {core_name}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"参数配置:")
        logger.info(f"  - num_solutions: {num_solutions}")
        logger.info(f"  - min_shared_length: {min_shared_length}")
        logger.info(f"  - grid_size: {grid_size}")
        logger.info(f"  - fixed_chiplet_idx: {fixed_chiplet_idx}")
        logger.info(f"  - min_pair_dist_diff: {min_pair_dist_diff}")
        logger.info(f"{'='*80}")
        
        try:
            # 运行求解搜索
            logger.info(f"调用 search_multiple_solutions...")
            sols = search_multiple_solutions(
                num_solutions=num_solutions,
                min_shared_length=min_shared_length,
                input_json_path=str(json_file.absolute()),
                grid_size=grid_size,
                fixed_chiplet_idx=fixed_chiplet_idx,
                min_pair_dist_diff=min_pair_dist_diff,
                output_dir=str(output_dir)  # 指定输出目录
            )
            
            logger.info(f"\n共找到 {len(sols)} 个不同的解。")
            print(f"  ✓ 成功：找到 {len(sols)} 个解")
            
            success_count += 1
            results_summary.append({
                'file': json_file.name,
                'core_name': core_name,
                'status': 'success',
                'num_solutions': len(sols),
                'output_dir': str(output_dir)
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
                'output_dir': str(output_dir)
            })
        finally:
            # 恢复原始的stdout
            sys.stdout = original_stdout
            # 关闭日志文件句柄
            if log_file_handle:
                log_file_handle.close()
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"批量测试完成")
    print(f"{'='*80}")
    print(f"成功: {success_count}/{len(json_files_list)}")
    print(f"失败: {fail_count}/{len(json_files_list)}")
    print(f"\n详细结果:")
    for result in results_summary:
        if result['status'] == 'success':
            print(f"  ✓ {result['file']} -> {result['core_name']}: {result['num_solutions']} 个解")
            print(f"    输出目录: {result['output_dir']}")
        else:
            print(f"  ✗ {result['file']} -> {result['core_name']}: 失败")
            print(f"    错误: {result.get('error', 'Unknown error')}")
            print(f"    输出目录: {result['output_dir']}")
    print(f"{'='*80}\n")


def main():
    """主函数，解析命令行参数并运行批量测试。"""
    parser = argparse.ArgumentParser(
        description='批量测试脚本：对 test_input 目录下的所有例子进行解的搜索',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认参数
  python3 batch_test_solutions.py
  
  # 指定参数
  python3 batch_test_solutions.py --num-solutions 4 --grid-size 0.5 --min-pos-diff 3.0 --min-pair-dist-diff 3.0
  
  # 只指定部分参数
  python3 batch_test_solutions.py --num-solutions 6 --grid-size 1.0
        """
    )
    
    parser.add_argument(
        '--num-solutions',
        type=int,
        default=DEFAULT_NUM_SOLUTIONS,
        help=f'需要搜索的解的数量（默认: {DEFAULT_NUM_SOLUTIONS}）'
    )
    
    parser.add_argument(
        '--min-shared-length',
        type=float,
        default=DEFAULT_MIN_SHARED_LENGTH,
        help=f'相邻chiplet之间的最小共享边长（默认: {DEFAULT_MIN_SHARED_LENGTH}）'
    )
    
    parser.add_argument(
        '--grid-size',
        type=float,
        default=DEFAULT_GRID_SIZE,
        help=f'网格大小（默认: {DEFAULT_GRID_SIZE}）'
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
        help=f'chiplet对之间距离差异的最小阈值，如果为None则使用grid_size或默认值1.0（默认: {DEFAULT_MIN_PAIR_DIST_DIFF}）'
    )
    
    parser.add_argument(
        '--test-input-dir',
        type=str,
        default=None,
        help='测试输入目录（默认: baseline/ICCAD23/test_input）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出基础目录（默认: output）'
    )
    
    parser.add_argument(
        '--files',
        type=str,
        nargs='+',
        default=None,
        metavar='FILE',
        help='指定要处理的JSON文件列表（文件名或完整路径），可以指定多个文件。例如: --files 5core.json 6core.json 或 --files 5core 6core'
    )
    
    args = parser.parse_args()
    
    # 转换路径
    test_input_dir = Path(args.test_input_dir) if args.test_input_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # 运行批量测试
    run_batch_tests(
        num_solutions=args.num_solutions,
        min_shared_length=args.min_shared_length,
        grid_size=args.grid_size,
        fixed_chiplet_idx=args.fixed_chiplet_idx,
        min_pair_dist_diff=args.min_pair_dist_diff,
        test_input_dir=test_input_dir,
        output_base_dir=output_dir,
        json_files=args.files,
    )


if __name__ == "__main__":
    main()

