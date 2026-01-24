# 批量测试脚本使用说明

## 概述

`batch_test_solutions.py` 是一个批量测试脚本，用于对 `baseline/ICCAD23/test_input` 目录下的所有测试用例进行解的搜索。

## 功能特点

- 自动遍历所有测试输入文件（`*.json`）
- 为每个测试用例创建独立的输出子目录（如 `5_core`, `6_core` 等）
- 保存所有输出文件（图片、.lp文件、.log文件）到对应的子目录
- 支持通过命令行参数自定义所有求解参数
- 提供详细的日志记录和结果统计

## 使用方法

### 基本用法（使用默认参数）

```bash
conda run -n pyside python3 tests/batch_test_solutions.py
```

### 指定参数

```bash
conda run -n pyside python tests/batch_test_solutions.py \
    --num-solutions 10 \
    --grid-size 0.5 \
    --min-pos-diff 3.0 \
    --min-pair-dist-diff 3.0
```

### 只处理指定的文件

```bash
# 处理指定的文件（可以带或不带.json扩展名）
conda run -n pyside python tests/batch_test_solutions.py --files 5core.json 6core.json

# 或者不带扩展名
conda run -n pyside python tests/batch_test_solutions.py --files 5core 6core 8core

# 结合其他参数
conda run -n pyside python tests/batch_test_solutions.py \
    --files 5core 6core \
    --num-solutions 4 \
    --grid-size 0.5
```

### 所有可用参数

- `--num-solutions N`: 需要搜索的解的数量（默认: 4）
- `--min-shared-length LENGTH`: 相邻chiplet之间的最小共享边长（默认: 0.1）
- `--grid-size SIZE`: 网格大小（默认: 0.5）
- `--fixed-chiplet-idx IDX`: 固定位置的chiplet索引（默认: 0）
- `--min-pos-diff DIFF`: 位置排除约束的最小变化量（默认: 3.0）
- `--min-pair-dist-diff DIFF`: chiplet对之间距离差异的最小阈值（默认: 3.0）
- `--test-input-dir DIR`: 测试输入目录（默认: `baseline/ICCAD23/test_input`）
- `--output-dir DIR`: 输出基础目录（默认: `output`）
- `--files FILE1 FILE2 ...`: 指定要处理的JSON文件列表（文件名或完整路径），可以指定多个文件。如果不指定，则处理所有文件。例如: `--files 5core.json 6core.json` 或 `--files 5core 6core`

## 输出结构

脚本会在 `output` 目录下为每个测试用例创建子目录：

```
output/
├── 2_core/
│   ├── 2_core.log          # 日志文件
│   ├── solution_1_layout.png
│   ├── solution_1.lp
│   ├── solution_2_layout.png
│   ├── solution_2.lp
│   └── ...
├── 3_core/
│   ├── 3_core.log
│   └── ...
├── 5_core/
│   ├── 5_core.log
│   └── ...
└── ...
```

## 输出文件说明

- **日志文件** (`{core_name}.log`): 包含完整的求解过程和调试信息
- **布局图片** (`solution_{n}_layout.png`): 每个解的布局可视化
- **LP文件** (`solution_{n}.lp`): ILP模型的LP格式文件

## 示例

### 示例1：使用默认参数运行所有测试

```bash
conda run -n pyside python3 tests/batch_test_solutions.py
```

### 示例2：只搜索2个解，使用更大的网格

```bash
conda run -n pyside python3 tests/batch_test_solutions.py \
    --num-solutions 2 \
    --grid-size 1.0
```

### 示例3：只处理指定的文件

```bash
conda run -n pyside python tests/batch_test_solutions.py --files 5core 6core
```

### 示例4：使用自定义输入和输出目录

```bash
conda run -n pyside python tests/batch_test_solutions.py \
    --test-input-dir /path/to/custom/input \
    --output-dir /path/to/custom/output
```

### 示例5：处理指定文件并自定义参数

```bash
conda run -n pyside python tests/batch_test_solutions.py \
    --files 5core 6core 8core \
    --num-solutions 6 \
    --grid-size 1.0 \
    --min-pos-diff 2.0
```

## 注意事项

1. 确保在 `conda pyside` 环境下运行脚本
2. 脚本会自动创建输出目录（如果不存在）
3. 如果某个测试用例失败，脚本会继续处理其他测试用例
4. 所有日志都会保存到对应的子目录中，方便后续分析

## 测试用例

当前支持的测试用例（来自 `baseline/ICCAD23/test_input`）：

- `2core.json` -> `2_core`
- `3core.json` -> `3_core`
- `5core.json` -> `5_core`
- `6core.json` -> `6_core`
- `8core.json` -> `8_core`
- `10core.json` -> `10_core`
- `12core.json` -> `12_core`
- `13.json` -> `13_core`

