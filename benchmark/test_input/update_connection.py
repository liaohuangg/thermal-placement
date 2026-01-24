#!/usr/bin/env python3
"""通过命令行参数指定 JSON 文件，更新 connections 中第四列（conn_type）的数值。"""
import argparse
import json
from pathlib import Path


def update_connections(file_path: Path, silicon_bridge_threshold: float = 128) -> bool:
    """为 connections 数组中的每个条目添加或更新第四列（>阈值则1，否则0）。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'connections' not in data:
        print(f"  {file_path.name}: 没有 connections 字段，跳过")
        return False

    updated = False
    for conn in data['connections']:
        if len(conn) == 3 and isinstance(conn[2], (int, float)):
            if conn[2] > silicon_bridge_threshold:
                conn.append(1)
            else:
                conn.append(0)
            updated = True
        elif len(conn) == 4 and isinstance(conn[3], (int, float)):
            if conn[2] > silicon_bridge_threshold:
                conn[3] = 1
            else:
                conn[3] = 0
            updated = True

    if updated:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  {file_path.name}: 已更新")
        return True
    else:
        print(f"  {file_path.name}: 无需更新（可能已有第四列或格式不同）")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='更新指定 JSON 文件中 connections 的第四列（conn_type）：第三列 > 阈值则为 1，否则为 0。'
    )
    parser.add_argument(
        '--files', '-f',
        nargs='+',
        metavar='FILE',
        help='要更新的 JSON 文件名（可多个）。相对路径相对于脚本所在目录；也可写绝对路径',
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=128,
        help='硅桥判定阈值：第三列数值大于此值则 conn_type=1，否则为 0（默认: 128）',
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.resolve()

    if args.files:
        paths = []
        for name in args.files:
            p = Path(name)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            if not p.exists():
                print(f"  警告: 文件不存在，跳过: {name}")
                continue
            if p.suffix.lower() != '.json':
                print(f"  警告: 非 JSON 文件，跳过: {name}")
                continue
            paths.append(p)
        json_files = paths
        print(f"指定了 {len(json_files)} 个 JSON 文件")
    else:
        json_files = sorted(base_dir.glob('*.json'))
        print(f"未指定 --files，处理当前目录下全部 {len(json_files)} 个 JSON 文件")

    print(f"阈值: {args.threshold}\n")

    updated_count = 0
    for json_file in json_files:
        if update_connections(json_file, silicon_bridge_threshold=args.threshold):
            updated_count += 1

    print(f"\n完成！共更新了 {updated_count} 个文件")


if __name__ == '__main__':
    main()
