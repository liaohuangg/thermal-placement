#!/usr/bin/env python3
import json
import os
from pathlib import Path

def update_connections(file_path, silicon_bridge_threshold=128):
    """为 connections 数组中的每个条目添加第四列"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'connections' not in data:
        print(f"  {file_path.name}: 没有 connections 字段，跳过")
        return False
    
    updated = False
    for conn in data['connections']:
        # 只处理有3列的连接（包含数字）
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
    dir_path = Path(__file__).parent
    json_files = sorted(dir_path.glob('*.json'))
    
    print(f"找到 {len(json_files)} 个 JSON 文件")
    print("开始处理...\n")
    
    updated_count = 0
    for json_file in json_files:
        if json_file.name == 'update_connections.py':
            continue
        if update_connections(json_file, silicon_bridge_threshold=128):
            updated_count += 1
    
    print(f"\n完成！共更新了 {updated_count} 个文件")

if __name__ == '__main__':
    main()
