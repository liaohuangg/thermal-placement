#!/usr/bin/env python3
"""
Convert .cfg files to JSON format for chiplet placement.
"""

import os
import re
import json
from pathlib import Path

def parse_cfg_file(cfg_path):
    """
    Parse a .cfg file and extract chiplet information.
    
    Args:
        cfg_path: Path to the .cfg file
        
    Returns:
        dict with keys: widths, heights, powers, connections_matrix
    """
    with open(cfg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract widths
    widths_match = re.search(r'widths\s*=\s*(.+?)(?:\n|$)', content, re.MULTILINE)
    if not widths_match:
        raise ValueError(f"Could not find 'widths' in {cfg_path}")
    widths_str = widths_match.group(1).strip()
    widths = [float(x.strip()) for x in widths_str.split(',') if x.strip()]
    
    # Extract heights
    heights_match = re.search(r'heights\s*=\s*(.+?)(?:\n|$)', content, re.MULTILINE)
    if not heights_match:
        raise ValueError(f"Could not find 'heights' in {cfg_path}")
    heights_str = heights_match.group(1).strip()
    heights = [float(x.strip()) for x in heights_str.split(',') if x.strip()]
    
    # Extract powers
    powers_match = re.search(r'powers\s*=\s*(.+?)(?:\n|$)', content, re.MULTILINE)
    if not powers_match:
        raise ValueError(f"Could not find 'powers' in {cfg_path}")
    powers_str = powers_match.group(1).strip()
    powers = [float(x.strip()) for x in powers_str.split(',') if x.strip()]
    
    # Extract connections matrix (may span multiple lines)
    # Find the connections section
    connections_match = re.search(r'connections\s*=\s*(.+?)(?=\n\n|\n\[|\n[a-z]+\s*=|$)', content, re.DOTALL)
    if not connections_match:
        raise ValueError(f"Could not find 'connections' in {cfg_path}")
    connections_str = connections_match.group(1).strip()
    
    # Parse the matrix: split by semicolon to get rows, then by comma to get values
    rows = []
    for row_str in connections_str.split(';'):
        row_str = row_str.strip()
        if not row_str:
            continue
        # Remove tabs and extra spaces, split by comma
        row = [int(x.strip()) for x in row_str.split(',') if x.strip()]
        if row:  # Only add non-empty rows
            rows.append(row)
    
    # Validate dimensions
    if len(widths) != len(heights) or len(widths) != len(powers):
        raise ValueError(f"Dimension mismatch: widths={len(widths)}, heights={len(heights)}, powers={len(powers)}")
    
    if len(rows) != len(widths):
        raise ValueError(f"Connections matrix rows ({len(rows)}) != chiplet count ({len(widths)})")
    
    for i, row in enumerate(rows):
        if len(row) != len(widths):
            raise ValueError(f"Connections matrix row {i} has {len(row)} columns, expected {len(widths)}")
    
    return {
        'widths': widths,
        'heights': heights,
        'powers': powers,
        'connections_matrix': rows
    }


def matrix_to_connections(connections_matrix):
    """
    Convert adjacency matrix to list of connections.
    
    Args:
        connections_matrix: 2D list representing adjacency matrix
        
    Returns:
        List of [chiplet_name1, chiplet_name2, bandwidth] tuples
    """
    connections = []
    num_chiplets = len(connections_matrix)
    
    # Convert chiplet index to name (0->A, 1->B, 2->C, ...)
    def index_to_name(idx):
        return chr(ord('A') + idx)
    
    # Only process upper triangle to avoid duplicates (matrix is symmetric)
    for i in range(num_chiplets):
        for j in range(i + 1, num_chiplets):
            bandwidth = connections_matrix[i][j]
            if bandwidth > 0:
                connections.append([index_to_name(i), index_to_name(j), int(bandwidth)])
    
    return connections


def cfg_to_json(cfg_path, output_dir):
    """
    Convert a .cfg file to JSON format.
    
    Args:
        cfg_path: Path to input .cfg file
        output_dir: Directory to save the JSON file
    """
    print(f"Processing: {cfg_path}")
    
    # Parse the .cfg file
    data = parse_cfg_file(cfg_path)
    
    # Create chiplets list
    chiplets = []
    for i in range(len(data['widths'])):
        chiplet_name = chr(ord('A') + i)  # 0->A, 1->B, 2->C, ...
        chiplets.append({
            'name': chiplet_name,
            'width': data['widths'][i],
            'height': data['heights'][i],
            'power': int(data['powers'][i])  # Power is typically an integer
        })
    
    # Convert connections matrix to list format
    connections = matrix_to_connections(data['connections_matrix'])
    
    # Create JSON structure
    json_data = {
        'chiplets': chiplets,
        'connections': connections
    }
    
    # Generate output filename
    cfg_name = Path(cfg_path).stem  # Get filename without extension
    output_path = os.path.join(output_dir, f"{cfg_name}.json")
    
    # Write JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"  -> Saved to: {output_path}")
    print(f"  -> Chiplets: {len(chiplets)}, Connections: {len(connections)}")
    return output_path


def main():
    """Main function to process all .cfg files."""
    # Set up paths
    config_dir = '/root/placement/thermal-placement/benchmark/config'
    output_dir = '/root/placement/thermal-placement/benchmark/test_input'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .cfg files
    cfg_files = list(Path(config_dir).glob('*.cfg'))
    
    if not cfg_files:
        print(f"No .cfg files found in {config_dir}")
        return
    
    print(f"Found {len(cfg_files)} .cfg file(s) to process\n")
    
    # Process each .cfg file
    success_count = 0
    error_count = 0
    
    for cfg_path in sorted(cfg_files):
        try:
            cfg_to_json(str(cfg_path), output_dir)
            success_count += 1
        except Exception as e:
            print(f"ERROR processing {cfg_path}: {e}")
            error_count += 1
        print()  # Empty line between files
    
    # Summary
    print("=" * 60)
    print(f"Processing complete!")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
