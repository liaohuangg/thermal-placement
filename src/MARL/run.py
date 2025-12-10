"""
Quick start script for MAPPO chiplet placement
Run this from the src directory
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 80)
    print("MAPPO Chiplet Placement - Quick Start")
    print("=" * 80)
    print()
    
    marl_dir = Path(__file__).parent
    
    # Check if test inputs exist
    test_input_dir = marl_dir.parent.parent / "baseline" / "ICCAD23" / "test_input"
    
    if not test_input_dir.exists():
        print(f"❌ Test input directory not found: {test_input_dir}")
        print("Please ensure the test cases are available.")
        return
    
    test_files = list(test_input_dir.glob("*.json"))
    print(f"Found {len(test_files)} test cases:")
    for f in sorted(test_files):
        print(f"  - {f.name}")
    print()
    
    # Ask user what to do
    print("Options:")
    print("  1. Run quick tests (recommended first time)")
    print("  2. Train on all test cases")
    print("  3. Exit")
    print()
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\nRunning quick tests...")
        test_script = marl_dir / "test_mappo.py"
        subprocess.run([sys.executable, str(test_script)])
    
    elif choice == "2":
        print("\nStarting training...")
        train_script = marl_dir / "train_mappo.py"
        subprocess.run([sys.executable, str(train_script)])
    
    elif choice == "3":
        print("Exiting...")
    
    else:
        print("Invalid choice. Exiting...")


if __name__ == "__main__":
    main()
