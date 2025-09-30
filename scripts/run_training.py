#!/usr/bin/env python3
"""
ProSetting TRL Training Launcher Script
Provides unified training entry point, supports semi-automated and fully automated modes
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='ProSetting TRL Training Launcher')
    parser.add_argument('--mode', choices=['auto', 'semi'], default='auto',
                       help='Training mode: auto (fully automated) or semi (semi-automated)')
    parser.add_argument('--config', type=str, help='Custom config file path')
    
    args = parser.parse_args()
    
    print(f"🚀 ProSetting TRL Training System")
    print(f"📊 Mode: {'Full Auto' if args.mode == 'auto' else 'Semi-Auto'}")
    
    if args.mode == 'auto':
        print("🤖 Starting fully automated TRL training...")
        from scripts.auto_trainer import main as auto_main
        return auto_main()
    else:
        print("🎮 Starting semi-automated TRL training...")
        from scripts.semi_auto_trainer import main as semi_main
        return semi_main()

if __name__ == "__main__":
    sys.exit(main())