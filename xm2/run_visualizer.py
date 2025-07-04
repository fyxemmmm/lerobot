#!/usr/bin/env python

"""
LeRobot数据集可视化工具启动脚本
使用方法:
    python run_visualizer.py --root /path/to/datasets --port 9091 --host 0.0.0.0
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="启动LeRobot数据集可视化工具")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="数据集根目录路径，例如: --root /path/to/datasets"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="服务器主机地址，默认为127.0.0.1，使用0.0.0.0可以从外部访问"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9091,
        help="服务器端口，默认为9091"
    )
    
    args = parser.parse_args()
    
    # 导入可视化模块
    try:
        from visualize_dataset_html import visualize_dataset_html
    except ImportError:
        print("错误: 无法导入visualize_dataset_html模块")
        sys.exit(1)
    
    # 启动可视化服务器
    print(f"启动数据集可视化服务器...")
    print(f"数据集根目录: {args.root}")
    print(f"服务器地址: {args.host}:{args.port}")
    print(f"访问地址: http://{args.host if args.host != '0.0.0.0' else '127.0.0.1'}:{args.port}/lerobothtml?subdir=你的数据集名称&episode=0")
    
    # 运行可视化工具
    visualize_dataset_html(
        dataset=None,
        root=Path(args.root),
        host=args.host,
        port=args.port
    )

if __name__ == "__main__":
    main() 