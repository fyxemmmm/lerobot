#!/usr/bin/env python

"""
兼容性模块，提供visualize_dataset_html.py所需的最小lerobot功能。
如果已安装lerobot包，可以直接使用原始导入；否则使用此模块提供的简化实现。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# 设置日志
def init_logging():
    """初始化日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# 简化版的IterableNamespace类
class IterableNamespace:
    """可迭代的命名空间，允许通过属性访问字典内容"""
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
    
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def get(self, key, default=None):
        return self.__dict__.get(key, default)
    
    def __iter__(self):
        return iter(self.__dict__)

# 简化版的LeRobotDataset类
class LeRobotDataset:
    """简化版的LeRobotDataset，仅用于本地数据集的可视化"""
    def __init__(self, repo_id: str, root: Optional[Path] = None, tolerance_s: float = 1e-4):
        self.repo_id = repo_id
        dataset_name = repo_id.split("/", 1)[1]
        
        if root is None:
            # 尝试从默认位置加载
            import os
            home = Path(os.path.expanduser("~"))
            root = home / ".cache" / "huggingface" / "datasets"
        
        self.root = root / dataset_name
        
        # 加载metadata
        with open(self.root / "meta" / "info.json", 'r') as f:
            self.info = json.load(f)
            
        with open(self.root / "meta" / "episodes.jsonl", 'r') as f:
            self.episodes = [json.loads(line) for line in f if line.strip()]
            
        # 加载必要的数据
        self.features = self.info["features"]
        self.fps = self.info["fps"]
        self.num_episodes = self.info["total_episodes"]
        self.num_frames = self.info["total_frames"]
        
        # 识别视频键
        self.video_keys = [key for key, ft in self.features.items() if ft.get("dtype") in ["video", "image"]]
        
        # 创建meta对象
        class Meta:
            def __init__(self, info, episodes, video_keys):
                self._version = info.get("codebase_version", "v2.1")
                self.shapes = {k: v["shape"] for k, v in info["features"].items()}
                self.video_keys = video_keys
                self.episodes = {ep["episode_index"]: ep for ep in episodes}
        
        self.meta = Meta(self.info, self.episodes, self.video_keys)
        
        # 创建episode_data_index
        self.episode_data_index = {"from": {}, "to": {}}
        
        # 如果有parquet文件，尝试加载
        try:
            import pandas as pd
            self.hf_dataset = None  # 实际使用时会按需加载
        except ImportError:
            self.hf_dataset = None
    
    def get_video_file_path(self, episode_id, video_key):
        """获取视频文件路径"""
        episode_chunk = episode_id // self.info["chunks_size"]
        video_path = self.info["video_path"].format(
            episode_chunk=episode_chunk,
            video_key=video_key,
            episode_index=episode_id
        )
        return self.root / video_path

# 可用数据集列表
available_datasets = []  # 在实际使用时会被填充 