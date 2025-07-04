#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesnt always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossly compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Example of usage:

- Visualize data stored on a local machine:
```bash
local$ python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht

local$ open http://localhost:9090
```

- Visualize data stored on a distant machine with a local viewer:
```bash
distant$ python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht

local$ ssh -L 9090:localhost:9090 distant  # create a ssh tunnel
local$ open http://localhost:9090
```

- Select episodes to visualize:
```bash
python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht \
    --episodes 7 3 5 1 4
```
"""

import argparse
import csv
import json
import logging
import re
import shutil
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from flask import Flask, redirect, render_template, request, url_for
from werkzeug.middleware.proxy_fix import ProxyFix

from lerobot import available_datasets
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import IterableNamespace
from lerobot.common.utils.utils import init_logging


def run_server(
    dataset: LeRobotDataset | IterableNamespace | None,
    episodes: list[int] | None,
    host: str,
    port: str,
    static_folder: Path,
    template_folder: Path,
    root: Path = None,
):
    app = Flask(__name__, static_folder=static_folder.resolve(), template_folder=template_folder.resolve())
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # specifying not to cache
    
    # 支持nginx反向代理
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    def get_static_url(filename):
        """生成静态文件URL，支持nginx反向代理子路径"""
        # 尝试使用Flask的url_for，如果在反向代理环境下会自动处理前缀
        try:
            static_url = url_for("static", filename=filename)
            print(f"Generated static URL: {static_url} for filename: {filename}")
            
            # 检查当前请求路径，如果包含前缀，确保静态URL也包含相同前缀
            current_path = request.path
            print(f"Current request path: {current_path}")
            
            # 如果当前路径包含前缀（比如 /xiangmu/lerobothtml），但静态URL没有包含
            if current_path.startswith('/') and '/' in current_path[1:]:
                path_parts = current_path[1:].split('/')
                if len(path_parts) > 1 and path_parts[0] != 'static':
                    # 检测到可能的前缀路径
                    prefix = path_parts[0]
                    print(f"Detected path prefix: {prefix}")
                    
                    # 如果静态URL不包含这个前缀，添加它
                    if not static_url.startswith(f'/{prefix}/static'):
                        if static_url.startswith('/static'):
                            static_url = f'/{prefix}{static_url}'
                            print(f"Added prefix to static URL: {static_url}")
            
            return static_url
        except Exception as e:
            print(f"url_for failed: {e}, using relative path")
            # 如果url_for失败，生成相对路径
            return f"./static/{filename}"

    # 新的路由：使用查询参数的形式 /lerobothtml?subdir=xxx&episode=xxx
    @app.route("/lerobothtml")
    def show_lerobothtml():
        # 从查询参数中获取subdir和episode
        subdir_path = request.args.get('subdir')
        episode_id = request.args.get('episode', type=int)
        
        print(f"Handling lerobothtml request - subdir: {subdir_path}, episode: {episode_id}")
        
        # 如果没有subdir参数，返回错误
        if not subdir_path:
            return "Missing 'subdir' parameter. Example: /lerobothtml?subdir=aloha_mobile_cabinet&episode=0", 400
        
        # 如果没有episode参数，自动重定向到episode_0
        if episode_id is None:
            print(f"No episode specified, redirecting to episode 0 for subdir: {subdir_path}")
            # 保留所有其他查询参数（如t=时间参数）
            args = request.args.copy()
            args['episode'] = 0
            return redirect(url_for('show_lerobothtml', **args))
        
        # 执行与原来subdir路由相同的逻辑
        try:
            if root:
                print(f"Loading dataset from subdir path: {subdir_path} from root: {root}")
                
                # 直接使用 root/subdir_path 作为数据集路径
                dataset_path = Path(root) / subdir_path
                print(f"Dataset path: {dataset_path}")
                
                expected_path = dataset_path / 'meta' / 'info.json'
                print(f"Expected path: {expected_path}")
                
                if not expected_path.exists():
                    return f"Dataset not found at path: {dataset_path}", 404
                
                try:
                    # 创建一个虚拟的repo_id用于兼容性
                    repo_id = f"local/{subdir_path.split('/')[-1]}"
                    dataset = LocalLeRobotDataset(repo_id, dataset_path, use_direct_path=True)
                    print(f"Successfully loaded dataset from subdir: {subdir_path}")
                    
                    # 为这个本地数据集创建视频符号链接
                    static_folder = Path(app.static_folder)
                    
                    # 创建subdir目录结构在static文件夹下
                    subdir_static_dir = static_folder / subdir_path
                    subdir_static_dir.mkdir(parents=True, exist_ok=True)
                    print(f"Created subdir static directory: {subdir_static_dir}")
                    
                    # 在subdir目录下创建videos符号链接
                    subdir_videos_dir = subdir_static_dir / "videos"
                    
                    # 删除现有的符号链接（无论是否损坏）
                    if subdir_videos_dir.exists() or subdir_videos_dir.is_symlink():
                        try:
                            subdir_videos_dir.unlink()
                            print(f"Removed existing subdir video symlink: {subdir_videos_dir}")
                        except Exception as e:
                            print(f"Warning: Could not remove existing subdir video symlink: {e}")
                            try:
                                subdir_videos_dir.unlink(missing_ok=True)
                            except:
                                pass
                    
                    # 创建新的符号链接
                    videos_target = (dataset.root / "videos").resolve()
                    try:
                        subdir_videos_dir.symlink_to(videos_target.as_posix())
                        print(f"Created subdir video symlink: {subdir_videos_dir} -> {videos_target}")
                        print(f"Symlink exists: {subdir_videos_dir.exists()}")
                        print(f"Target exists: {videos_target.exists()}")
                    except Exception as e:
                        print(f"Error creating subdir video symlink: {e}")
                        # 如果符号链接创建失败，尝试清理并重试
                        import os
                        if subdir_videos_dir.exists():
                            os.remove(subdir_videos_dir)
                        subdir_videos_dir.symlink_to(videos_target.as_posix())
                        print(f"Retry: Created subdir video symlink: {subdir_videos_dir} -> {videos_target}")
                    
                except Exception as e:
                    print(f"Error loading dataset from subdir: {e}")
                    import traceback
                    traceback.print_exc()
                    return f"Error loading dataset from subdir {subdir_path}: {e}", 500
            else:
                return "Root directory not specified", 400
                
        except Exception as e:
            print(f"Error in lerobothtml route: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing lerobothtml request: {e}", 500

        # 处理数据集版本检查和数据获取（复用原来的逻辑）
        try:
            if isinstance(dataset, LocalLeRobotDataset):
                dataset_version = dataset.info.get("codebase_version", "v2.1")
            else:
                dataset_version = (
                    str(dataset.meta._version) if isinstance(dataset, LeRobotDataset) else dataset.codebase_version
                )
            
            match = re.search(r"v(\d+)\.", dataset_version)
            if match:
                major_version = int(match.group(1))
                if major_version < 2:
                    return "Make sure to convert your LeRobotDataset to v2 & above."

            print(f"Getting episode data for episode {episode_id}")
            episode_data_csv_str, columns, ignored_columns = get_episode_data(dataset, episode_id)
            print(f"Episode data retrieved successfully")
        except Exception as e:
            print(f"Error in episode data processing: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing episode data: {e}", 500

        try:
            print(f"Creating dataset_info for {type(dataset)}")
            dataset_info = {
                "repo_id": f"lerobothtml/{subdir_path}",
                "num_samples": dataset.num_frames,
                "num_episodes": dataset.num_episodes,
                "fps": dataset.fps,
            }
            print(f"Dataset info created successfully: {dataset_info}")
        except Exception as e:
            print(f"Error creating dataset_info: {e}")
            import traceback
            traceback.print_exc()
            return f"Error creating dataset info: {e}", 500

        # 处理视频信息
        try:
            print(f"Processing dataset videos for episode {episode_id}")
            print(f"Dataset video_keys: {dataset.video_keys}")
            print(f"Dataset root: {dataset.root}")
            print(f"Videos directory exists: {(dataset.root / 'videos').exists()}")
            
            if dataset.video_keys:
                video_paths = [
                    dataset.get_video_file_path(episode_id, key) for key in dataset.video_keys
                ]
                print(f"Video paths: {video_paths}")
                
                # 检查视频文件是否真实存在
                for i, video_path in enumerate(video_paths):
                    print(f"Video {i}: {video_path}")
                    print(f"  - Path exists: {video_path.exists()}")
                    if video_path.exists():
                        print(f"  - File size: {video_path.stat().st_size} bytes")
                    else:
                        print(f"  - Parent directory exists: {video_path.parent.exists()}")
                        if video_path.parent.exists():
                            print(f"  - Files in parent directory: {list(video_path.parent.iterdir())}")
            else:
                print("No video keys found - trying to manually scan for videos")
                videos_dir = dataset.root / "videos"
                if videos_dir.exists():
                    print(f"Contents of videos directory: {list(videos_dir.iterdir())}")
                    # 尝试手动查找视频文件
                    video_paths = []
                    for chunk_dir in videos_dir.iterdir():
                        if chunk_dir.is_dir() and chunk_dir.name.startswith('chunk-'):
                            print(f"Found chunk directory: {chunk_dir}")
                            for item in chunk_dir.iterdir():
                                if item.is_dir():
                                    # 查找视频文件
                                    for video_file in item.iterdir():
                                        if video_file.suffix.lower() in ['.mp4', '.avi', '.mov']:
                                            if f"episode_{episode_id:06d}" in video_file.name:
                                                video_paths.append(video_file)
                                                print(f"Found matching video: {video_file}")
                else:
                    print(f"Videos directory does not exist: {videos_dir}")
                    video_paths = []
            
            # 生成相对于videos目录的路径，使用subdir前缀
            videos_info = []
            for video_path in video_paths:
                try:
                    # 获取相对于dataset root/videos的路径
                    relative_video_path = video_path.relative_to(dataset.root / "videos")
                    # 使用subdir/videos的路径格式，通过辅助函数生成URL以支持反向代理
                    video_url = get_static_url(f"{subdir_path}/videos/{relative_video_path}")
                    
                    # 尝试从路径中提取合适的文件名
                    if video_path.parent.name.startswith('observation.images.'):
                        filename = video_path.parent.name
                    else:
                        filename = video_path.parent.name if video_path.parent.name != 'videos' else video_path.stem
                    
                    videos_info.append({
                        "url": video_url,
                        "filename": filename,
                    })
                    print(f"Video URL: {video_url}")
                    print(f"Video filename: {filename}")
                except ValueError as e:
                    print(f"Error processing video path {video_path}: {e}")
                    # 如果无法计算相对路径，尝试直接使用文件名
                    try:
                        # 构建一个简单的相对路径
                        if 'chunk-' in str(video_path):
                            # 提取chunk-xxx/folder/file.mp4部分
                            path_parts = video_path.parts
                            chunk_idx = next(i for i, part in enumerate(path_parts) if part.startswith('chunk-'))
                            relative_path = '/'.join(path_parts[chunk_idx:])
                            video_url = get_static_url(f"{subdir_path}/videos/{relative_path}")
                            filename = video_path.parent.name
                            videos_info.append({
                                "url": video_url,
                                "filename": filename,
                            })
                            print(f"Fallback Video URL: {video_url}")
                    except Exception as e2:
                        print(f"Failed to create fallback video info: {e2}")
            
            print(f"Videos info created: {len(videos_info)} videos")
            
            # 获取任务信息
            if episode_id in dataset.meta.episodes:
                episode_data = dataset.meta.episodes[episode_id]
                tasks = episode_data.get("tasks", [])  # 使用.get()方法安全地获取tasks，如果不存在则返回空列表
                print(f"Tasks retrieved: {tasks}")
                print(f"Episode data keys: {list(episode_data.keys())}")  # 调试信息：显示episode数据的所有键
            else:
                print(f"Episode {episode_id} not found, using empty tasks")
                tasks = []
                
        except Exception as e:
            print(f"Error in video processing: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing videos: {e}", 500

        # 设置语言指令（如果有视频的话）
        if videos_info:
            videos_info[0]["language_instruction"] = tasks
        else:
            print("Warning: No videos found for this episode")

        episodes = list(range(dataset.num_episodes))

        print(f"Rendering template with {len(videos_info)} videos")
        try:
            return render_template(
                "visualize_dataset_template.html",
                episode_id=episode_id,
                episodes=episodes,
                dataset_info=dataset_info,
                videos_info=videos_info,
                episode_data_csv_str=episode_data_csv_str,
                columns=columns,
                ignored_columns=ignored_columns,
                subdir_path=subdir_path,
                is_lerobothtml=True,
            )
        except Exception as e:
            print(f"Error rendering template: {e}")
            import traceback
            traceback.print_exc()
            return f"Error rendering template: {e}", 500


    app.run(host=host, port=port)


def get_episode_data(dataset, episode_index):
    """Get a csv str containing timeseries data of an episode (e.g. state and action).
    This file will be loaded by Dygraph javascript to plot data in real time."""
    columns = []

    # 扩展列选择逻辑，支持object类型的数值数据
    selected_columns = []
    for col in dataset.features.keys():
        if col == "timestamp":
            continue
        
        feature_info = dataset.features[col]
        if feature_info["dtype"] in ["float32", "int32"]:
            selected_columns.append(col)
        elif feature_info["dtype"] == "object":
            # 检查是否是数值数组（不是视频数据）
            if "video" not in feature_info.get("info", {}):
                selected_columns.append(col)
    
    # 移除timestamp后再检查
    if "timestamp" in selected_columns:
        selected_columns.remove("timestamp")

    ignored_columns = []
    for column_name in selected_columns[:]:  # 使用切片复制
        shape = dataset.features[column_name]["shape"]
        shape_dim = len(shape)
        if shape_dim > 1:
            selected_columns.remove(column_name)
            ignored_columns.append(column_name)

    # init header of csv with state and action names
    header = ["timestamp"]

    for column_name in selected_columns:
        if isinstance(dataset, LeRobotDataset):
            dim_state = dataset.meta.shapes[column_name][0]
        elif isinstance(dataset, LocalLeRobotDataset):
            dim_state = dataset.features[column_name]["shape"][0]
        else:
            dim_state = dataset.features[column_name].shape[0]

        if "names" in dataset.features[column_name] and dataset.features[column_name]["names"]:
            column_names = dataset.features[column_name]["names"]
            while not isinstance(column_names, list):
                column_names = list(column_names.values())[0]
        else:
            column_names = [f"{column_name}_{i}" for i in range(dim_state)]
        columns.append({"key": column_name, "value": column_names})

        header += column_names

    selected_columns.insert(0, "timestamp")

    if isinstance(dataset, LeRobotDataset):
        from_idx = dataset.episode_data_index["from"][episode_index]
        to_idx = dataset.episode_data_index["to"][episode_index]
        data = (
            dataset.hf_dataset.select(range(from_idx, to_idx))
            .select_columns(selected_columns)
            .with_format("pandas")
        )
    elif isinstance(dataset, LocalLeRobotDataset):
        # 对于我们自己的本地数据集实现，直接从parquet读取数据
        episode_chunk = int(episode_index) // dataset.info["chunks_size"]
        data_path = dataset.root / dataset.info["data_path"].format(
            episode_chunk=episode_chunk, episode_index=episode_index
        )
        try:
            df = pd.read_parquet(data_path)
            data = df[selected_columns]  # Select specific columns
        except Exception as e:
            print(f"Error reading parquet data: {e}")
            # 创建一个空的DataFrame作为备用
            dummy_data = {col: [0] for col in selected_columns}
            data = pd.DataFrame(dummy_data)
    else:
        repo_id = dataset.repo_id

        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/" + dataset.data_path.format(
            episode_chunk=int(episode_index) // dataset.chunks_size, episode_index=episode_index
        )
        df = pd.read_parquet(url)
        data = df[selected_columns]  # Select specific columns

    rows = np.hstack(
        (
            np.expand_dims(data["timestamp"], axis=1),
            *[np.vstack(data[col]) for col in selected_columns[1:]],
        )
    ).tolist()

    # Convert data to CSV string
    csv_buffer = StringIO()
    csv_writer = csv.writer(csv_buffer)
    # Write header
    csv_writer.writerow(header)
    # Write data rows
    csv_writer.writerows(rows)
    csv_string = csv_buffer.getvalue()

    return csv_string, columns, ignored_columns



class LocalLeRobotDataset:
    """简化版的LeRobotDataset，仅用于本地数据集的可视化"""
    def __init__(self, repo_id: str, root: Path, use_direct_path: bool = False):
        self.repo_id = repo_id
        dataset_name = repo_id.split("/", 1)[1]
        
        # 如果use_direct_path=True，直接使用root作为数据集路径
        # 否则使用原来的逻辑：root/dataset_name
        if use_direct_path:
            self.root = root
        else:
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
        self.total_episodes = self.info["total_episodes"]  # 兼容性
        self.total_frames = self.info["total_frames"]      # 兼容性
        # 识别视频/图像键：支持video和image类型，因为LeRobot中图像通常压缩为视频存储
        self.video_keys = [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]
        
        # 调试信息：打印视频键识别结果
        print(f"Dataset features: {list(self.features.keys())}")
        print(f"Features with video dtype: {[(key, ft['dtype']) for key, ft in self.features.items() if ft['dtype'] == 'video']}")
        print(f"Features with image dtype: {[(key, ft['dtype']) for key, ft in self.features.items() if ft['dtype'] == 'image']}")
        print(f"All feature dtypes: {[(key, ft['dtype']) for key, ft in self.features.items()]}")
        print(f"Identified video_keys (video + image): {self.video_keys}")
        
        # 如果没有找到视频键，尝试通过其他方式识别
        if not self.video_keys:
            # 尝试通过文件名模式识别视频键
            video_candidates = []
            for key, ft in self.features.items():
                if any(word in key.lower() for word in ['camera', 'image', 'video', 'observation']):
                    video_candidates.append(key)
            print(f"Video candidates based on name pattern: {video_candidates}")
            
            # 如果info中有video_path配置，尝试解析出视频键
            if "video_path" in self.info:
                video_path_template = self.info["video_path"]
                print(f"Video path template: {video_path_template}")
                # 尝试从路径模板中提取视频键的占位符
                import re
                video_key_matches = re.findall(r'\{video_key\}', video_path_template)
                if video_key_matches:
                    # 如果模板中有video_key占位符，说明应该有视频
                    print("Video path template contains video_key placeholder, manually checking for videos...")
                    self.video_keys = video_candidates  # 使用候选键
        
        print(f"Final video_keys: {self.video_keys}")
        
        # 对象兼容性 - 创建一个简单的meta对象
        class SimpleMeta:
            def __init__(self, video_keys, episodes, get_video_file_path):
                self.video_keys = video_keys
                self.episodes = {ep["episode_index"]: ep for ep in episodes}
                self.get_video_file_path = get_video_file_path
        
        self.meta = SimpleMeta(self.video_keys, self.episodes, self.get_video_file_path)
        
    def get_video_file_path(self, episode_id, video_key):
        """获取视频文件路径"""
        episode_chunk = episode_id // self.info["chunks_size"]
        video_path = self.info["video_path"].format(
            episode_chunk=episode_chunk,
            video_key=video_key,
            episode_index=episode_id
        )
        return self.root / video_path

def get_dataset_info(repo_id: str, root: Path = None, subdir: str = None) -> IterableNamespace:
    # 首先尝试从本地加载数据集
    if root and repo_id.startswith("local/"):
        dataset_name = repo_id.split("/", 1)[1]
        
        # 构建实际的数据集路径：
        # 如果有subdir，直接使用 root/subdir 作为数据集路径
        # 如果没有subdir，使用 root/dataset_name
        if subdir:
            local_dataset_path = Path(root) / subdir
        else:
            local_dataset_path = Path(root) / dataset_name
        
        if local_dataset_path.exists() and (local_dataset_path / "meta" / "info.json").exists():
            try:
                # 直接读取本地的info.json文件
                with open(local_dataset_path / "meta" / "info.json", 'r') as f:
                    dataset_info = json.load(f)
                dataset_info["repo_id"] = repo_id
                return IterableNamespace(dataset_info)
            except Exception as e:
                print(f"Failed to load local dataset info {repo_id}: {e}")
    
    # 如果不是本地数据集或加载失败，从HF Hub获取
    response = requests.get(
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/info.json", timeout=5
    )
    response.raise_for_status()  # Raises an HTTPError for bad responses
    dataset_info = response.json()
    dataset_info["repo_id"] = repo_id
    return IterableNamespace(dataset_info)


def visualize_dataset_html(
    dataset: LeRobotDataset | None,
    episodes: list[int] | None = None,
    output_dir: Path | None = None,
    serve: bool = True,
    host: str = "127.0.0.1",
    port: int = 9090,
    force_override: bool = False,
    root: Path = None,
):
    init_logging()

    # 修改模板目录为当前文件所在目录的template子目录
    template_dir = Path(__file__).resolve().parent / "template"
    print(f"Using template directory: {template_dir}")

    if output_dir is None:
        print("output_dir is None")
        # Create a temporary directory that will be automatically cleaned up
        output_dir = tempfile.mkdtemp(prefix="lerobot_visualize_dataset_")

    output_dir = Path(output_dir)
    print("output_dir", output_dir)
    if output_dir.exists():
        if force_override:
            shutil.rmtree(output_dir)
        else:
            logging.info(f"Output directory already exists. Loading from it: '{output_dir}'")

    output_dir.mkdir(parents=True, exist_ok=True)

    static_dir = output_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    print("static_dir", static_dir)

    print("dssss", dataset)

    if dataset is None:
        if serve:
            run_server(
                dataset=None,
                episodes=None,
                host=host,
                port=port,
                static_folder=static_dir,
                template_folder=template_dir,
                root=root,
            )
    else:
        # Create a simlink from the dataset video folder containing mp4 files to the output directory
        # so that the http server can get access to the mp4 files.
        if isinstance(dataset, LeRobotDataset):
            ln_videos_dir = static_dir / "videos"
            if not ln_videos_dir.exists():
                ln_videos_dir.symlink_to((dataset.root / "videos").resolve().as_posix())
        elif isinstance(dataset, LocalLeRobotDataset):
            ln_videos_dir = static_dir / "videos"
            if not ln_videos_dir.exists():
                ln_videos_dir.symlink_to((dataset.root / "videos").resolve().as_posix())

        if serve:
            run_server(dataset, episodes, host, port, static_dir, template_dir, root)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Name of hugging face repositery containing a LeRobotDataset dataset (e.g. `lerobot/pusht` for https://huggingface.co/datasets/lerobot/pusht).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for a dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--load-from-hf-hub",
        type=int,
        default=0,
        help="Load videos and parquet files from HF Hub rather than local system.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to visualize (e.g. `0 1 5 6` to load episodes of index 0, 1, 5 and 6). By default loads all episodes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write html files and kickoff a web server. By default write them to 'outputs/visualize_dataset/REPO_ID'.",
    )
    parser.add_argument(
        "--serve",
        type=int,
        default=1,
        help="Launch web server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Web host used by the http server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="Web port used by the http server.",
    )
    parser.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="Delete the output directory if it exists already.",
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    load_from_hf_hub = kwargs.pop("load_from_hf_hub")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")

    dataset = None
    if repo_id:
        dataset = (
            LeRobotDataset(repo_id, root=root, tolerance_s=tolerance_s)
            if not load_from_hf_hub
            else get_dataset_info(repo_id, root)
        )

    visualize_dataset_html(dataset, root=root, **{k: v for k, v in vars(args).items() if k != 'root'})


if __name__ == "__main__":
    main()
