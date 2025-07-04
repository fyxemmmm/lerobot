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
    --root dataset --host 0.0.0.0 --port 9091

local$ open http://127.0.0.1:9091/lerobothtml?subdir=cc&episode=1&t=6.11
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
from flask import Flask, redirect, render_template, request, url_for

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

    # 使用查询参数的形式 /lerobothtml?subdir=xxx&episode=xxx
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
                    # 使用完整的subdir_path，将斜杠替换为下划线避免冲突
                    safe_subdir_name = subdir_path.replace('/', '_')
                    dataset_videos_dir = static_folder / f"videos_{safe_subdir_name}"
                    
                    # 删除现有的符号链接（无论是否损坏）
                    if dataset_videos_dir.exists() or dataset_videos_dir.is_symlink():
                        try:
                            dataset_videos_dir.unlink()
                            print(f"Removed existing dataset symlink: {dataset_videos_dir}")
                        except Exception as e:
                            print(f"Warning: Could not remove existing dataset symlink: {e}")
                            try:
                                dataset_videos_dir.unlink(missing_ok=True)
                            except:
                                pass
                    
                    # 创建新的符号链接
                    try:
                        dataset_videos_dir.symlink_to((dataset.root / "videos").resolve().as_posix())
                        print(f"Created video symlink: {dataset_videos_dir} -> {dataset.root / 'videos'}")
                    except Exception as e:
                        print(f"Error creating dataset video symlink: {e}")
                        # 如果符号链接创建失败，尝试清理并重试
                        import os
                        if dataset_videos_dir.exists():
                            os.remove(dataset_videos_dir)
                        dataset_videos_dir.symlink_to((dataset.root / "videos").resolve().as_posix())
                        print(f"Retry: Created video symlink: {dataset_videos_dir} -> {dataset.root / 'videos'}")
                    
                    # 创建通用的videos链接指向当前数据集
                    generic_videos_dir = static_folder / "videos"
                    
                    # 删除现有的通用符号链接（无论是否损坏）
                    if generic_videos_dir.exists() or generic_videos_dir.is_symlink():
                        try:
                            generic_videos_dir.unlink()
                            print(f"Removed existing generic symlink: {generic_videos_dir}")
                        except Exception as e:
                            print(f"Warning: Could not remove existing generic symlink: {e}")
                            try:
                                generic_videos_dir.unlink(missing_ok=True)
                            except:
                                pass
                    
                    # 创建新的通用符号链接
                    videos_target = (dataset.root / "videos").resolve()
                    try:
                        generic_videos_dir.symlink_to(videos_target.as_posix())
                        print(f"Created generic video symlink: {generic_videos_dir} -> {videos_target}")
                        print(f"Symlink exists: {generic_videos_dir.exists()}")
                        print(f"Target exists: {videos_target.exists()}")
                    except Exception as e:
                        print(f"Error creating generic symlink: {e}")
                        # 如果符号链接创建失败，尝试清理并重试
                        import os
                        if generic_videos_dir.exists():
                            os.remove(generic_videos_dir)
                        generic_videos_dir.symlink_to(videos_target.as_posix())
                        print(f"Retry: Created generic video symlink: {generic_videos_dir} -> {videos_target}")
                    
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

        # 处理数据集版本检查和数据获取
        try:
            dataset_version = dataset.info.get("codebase_version", "v2.1")
            
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
            video_paths = [
                dataset.get_video_file_path(episode_id, key) for key in dataset.video_keys
            ]
            print(f"Video paths: {video_paths}")
            
            videos_info = []
            for video_path in video_paths:
                # 获取相对于dataset root/videos的路径
                relative_video_path = video_path.relative_to(dataset.root / "videos")
                video_url = url_for("static", filename=f"videos/{relative_video_path}")
                videos_info.append({
                    "url": video_url,
                    "filename": video_path.parent.name,
                })
                print(f"Video URL: {video_url}")
            
            print(f"Videos info created: {len(videos_info)} videos")
            
            # 获取任务信息
            if episode_id in dataset.meta.episodes:
                tasks = dataset.meta.episodes[episode_id]["tasks"]
                print(f"Tasks retrieved: {tasks}")
            else:
                print(f"Episode {episode_id} not found, using empty tasks")
                tasks = []
                
        except Exception as e:
            print(f"Error in video processing: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing videos: {e}", 500

        videos_info[0]["language_instruction"] = tasks

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
        dim_state = dataset.features[column_name]["shape"][0]

        if "names" in dataset.features[column_name] and dataset.features[column_name]["names"]:
            column_names = dataset.features[column_name]["names"]
            while not isinstance(column_names, list):
                column_names = list(column_names.values())[0]
        else:
            column_names = [f"{column_name}_{i}" for i in range(dim_state)]
        columns.append({"key": column_name, "value": column_names})

        header += column_names

    selected_columns.insert(0, "timestamp")

    # 只处理本地数据集
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
        self.video_keys = [key for key, ft in self.features.items() if ft["dtype"] == "video"]
        
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

    template_dir = Path(__file__).resolve().parent.parent / "templates"

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
        if isinstance(dataset, LocalLeRobotDataset):
            ln_videos_dir = static_dir / "videos"
            if not ln_videos_dir.exists():
                ln_videos_dir.symlink_to((dataset.root / "videos").resolve().as_posix())

        if serve:
            run_server(dataset, episodes, host, port, static_dir, template_dir, root)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for a dataset stored locally (e.g. `--root dataset`). This is required for local datasets.",
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
        help="Directory path to write html files and kickoff a web server. By default write them to a temporary directory.",
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

    args = parser.parse_args()
    kwargs = vars(args)
    root = kwargs.pop("root")

    if not root:
        print("Error: --root parameter is required for local datasets")
        return

    dataset = None

    visualize_dataset_html(dataset, root=root, **kwargs)


if __name__ == "__main__":
    main()
