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

    @app.route("/")
    def hommepage(dataset=dataset):
        print("ddddd1111", dataset)
        if dataset:  # none
            dataset_namespace, dataset_name = dataset.repo_id.split("/")
            return redirect(
                url_for(
                    "show_episode",
                    dataset_namespace=dataset_namespace,
                    dataset_name=dataset_name,
                    episode_id=0,
                )
            )

        dataset_param, episode_param = None, None
        all_params = request.args
        if "dataset" in all_params:
            dataset_param = all_params["dataset"]
        if "episode" in all_params:
            episode_param = int(all_params["episode"])

        print("xxxx",dataset_param, episode_param) # none none
        if dataset_param:
            dataset_namespace, dataset_name = dataset_param.split("/")
            return redirect(
                url_for(
                    "show_episode",
                    dataset_namespace=dataset_namespace,
                    dataset_name=dataset_name,
                    episode_id=episode_param if episode_param is not None else 0,
                )
            )

        featured_datasets = [
            "lerobot/aloha_static_cups_open",
            "lerobot/columbia_cairlab_pusht_real",
            "lerobot/taco_play",
        ]
        
        # 查找本地数据集
        local_datasets = []
        if root and Path(root).exists():
            for item in Path(root).iterdir():
                if item.is_dir() and (item / "meta").exists() and (item / "videos").exists():
                    local_datasets.append(f"local/{item.name}")
        
        return render_template(
            "visualize_dataset_homepage.html",
            featured_datasets=featured_datasets,
            lerobot_datasets=available_datasets,
            local_datasets=local_datasets,
        )

    @app.route("/<string:dataset_namespace>/<string:dataset_name>")
    def show_first_episode(dataset_namespace, dataset_name):
        first_episode_id = 0
        return redirect(
            url_for(
                "show_episode",
                dataset_namespace=dataset_namespace,
                dataset_name=dataset_name,
                episode_id=first_episode_id,
            )
        )

    @app.route("/<string:dataset_namespace>/<string:dataset_name>/episode_<int:episode_id>")
    def show_episode(dataset_namespace, dataset_name, episode_id, dataset=dataset, episodes=episodes):
        repo_id = f"{dataset_namespace}/{dataset_name}"
        print(f"Handling request for: {repo_id}, episode: {episode_id}")
        
        # 从URL参数中获取subdir（保持向后兼容）
        subdir = request.args.get('subdir', None)
        print(f"URL subdir parameter: {subdir}")
        
        try:
            if dataset is None:
                if repo_id.startswith("local/") and root:
                    # 对于本地数据集，使用我们自己的简化实现，完全不依赖HF Hub
                    dataset_name = repo_id.split("/", 1)[1]
                    print(f"Loading local dataset: {dataset_name} from root: {root}, subdir: {subdir}")
                    
                    # 构建实际的数据集路径：
                    # 如果有subdir，直接使用 root/subdir 作为数据集路径
                    # 如果没有subdir，使用 root/dataset_name
                    if subdir:
                        dataset_path = Path(root) / subdir
                        print(f"Using subdir from URL, dataset path: {dataset_path}")
                    else:
                        dataset_path = Path(root) / dataset_name
                        print(f"No subdir in URL, dataset path: {dataset_path}")
                    
                    expected_path = dataset_path / 'meta' / 'info.json'
                    print(f"Expected path: {expected_path}")
                    
                    try:
                        # 直接传递数据集路径，不再在LocalLeRobotDataset中添加dataset_name
                        dataset = LocalLeRobotDataset(repo_id, dataset_path, use_direct_path=True)
                        print(f"Successfully loaded local dataset: {repo_id}")
                        
                        # 为这个本地数据集创建视频符号链接
                        static_folder = Path(app.static_folder)
                        dataset_videos_dir = static_folder / f"videos_{dataset_name}"
                        if not dataset_videos_dir.exists():
                            dataset_videos_dir.symlink_to((dataset.root / "videos").resolve().as_posix())
                            print(f"Created video symlink: {dataset_videos_dir} -> {dataset.root / 'videos'}")
                        
                        # 创建通用的videos链接指向当前数据集
                        generic_videos_dir = static_folder / "videos"
                        
                        # 删除现有的符号链接（无论是否损坏）
                        if generic_videos_dir.exists() or generic_videos_dir.is_symlink():
                            try:
                                generic_videos_dir.unlink()
                                print(f"Removed existing symlink: {generic_videos_dir}")
                            except Exception as e:
                                print(f"Warning: Could not remove existing symlink: {e}")
                                # 强制删除，即使是损坏的符号链接
                                try:
                                    generic_videos_dir.unlink(missing_ok=True)
                                except:
                                    pass
                        
                        videos_target = (dataset.root / "videos").resolve()
                        try:
                            generic_videos_dir.symlink_to(videos_target.as_posix())
                            print(f"Created video symlink: {generic_videos_dir} -> {videos_target}")
                            print(f"Symlink exists: {generic_videos_dir.exists()}")
                            print(f"Target exists: {videos_target.exists()}")
                        except Exception as e:
                            print(f"Error creating symlink: {e}")
                            # 如果符号链接创建失败，尝试清理并重试
                            import os
                            if generic_videos_dir.exists():
                                os.remove(generic_videos_dir)
                            generic_videos_dir.symlink_to(videos_target.as_posix())
                            print(f"Retry: Created video symlink: {generic_videos_dir} -> {videos_target}")
                        
                    except Exception as e:
                        print(f"Error loading local dataset with custom loader: {e}")
                        import traceback
                        traceback.print_exc()
                        return f"Error loading local dataset {repo_id}: {e}", 500
                else:
                    dataset = get_dataset_info(repo_id, root, subdir)
        except Exception as e:
            print(f"Error loading dataset {repo_id}: {e}")
            import traceback
            traceback.print_exc()
            return f"Error loading dataset {repo_id}: {e}", 500
        except FileNotFoundError:
            return (
                "Make sure to convert your LeRobotDataset to v2 & above. See how to convert your dataset at https://github.com/huggingface/lerobot/pull/461",
                400,
            )
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
            if isinstance(dataset, LocalLeRobotDataset):
                dataset_info = {
                    "repo_id": f"{dataset_namespace}/{dataset_name}",
                    "num_samples": dataset.num_frames,
                    "num_episodes": dataset.num_episodes,
                    "fps": dataset.fps,
                }
            else:
                dataset_info = {
                    "repo_id": f"{dataset_namespace}/{dataset_name}",
                    "num_samples": dataset.num_frames
                    if isinstance(dataset, LeRobotDataset)
                    else dataset.total_frames,
                    "num_episodes": dataset.num_episodes
                    if isinstance(dataset, LeRobotDataset)
                    else dataset.total_episodes,
                    "fps": dataset.fps,
                }
            print(f"Dataset info created successfully: {dataset_info}")
        except Exception as e:
            print(f"Error creating dataset_info: {e}")
            import traceback
            traceback.print_exc()
            return f"Error creating dataset info: {e}", 500
        if isinstance(dataset, LeRobotDataset):
            video_paths = [
                dataset.meta.get_video_file_path(episode_id, key) for key in dataset.meta.video_keys
            ]
            videos_info = [
                {
                    "url": url_for("static", filename=str(video_path).replace("\\", "/")),
                    "filename": video_path.parent.name,
                }
                for video_path in video_paths
            ]
            tasks = dataset.meta.episodes[episode_id]["tasks"]
        elif isinstance(dataset, LocalLeRobotDataset):
            print(f"Processing LocalLeRobotDataset videos for episode {episode_id}")
            try:
                video_paths = [
                    dataset.get_video_file_path(episode_id, key) for key in dataset.video_keys
                ]
                print(f"Video paths: {video_paths}")
                
                # 生成相对于videos目录的路径，因为我们创建了符号链接
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
                
                # print(f"Available episode keys: {list(dataset.meta.episodes.keys())}")
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
        else:
            video_keys = [key for key, ft in dataset.features.items() if ft["dtype"] == "video"]
            videos_info = [
                {
                    "url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
                    + dataset.video_path.format(
                        episode_chunk=int(episode_id) // dataset.chunks_size,
                        video_key=video_key,
                        episode_index=episode_id,
                    ),
                    "filename": video_key,
                }
                for video_key in video_keys
            ]

            response = requests.get(
                f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/episodes.jsonl", timeout=5
            )
            response.raise_for_status()
            # Split into lines and parse each line as JSON
            tasks_jsonl = [json.loads(line) for line in response.text.splitlines() if line.strip()]

            filtered_tasks_jsonl = [row for row in tasks_jsonl if row["episode_index"] == episode_id]
            tasks = filtered_tasks_jsonl[0]["tasks"]

        videos_info[0]["language_instruction"] = tasks

        if episodes is None:
            if isinstance(dataset, LocalLeRobotDataset):
                episodes = list(range(dataset.num_episodes))
            else:
                episodes = list(
                    range(dataset.num_episodes if isinstance(dataset, LeRobotDataset) else dataset.total_episodes)
                )

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
            )
        except Exception as e:
            print(f"Error rendering template: {e}")
            import traceback
            traceback.print_exc()
            return f"Error rendering template: {e}", 500

    # 新的路由：处理subdir作为路径参数的情况
    @app.route("/subdir/<path:subdir_path>/episode_<int:episode_id>")
    def show_episode_with_subdir(subdir_path, episode_id, dataset=dataset, episodes=episodes):
        print(f"Handling subdir request for: {subdir_path}, episode: {episode_id}")
        
        try:
            if dataset is None and root:
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
                    dataset_name = subdir_path.split('/')[-1]  # 使用最后一部分作为数据集名称
                    dataset_videos_dir = static_folder / f"videos_{dataset_name}"
                    if not dataset_videos_dir.exists():
                        dataset_videos_dir.symlink_to((dataset.root / "videos").resolve().as_posix())
                        print(f"Created video symlink: {dataset_videos_dir} -> {dataset.root / 'videos'}")
                    
                    # 创建通用的videos链接指向当前数据集
                    generic_videos_dir = static_folder / "videos"
                    
                    # 删除现有的符号链接（无论是否损坏）
                    if generic_videos_dir.exists() or generic_videos_dir.is_symlink():
                        try:
                            generic_videos_dir.unlink()
                            print(f"Removed existing symlink: {generic_videos_dir}")
                        except Exception as e:
                            print(f"Warning: Could not remove existing symlink: {e}")
                            try:
                                generic_videos_dir.unlink(missing_ok=True)
                            except:
                                pass
                    
                    videos_target = (dataset.root / "videos").resolve()
                    try:
                        generic_videos_dir.symlink_to(videos_target.as_posix())
                        print(f"Created video symlink: {generic_videos_dir} -> {videos_target}")
                        print(f"Symlink exists: {generic_videos_dir.exists()}")
                        print(f"Target exists: {videos_target.exists()}")
                    except Exception as e:
                        print(f"Error creating symlink: {e}")
                        import os
                        if generic_videos_dir.exists():
                            os.remove(generic_videos_dir)
                        generic_videos_dir.symlink_to(videos_target.as_posix())
                        print(f"Retry: Created video symlink: {generic_videos_dir} -> {videos_target}")
                    
                except Exception as e:
                    print(f"Error loading dataset from subdir: {e}")
                    import traceback
                    traceback.print_exc()
                    return f"Error loading dataset from subdir {subdir_path}: {e}", 500
            else:
                return "Dataset already loaded or root not specified", 400
                
        except Exception as e:
            print(f"Error in subdir route: {e}")
            import traceback
            traceback.print_exc()
            return f"Error processing subdir request: {e}", 500

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
                "repo_id": f"subdir/{subdir_path}",
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
            
            # 生成相对于videos目录的路径，因为我们创建了符号链接
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

        if episodes is None:
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
            )
        except Exception as e:
            print(f"Error rendering template: {e}")
            import traceback
            traceback.print_exc()
            return f"Error rendering template: {e}", 500

    app.run(host=host, port=port)


def get_ep_csv_fname(episode_id: int):
    ep_csv_fname = f"episode_{episode_id}.csv"
    return ep_csv_fname


def get_episode_data(dataset, episode_index):
    """Get a csv str containing timeseries data of an episode (e.g. state and action).
    This file will be loaded by Dygraph javascript to plot data in real time."""
    columns = []

    selected_columns = [col for col, ft in dataset.features.items() if ft["dtype"] in ["float32", "int32"]]
    selected_columns.remove("timestamp")

    ignored_columns = []
    for column_name in selected_columns:
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


def get_episode_video_paths(dataset: LeRobotDataset, ep_index: int) -> list[str]:
    # get first frame of episode (hack to get video_path of the episode)
    first_frame_idx = dataset.episode_data_index["from"][ep_index].item()
    return [
        dataset.hf_dataset.select_columns(key)[first_frame_idx][key]["path"]
        for key in dataset.meta.video_keys
    ]


def get_episode_language_instruction(dataset: LeRobotDataset, ep_index: int) -> list[str]:
    # check if the dataset has language instructions
    if "language_instruction" not in dataset.features:
        return None

    # get first frame index
    first_frame_idx = dataset.episode_data_index["from"][ep_index].item()

    language_instruction = dataset.hf_dataset[first_frame_idx]["language_instruction"]
    # TODO (michel-aractingi) hack to get the sentence, some strings in openx are badly stored
    # with the tf.tensor appearing in the string
    return language_instruction.removeprefix("tf.Tensor(b'").removesuffix("', shape=(), dtype=string)")


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
