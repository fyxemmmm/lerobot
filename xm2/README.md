# LeRobot数据集可视化工具

这是一个独立的LeRobot数据集可视化工具，可以在浏览器中查看LeRobot格式的数据集，包括视频和时间序列数据。

## 安装

1. 克隆或下载此项目
2. 安装依赖项：

```bash
pip install -r requirements.txt
```

## 使用方法

### 快速启动

使用提供的启动脚本：

```bash
python run_visualizer.py --root /path/to/datasets --host 0.0.0.0 --port 9091
```

参数说明：
- `--root`: 数据集根目录路径（必需）
- `--host`: 服务器主机地址，默认为127.0.0.1，使用0.0.0.0可以从外部访问
- `--port`: 服务器端口，默认为9091

### 直接使用可视化模块

也可以直接使用`visualize_dataset_html.py`：

```bash
python visualize_dataset_html.py --root /path/to/datasets --host 0.0.0.0 --port 9091
```

### 访问数据集

启动服务器后，通过浏览器访问：

```
http://localhost:9091/lerobothtml?subdir=数据集名称&episode=0
```

其中：
- `subdir`: 数据集子目录名称
- `episode`: 要查看的数据集片段编号（从0开始）

## 数据集格式要求

数据集应符合LeRobot格式，包含以下结构：

```
dataset_name/
├── meta/
│   ├── info.json
│   └── episodes.jsonl
└── videos/
    └── ...
```

## 功能特性

- 视频播放与同步
- 时间序列数据可视化
- 支持多种数据类型
- 支持键盘快捷键控制
- 支持通过URL参数指定时间点

## 注意事项

- 此工具优先使用已安装的lerobot包，如果未安装则使用内置的兼容模块
- 视频文件必须放在数据集目录的videos子目录中
- 数据文件应为parquet格式 