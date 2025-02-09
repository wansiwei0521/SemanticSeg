# 数据增强前
import os
import pickle
import networkx as nx
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from utils.dataset_utils import cluster_hr_data, precompute_pyg_data
from utils.data_aug_model import Generator
from collections import Counter
from utils.dataset_utils import NODE_TYPE_MAP, EDGE_TYPE_MAP
from utils.utils import ModelConfig
from stnet.stnet import SpatioTemporalModel
import re


class ScenarioGraphDataset(Dataset):
    def __init__(self, root_dirs: List[str], config: ModelConfig, device: torch.device, cache_path: str = None):
        super().__init__()
        self.root_dirs = root_dirs
        self.config = config
        self.window_size = config.window_size
        self.step_size = config.step_size
        self.device = device
        self.cache_path = cache_path
        self.data_list = []

         # 检查缓存
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"从缓存文件加载数据: {self.cache_path}")
            self._load_from_cache()
        else:
            print("未找到缓存文件，重新加载数据...")
            self._load_all_data()
            if self.cache_path:
                self._save_to_cache()
                
        # 将数据直接加载到目标设备
        # self._move_data_to_device()

    def _load_all_data(self):
        for root_dir in self.root_dirs:
            print(f"正在加载数据目录: {root_dir}")
            for dirpath, dirnames, filenames in os.walk(root_dir):
                scenario_files = [f for f in filenames if f.endswith(".pkl")]
                for sf in scenario_files:
                    scenario_file_full = os.path.join(dirpath, sf)
                    csv_files = sorted([f for f in filenames if f.endswith(".csv")])
                    if not csv_files:
                        print(f"[Warning] 当前路径下没有找到任何 CSV 文件，跳过: {dirpath}")
                        continue
                    tag_file_full = os.path.join(dirpath, csv_files[0])

                    with open(scenario_file_full, "rb") as f:
                        scenario_graphs = pickle.load(f)

                    labels_df = pd.read_csv(tag_file_full).dropna(subset=["HR"])
                    window_size_cls = 5
                    ranked_labels, _ = cluster_hr_data(np.array(labels_df["HR"]), window_size_cls, num_classes=self.config.num_classes)

                    if len(scenario_graphs) > len(ranked_labels):
                        print(f"[Warning] 图和标签数量不匹配, 裁剪: {scenario_file_full}")
                        print(f"[Warning] graph: {len(scenario_graphs)}, ranked_labels: {len(ranked_labels)}")
                        scenario_graphs = scenario_graphs[: len(ranked_labels)]

                    windowed_data = self._nx_to_pyg_data(
                        scenario_graphs,
                        ranked_labels,
                        self.window_size,
                        self.step_size
                    )
                    self.data_list.extend(windowed_data)

    def _nx_to_pyg_data(self,
        scenario_graphs: List[nx.Graph],
        labels: np.ndarray,
        window_size: int,
        step_size: int
    ) -> List[Tuple[List[Data], float]]:
        # 先一次性预处理所有Graph，提高滑动窗口切分效率
        precomputed_data = precompute_pyg_data(scenario_graphs)
        data_list = []
        for i in range(0, len(precomputed_data) - window_size + 1, step_size):
            window_data = precomputed_data[i : i + window_size]
            window_label = labels[i + window_size - 1]
            data_list.append((window_data, window_label))
        return data_list

    def _save_to_cache(self):
        """
        将处理后的数据保存到缓存文件。
        """
        print(f"正在将数据保存到缓存文件: {self.cache_path}")
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.data_list, f)

    def _load_from_cache(self):
        """
        从缓存文件加载数据。
        """
        with open(self.cache_path, "rb") as f:
            self.data_list = pickle.load(f)

    def _move_data_to_device(self):
        """
        将所有数据移到指定设备并转换为 GPU 张量。
        """
        print(f"将数据加载到 {self.device}...")
        for idx, (window_graphs, window_label) in enumerate(self.data_list):
            # 将每个图的数据移动到 GPU
            for graph in window_graphs:
                graph.x = graph.x.to(self.device, dtype=torch.float32)
                graph.edge_attr = graph.edge_attr.to(self.device, dtype=torch.float32)
                graph.edge_index = graph.edge_index.to(self.device, dtype=torch.long)
            # 将标签也移到 GPU
            self.data_list[idx] = (window_graphs, torch.tensor([window_label], device=self.device, dtype=torch.float32))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        window_graphs, window_label = self.data_list[idx]
        return Batch.from_data_list(window_graphs).to(self.device), torch.tensor(window_label,dtype=torch.long).to(self.device)
    
    def compute_class_weights(self):
        # 假设 dataset 中每个样本的标签在 data.y 属性中，且为标量
        labels = [int(label) for _, label in self.data_list]  
        label_counter = Counter(labels)
        
        total_samples = len(self.data_list)
        num_classes = len(label_counter)
        
        # 根据公式计算每个类别的权重
        weights = {}
        for label, count in label_counter.items():
            weights[label] = total_samples / (num_classes * count)
        
        # 转换为 tensor，按照类别索引顺序排列
        weight_list = [weights[i] for i in range(num_classes)]
        return torch.tensor(weight_list, dtype=torch.float)


class AugmentedScenarioGraphDataset(Dataset):
    def __init__(self, root_dirs: List[str], window_size: int, step_size: int, generator_model_path: str, node_feature_dim: int, num_classes:int, device: torch.device, cache_path: str = None):
        """
        初始化 AugmentedScenarioGraphDataset。
        """
        super().__init__()
        self.root_dirs = root_dirs
        self.window_size = window_size
        self.step_size = step_size
        self.device = device
        self.cache_path = cache_path  # 缓存文件路径
        self.data_list = []
        self.num_classes = num_classes

        # 检查缓存
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"从缓存文件加载数据: {self.cache_path}")
            self._load_from_cache()
        else:
            print("未找到缓存文件，重新加载数据...")
            # 加载预训练的生成器模型
            # 从文件路径中提取文件名，再用正则表达式提取参数
            model_basename = os.path.basename(generator_model_path)  # 例如 "ebike_30_1_2_16_best_model.pth"
            pattern = r"([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)"
            match = re.search(pattern, model_basename)
            if match:
                num_layers = int(match.group(3))
                hidden_dim = int(match.group(4))
                num_classes = int(match.group(5))
                print(f"Extracted num_layers: {num_layers}, hidden_dim: {hidden_dim},num_classes: {num_classes}")
            else:
                raise ValueError("模型文件名格式不正确")

            config = ModelConfig(
                num_layers=num_layers,
                num_features=node_feature_dim,
                hidden_dim=hidden_dim,
                window_size=window_size,
                step_size=step_size,
                num_classes=num_classes
            )
            self.generator = SpatioTemporalModel(config).to(device)
            state_dict = torch.load(generator_model_path, map_location=self.device, weights_only=False)
            self.generator.load_state_dict(state_dict)
            self.generator.eval()  # 切换到评估模式
            self._load_all_data()
            if self.cache_path:
                self._save_to_cache()

    def _load_all_data(self):
        """
        遍历所有 root_dirs 下的子文件夹，读取 .pkl 文件和标签 csv 文件。
        如果缺失标签或标签文件，使用生成器生成伪标签。
        """
        for root_dir in self.root_dirs:
            print(f"正在加载数据目录: {root_dir}")
            for dirpath, dirnames, filenames in os.walk(root_dir):
                scenario_files = [f for f in filenames if f.endswith(".pkl")]
                for sf in scenario_files:
                    scenario_file_full = os.path.join(dirpath, sf)
                    csv_files = sorted([f for f in filenames if f.endswith(".csv")])

                    if not csv_files:
                        print(f"[Warning] 当前路径下没有找到 CSV 文件，使用伪标签: {dirpath}")
                        tag_file_full = None
                    else:
                        tag_file_full = os.path.join(dirpath, csv_files[0])

                    # 读取场景图
                    with open(scenario_file_full, "rb") as f:
                        scenario_graphs = pickle.load(f)

                    # 处理标签
                    if tag_file_full:
                        labels_df = pd.read_csv(tag_file_full).dropna(subset=['HR'])
                        window_size_cls = 5
                        ranked_labels, _ = cluster_hr_data(np.array(labels_df["HR"]), window_size_cls)
                        # 如果图和标签长度不匹配，裁剪图数据
                        if len(scenario_graphs) > len(ranked_labels):
                            print(f"[Warning] 图和标签数量不匹配, 补充: {scenario_file_full}")
                            scenario_graphs = scenario_graphs[: len(ranked_labels)]
                    else:
                        ranked_labels = np.array([None] * len(scenario_graphs))

                    windowed_data = self._nx_to_pyg_data(
                        scenario_graphs,
                        ranked_labels,
                        self.window_size,
                        self.step_size
                    )
                    self.data_list.extend(windowed_data)

    def _generate_fake_label(self, window_graphs: List[Data]) -> float:
        """
        使用预训练的生成器模型为窗口图数据生成伪标签。

        :param window_graphs: 一个包含多个图数据的列表，每个图数据是一个 torch_geometric.data.Data 对象。
        :return: 生成的伪标签，一个整数。
        """
        # 将 window_graphs 移动到 self.device
        batch_data = Batch.from_data_list(window_graphs).to(self.device)

        fake_labels, _ = self.generator(batch_data.x,
                        batch_data.edge_index,
                        batch_data.edge_attr,
                        batch_data.batch)
        # 由于生成器现在返回的是多个标签的概率分布，我们需要取一个具有代表性的值
        # 这里取概率最大的类别的索引作为伪标签（你可以根据需要修改）
        fake_label = fake_labels.argmax(dim=1).squeeze(0).item()  # 添加 .item()
        return fake_label

    def _nx_to_pyg_data(self,
                        scenario_graphs: List[nx.Graph],
                        labels: np.ndarray,
                        window_size: int,
                        step_size: int) -> List[Tuple[List[Data], float]]:
        """
        将图数据转换为 PyG 数据，并为每个图指定标签。
        """
        preprocessed_data = precompute_pyg_data(scenario_graphs)
        data_list = []

        for i in range(0, len(scenario_graphs) - window_size + 1, step_size):
            window_data = preprocessed_data[i:i + window_size]
            real_label = labels[i + window_size - 1]

            if pd.isna(real_label):
                fake_label = self._generate_fake_label(window_data)
                label = fake_label
            else:
                label = real_label

            data_list.append((window_data, label))
        return data_list

    def _save_to_cache(self):
        """
        将处理后的数据保存到缓存文件。
        """
        print(f"正在将数据保存到缓存文件: {self.cache_path}")
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.data_list, f)

    def _load_from_cache(self):
        """
        从缓存文件加载数据。
        """
        with open(self.cache_path, "rb") as f:
            self.data_list = pickle.load(f)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        window_graphs, window_label = self.data_list[idx]
        # return window_graphs, window_label
        return Batch.from_data_list(window_graphs).to(self.device), torch.tensor(window_label,dtype=torch.long).to(self.device)
    
    def compute_class_weights(self):
        # 假设 dataset 中每个样本的标签在 data.y 属性中，且为标量
        labels = [int(label) for _, label in self.data_list]  
        label_counter = Counter(labels)
        
        total_samples = len(self.data_list)
        num_classes = len(label_counter)
        
        # 根据公式计算每个类别的权重
        weights = {}
        for label, count in label_counter.items():
            weights[label] = total_samples / (num_classes * count)
        
        # 转换为 tensor，按照类别索引顺序排列
        weight_list = [weights[i] for i in range(num_classes)]
        return torch.tensor(weight_list, dtype=torch.float)