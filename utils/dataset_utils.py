
import networkx as nx
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# Node and Edge Type Maps
NODE_TYPE_MAP = {
    "ego_car": 0,
    "road": 1,
    "lane": 2,
    "other": 3
}

EDGE_TYPE_MAP = {
    "ego_to_other": 0,
    "other_to_other": 1,
    "obj_in_front_lane": 2,
    "obj_in_left_lane": 3,
    "obj_in_right_lane": 4,
    "lane_to_road": 5,
    "ego_to_front_lane": 6
}

# 计算滑动窗口斜率
def calculate_slope(hr_data: np.ndarray, window_size: int) -> np.ndarray:
    slopes = []
    for i in range(len(hr_data) - window_size + 1):
        window = hr_data[i:i+window_size]
        x = np.arange(window_size).reshape(-1, 1)  # 时间索引
        y = window.reshape(-1, 1)  # HR数据
        model = LinearRegression()
        model.fit(x, y)
        slopes.append(model.coef_[0][0])  # 获取斜率
    return np.array(slopes)

# 计算二阶差分
def calculate_second_diff(hr_data: np.ndarray) -> np.ndarray:
    return np.diff(np.diff(hr_data))

# 计算滑动标准差
def calculate_rolling_std(hr_data: np.ndarray, window_size: int) -> np.ndarray:
    return pd.Series(hr_data).rolling(window=window_size).std().dropna().values

# 计算所有特征并进行KMeans++聚类
def cluster_hr_data(hr_data: np.ndarray, window_size: int = 10, num_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    对心率数据进行特征提取、标准化、KMeans 聚类，并对各簇的均值进行排序和标记。
    
    Args:
        hr_data (np.ndarray): 心率数据。
        window_size (int): 滑动窗口大小。
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 排序后的簇标签和每个数据点的排名。
    """
    # 计算指标
    slopes = calculate_slope(hr_data, window_size)
    second_diff = calculate_second_diff(hr_data)
    rolling_std = calculate_rolling_std(hr_data, window_size)

    # 对各个指标进行合并，确保它们的长度一致
    min_len = min(len(slopes), len(second_diff), len(rolling_std))
    features = np.vstack([
        slopes[:min_len],
        second_diff[:min_len],
        rolling_std[:min_len]
    ]).T  # 转置使得每一行代表一个数据点的特征

    # 对特征进行标准化
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # 使用KMeans++进行聚类
    kmeans = KMeans(n_clusters=num_classes, init='k-means++', n_init='auto', random_state=42)
    labels = kmeans.fit_predict(features)

    # 计算每个簇的心率均值
    cluster_means = []
    for cluster_id in np.unique(labels):
        cluster_mean = hr_data[np.where(labels == cluster_id)].mean()
        cluster_means.append((cluster_id, cluster_mean))
    
    # 按均值降序排序
    sorted_clusters = sorted(cluster_means, key=lambda x: x[1], reverse=True)
    cluster_rank = {cluster_id: rank for rank, (cluster_id, _) in enumerate(sorted_clusters)}

    # 根据排名重新标记每个数据点
    ranked_labels = np.array([cluster_rank[label] for label in labels])

    return ranked_labels, labels

CATEGORIES = [
    "Person", "Bicycle", "Cyclist", "Car", "Motorcycle", "Motorcyclist",
    "Bus", "Truck", "Traffic Light", "Animal", "Obstacle", "Pedestrian",
    "Skater", "Scooter", "Emergency Vehicle", "Stroller", "Traffic Camera",
    "Lane Control Sign", "Informational Sign", "Construction Sign", "Guide Sign",
    "Warning Sign", "Regulatory Sign", "other sign", "Lane Control Signs",
    "Informational Signs", "Construction Signs", "Guide Signs", "Warning Signs",
    "Regulatory Signs", "other signs"
]

# LabelEncoder用于编码`class_name`属性
class_encoder = LabelEncoder()
class_encoder.fit(CATEGORIES)

lane_encoder = LabelEncoder()
lane_encoder.fit(["lane_front", "lane_left", "lane_right"])

def encode_node_features(node_name: str, node_attr: dict) -> torch.Tensor:
    """
    将单个节点的属性转换为固定维度的特征向量。不同节点类型具有不同的特征提取逻辑。
    new_feat = [node_type[4], speed[3], CTO[3], class[1], track[1], entropy[4]]
    """
    feature_dim = len(NODE_TYPE_MAP) + 12  # 扩展特征维度，以容纳更多的属性
    feat = np.zeros((feature_dim,), dtype=np.float32)

    # 根据 node_type 提取不同的特征
    node_type_str = node_attr.get("node_type", "other")
    
    if node_type_str == "ego_car":
        # One-hot encoding for node type
        feat[NODE_TYPE_MAP["ego_car"]] = 1.0
        # 特定于 ego_car 类型的特征
        feat[5] = node_attr.get("speed", 0.0)  # speed   

    elif node_type_str == "road":
        # One-hot encoding for node type
        feat[NODE_TYPE_MAP["road"]] = 1.0
        # 特定于 road 类型的特征
        feat[7] = node_attr.get("ComplexityLevel", 0.0)
        feat[8] = node_attr.get("Tolerance", 0.0)
        feat[9] = node_attr.get("Occlusion", 0.0)
        

    elif node_type_str == "lane":
        # One-hot encoding for node type
        feat[NODE_TYPE_MAP["lane"]] = 1.0
        # 特定于 lane 类型的特征
        feat[7] = node_attr.get("Complexity", 0.0)
        feat[8] = node_attr.get("Tolerance", 0.0)
        feat[9] = node_attr.get("Occlusion", 0.0)
        feat[10] = lane_encoder.transform([node_name])[0]

    elif node_type_str == "other":
        # One-hot encoding for node type
        feat[NODE_TYPE_MAP["other"]] = 1.0
            
        # 提取 speed 属性
        speed_3d = node_attr.get("average_velocity_3d", [0.0, 0.0, 0.0])
        for i in range(3):
            feat[4 + i] = speed_3d[i]
        
        # 对 class_name 进行编码
        class_name = node_attr.get("class_name", "unknown")  # 默认为 "unknown"
        class_idx = class_encoder.transform([class_name])[0]  # 使用 LabelEncoder 编码
        feat[10] = class_idx
        
        # 对 track_id 进行编码
        feat[11] = node_attr.get("tracker_id", -1)
        
        # 其他属性提取
        feat[12] = node_attr.get("color_entropy", 0.0)
        feat[13] = node_attr.get("brightness_entropy", 0.0)
        feat[14] = node_attr.get("texture_entropy", 0.0)
        feat[15] = node_attr.get("traffic_entropy", 0.0)

    return torch.tensor(feat, dtype=torch.float32)


def encode_edge_features(edge_attr: dict) -> torch.Tensor:
    """
    将单个边的属性转换为固定维度的特征向量。
    """
    feature_dim = 8
    feat = np.zeros((feature_dim,), dtype=np.float32)

    # 1) weight -> 索引 0
    weight = edge_attr.get("weight", 0.0)
    weight_clipped = np.clip(weight, -500, 500)
    feat[0] = 1 / (1 + np.exp(-weight_clipped))

    # 2) edge_type -> one-hot -> 索引 [1..7]
    edge_type_str = edge_attr.get("edge_type", "other_to_other")
    type_idx = EDGE_TYPE_MAP.get(edge_type_str, EDGE_TYPE_MAP["other_to_other"])
    feat[1 + type_idx] = 1.0

    return torch.tensor(feat, dtype=torch.float32)

def precompute_pyg_data(scenario_graphs: List[nx.Graph]) -> List[Data]:
    """
    将 NetworkX 图转换为 PyG 数据。
    """
    pyg_data_list = []
    for g in scenario_graphs:
        nodes = list(g.nodes())
        node_idx_map = {n: j for j, n in enumerate(nodes)}
        x_list = [encode_node_features(n, g.nodes[n]) for n in nodes]
        x = torch.stack(x_list, dim=0)

        edge_index_list = []
        edge_attr_list = []
        for u, v, attr in g.edges(data=True):
            u_idx = node_idx_map[u]
            v_idx = node_idx_map[v]
            edge_index_list.append([u_idx, v_idx])
            edge_attr_list.append(encode_edge_features(attr))

        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr_list, dim=0)
        data = Data(
            x=x.to(torch.float32),
            edge_index=edge_index.to(torch.long),
            edge_attr=edge_attr.to(torch.float32),
            y=torch.tensor([0], dtype=torch.long)
        )
        pyg_data_list.append(data)
    return pyg_data_list