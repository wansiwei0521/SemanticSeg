
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from stnet.stnet import SpatioTemporalModel
from utils.utils import ModelConfig, FocalLoss, train_model
from utils.dataset import AugmentedScenarioGraphDataset
from utils.dataset_utils import NODE_TYPE_MAP, EDGE_TYPE_MAP

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')

# 滑动窗口配置
window_size = 30
step_size = 1

# 初始化配置
config = ModelConfig(
    num_layers=1,
    num_features=12 + len(NODE_TYPE_MAP),
    hidden_dim=32,
    num_relations=8,
    edge_dim=8,
    num_epochs=300,
    num_classes=3,
    window_size=window_size,
    step_size=step_size
)

working_dir = "./"
model_dir = f"{working_dir}/model"
dataset_dir = "./dataset"

# 数据集配置
scene_datasets = {
    "secondary_road": ["/secondary-road"],
    "main_secondary": ['/main-secondary'],
    "motor": [
        "/secondary-road",
        '/main-secondary'
    ],
    "ebike": ['/ebike'],
    "total": [
        "/secondary-road",
        '/main-secondary',
        '/ebike'
    ]
}

# 训练循环
for scene_name, data_dirs in scene_datasets.items():
    print(f"\n=== Training Scene: {scene_name} ===")
    
    data_dirs = [f"{dataset_dir}\driving-scene-graph{d}" for d in data_dirs]
    
    # 数据集加载
    cache_path = f"{dataset_dir}/cache/{scene_name}_{window_size}_{step_size}_dataset_aug_cache.pkl"
    generator_model_path = f"{working_dir}/model/data_aug/{scene_name}_{window_size}_{step_size}_generator_model.pth"

    # 加载数据集
    dataset = AugmentedScenarioGraphDataset(
        root_dirs=data_dirs,
        window_size=config.window_size,
        step_size=config.step_size,
        generator_model_path=generator_model_path,
        node_feature_dim=config.num_features,
        hidden_dim=64,
        device=device,
        cache_path=cache_path,
        num_classes=config.num_classes
    )
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, 
                                                                [train_size, val_size], 
                                                                generator=torch.Generator().manual_seed(0))

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # 模型初始化
    model = SpatioTemporalModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.25, gamma=2)
    # criterion = nn.CrossEntropyLoss()

    # 开始训练
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        checkpoint_dir=f"{model_dir}/checkpoint",
        bestmodel_dir=f"{model_dir}/bestmodel",
        scene_name=scene_name,
        patience=15
    )