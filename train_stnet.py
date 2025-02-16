# model training using pyg_dataset
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch_geometric.loader import DataLoader
from stnet.stnet import SpatioTemporalModel
from utils.utils import ModelConfig, FocalLoss, train_model
from utils.pyg_dataset import STGraphDataset, MultiGraphData
from utils.dataset_utils import NODE_TYPE_MAP, EDGE_TYPE_MAP

if __name__ == '__main__':
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision('high')

    # 滑动窗口配置
    window_size = 30
    step_size = 1
    num_classes = 7

    # 初始化配置
    config = ModelConfig(
        num_layers=3,
        num_features=12 + len(NODE_TYPE_MAP),
        hidden_dim=16,
        num_relations=8,
        edge_dim=8,
        num_epochs=200,
        num_classes=num_classes,
        window_size=window_size,
        step_size=step_size,
        learning_rate=1e-3,
        weight_decay=1e-4,
        graph_num_head=4,
        pool_ratio=0.9,
        num_seed_points=4,
        lstm_hidden_dim=32,
        lstm_bidirectional=False,
        lstm_num_layers=1,
        fc_dropout=0.3,
        batch_size=32,
        gnn_query_dim=16,
        gnn_num_head=1,
        gnn_num_block=8,
        agg_num_encoder_blocks=4,
        agg_num_decoder_blocks=4
    )

    working_dir = "./"
    model_dir = f"{working_dir}/model"

    scene_name = "motor"
    print(f"\n=== Training Scene: {scene_name} ===")
    # initialize wandb
    wandb.init(project="stnet-train-pygdata", config=config.__dict__)

    # 加载数据集
    work_dir = r'D:/OneDrive - chd.edu.cn/Desktop/毕业论文数据/code/STNet'
    dataset = STGraphDataset(
        root=os.path.join(work_dir, 'dataset', 'pyg_cache'),
        pkl_file_path=os.path.join(work_dir, 'dataset', 'cache', 'motor_30_1_7_dataset_aug_cache.pkl')
    )
    print(f"Dataset: {dataset}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    weights = dataset.compute_class_weights()
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, 
                                                                [train_size, val_size], 
                                                                generator=torch.Generator().manual_seed(0))

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # 模型初始化
    model = SpatioTemporalModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    print(f"class weights: {weights}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # wandb.log({"trainable_params": trainable_params})
    print(f"Trainable parameters: {trainable_params}")
    for name, module in model.named_children():
        params_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        # wandb.log({f"{name}_params": params_count})
        print(f"{name} parameters: {params_count}")
    print(model)

    # 开始训练
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        checkpoint_dir=f"{model_dir}/checkpoint",
        bestmodel_dir=f"{model_dir}/bestmodel",
        scene_name=scene_name,
        patience=15
    )