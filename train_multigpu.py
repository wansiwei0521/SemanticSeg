import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader
from stnet.stnet import SpatioTemporalModel
from utils.utils import ModelConfig, FocalLoss, train_model
from utils.pyg_dataset import STGraphDataset, MultiGraphData
from utils.dataset_utils import NODE_TYPE_MAP, EDGE_TYPE_MAP

def run(rank: int, world_size: int):
    try:
        # 分布式初始化：根据环境变量获取 rank 和 world_size
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        
        device = torch.device("cuda", rank)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.set_float32_matmul_precision('high')
    
        # 滑动窗口配置及其他参数保持不变
        window_size = 30
        step_size = 1
        num_classes = 7
    
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
            learning_rate=0.001,
            weight_decay=0.01,
            graph_num_head=4,
            pool_ratio=0.9,
            num_seed_points=1,
            lstm_hidden_dim=16,
            lstm_bidirectional=False,
            lstm_num_layers=1,
            fc_dropout=0.3,
            batch_size=16,
            gnn_query_dim=16,
            gnn_num_head=1
        )
    
        working_dir = "./"
        model_dir = f"{working_dir}/model"
        dataset_dir = "/kaggle/input/pyg-scene-dataset"
    
        scene_name = "motor"
        if rank == 0:
            print(f"\n=== Training Scene: {scene_name} ===")
        
        dataset = STGraphDataset(
            root=os.path.join(working_dir, 'dataset', 'pyg_cache'),
            pkl_file_path=os.path.join(dataset_dir, 'motor_30_1_7_dataset_aug_cache.pkl')
        )
        if rank == 0:
            print(f"Dataset: {dataset}")
            print(f"Number of graphs: {len(dataset)}")
            print(f"Number of classes: {dataset.num_classes}")
        weights = dataset.compute_class_weights()
    
        # 划分训练集和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0)
        )
    
        # 使用 DistributedSampler 包装数据集（DDP 中 shuffle 应由 sampler 控制）
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler)
    
        # 初始化模型，并包装为 DDP 模型，启用未使用参数检测
        model = SpatioTemporalModel(config).to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        if rank == 0:
            print(f"class weights: {weights}")
    
        # 假设你使用 wandb 记录训练过程
        if rank == 0:
            import wandb
            wandb.init(project="spatio-temporal", config=config.__dict__)
        
        
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
    
        # 结束分布式进程
        dist.destroy_process_group()
    except Exception as e:
        print(f"Error in process {rank}: {e}")
        raise e

if __name__ == '__main__':
    import wandb
    wandb.login(key="b11e2943a6b14159c3d02d1c25297fadab22e7af")
    import sys
    sys.path.append('/kaggle/working/stnet')
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
