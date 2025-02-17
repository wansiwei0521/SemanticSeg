import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader
from stnet.stnet import SpatioTemporalModel
from utils.utils import ModelConfig, train_model_multigpu
from utils.pyg_dataset import STGraphDataset, MultiGraphData
from utils.dataset_utils import NODE_TYPE_MAP, EDGE_TYPE_MAP

def get_args():
    parser = argparse.ArgumentParser(description="Train STNet on multi-GPU")
    # 模型与训练参数
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in the model")
    parser.add_argument("--num_features", type=int, default=12 + len(NODE_TYPE_MAP), help="Input feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden dimension")
    parser.add_argument("--num_relations", type=int, default=8, help="Number of relations (edge types)")
    parser.add_argument("--edge_dim", type=int, default=8, help="Edge feature dimension")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--num_classes", type=int, default=7, help="Number of classes")
    parser.add_argument("--window_size", type=int, default=30, help="Temporal window size")
    parser.add_argument("--step_size", type=int, default=1, help="Step size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--graph_num_head", type=int, default=4, help="Graph attention head count")
    parser.add_argument("--pool_ratio", type=float, default=0.9, help="Pooling ratio")
    parser.add_argument("--num_seed_points", type=int, default=1, help="Number of seed points")
    parser.add_argument("--lstm_hidden_dim", type=int, default=16, help="LSTM hidden dimension")
    parser.add_argument("--lstm_bidirectional", type=lambda x: (str(x).lower() == 'true'),
                        default=False, help="LSTM bidirectional flag (True/False)")
    parser.add_argument("--lstm_num_layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--fc_dropout", type=float, default=0.3, help="Dropout rate for fully connected layers")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--gnn_query_dim", type=int, default=16, help="GNN query dimension")
    parser.add_argument("--gnn_num_head", type=int, default=1, help="GNN head count")
    # 数据集及保存路径参数
    parser.add_argument("--scene_name", type=str, default="motor", help="Scene name")
    parser.add_argument("--data_dir", type=str, default="/kaggle/input/pyg-scene-dataset", help="Dataset directory")
    parser.add_argument("--cache_dir", type=str, default="dataset/pyg_cache", help="Cache directory for processed dataset")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to save models")
    # 其他训练设置
    parser.add_argument("--patience", type=int, default=15, help="Training patience for early stopping")
    parser.add_argument("--key", type=str, default="", help="Wandb API key")
    return parser.parse_args()

def run(rank: int, world_size: int, args):
    try:
        # 分布式初始化：环境变量配置
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        device = torch.device("cuda", rank)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.set_float32_matmul_precision('high')

        # 使用命令行参数创建配置对象
        config = ModelConfig(
            num_layers=args.num_layers,
            num_features=args.num_features,
            hidden_dim=args.hidden_dim,
            num_relations=args.num_relations,
            edge_dim=args.edge_dim,
            num_epochs=args.num_epochs,
            num_classes=args.num_classes,
            window_size=args.window_size,
            step_size=args.step_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            graph_num_head=args.graph_num_head,
            pool_ratio=args.pool_ratio,
            num_seed_points=args.num_seed_points,
            lstm_hidden_dim=args.lstm_hidden_dim,
            lstm_bidirectional=args.lstm_bidirectional,
            lstm_num_layers=args.lstm_num_layers,
            fc_dropout=args.fc_dropout,
            batch_size=args.batch_size,
            gnn_query_dim=args.gnn_query_dim,
            gnn_num_head=args.gnn_num_head
        )

        working_dir = "./"
        model_dir = os.path.join(working_dir, args.model_dir)
        dataset_dir = args.data_dir

        if rank == 0:
            print(f"\n=== Training Scene: {args.scene_name} ===")

        dataset = STGraphDataset(
            root=os.path.join(working_dir, args.cache_dir),
            pkl_file_path=os.path.join(dataset_dir, f'{args.scene_name}_{config.window_size}_1_{config.num_classes}_dataset_aug_cache.pkl')
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

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler)

        # 初始化模型，并包装为 DDP 模型
        torch.manual_seed(12345)
        model = SpatioTemporalModel(config).to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        if rank == 0:
            print(f"class weights: {weights}")

        # 使用 wandb 记录训练过程
        if rank == 0:
            import wandb
            wandb.init(project="spatio-temporal", config=config.__dict__)

        # 开始训练
        train_model_multigpu(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=config,
            checkpoint_dir=os.path.join(model_dir, "checkpoint"),
            bestmodel_dir=os.path.join(model_dir, "bestmodel"),
            scene_name=args.scene_name,
            patience=args.patience
        )

        # 销毁分布式进程组
        dist.destroy_process_group()
    except Exception as e:
        print(f"Error in process {rank}: {e}")
        raise e

if __name__ == '__main__':
    args = get_args()
    import wandb
    wandb.login(key=args.key)
    sys.path.append('/kaggle/working/stnet')
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
