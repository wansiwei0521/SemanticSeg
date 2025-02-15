
import os
import sys
import signal
import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from tqdm import tqdm
from collections import Counter
import wandb
from sklearn.metrics import balanced_accuracy_score
import torch
import torch.distributed as dist


class ModelConfig:
    def __init__(self, **kwargs):
        """支持参数动态初始化的模型配置类
        
        参数说明：
        ----------------------------
        # 图结构参数
        num_features: 节点特征维度，默认 64
        hidden_dim: 隐藏层维度，默认 128
        num_relations: 关系类型数，默认 5
        edge_dim: 边特征维度，默认 8
        
        # 图编码器参数
        graph_num_head: 图注意力头数，默认 2
        pool_ratio: 池化保留比例，默认 0.8
        num_seed_points: SetTransformer种子点数，默认 4
        graph_dropout: 图结构Dropout率，默认 0.2
        
        # 时序建模参数
        window_size: 时序窗口长度，默认 10
        lstm_hidden_dim: LSTM隐藏维度，默认 128
        lstm_bidirectional: 是否双向LSTM，默认 True
        lstm_num_layers: LSTM层数，默认 1
        
        # 分类器参数
        num_classes: 分类类别数，默认 10
        fc_dropout: 全连接层Dropout率，默认 0.5
        """
        # 训练参数
        self.num_epochs = 100
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.batch_size = 64
        
        
        # 默认参数值
        self.num_layers = 3
        self.num_features = 16
        self.hidden_dim = 32
        self.num_relations = 8
        self.edge_dim = 8
        
        self.graph_num_head = 1
        self.pool_ratio = 0.8
        self.num_seed_points = 4
        self.graph_dropout = 0
        self.gnn_query_dim = 16
        self.gnn_num_head = 2
        self.gnn_num_block = 2
        self.gnn_attention_mode = 'multiplicative-self-attention'
        
        self.window_size = 30
        self.step_size = 1
        self.lstm_hidden_dim = 32
        self.lstm_bidirectional = False
        self.lstm_num_layers = 1
        
        self.num_classes = 3
        self.fc_dropout = 0.1

        # 应用用户自定义参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid config parameter: {key}")

    def __repr__(self):
        """可视化配置参数的展示形式"""
        params = "\n".join([f"{k:20} = {v}" for k,v in self.__dict__.items()])
        return f"ModelConfig:\n{params}"

class SklearnBalancedAccuracy:
        def __init__(self):
            self.all_preds = []
            self.all_labels = []
            
        def update(self, preds, labels):
            self.all_preds.extend(preds.detach().cpu().numpy().tolist())
            self.all_labels.extend(labels.detach().cpu().numpy().tolist())
        
        def compute(self):
            score = balanced_accuracy_score(self.all_labels, self.all_preds)
            return torch.tensor(score)
        
def train_model_multigpu(model, train_loader, val_loader, optimizer, criterion, device, config, checkpoint_dir, bestmodel_dir, scene_name, patience=10, record_freq=5):
    """
    优化后的训练函数，增加了 wandb 记录训练过程
    
    :param model: 要训练的模型
    :param train_loader: 训练数据加载器
    :param val_loader: 验证数据加载器
    :param optimizer: 优化器
    :param criterion: 损失函数
    :param device: 训练设备
    :param config: 模型配置对象
    :param checkpoint_dir: 检查点保存目录
    :param scene_name: 场景名称
    :param patience: 早停耐心值
    """
    model.train()
    torch.autograd.set_detect_anomaly(True)
    best_val_metric = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }

    epochs_without_improvement = 0
    start_epoch = 0

    # 检查点路径处理
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(bestmodel_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{scene_name}_{config.window_size}_{config.step_size}_{config.num_layers}_{config.hidden_dim}_{config.num_classes}_checkpoint.pth")
    best_model_path = os.path.join(bestmodel_dir, f"{scene_name}_{config.window_size}_{config.step_size}_{config.num_layers}_{config.hidden_dim}_{config.num_classes}_best_model.pth")

    # # 加载现有检查点
    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     best_val_metric = checkpoint['best_val_metric']
    #     start_epoch = checkpoint['epoch'] + 1
    #     print(f"Loaded checkpoint from epoch {start_epoch}")

    # # 信号处理优化
    # def _signal_handler(sig, frame):
    #     print("\nInterrupt received, saving checkpoint...")
    #     _save_checkpoint(epoch, force_save=True)
    #     exit(0)

    # signal.signal(signal.SIGINT, _signal_handler)

    # 检查点保存函数
    def _save_checkpoint(current_epoch, force_save=False):
        nonlocal best_val_metric
        checkpoint = {
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_metric': best_val_metric,
            'config': vars(config)
        }
        torch.save(checkpoint, checkpoint_path)
        if force_save:
            print(f"Checkpoint saved at epoch {current_epoch}")

    # 训练循环
    try:
        for epoch in range(start_epoch, config.num_epochs):
            model.train()
            epoch_loss = 0
            
            for batch_data, batch_labels in train_loader:
                # 数据预处理
                if isinstance(batch_data, list):
                    batch_data = Batch.from_data_list(batch_data).to(device)
                else:
                    batch_data = batch_data.to(device)
                    
                batch_labels = batch_labels.to(device).squeeze()
                if batch_labels.dim() == 0:
                    batch_labels = batch_labels.unsqueeze(0)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                outputs, _ = model(
                    batch_data.x,
                    batch_data.edge_index,
                    batch_data.edge_attr,
                    batch_data.batch
                )

                # 计算损失
                loss = criterion(outputs, batch_labels)
                loss = loss ** 2
                
                # 反向传播
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                optimizer.step()

                # 更新进度
                epoch_loss += loss.item()
                torch.cuda.empty_cache()
            
            dist.barrier()
            if device.index == 0:
                print(f"Epoch {epoch+1}/{config.num_epochs} - Loss: {epoch_loss:.4f}")

            # 验证阶段
            if device.index == 0:
                val_metric = evaluate_model(model, val_loader, device, config.num_classes)
                train_loss = epoch_loss / len(train_loader)

                # 早停机制
                if val_metric['accuracy'] > best_val_metric['accuracy']:
                    best_val_metric = val_metric
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), best_model_path)
                    wandb.run.summary["best_val_f1"] = val_metric['f1']
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        break

                # 保存检查点
                _save_checkpoint(epoch)
                
                # wandb 记录
                wandb.log({
                    'epoch': epoch+1,
                    'train_loss': train_loss,
                    'val_accuracy': val_metric['accuracy'],
                    'val_precision': val_metric['precision'],
                    'val_recall': val_metric['recall'],
                    'val_f1': val_metric['f1'],
                    'balanced_accuracy': val_metric['balanced_accuracy'],
                    'cohen_kappa': val_metric['cohen_kappa'],
                    'matthews_corrcoef': val_metric['matthews_corrcoef'],
                    'best_val_accuracy': best_val_metric['accuracy'],
                    'best_val_precision': best_val_metric['precision'],
                    'best_val_recall': best_val_metric['recall'],
                    'best_val_f1': best_val_metric['f1'],
                    'best_balanced_accuracy': best_val_metric['balanced_accuracy'],
                    'best_cohen_kappa': best_val_metric['cohen_kappa'],
                    'best_matthews_corrcoef': best_val_metric['matthews_corrcoef']
                })
                
                # 每五个 epoch 验证一次 train_loader 并记录评价指标
                if (epoch + 1) % record_freq == 0:
                    train_metric = evaluate_model(model, train_loader, device, config.num_classes)
                    wandb.log({
                        'train_accuracy': train_metric['accuracy'],
                        'train_precision': train_metric['precision'],
                        'train_recall': train_metric['recall'],
                        'train_f1': train_metric['f1'],
                        'train_balanced_accuracy': train_metric['balanced_accuracy'],
                        'train_cohen_kappa': train_metric['cohen_kappa'],
                        'train_matthews_corrcoef': train_metric['matthews_corrcoef']
                    })
            
            dist.barrier()

    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        print(f"Best validation F1: {best_val_metric['f1']:.4f}")
        _save_checkpoint(epoch, force_save=True)
        wandb.finish()
    
def train_model(model, train_loader, val_loader, optimizer, criterion, device, config, checkpoint_dir, bestmodel_dir, scene_name, patience=10, record_freq=5):
    """
    优化后的训练函数，增加了 wandb 记录训练过程
    
    :param model: 要训练的模型
    :param train_loader: 训练数据加载器
    :param val_loader: 验证数据加载器
    :param optimizer: 优化器
    :param criterion: 损失函数
    :param device: 训练设备
    :param config: 模型配置对象
    :param checkpoint_dir: 检查点保存目录
    :param scene_name: 场景名称
    :param patience: 早停耐心值
    """
    model.train()
    torch.autograd.set_detect_anomaly(True)
    best_val_metric = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }

    epochs_without_improvement = 0
    start_epoch = 0

    # 检查点路径处理
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(bestmodel_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{scene_name}_{config.window_size}_{config.step_size}_{config.num_layers}_{config.hidden_dim}_{config.num_classes}_checkpoint.pth")
    best_model_path = os.path.join(bestmodel_dir, f"{scene_name}_{config.window_size}_{config.step_size}_{config.num_layers}_{config.hidden_dim}_{config.num_classes}_best_model.pth")

    # # 加载现有检查点
    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     best_val_metric = checkpoint['best_val_metric']
    #     start_epoch = checkpoint['epoch'] + 1
    #     print(f"Loaded checkpoint from epoch {start_epoch}")

    # # 信号处理优化
    # def _signal_handler(sig, frame):
    #     print("\nInterrupt received, saving checkpoint...")
    #     _save_checkpoint(epoch, force_save=True)
    #     exit(0)

    # signal.signal(signal.SIGINT, _signal_handler)

    # 检查点保存函数
    def _save_checkpoint(current_epoch, force_save=False):
        nonlocal best_val_metric
        checkpoint = {
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_metric': best_val_metric,
            'config': vars(config)
        }
        torch.save(checkpoint, checkpoint_path)
        if force_save:
            print(f"Checkpoint saved at epoch {current_epoch}")

    # 训练循环
    try:
        for epoch in range(start_epoch, config.num_epochs):
            model.train()
            epoch_loss = 0
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}") as pbar:
                for batch_data, batch_labels in pbar:
                    # 数据预处理
                    if isinstance(batch_data, list):
                        batch_data = Batch.from_data_list(batch_data).to(device)
                    else:
                        batch_data = batch_data.to(device)
                        
                    batch_labels = batch_labels.to(device).squeeze()
                    if batch_labels.dim() == 0:
                        batch_labels = batch_labels.unsqueeze(0)

                    # 梯度清零
                    optimizer.zero_grad()

                    # 前向传播
                    outputs, _ = model(
                        batch_data.x,
                        batch_data.edge_index,
                        batch_data.edge_attr,
                        batch_data.batch
                    )

                    # 计算损失
                    loss = criterion(outputs, batch_labels)
                    loss = loss ** 2
                    
                    # 反向传播
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=30, norm_type=2)
                    optimizer.step()

                    # 更新进度
                    epoch_loss += loss.item()
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                    torch.cuda.empty_cache()

            # 验证阶段
            val_metric = evaluate_model(model, val_loader, device, config.num_classes)
            train_loss = epoch_loss / len(train_loader)

            # 早停机制
            if val_metric['accuracy'] > best_val_metric['accuracy']:
                best_val_metric = val_metric
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_model_path)
                wandb.run.summary["best_val_f1"] = val_metric['f1']
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

            # 保存检查点
            _save_checkpoint(epoch)
            
            # wandb 记录
            wandb.log({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'val_accuracy': val_metric['accuracy'],
                'val_precision': val_metric['precision'],
                'val_recall': val_metric['recall'],
                'val_f1': val_metric['f1'],
                'balanced_accuracy': val_metric['balanced_accuracy'],
                'cohen_kappa': val_metric['cohen_kappa'],
                'matthews_corrcoef': val_metric['matthews_corrcoef'],
                'best_val_accuracy': best_val_metric['accuracy'],
                'best_val_precision': best_val_metric['precision'],
                'best_val_recall': best_val_metric['recall'],
                'best_val_f1': best_val_metric['f1'],
                'best_balanced_accuracy': best_val_metric['balanced_accuracy'],
                'best_cohen_kappa': best_val_metric['cohen_kappa'],
                'best_matthews_corrcoef': best_val_metric['matthews_corrcoef']
            })
            
            # 每五个 epoch 验证一次 train_loader 并记录评价指标
            if (epoch + 1) % record_freq == 0:
                train_metric = evaluate_model(model, train_loader, device, config.num_classes)
                wandb.log({
                    'train_accuracy': train_metric['accuracy'],
                    'train_precision': train_metric['precision'],
                    'train_recall': train_metric['recall'],
                    'train_f1': train_metric['f1'],
                    'train_balanced_accuracy': train_metric['balanced_accuracy'],
                    'train_cohen_kappa': train_metric['cohen_kappa'],
                    'train_matthews_corrcoef': train_metric['matthews_corrcoef']
                })

    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        print(f"Best validation F1: {best_val_metric['f1']:.4f}")
        _save_checkpoint(epoch, force_save=True)
        wandb.finish()


def evaluate_model(model, data_loader, device, num_classes):
    model.eval()
    
    # 定义多分类评价指标
    acc_metric = torchmetrics.Accuracy(
        task="multiclass",
        num_classes=num_classes,
        average="macro"
    ).to(device)
    
    precision_metric = torchmetrics.Precision(
        task="multiclass",
        num_classes=num_classes,
        average='macro'
    ).to(device)
    
    recall_metric = torchmetrics.Recall(
        task="multiclass",
        num_classes=num_classes,
        average='macro'
    ).to(device)
    
    f1_metric = torchmetrics.F1Score(
        task="multiclass",
        num_classes=num_classes,
        average='macro'
    ).to(device)
    
    # 使用sklearn的balanced_accuracy_score替代torchmetrics版本，定义一个自定义累积器
    balanced_acc_metric = SklearnBalancedAccuracy()
    cohen_kappa_metric = torchmetrics.CohenKappa(task="multiclass", num_classes=num_classes).to(device)
    matthews_corrcoef_metric = torchmetrics.MatthewsCorrCoef(task="multiclass", num_classes=num_classes).to(device)

    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            # 如果 batch_data 是一个 list[Data, Data, ...] 就用 Batch 来合并
            if isinstance(batch_data, list):
                batch_data = Batch.from_data_list(batch_data).to(device)
            else:
                batch_data = batch_data.to(device)
                
            # 前向传播得到输出
            outputs, _ = model(
                batch_data.x,
                batch_data.edge_index,
                batch_data.edge_attr,
                batch_data.batch
            )
            
            # 计算预测值
            probs = F.softmax(outputs, dim=-1)
            preds = probs.argmax(dim=1)

            # 处理标签维度，确保 preds 和 batch_labels 可以对应
            batch_labels = batch_labels.squeeze()
            if batch_labels.dim() == 0:
                batch_labels = batch_labels.unsqueeze(0)
            batch_labels = batch_labels.to(device)

            # 更新各个评价指标
            acc_metric.update(preds, batch_labels)
            precision_metric.update(preds, batch_labels)
            recall_metric.update(preds, batch_labels)
            f1_metric.update(preds, batch_labels)
            balanced_acc_metric.update(preds, batch_labels)
            cohen_kappa_metric.update(preds, batch_labels)
            matthews_corrcoef_metric.update(preds, batch_labels)

    # 计算最终评价指标
    accuracy = acc_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1 = f1_metric.compute().item()
    balanced_accuracy = balanced_acc_metric.compute().item()
    cohen_kappa = cohen_kappa_metric.compute().item()
    matthews_corrcoef = matthews_corrcoef_metric.compute().item()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'balanced_accuracy': balanced_accuracy,
        'cohen_kappa': cohen_kappa,
        'matthews_corrcoef': matthews_corrcoef
    }


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # 交叉熵损失
        pt = torch.exp(-ce_loss)  # 计算 p_t
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # Focal Loss 计算公式
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()
    
