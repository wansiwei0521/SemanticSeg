# 测试
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import signal  # 导入 signal 模块
import os  # 导入 os 模块
import torchmetrics



# ------------------ 生成器网络 ------------------
class Generator(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, window_size, num_classes, dropout_rate=0.5):
        super().__init__()
        self.window_size = window_size
        self.gat = GATConv(node_feature_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)  # 添加 Dropout
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, graphs, z):
        # 图编码器
        graph_features = []
        for graph in graphs:
            x = self.gat(graph.x, graph.edge_index)
            x = global_mean_pool(x, graph.batch)
            graph_features.append(x)
        graph_features = torch.stack(graph_features, dim=1)  # (batch_size, window_size, hidden_dim)

        # 序列编码器
        output, (h_n, c_n) = self.lstm(graph_features)
        hidden = h_n[-1]  # (batch_size, hidden_dim)
        hidden = self.dropout(hidden)  # 应用 Dropout

        # 标签生成器
        fake_labels = self.fc(torch.cat([hidden, z], dim=1))
        return fake_labels


# ------------------ 判别器网络 ------------------
class Discriminator(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, window_size, dropout_rate=0.5):
        super().__init__()
        self.window_size = window_size
        self.gat = GATConv(node_feature_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)  # 添加 Dropout
        self.fc = nn.Linear(hidden_dim, 1)  # 输出真伪概率

    def forward(self, graphs, labels):
        # 图编码器
        graph_features = []
        for graph in graphs:
            x = self.gat(graph.x, graph.edge_index)
            x = global_mean_pool(x, graph.batch)
            graph_features.append(x)
        graph_features = torch.stack(graph_features, dim=1)  # (batch_size, window_size, hidden_dim)

        # 序列编码器
        output, (h_n, c_n) = self.lstm(graph_features)
        hidden = h_n[-1]  # (batch_size, hidden_dim)
        hidden = self.dropout(hidden)  # 应用 Dropout

        # 标签判别器
        validity = torch.sigmoid(self.fc(hidden))  # 将输出转换为概率
        return validity


# ------------------ 训练流程 ------------------
def train_gan(dataloader, generator, discriminator, g_optimizer, d_optimizer, num_epochs, device, window_size, val_dataloader=None, checkpoint_path=None, scene_name=None):
    adversarial_loss = nn.BCEWithLogitsLoss()

    best_val_metric = 0  # 初始化最佳验证指标,例如F1-score # 初始化最佳验证损失
    patience = 20  # 容忍的 epochs 数量
    epochs_without_improvement = 0  # 记录验证损失没有改善的 epochs 数量
    start_epoch = 0  # 初始化起始 epoch

    # 加载检查点（如果存在）
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        best_val_metric = checkpoint['best_val_metric']
        epochs_without_improvement = checkpoint['epochs_without_improvement']
        start_epoch = checkpoint['epoch'] + 1  # 从下一轮开始
        print(f"加载检查点 '{checkpoint_path}'，从 epoch {start_epoch} 继续训练")

    generator.train()
    discriminator.train()

    # 定义信号处理函数
    def signal_handler(sig, frame):
        print("接收到中断信号，保存检查点...")
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'best_val_metric': best_val_metric,
            'epochs_without_improvement': epochs_without_improvement
        }
        if checkpoint_path:
            torch.save(checkpoint, checkpoint_path)
        else:
            # 如果没有提供 checkpoint_path，则保存在默认路径
            default_checkpoint_path = f"/kaggle/working/model/{scene_name}_checkpoint.pth"
            torch.save(checkpoint, default_checkpoint_path)
            print(f"检查点已保存至 {default_checkpoint_path}")
        print("退出程序")
        # exit(0)

    # 注册信号处理函数 (捕捉 SIGINT 信号，即 Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    for epoch in range(start_epoch, num_epochs):
        # 使用 tqdm 包裹 dataloader，并设置 desc（描述）
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as tepoch:
            for batch_idx, (graphs, real_labels) in enumerate(tepoch):
                # 将 graphs 序列转换为 Batch 对象
                batch = Batch.from_data_list([g.to(device) for g in graphs])
                real_labels = real_labels.to(device)

                # ---------------------- 训练判别器 ----------------------
                d_optimizer.zero_grad()
                # 判别真实数据
                real_validity = discriminator(graphs, real_labels)
                real_loss = adversarial_loss(real_validity, torch.ones_like(real_validity))

                # 判别伪数据
                z = torch.randn(real_labels.size(0), 64, device=device)  # 生成随机噪声
                fake_labels = generator(graphs, z)
                fake_validity = discriminator(graphs, fake_labels)
                fake_loss = adversarial_loss(fake_validity, torch.zeros_like(fake_validity))

                d_loss = (real_loss + fake_loss) / 2

                # L2 正则化 (可以根据需要调整 weight_decay)
                l2_reg = torch.tensor(0., device=device)
                for param in discriminator.parameters():
                    l2_reg += torch.norm(param, p=2)
                d_loss += 1e-5 * l2_reg  # 添加 L2 正则化项

                d_loss.backward()  # 反向传播
                d_optimizer.step()  # 更新参数

                # ---------------------- 训练生成器 ----------------------
                g_optimizer.zero_grad()
                # 生成伪数据
                z = torch.randn(real_labels.size(0), 64, device=device)
                fake_labels = generator(graphs, z)
                fake_validity = discriminator(graphs, fake_labels)

                g_loss = adversarial_loss(fake_validity, torch.ones_like(fake_validity))

                # L2 正则化
                l2_reg = torch.tensor(0., device=device)
                for param in generator.parameters():
                    l2_reg += torch.norm(param, p=2)
                g_loss += 1e-5 * l2_reg  # 添加 L2 正则化项

                g_loss.backward()
                g_optimizer.step()

                # 更新 tqdm 的后缀，显示当前的损失
                tepoch.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())

        # 在每个 epoch 结束后评估验证集
        if val_dataloader:
            val_metric = evaluate_generator(val_dataloader, generator, device, num_classes)
            print(f"Epoch {epoch+1}: Validation F1 Score: {val_metric:.4f}") # 假设使用F1-score评估

            # 早停机制
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                epochs_without_improvement = 0
                # 在这里保存表现最好的模型
                torch.save(generator.state_dict(), generator_model_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        # 每个 epoch 结束后保存检查点
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
            'best_val_metric': best_val_metric,
            'epochs_without_improvement': epochs_without_improvement
        }
        if checkpoint_path:
            torch.save(checkpoint, checkpoint_path)
        else:
            # 如果没有提供 checkpoint_path，则保存在默认路径
            default_checkpoint_path = f"/kaggle/working/model/{scene_name}_checkpoint.pth"
            torch.save(checkpoint, default_checkpoint_path)
            print(f"检查点已保存至 {default_checkpoint_path}")
            



def evaluate_generator(val_dataloader, generator, device, num_classes):
    generator.eval()
    # accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device) # 使用Accuracy评估
    # 使用 F1Score 评估, average='micro' 表示计算所有类别的总的 F1 分数
    f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='micro').to(device) 
    with torch.no_grad():
        for graphs, labels in val_dataloader:
            batch = Batch.from_data_list([g.to(device) for g in graphs])
            labels = labels.to(device)
            z = torch.randn(labels.size(0), 64, device=device)
            fake_labels = generator(graphs, z)

            # fake_labels: [batch_size, num_classes]
            # labels: [batch_size, 1] -> 需要转换为 [batch_size]
            predictions = fake_labels.argmax(dim=1)
            # accuracy.update(predictions, labels.squeeze()) # 使用Accuracy评估
            f1_score.update(predictions, labels.squeeze())

    # return accuracy.compute().item() # 使用Accuracy评估
    return f1_score.compute().item()
    
# ------------------ 主程序示例 ------------------
if __name__ == "__main__":
    from utils.dataset import ScenarioGraphDataset
    from utils.dataset_utils import NODE_TYPE_MAP
    
    # 假设已有场景及数据路径
    scene_datasets = {
        # "secondary_road": ["/kaggle/input/driving-scene-graph/dataset/secondary-road"],
        # "ebike": ['/kaggle/input/driving-scene-graph/dataset/ebike'],
        # "main_secondary": ['/kaggle/input/driving-scene-graph/dataset/main-secondary'],
        # "motor": [
        #     "/kaggle/input/driving-scene-graph/dataset/secondary-road",
        #     '/kaggle/input/driving-scene-graph/dataset/main-secondary'
        # ],
        "total": [
            "/kaggle/input/driving-scene-graph/dataset/secondary-road",
            '/kaggle/input/driving-scene-graph/dataset/main-secondary',
            '/kaggle/input/driving-scene-graph/dataset/ebike'
        ]
    }
    # 数据集参数
    window_size = 30  # 窗口步长
    step_size = 1  # 假设的步长值
    node_feature_dim = 12 + len(NODE_TYPE_MAP)  # 节点特征维度 # 假设改为10
    num_classes = 3  # 标签类别数量

    # 训练参数
    hidden_dim = 64
    num_epochs = 100
    batch_size = 512
    num_workers = 4  # 使用 4 个进程加载数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 逐场景训练
    for scene_name, root_dirs in scene_datasets.items():
        print(f"\n=== 开始训练场景: {scene_name} ===")

        # 定义缓存路径，针对每个场景使用单独的缓存文件
        cache_path = f"/kaggle/working/cache/{scene_name}_{window_size}_{step_size}_dataset_pre_cache.pkl" 
        generator_model_path = f"/kaggle/working/model/{scene_name}_{window_size}_{step_size}_generator_model.pth"
        checkpoint_path = f"/kaggle/working/model/{scene_name}_checkpoint.pth" # 定义检查点路径

        dataset = ScenarioGraphDataset(root_dirs, window_size, step_size, device, cache_path)
        # 假设数据集的 80% 用于训练，20% 用于验证
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # 在初始化 DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # 初始化生成器和判别器, 并添加 Dropout
        generator = Generator(node_feature_dim, hidden_dim, window_size, num_classes, dropout_rate=0.5).to(device)
        discriminator = Discriminator(node_feature_dim, hidden_dim, window_size, dropout_rate=0.5).to(device)
    
        # 初始化优化器
        g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
        d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)
    
        # 训练 GAN
        train_gan(train_dataloader, generator, discriminator, g_optimizer, d_optimizer, num_epochs, device, window_size, val_dataloader, checkpoint_path, scene_name)

        # 保存模型
        # torch.save(generator.state_dict(), generator_model_path) # 已在早停中保存
        # torch.save(discriminator.state_dict(), f"{scene_name}_discriminator.pth")
        print(f"场景 {scene_name} 训练完成\n")