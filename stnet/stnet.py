import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv, SAGPooling, LayerNorm, BatchNorm
from torch_geometric.nn.aggr import SetTransformerAggregation

class TemporalAttention(nn.Module):
    """可学习的时间注意力机制"""
    def __init__(self, feat_dim, seg_len):
        super().__init__()
        self.query = nn.Linear(feat_dim, feat_dim)
        self.key = nn.Linear(feat_dim, feat_dim)
        self.value = nn.Linear(feat_dim, feat_dim)
        
        # 可学习缩放参数
        self.alpha = nn.Parameter(torch.tensor(1.0))  

        # 可学习注意力偏置
        self.attention_bias = nn.Parameter(torch.zeros(feat_dim, seg_len))

        # 门控机制
        self.attention_gate = nn.Linear(feat_dim, feat_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        Q, K, V = self.query(x), self.key(x), self.value(x)

        # 计算注意力分数 + 可学习偏置
        scores = torch.matmul(Q, K.transpose(1,2)) / (x.size(-1)**0.5)
        scores = self.alpha * scores + torch.matmul(Q, self.attention_bias)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # 计算门控值
        gate = self.sigmoid(self.attention_gate(Q))  # (B, T, F)

        # 门控机制让注意力动态调节
        attn_output = torch.matmul(attn_weights, V) * gate

        return attn_output, attn_weights



class GraphEncoder(nn.Module):
    """图编码器"""
    def __init__(self, config):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for idx in range(config.num_layers):
            self.convs.append(
                RGATConv(
                    config.hidden_dim if idx > 0 else config.num_features,
                    config.hidden_dim,
                    config.num_relations,
                    edge_dim=config.edge_dim
                )
            )
            
            self.norms.append(BatchNorm(config.hidden_dim))

        self.pool = SAGPooling(config.hidden_dim, config.pool_ratio)
        self.aggr = SetTransformerAggregation(
            channels=config.hidden_dim,
            heads=config.graph_num_head,
            num_seed_points=config.num_seed_points,
            dropout=config.graph_dropout
        )

    def forward(self, x, edge_index, edge_attr, batch):
        edge_type = edge_attr.argmax(dim=1) if edge_attr.dim()>1 else edge_attr

        for idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x
            out = conv(x, edge_index, edge_type, edge_attr)  # (N, hidden_dim)
            out = norm(out)  # BatchNorm1d 不需要额外的 batch 索引
            out = F.relu(out + residual if idx > 0 else out)
            x = out
        
        x, _, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        return self.aggr(x, batch), batch

class SpatioTemporalModel(nn.Module):
    """时空联合模型"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.graph_encoder = GraphEncoder(config)
        
        # 时序模块
        temporal_dim = config.hidden_dim * config.num_seed_points
        self.temporal_attn = TemporalAttention(temporal_dim, config.window_size)
        self.lstm = nn.LSTM(
            input_size=temporal_dim,
            hidden_size=config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers,
            bidirectional=config.lstm_bidirectional,
            batch_first=True
        )
        
        # 分类头
        lstm_output_dim = config.lstm_hidden_dim * (2 if config.lstm_bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim + lstm_output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.fc_dropout),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # 图编码
        graph_feats, batch = self.graph_encoder(x, edge_index, edge_attr, batch)
        
        # 时序重组
        seq_len = self.config.window_size
        batch_size = graph_feats.shape[0] // seq_len
        temporal_feats = graph_feats.view(batch_size, seq_len, -1)
        
        # 时间注意力
        attn_feats, attn_weights = self.temporal_attn(temporal_feats)
        
        # LSTM处理
        lstm_out, _ = self.lstm(temporal_feats)
        lstm_feats = lstm_out.mean(dim=1)
        
        # 特征融合
        combined = torch.cat([attn_feats.mean(dim=1), lstm_feats], dim=-1)
        return self.classifier(combined), attn_weights