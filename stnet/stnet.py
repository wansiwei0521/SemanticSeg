import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv, SAGPooling, LayerNorm, BatchNorm, GraphNorm
from torch_geometric.nn.aggr import SetTransformerAggregation

class TemporalAttention(nn.Module):
    """时间注意力机制，增强时序特征提取"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        Q = self.query(x)  # [B,T,H]
        K = self.key(x)    # [B,T,H]
        V = self.value(x)  # [B,T,H]
        
        scores = torch.matmul(Q, K.transpose(1,2)) / (x.size(-1)**0.5)
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V), attn_weights  # [B,T,H], [B,T,T]



class GraphEncoder(nn.Module):
    """图编码器"""
    def __init__(self, config):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        gnn_output_dim = config.hidden_dim * config.gnn_num_head * config.gnn_query_dim
        
        self.feat_encoder = nn.Linear(config.num_features, gnn_output_dim)
        self.feat_decoder = nn.Linear(gnn_output_dim, config.hidden_dim)
        self.norms_input = GraphNorm(gnn_output_dim)
        

        for _ in range(config.num_layers):
            self.convs.append(
                RGATConv(
                    in_channels=gnn_output_dim,
                    out_channels=config.hidden_dim,
                    num_relations=config.num_relations,
                    edge_dim=config.edge_dim,
                    heads=config.gnn_num_head,
                    dim=config.gnn_query_dim,
                    num_blocks=config.gnn_num_block,
                    attention_mode=config.gnn_attention_mode,
                )
            )
            self.norms.append(GraphNorm(gnn_output_dim))
        

        self.pool = SAGPooling(config.hidden_dim, config.pool_ratio)
        self.aggr = SetTransformerAggregation(
            channels=config.hidden_dim,
            heads=config.graph_num_head,
            num_seed_points=config.num_seed_points,
            dropout=config.graph_dropout
        )

    def forward(self, x, edge_index, edge_attr, batch):
        edge_type = edge_attr.argmax(dim=1) if edge_attr.dim()>1 else edge_attr
        x = F.leaky_relu(self.feat_encoder(x), negative_slope=0.1)
        x = self.norms_input(x, batch)
        
        for _, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = x
            out = conv(x, edge_index, edge_type, edge_attr)  # (N, hidden_dim)
            out = norm(out, batch)  # BatchNorm1d 不需要额外的 batch 索引
            out = F.leaky_relu(out + residual, negative_slope=0.1)
            x = out
            
        x = F.leaky_relu(self.feat_decoder(x), negative_slope=0.1)
        
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
        self.temporal_attn = TemporalAttention(temporal_dim)
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
        lstm_feats = lstm_out[:, -1, :]
        
        # 特征融合
        combined = torch.cat([attn_feats.mean(dim=1), lstm_feats], dim=-1)
        return self.classifier(combined), attn_weights