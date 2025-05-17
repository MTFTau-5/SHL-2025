import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModalProjector(nn.Module):
    def __init__(self, input_dim=1071, hidden_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.proj(x)

class TemporalCNN(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.conv_blocks(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.attention_weights = None 
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        self.attention_weights = attn
        return self.out(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_ratio * embed_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * embed_dim, embed_dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MultiModalCNNTransformer(nn.Module):
    def __init__(self, num_modes=4, num_classes=10):
        super().__init__()
        self.num_modes = num_modes

        self.modal_projectors = nn.ModuleList([
            ModalProjector() for _ in range(num_modes)
        ])
        self.temporal_cnns = nn.ModuleList([
            TemporalCNN() for _ in range(num_modes)
        ])
        self.transformer = nn.Sequential(
            TransformerBlock(256, num_heads=8),
            TransformerBlock(256, num_heads=8)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * num_modes, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        modal_features = []
        for i in range(self.num_modes):
            proj_feat = self.modal_projectors[i](x[:, i]) 
            cnn_feat = self.temporal_cnns[i](proj_feat.permute(0, 2, 1))
            cnn_feat = cnn_feat.permute(0, 2, 1)
            modal_feat = cnn_feat.mean(dim=1)
            modal_features.append(modal_feat)
        combined = torch.stack(modal_features, dim=1)
        combined = self.transformer(combined) 
        combined = combined.reshape(batch_size, -1)
        return self.classifier(combined)