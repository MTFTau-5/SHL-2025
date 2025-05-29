import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModalProjector(nn.Module):
    def __init__(self, input_dim=1071, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.proj(x)

class SimplifiedTemporalCNN(nn.Module):
    def __init__(self, input_dim=256, dropout=0.2):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(128, 256, kernel_size=1), 
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.conv_blocks(x)

class SimplifiedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Simplified feed-forward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Single attention + FF block
        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm(x)
        x = x + self.dropout(self.ff(x))
        return x

class SimplifiedMultiModalCNNTransformer(nn.Module):
    def __init__(self, num_modes=4, num_classes=4, dropout=0.3):
        super().__init__()
        self.num_modes = num_modes

        # Projectors and CNNs
        self.modal_projectors = nn.ModuleList([
            ModalProjector(dropout=dropout) for _ in range(num_modes)
        ])
        self.temporal_cnns = nn.ModuleList([
            SimplifiedTemporalCNN(dropout=dropout) for _ in range(num_modes)
        ])
        self.transformer = SimplifiedTransformerBlock(256, dropout=dropout)
        
        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 * num_modes, num_classes),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        modal_features = []
        
        # Process each modality
        for i in range(self.num_modes):
            proj_feat = self.modal_projectors[i](x[:, i]) 
            cnn_feat = self.temporal_cnns[i](proj_feat.permute(0, 2, 1))
            modal_feat = cnn_feat.mean(dim=-1)  # Global average pooling
            modal_features.append(modal_feat)
        
        # Combine and transform
        combined = torch.stack(modal_features, dim=1)
        combined = self.transformer(combined)
        
        # Classify
        combined = combined.reshape(batch_size, -1)
        return self.classifier(combined)