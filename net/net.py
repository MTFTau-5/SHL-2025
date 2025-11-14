import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

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
    def __init__(self, num_modes=4, num_classes=4, dropout=0.3, latent_dim=128):
        super().__init__()
        self.num_modes = num_modes
        self.latent_dim = latent_dim
        self.modal_projectors = nn.ModuleList([
            ModalProjector(dropout=dropout) for _ in range(num_modes)
        ])
        self.temporal_cnns = nn.ModuleList([
            SimplifiedTemporalCNN(dropout=dropout) for _ in range(num_modes)
        ])
        self.transformer = SimplifiedTransformerBlock(256, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(256 * num_modes, num_classes),
            nn.Dropout(dropout)
        )
        
        # VAE模块：用于缺失模态的重建
        self.encoder_mu = nn.Linear(256, latent_dim)
        self.encoder_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        # 重建头：将解码后的特征重建为原始维度
        self.reconstruction_head = nn.Linear(256, 256)
       
    def forward(self, x, is_training=False, mask_prob=0.5):
        """
        Args:
            x: 输入张量，形状为 (batch_size, num_modes, time_steps, feature_dim)
            is_training: 是否在训练模式下进行模态掩蔽
            mask_prob: 模态掩蔽的概率
            
        Returns:
            logits: 分类逻辑值
            vae_outputs: VAE相关输出 (recon, mu, logvar, original_modal)，如果进行了掩蔽则返回，否则为None
        """
        batch_size = x.size(0)
        masked_idx = None
        original_modal = None
        recon = None
        mu = None
        logvar = None
        
        if is_training and random.random() < mask_prob:
            masked_idx = random.randint(0, self.num_modes - 1)
            original_x = x[:, masked_idx].clone()
            x[:, masked_idx] = torch.zeros_like(x[:, masked_idx])            

            original_proj = self.modal_projectors[masked_idx](original_x)
            original_cnn = self.temporal_cnns[masked_idx](original_proj.permute(0, 2, 1))
            original_modal = original_cnn.mean(dim=-1)
        

        modal_features = []       
        for i in range(self.num_modes):
            proj_feat = self.modal_projectors[i](x[:, i])
            cnn_feat = self.temporal_cnns[i](proj_feat.permute(0, 2, 1))
            modal_feat = cnn_feat.mean(dim=-1)
            modal_features.append(modal_feat)
       
        combined = torch.stack(modal_features, dim=1)
        combined = self.transformer(combined)
        if masked_idx is not None:
            vae_input = combined[:, masked_idx, :]
            mu = self.encoder_mu(vae_input)
            logvar = self.encoder_logvar(vae_input)
            z = self.reparameterize(mu, logvar)
            recon = self.reconstruction_head(self.decoder(z))
        
        combined = combined.reshape(batch_size, -1)
        logits = self.classifier(combined)
        vae_outputs = None
        if masked_idx is not None:
            vae_outputs = {
                'recon': recon,
                'mu': mu,
                'logvar': logvar,
                'original_modal': original_modal,
                'masked_idx': masked_idx
            }
        
        return logits, vae_outputs
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def compute_vae_loss(self, vae_outputs, beta=1.0):
        """
        计算VAE损失（重建损失 + KL散度）
        
        Args:
            vae_outputs: VAE输出字典，包含 recon, mu, logvar, original_modal
            beta: KL散度的权重参数
            
        Returns:
            总VAE损失、重建损失、KL损失
        """
        if vae_outputs is None:
            return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        
        recon = vae_outputs['recon']
        mu = vae_outputs['mu']
        logvar = vae_outputs['logvar']
        original_modal = vae_outputs['original_modal']
        recon_loss = F.mse_loss(recon, original_modal, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_vae_loss = recon_loss + beta * kl_loss
        
        return total_vae_loss, recon_loss, kl_loss

MultiModalCNNTransformer = SimplifiedMultiModalCNNTransformer