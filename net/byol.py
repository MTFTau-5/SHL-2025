import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from net import CNN

class EMA:
    def __init__(self, beta):
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Predictor(nn.Module):
    def __init__(self, in_dim=32, hidden_dim=64, out_dim=32):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class BYOL(CNN):
    def __init__(self, cnn_channels, num_classes, moving_average_decay=0.996):
        super().__init__(cnn_channels, num_classes)
        self.predictor = Predictor()
        self.target_encoder = CNN(cnn_channels, num_classes)
        self.target_ema_updater = EMA(beta=moving_average_decay)
        self.update_moving_average()
        for p in self.target_encoder.parameters():
            p.requires_grad = False

   
    
    def update_moving_average(self):
        for current_params, ma_params in zip(self.parameters(), 
                                             self.target_encoder.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.target_ema_updater.update_average(old_weight, up_weight)

    def forward(self, x1, x2):
        _, z1 = super().forward(x1)
        p1 = self.predictor(z1)

        with torch.no_grad():
            _, z2 = self.target_encoder(x2)

        
        
        loss = self.loss_fn(p1, z2)
        self.update_moving_average()
        return loss

    
    
    
    def extract_features(self, x):

        _, features = super().forward(x)
        return features

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        
        
        return 2 - 2 * (x * y).sum(dim=-1)

