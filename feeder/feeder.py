import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple, Union

class Feeder(Dataset):
    
    def __init__(self, 
                 data_path: str, 
                 label_path: Optional[str] = None, 
                 window_len: int = 32, 
                 stride: int = 16, 
                 modal_dropout: bool = True, 
                 drop_strategy: str = 'random') -> None:
        self.window_len = window_len
        self.stride = stride
        self.modal_dropout = modal_dropout
        # self.modal_dropout = False
        self.drop_strategy = drop_strategy
        self.modal_importance = torch.tensor([0.25, 0.25, 0.25, 0.25]) 
        
        self.data_path = data_path
        self.label_path = label_path
        self._training = False 
        
        self._load_and_preprocess_data()
        
    def _load_and_preprocess_data(self) -> None:
        print(f"加载数据文件: {self.data_path}")
        data = np.load(self.data_path, mmap_mode='r')
        print(f"原始数据的尺寸: {data.shape}")

        if data.ndim == 4:
            self.raw_data = data.mean(axis=1) 
        elif data.ndim == 3:
            self.raw_data = data
        else:
            raise ValueError(f"不支持的数据维度: {data.ndim}")

        self.num_modes = self.raw_data.shape[0]
        self.total_frames = self.raw_data.shape[1]
        self.feature_dim = self.raw_data.shape[2]

        self.num_windows = (self.total_frames - self.window_len) // self.stride + 1
        print(f"滑动窗口分段后窗口数: {self.num_windows} (长度={self.window_len}, 步长={self.stride})")
        if self.label_path:
            print(f"加载标签文件: {self.label_path}")
            self.raw_labels = np.load(self.label_path, mmap_mode='r')
            print(f"标签数据的尺寸: {self.raw_labels.shape}")
            self.labels = self.raw_labels[self.window_len//2 :: self.stride][:self.num_windows]
        else:
            self.labels = None
    
    def __len__(self) -> int:
        return self.num_windows
    
    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        start = index * self.stride
        end = start + self.window_len
        window_data = self.raw_data[:, start:end, :]
        window_data = torch.from_numpy(window_data.copy()).float()
        if self.modal_dropout and self.training:
            window_data = self._apply_modal_dropout(window_data)
        if self.labels is not None:
            return window_data, self.labels[index]
        else:
            return window_data
    
    def _apply_modal_dropout(self, data: torch.Tensor) -> torch.Tensor:
        if self.drop_strategy == 'random':
            return self._apply_random_dropout(data)
        elif self.drop_strategy == 'importance':
            return self._apply_importance_dropout(data)
        else:
            raise ValueError(f"未知的遮挡策略: {self.drop_strategy}")
    
    def _apply_random_dropout(self, data: torch.Tensor) -> torch.Tensor:
        num_to_drop = torch.randint(1, self.num_modes, (1,)).item()
        drop_indices = torch.randperm(self.num_modes)[:num_to_drop]
        data[drop_indices] = 0
        return data
    
    def _apply_importance_dropout(self, data: torch.Tensor) -> torch.Tensor:

        probs = 0.5 * (1 - self.modal_importance)
        mask = (torch.rand(self.num_modes) > probs).float().unsqueeze(1).unsqueeze(2)
        return data * mask
    
    @property
    def training(self) -> bool:
        return self._training
    
    @training.setter
    def training(self, value: bool) -> None:
        self._training = value
