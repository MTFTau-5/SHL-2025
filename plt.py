import matplotlib.pyplot as plt
import numpy as np
import os

data = np.load('data/npy_data/train/data.npy')

def plot_data(data, modal_idx, channel_idx=0, frame_idx=0):
    plt.figure(figsize=(10, 6))
    plt.plot(data[modal_idx, channel_idx, frame_idx], label=f'Modal {modal_idx}, Channel {channel_idx}, Frame {frame_idx}')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title(f'Data for Modal {modal_idx}, Channel {channel_idx}, Frame {frame_idx}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plot_modal{modal_idx}_channel{channel_idx}_frame{frame_idx}.png')
    plt.close()
# 示例：画4个模态的第0通道第0帧
for i in range(4):
    plot_data(data, i, channel_idx=0, frame_idx=0)