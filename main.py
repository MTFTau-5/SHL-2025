import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from feeder import Feeder
import time
import numpy as np
import os
import math
from net import MultiModalCNNTransformer  
from util import yaml_parser
import zhplot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_sensor_interaction(weights, epoch, save_dir="attention_maps"):
    os.makedirs(save_dir, exist_ok=True)
    
    # 平均所有头和批次维度
    weights = weights.mean(dim=0).mean(dim=0).cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        weights, 
        annot=True, 
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=["传感器1", "传感器2", "传感器3", "传感器4"],
        yticklabels=["传感器1", "传感器2", "传感器3", "传感器4"],
        vmin=0, vmax=1
    )
    plt.title(f"传感器交互强度 (Epoch {epoch})")
    plt.savefig(f"{save_dir}/sensor1234_epoch{epoch}.png", bbox_inches='tight', dpi=300)
    plt.close()

def main():
    (  
        train_data_path, 
        test_data_path, 
        valid_data_path, 
        valid_label_path,
        train_label_path,
        batch_size,
        epochs,
        cnn_channels,
        num_classes,
        num_clusters,
        update_interval,
        num_epochs,
        lr
    ) = yaml_parser()

    train_data_path = '/home/mtftau-5/work3/SHL-2025/data/fft_data/train/data.npy'
    train_label_path = '/home/mtftau-5/work3/SHL-2025/data/fft_data/train/label.npy'
    test_data_path = '/home/mtftau-5/work3/SHL-2025/data/fft_data/test/data.npy'
    valid_label_path = '/home/mtftau-5/work3/SHL-2025/data/fft_data/valid/label.npy'
    valid_data_path = '/home/mtftau-5/work3/SHL-2025/data/fft_data/valid/data.npy'

    train_dataset = Feeder(train_data_path, train_label_path)
    test_dataset = Feeder(test_data_path) 
    valid_dataset = Feeder(valid_data_path, valid_label_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    for inputs, labels in train_loader:
        print(f"输入数据的形状: {inputs.shape}") 
        break

    model = MultiModalCNNTransformer(
        num_modes=4,         
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_valid_loss = float('inf')
    best_model_path = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # ========== 新增：获取并可视化注意力权重 ==========
        model.eval()
        with torch.no_grad():

            sample_inputs, _ = next(iter(valid_loader))
            sample_inputs = sample_inputs.to(device).float()
            _ = model(sample_inputs) 

            if hasattr(model.transformer[-1].attn, 'attention_weights') and \
               model.transformer[-1].attn.attention_weights is not None:
                plot_sensor_interaction(
                    model.transformer[-1].attn.attention_weights,
                    epoch+1
                )

                np.save(
                    f"attention_maps/raw_weights_epoch{epoch+1}.npy", 
                    model.transformer[-1].attn.attention_weights.cpu().numpy()
                )

        model.eval()
        valid_loss = 0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device).float()
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                valid_total += labels.size(0)
                valid_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        valid_loss /= len(valid_loader)
        valid_acc = 100. * valid_correct / valid_total
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | '
              f'Valid Loss: {valid_loss:.4f}, Acc: {valid_acc:.2f}%')
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            best_model_path = f'/home/mtftau-5/work3/SHL-2025/output/best_model_{timestamp}.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model to {best_model_path}')

    if best_model_path is not None:
        model.load_state_dict(torch.load(best_model_path))
        print(f'Loaded best model from {best_model_path}')
    else:
        print('Warning: No best model saved during training! Using the last epoch model.')

    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, tuple) or isinstance(batch, list):
                inputs = batch[0] 
            else:
                inputs = batch
                
            inputs = inputs.to(device).float()
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
    
    np.save('test_predictions.npy', np.array(all_preds))
    print('Predictions saved to test_predictions.npy')

if __name__ == '__main__':
    main()