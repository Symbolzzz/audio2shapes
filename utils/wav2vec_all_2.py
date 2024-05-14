# -*- coding: utf-8 -*-



from transformers import AutoProcessor, Wav2Vec2Model
import torch
from sklearn.model_selection import train_test_split
import librosa
import os
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def process_folder(folder_path, model_name="facebook/wav2vec2-base-960h"):

    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    # Get a list of all .wav files in the folder
    wav_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".wav")]
    csv_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".csv")]

    all_hidden_states = []
    jawopen_values_list = []



    print(csv_files)


    for i in range(len(csv_files)):

      # Load audio file
      audio_input, rate = librosa.load(os.path.join(folder_path, wav_files[i]), sr=16000)

      # Process audio input with the processor
      inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt")

      # Forward pass through the model
      with torch.no_grad():
          outputs = model(**inputs)

      # Get the last hidden states
      last_hidden_states = outputs.last_hidden_state

      with open(os.path.join(folder_path, csv_files[i]), 'r') as file:

          data = pd.read_csv(file)
          jaw_open_values = data['jawOpen'].tolist()
          n = len(jaw_open_values)
          print("n=",n)

      n = len(last_hidden_states[0])
      jaw_open_values = jaw_open_values[0:n]
      jawopen_values_list.extend(jaw_open_values)
      all_hidden_states.append(last_hidden_states)

    return all_hidden_states,jawopen_values_list



path = "/content"

all_hidden_states,jawopen_values = process_folder(path)
print(jawopen_values)
print(all_hidden_states)

cat_hidden_states_list = torch.cat(all_hidden_states,dim=1)
cat_hidden_states_list

x_reshaped = cat_hidden_states_list.reshape(-1, 768)
x_reshaped

print(x_reshaped.shape)

len(jawopen_values)

jawopen_tensor = torch.tensor(jawopen_values, dtype=torch.float32)
y_reshaped = jawopen_tensor.reshape(-1, 1)

print(x_reshaped)
print(y_reshaped)
print(x_reshaped.shape)
print(y_reshaped.shape)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

# 假设 x 和 y 是你的数据张量
x_train, x_test, y_train, y_test = train_test_split(x_reshaped, y_reshaped, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
val_dataset = TensorDataset(x_val, y_val)

# 创建数据加载器
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=2, num_heads=8, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: [batch_size, sequence_length, input_dim]

        # Transformer Encoder
        x = self.transformer_encoder(x)  # [batch_size, sequence_length, input_dim]

        # Decoder
        x = self.decoder(x)  # [batch_size, sequence_length, output_dim]

        return x

# 定义模型
input_dim = 768
output_dim = 1
model = TransformerModel(input_dim, output_dim).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        # inputs 和 targets 已经在合适的设备上，无需再次移动

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# 定义评估函数
def evaluate_model(model, dataloader, criterion):
    model.eval()  # 将模型设置为评估模式
    total_loss = 0.0
    with torch.no_grad():  # 不计算梯度
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

    # 计算平均损失
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

# 假设你有测试数据集 test_dataset 和对应的数据加载器 test_dataloader

# 使用评估函数评估模型
test_loss = evaluate_model(model, test_dataloader, criterion)
print(f'Test Loss: {test_loss:.4f}')

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    # 训练
    model.train()
    for inputs, targets in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # 评估
    model.eval()
    val_loss = evaluate_model(model, val_dataloader, criterion)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

# 测试
test_loss = evaluate_model(model, test_dataloader, criterion)
print(f'Test Loss: {test_loss:.4f}')

import numpy as np

# 预测
model.eval()
predictions = []
with torch.no_grad():
    for inputs, _ in test_dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)

predictions.shape

def calculate_r_squared(predictions, targets):
    # 计算平均值
    y_mean = torch.mean(targets)

    # 计算总平方和
    total_sum_of_squares = torch.sum((targets - y_mean) ** 2)

    # 计算残差平方和
    residual_sum_of_squares = torch.sum((targets - predictions) ** 2)

    # 计算 R 方
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)

    return r_squared.item()

# 计算 R 方
model.eval()
predictions = []
with torch.no_grad():
    for inputs, targets in val_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        predictions.append(outputs.cpu())

predictions = torch.cat(predictions, dim=0)
r_squared = calculate_r_squared(predictions, y_val)
print(f'R squared on validation set: {r_squared:.4f}')