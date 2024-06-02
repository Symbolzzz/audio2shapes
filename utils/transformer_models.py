import os
import torch
import librosa
import pandas as pd
import numpy as np
from transformers import AutoProcessor, Wav2Vec2Model
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(input_dim, max_seq_length)
        self.transformer = nn.Transformer(d_model=input_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, batch_first=True)
        self.fc = nn.Linear(input_dim, output_dim)
        self.target_embedding = nn.Linear(1, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def create_mask(self, src_len, tgt_len, batch_size, device):
        src_mask = torch.zeros((src_len, src_len), device=device).type(torch.bool)
        tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(device) if tgt_len > 0 else None
        src_padding_mask = torch.zeros((batch_size, src_len), device=device).type(torch.bool)
        tgt_padding_mask = torch.zeros((batch_size, tgt_len), device=device).type(torch.bool) if tgt_len > 0 else None
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def encode(self, src, src_key_padding_mask):
        src_len = src.size(1)
        batch_size = src.size(0)
        device = src.device
        src_mask, _, _, _ = self.create_mask(src_len, 0, batch_size, device)
        src = self.layer_norm(self.pos_encoder(src))
        memory = self.transformer.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(self, tgt, memory, tgt_key_padding_mask):
        tgt_len = tgt.size(1)
        batch_size = tgt.size(0)
        device = tgt.device
        _, tgt_mask, _, _ = self.create_mask(memory.size(1), tgt_len, batch_size, device)
        tgt = self.layer_norm(self.pos_encoder(self.target_embedding(tgt.unsqueeze(-1))))
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.fc(output)
        return output

    def forward(self, src, tgt=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        memory = self.encode(src, src_key_padding_mask)
        if tgt is not None:
            output = self.decode(tgt, memory, tgt_key_padding_mask)
            return output
        return memory

    def predict(self, src, src_key_padding_mask, max_len):
        self.eval()
        memory = self.encode(src, src_key_padding_mask)
        outputs = []
        tgt = torch.zeros((src.size(0), 1), device=src.device)  # 初始目标序列
        tgt_key_padding_mask = torch.zeros((src.size(0), 1), device=src.device).type(torch.bool)  # 初始目标填充掩码
        for _ in range(max_len):
            output = self.decode(tgt, memory, tgt_key_padding_mask)
            output = output[:, -1, :]
            outputs.append(output.unsqueeze(1))
            tgt = torch.cat((tgt, output), dim=1)
            tgt_key_padding_mask = torch.cat((tgt_key_padding_mask, torch.zeros((src.size(0), 1), device=src.device).type(torch.bool)), dim=1)
        return torch.cat(outputs, dim=1)


# 音频处理函数
def process_audio(audio_base64, model_name="facebook/wav2vec2-base-960h"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    # 不需要编码成 base64
    # audio_input, rate = librosa.load(file_path, sr=16000)
    inputs = processor(audio_base64, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state.squeeze(0)
    return last_hidden_states

# 反归一化函数
def denormalize(target_list, min_value, max_value):
    return [target * (max_value - min_value) + min_value for target in target_list]

# 使用自己的音频进行预测
def predict_with_audio(model, audio_base64, min_jaw_open_value, max_jaw_open_value):
    # 处理音频
    processed_audio = process_audio(audio_base64)
    
    # 创建输入掩码
    src = processed_audio.unsqueeze(0)  # (1, seq_len, 768)
    src_mask = torch.ones(src.shape[:2], device=src.device, dtype=torch.bool)  # (1, seq_len)

    # 预测
    with torch.no_grad():
        output = model.predict(src, src_key_padding_mask=~src_mask, max_len=processed_audio.shape[0])
    
    # 反归一化
    denormalized_output = denormalize(output.squeeze(-1).cpu().numpy(), min_jaw_open_value, max_jaw_open_value)
    return denormalized_output


# 加载训练好的模型
# model_path = "transformer_model.pth"
# input_dim = 768
# output_dim = 1
# nhead = 8
# num_encoder_layers = 8
# num_decoder_layers = 8
# dim_feedforward = 4096
# max_seq_length = 390  # 与训练时一致

# model = TransformerModel(input_dim, output_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length)
# model.load_state_dict(torch.load(model_path))
# model.eval()

# # 示例音频文件路径
# audio_file_path = "test.wav"

# # 使用训练好的模型进行预测
# min_jaw_open_value = 0.0  # 替换为你的实际最小值
# max_jaw_open_value = 1.0   # 替换为你的实际最大值

# predicted_jaw_open_values = predict_with_audio(model, audio_file_path, min_jaw_open_value, max_jaw_open_value)

# print(f"Predicted jaw open values: {predicted_jaw_open_values}")
