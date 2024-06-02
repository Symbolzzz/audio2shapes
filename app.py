from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.transformer_models import predict_with_audio, TransformerModel
import torch
import base64
import librosa
import json
import io
import numpy as np

app = Flask(__name__)
CORS(app)
app.config['JSON_SORT_KEYS'] = False

@app.route('/', methods = ['GET', 'POST'])
def main():
    return 'This is an empty page. Please visit xxxx/audio2shapes.'

@app.route('/audio2shapes', methods = ['GET', 'POST'])
def audio():
    if request.method == 'GET':
        return 'audio2shapes.'
    else:
        try:
            data = request.json

            audio_base64 = data['speech']

            # 添加需要处理删除speech头部的处理
            # 删除 data:audio/wav;base64, ...
            audio_base64 = audio_base64.split(',')
            if (len(audio_base64) > 1) :
                audio_base64 = audio_base64[1]
            else:
                audio_base64 = audio_base64[0]

            audio_decoded = base64.b64decode(audio_base64)

            audio_bytes, fps = librosa.load(io.BytesIO(audio_decoded), sr=data['rate'])

            duration = librosa.get_duration(y=audio_bytes, sr=data['rate'])

            startTime = 0.0
            endTime = duration

            audio_numpy = np.frombuffer(audio_bytes, dtype=np.float32)

            input_dim = 768
            output_dim = 1
            nhead = 8
            num_encoder_layers = 8
            num_decoder_layers = 8
            dim_feedforward = 4096
            max_seq_length = 390  # 与训练时一致
            
            model = TransformerModel(input_dim=input_dim, output_dim=output_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, max_seq_length=max_seq_length)

            m_state_dict = torch.load('./utils/transformer_model.pth')
            model.load_state_dict(m_state_dict)
            
            # 使用训练好的模型进行预测
            min_jaw_open_value = 0.0  # 替换为你的实际最小值
            max_jaw_open_value = 1.0   # 替换为你的实际最大值


            with torch.no_grad():
                predicted_jaw_open_values = predict_with_audio(model, audio_numpy, min_jaw_open_value, max_jaw_open_value)
            
            res = {
                'code' : 0, 
                'message' : 'success', 
                'data' : {
                    'startTime' : startTime,
                    'endTime' : endTime,
                    'fps' : fps,
                    'blendshapes' : {
                        'jawOpen' : np.squeeze(predicted_jaw_open_values).tolist()
                    },
                },
            }
            res_json = json.dumps(res, sort_keys=False)
            
            return res_json
    
        except Exception as e:
        # 处理异常情况
            return jsonify({
                'code' : 1,
                'message': 'Error', 
                'error': str(e)})
        

if __name__ == '__main__':
    app.run("0.0.0.0", debug=True, port=8888)

