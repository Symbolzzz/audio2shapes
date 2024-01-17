from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.wav2vec_all_2 import DeepRegressionModel, process_wav, normalize_hidden_states
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

            audio_tensor =  process_wav(audio_numpy)
            # print(audio_tensor)
            x = normalize_hidden_states(audio_tensor)
            # print(x)
            # print(x.shape)
            x_test = x.reshape(-1, 768)
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
            

            input_size = 768
            hidden_sizes = [256, 128, 64] 
            output_size = 1
            model = DeepRegressionModel(input_size, hidden_sizes, output_size)

            m_state_dict = torch.load('./utils/model.pt')
            model.load_state_dict(m_state_dict)


            with torch.no_grad():
                y_preds_tensor = model(x_test_tensor)

            y_preds = y_preds_tensor.numpy()
            y = y_preds.tolist()
            
            res = {
                'code' : 0, 
                'message' : 'success', 
                'data' : {
                    'startTime' : startTime,
                    'endTime' : endTime,
                    'fps' : fps,
                    'blendshapes' : {
                        'jawOpen' : np.squeeze(y).tolist()
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

