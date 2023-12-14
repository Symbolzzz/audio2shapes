import base64
import requests
import json

# 准备POST请求的数据
data = {
    "format": "wav",
    "rate": 16000,
    "dev_pid": 1537,
    "channel": 1,
    "speech": ""
}

# 读取音频文件
file_path = 'audios/BAC009S0003W0200.wav'

# 将音频文件以字节流存储
with open(file_path, 'rb') as f:
    audio_content = f.read()
    f.close()

# 将音频数据转换为base64编码
audio_base64 = base64.b64encode(audio_content).decode('utf-8')

data["speech"] = audio_base64

# 发送POST请求
url = 'http://172.31.70.115:8888/audio2shapes'
headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json.dumps(data), headers=headers)

# 打印服务器的响应
# print(response.text)

with open('res.json', 'w') as f:
    json.dump(json.loads(response.text), f, indent=4, sort_keys=False)

