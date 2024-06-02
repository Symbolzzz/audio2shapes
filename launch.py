import base64
import requests
import json
import argparse

parser = argparse.ArgumentParser(description='config of the project.')
parser.add_argument('-c', type=str, required=True, help='Path to the config file')

args = parser.parse_args()

print(args.c)

# 读取配置文件
with open(args.c, 'r') as config_file:
    config = json.load(config_file)
    
file_path = config['file_path']
url = config['url']
headers = config['headers']
data = config['post_data']


# 将音频文件以字节流存储
with open(file_path, 'rb') as f:
    audio_content = f.read()
    f.close()

# 将音频数据转换为base64编码
audio_base64 = base64.b64encode(audio_content).decode('utf-8')

data["speech"] = audio_base64

# 发送POST请求
response = requests.post(url, data=json.dumps(data), headers=headers)

# 打印服务器的响应
# print(response.text)

with open('res.json', 'w') as f:
    json.dump(json.loads(response.text), f, indent=4, sort_keys=False)

