# audio2shapes API

从语音预测每一帧的 52 个 blendshape 值，输入语音，返回以 json 格式存储的 blendshape。

## 调用方式

POST

## 运行

### 实验环境

```shell
conda create -n flask python=3.9
conda activate flask
pip install -r requirements.txt
```

### 预训练模型

下载[transformer_model](https://drive.google.com/file/d/1qXvvbSW_L9mG9K9Mh_HwvoD7y2lbaIPo/view?usp=drive_link) ，并放在 `utils/` 目录下。

### 服务端

启动服务端

```shell
python app.py
```

### 客户端

先在 `config.json` 修改服务端返回的公网地址，如下图中的 `http://172.31.70.115:8888`。务必在 `config.json` 中配置正确的语音文件路径，目前支持格式：`wav`。

![截屏2023-12-13 16.58.35](https://github.com/Symbolzzz/audio2shapes/blob/main/static/images/2023-12-13%2016.58.35.png?raw=true)

然后启动客户端请求

```shell
python launch.py -c ./config.json
```

## 结果

结果会以json格式写在 `res.json` 文件中。
