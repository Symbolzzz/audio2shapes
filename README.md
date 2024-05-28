# API

## API名称

audio2shapes

## 调用方式

POST

## 运行

```
conda create -n flask python=3.9
conda activate flask
pip install -r requirements.txt
```

然后先运行 `app.py` 再运行 `launch.py` 。

![截屏2023-12-13 16.58.35](https://github.com/Symbolzzz/audio2shapes/blob/main/static/images/2023-12-13%2016.58.35.png?raw=true)

运行 `app.py` 之后会出现上述界面，将上述地址复制到 `launch.py` 中url字段。

![截屏2023-12-13 16.59.31](https://github.com/Symbolzzz/audio2shapes/blob/main/static/images/2023-12-13%2016.59.31.png?raw=true)

然后再运行 `launch.py`。

`audios` 文件夹中存储了测试音频。

## 结果

结果会以json格式写在 `res.json` 文件中。

目前使用的模型还不够完善，存在问题：

* 模型准确率不高
* 只预测了 `jawOpen` 值

接下来的工作：

* 改善模型并保存，替换目前的模型
* 再训练好 `blendshapes` 中的其他参数的模型，加入进来
