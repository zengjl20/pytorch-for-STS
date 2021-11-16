## STS短文本对分类任务
* 本项目是针对STS短文本分类任务提出的解决方案。项目采用的是卷积神经网络和循环神经网络（LSTM）两种模型，因为数据量和模型均较小，因此可以在CPU上运行。
* 其中在LSTM模型上提供了attention机制的支持。

## Requirement
* python 3
* pytorch > 1.0
* torchtext > 0.1
* numpy

## 模型训练启动命令
* 卷积神经网络模型:  
    `python main.py`
* LSTM模型:  
    `python main.py -use-cnn=False`
* 带注意力机制的LSTM模型:  
    `python main.py -use-cnn=False -use-att=True`

## Tips
