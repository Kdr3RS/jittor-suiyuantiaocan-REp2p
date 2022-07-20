| 第二届计图挑战赛-草图生成风景比赛

# Jittor 随缘调参 REp2p


![主要结果](https://s1.328888.xyz/2022/07/19/l8vn5.jpg) ![主要结果](https://s1.328888.xyz/2022/07/19/l8dfS.jpg)


## 简介


本项目包含了第二届计图挑战赛计图 - 草图生成风景比赛的代码实现。本项目的特点是：在pix2pix的模型基础上引入自注意力机制，同时使用多级特征跳跃连接
。优化后的模型可在不损失mask accuracy的情况下显著提升图像的FID指标。

## 安装 


本项目运行推理需求显存为10G，如要训练请根据显卡显存调整--batch_size大小，最低需求16G。
可在RTX Titan，RTX 3090上以默认设置进行训练

#### 运行环境
- ubuntu 20.04 LTS
- python >= 3.7
- jittor >= 1.3.0

本项目基于开源机器学习框架计图(jittor)实现，计图安装方法参考 https://cg.cs.tsinghua.edu.cn/jittor/download/

#### 安装依赖
执行以下命令安装 python 依赖
```
pip install -r requirements.txt
```
#### 数据集
本项目使用的数据集由第二届计图人工智能挑战赛-风景图片生成赛题提供，[下载地址](https://www.educoder.net/competitions/index/Jittor-3).
将数据下载解压到 `<root>/Data/` 下,训练数据存放于`<root>/Data/train` ，测试数据存放于`<root>/Data/val`。
#### 预训练模型
预训练模型模型下载地址为 [点击此处](https://drive.google.com/file/d/1MhSOqBX0HAQBgcrFYGcP_ZEquW7TSqQr/view?usp=sharing)
，下载后放入目录 `<root>/result/saved_models` 下。

## 数据预处理

执行以下命令对数据预处理：
```
python imgarg.py ./Data/train/imgs
python imgarg.py ./Data/train/labels
```
注意：使用预处理扩充数据集可以提高生成图像质量,但会导致训练过程更不稳定,使用时去掉模型中spectralnorm的注释。
## 训练


运行以下命令进行训练：
```
python train.py
```
从检查点继续训练，运行
```
python train.py --epoch [epoch_number]
```
## 推理


生成测试集上的结果可以运行以下命令：

```
python test.py  --epoch [epoch_number]
```

## 致谢


此项目基于论文 *Image-to-image translation with conditional adversarial networks* 实现，代码主体结构在 [jittor-gan](https://github.com/Jittor/gan-jittor) 的基础上进行优化。

