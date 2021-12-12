## CHIP2021-Task3-临床术语标准化任务
评测网站: http://cips-chip.org.cn/2021/eval3

所有的代码都是基于我们开源的ark-nlp实现。
本次CHIP2021的临床术语标准化任务是没有A榜的，所以代码调试都是在天池的中文医疗信息处理数据集CBLUE的临床术语标准化任务上完成的

ark-nlp地址：https://github.com/xiangking/ark-nlp

中文医疗信息处理数据集CBLUE:https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414

#### 运行设备

```
Cuda:11.0
GPU:GeForce RTX 3060 * 1
显存:12GB
CPU:7核 Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
内存:20GB
硬盘:100GB SSD
```

#### 运行python环境

```
Python 3.8.10
pip install ark-nlp
pip install scikit-learn 
pip install pandas
pip install elasticsearch
pip install openpyxl
pip install python-Levenshtein
```

#### 模型简介 

模型主要分为三部分：召回、个数预测、相似预测

##### 1. 召回

将ICD文件、训练集合、清洗后的训练集加入ES创建索引

##### 2.个数预测

![1637117921242](https://upload-images.jianshu.io/upload_images/24540525-318b6f6afd7701a8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

对非标准词，直接使用文本分类对其对应的标准词个数进行预测（label分别为对应标准词一个，标准词两个和标准词两个以上）

##### 3.相似预测

将非标准词和标准词拼接成如下bert输入，预测相似性

![1637118179438](https://upload-images.jianshu.io/upload_images/24540525-2b2ae07b30bbc62e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



#### 复现步骤 

##### 1. 创建ES索引

使用docker创建容器

```
docker pull nshou/elasticsearch-kibana
docker run -it --name es-kibana -d -p 8080:9200 -p 5601:5601 nshou/elasticsearch-kibana
```

使用如下代码创建索引

```
python es_index.py
```

PS：已经于39.99.190.185:8080上提供该服务

##### 2.整体复现

```
bash ./run.sh
```

PS：创建conda环境时会出现y/n选项，请手动输入y进行环境创建

##### 3.run.sh各命令说明

* 必要的模型存储文件夹创建
  * mkdir -p ./checkpoint/textsim
  * mkdir -p ./checkpoint/predict_num

* 创建代码执行的虚拟环境
  * conda create -n goodwang python=3.8.10
  * conda activate goodwang

* 安装依赖包
  * pip install ark-nlp
  * pip install scikit-learn 
  * pip install pandas
  * pip install elasticsearch
  * pip install openpyxl
  * pip install python-Levenshtein
* 数据预处理，生成相似模型训练所需的训练数据
  * python data_process.py
* 训练
  * 训练相似模型：python textsim.py
  * 训练个数预测模型：python predictnum.py
* 预测
  * python predict.py

#### 预训练模型 

##### 1. 下载链接 

```
https://huggingface.co/nghuyong/ernie-1.0
```

##### 2. 下载完成后的处理步骤

```
不需要处理，代码中会自动下载和处理
```