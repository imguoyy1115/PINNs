# PINNs（Colab）
## 一、安装依赖环境
- MIniconda3
```python
# 安装Miniconda3
!wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!bash Miniconda3-latest-Linux-x86_64.sh -b -p /usr/local/miniconda
!rm Miniconda3-latest-Linux-x86_64.sh

# 写入环境变量（全局生效）
import os
os.environ["PATH"] = "/usr/local/miniconda/bin:" + os.environ["PATH"]

# conda授权
!conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
!conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```
- 激活虚拟环境
```python
!conda create -n PINNs python=3.7 -y
```
- CUDA 10.0
```python
!/usr/local/miniconda/bin/conda install -y -n PINNs cudatoolkit=10.0 cudnn=7.6
```

## 二、安装相关库
参考requirement.py中的库安装
- numpy 1.19.5
- scipy 1.5.4
- matplotlib 3.3.4
- mkl-service
- tensorflow-gpu 1.15.0
- pyDOE 0.3.8
- IPython
- matplotlib-inline

## 三、运行Schrodinger.py
```python
!/usr/local/miniconda/envs/PINNs/bin/python /content/PINNs/main/continuous_time_inference\ \(Schrodinger\)/Colab/Schrodinger.py
```
