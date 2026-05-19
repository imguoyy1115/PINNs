## 运行程序
```python
!/usr/local/miniconda/envs/PINNs/bin/python /content/PINNs/Colab/discrete_time_identification\ \(Burgers\)/Burgers.py
```
## 运行结果

```python
============================================================
FINAL SUMMARY
============================================================

[Clean Data : noise = 0.0]
Error lambda_1 : 0.000423%
Error lambda_2 : 0.326319%
Lambda_1       : 0.999996
Lambda_2       : 0.00319349

[Noisy Data : noise = 0.01]
Error lambda_1 : 0.251639%
Error lambda_2 : 0.355218%
Lambda_1       : 0.997484
Lambda_2       : 0.00319441
============================================================

```
此输出结果对应的相关参数为：
```python
skip = 80
N0 = 199
N1 = 201
layers = [1, 50, 50, 50, 50, q]
```
