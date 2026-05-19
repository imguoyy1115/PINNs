## 运行程序
```python
!/usr/local/miniconda/envs/PINNs/bin/python /content/PINNs/Colab/continuous_time_identification\ \(Burgers\)/Burgers.py
```
## 运行结果

```python
============================================================
FINAL SUMMARY
============================================================

[Clean Data : noise = 0.0]
Error u        : 1.190491e-03
Error lambda_1 : 0.04199%
Error lambda_2 : 1.07019%
Lambda_1       : 0.999580
Lambda_2       : 0.00321716

[Noisy Data : noise = 0.01]
Error lambda_1 : 0.14758%
Error lambda_2 : 0.26941%
Lambda_1       : 0.998524
Lambda_2       : 0.00317452
============================================================
```
此输出结果对应的相关参数为：
```python
N_u = 2000
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
```
