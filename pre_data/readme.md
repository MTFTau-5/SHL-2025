## 流程

1. 解压数据集：将原始数据在`data`目录下解压，手动调整目录结构为
```
data
└───raw_data
    ├───train
    ├───valid

```
2. 数据集处理为npy格式：运行`python gen_data.py`得到数据
3. 数据集rfft处理：运行`python gen_rfft.py`得到数据
3. 最终你会得到如下所展示的目录结构与文件
```
data
└───fft_data
    ├───train
    ├───valid
├───npy_data
└───raw_data
```