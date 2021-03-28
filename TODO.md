


| 任务 | 状态 |
|:---:|:----:|
|修改conda源|$\checkmark$|
|创建conda环境|$\checkmark$|
|下载Reddit|$\checkmark$|
|下载Weibo||



## 修改conda源
```
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

## 创建conda环境
查看所有conda环境
```
conda env list
```

使用cmy conda环境
```
conda activate cmy
```

退出cmy conda环境
```
conda deactivate
```

cmy环境配置
- Python 3.6.13
- Pytorch 1.8.1
- DGL     0.6.0
- scikit-learn 0.24.1


## 下载DGL数据集

他们在$/home/wuyao/.dgl$文件夹下

- cora
  
  使用
  ```python
  dgl.data.CoraGraphDataset(raw_dir="/home/wuyao/.dgl/citeseer")[0]
  ```

- citeseer

  使用
  ```python
  dgl.data.CiteseerGraphDataset(raw_dir="/home/wuyao/.dgl/citeseer")[0]
  ```

- reddit

  使用
  ```python
  g = dgl.data.RedditDataset(raw_dir="/home/wuyao/.dgl/reddit")[0]
  ```