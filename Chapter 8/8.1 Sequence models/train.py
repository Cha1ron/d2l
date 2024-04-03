import os 
import torch
from torch import nn
import matplotlib.pyplot as plt
#from d2l import torch as d2l

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# generate data
T = 1000
time = torch.arange(1, T+1, dtype = torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,)) #一维张量

plt.plot(time, x)
plt.xlabel('time')
plt.ylabel('x')
plt.show()

# features and labels
tau = 4
features = torch.zeros((T - tau, tau)) # T-tau 为样本数，tau为特征数（用前4个时刻t-tau到t-1来预测下一个时刻）
for i in range(tau):
    features[:, i] = x[i: T- tau + i] # i=0为样本的第一个特征，那么第一个样本t4的四个特征为[0,1,2,3]，
                                      # 第二个样本t5的四个特征为[1,2,3,4],那么i到T-tau+i的值就分别是第一、第二个一直到第T-tau个样本的第一个特征
labels = x[tau:].reshape(-1, 1)

batch_size, n_train = 16, 600   # 600个训练样本
dataset = torch.utils.data.TensorDataset(features[:n_train], labels[:n_train])
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle = True)

# define the model,拥有两个全连接层的多层感知机，ReLU激活函数和平方损失

