import torch
from IPython import display
import torchvision
import torch.nn as nn

# 读取数据
batch_size = 256 

transform = torchvision.transforms.ToTensor()
minist = torchvision.datasets.FashionMNIST(root='D:\project\data\MNIST', train=True, transform=transform, download=True)
dataload_workers = 4
train_iter = torch.utils.data.DataLoader(minist, batch_size=batch_size, shuffle=True, num_workers=dataload_workers)
test_iter  = torch.utils.data.DataLoader(minist, batch_size=batch_size, shuffle=False, num_workers=dataload_workers)

# 定义模型参数
nums_inputs, nums_outputs = 784, 10
W = nn.Parameter(torch.randn(nums_inputs, nums_outputs, requires_grad=True)*0.01)
b = nn.Parameter(torch.zeros(nums_outputs, requires_grad=True))

# 定义softmax操作
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


# 定义模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, nums_inputs)), W)+b)

# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')
