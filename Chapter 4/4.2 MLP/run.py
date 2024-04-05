import torch
from torch import nn
import torchvision
from torchvision import transforms


# 读取数据
batch_size = 256
trans = transforms.ToTensor()
#mnist = torchvision.datasets.FashionMNIST(root='../data/MNIST', train=True, transform=trans, download=True)
mnist = torchvision.datasets.FashionMNIST(root='D:\project\data\MNIST', train=True, transform=trans, download=True)
dataloader_workers = 4
train_iter= torch.utils.data.DataLoader(mnist, batch_size, shuffle=True, num_workers=4)
test_iter = torch.utils.data.DataLoader(mnist, batch_size, shuffle=False, num_workers=4)

# 定义模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256 # 28*28=784, 10, 2^8=256

W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True)*0.01, requires_grad=True)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True)) # 生成一维向量，不用指定行数1
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True)*0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

# 定义激活函数
def relu(X):
    d = torch.zeros_like(X)
    return torch.max(X, d)

# 定义模型
def mlp(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)
    return (H@W2 + b2)

# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# train
def train_epoch(net, train_iter, loss, updater):


num_epoches, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
