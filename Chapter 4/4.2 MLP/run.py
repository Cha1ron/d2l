import torch
from torch import nn
import torchvision
from torchvision import transforms

batch_size = 256
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root='../data/MNIST', train=True, transform=trans, download=True)
dataloader_workers = 4
train_iter, test_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4), torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=False, num_workers=4)