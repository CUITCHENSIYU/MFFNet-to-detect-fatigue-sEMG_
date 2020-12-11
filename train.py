import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from DataHelper import *
from torch.autograd import Variable
import torch.optim as optim

from model.MFFNet import *
from tools import *

#检查设备
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

#加载数据,数据格式为：（样本数，通道数。采样数）
X_cnn, X, y, _ = np.load('EMG_4class_fusion.npy',allow_pickle=True)
#Shuffle data
state = np.random.get_state()
np.random.shuffle(X_cnn)
np.random.set_state(state)
np.random.shuffle(X)
np.random.set_state(state)
np.random.shuffle(y)

X_cnn_train = X_cnn[0:18000]
X_train = X[0:18000]
y_train = y[0:18000]
batch_size=64

train_cnn_tensor =torch.tensor(X_cnn_train,dtype=torch.float)
train_tensor =torch.tensor(X_train,dtype=torch.float)
train_label_tensor = torch.from_numpy(np.array(y_train)).float()
train_dataset = TensorDataset(train_cnn_tensor, train_tensor, train_label_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

#加载模型
num_classes=4

model = FFNet().to(device)

lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr )
loss_func = My_categorical_crossentropy(device).to(device)

#训练模型
def train(model, device, train_loader, optimizer, epoch, log_interval=1):
    model.train()
    for i, data in enumerate(train_loader, 0):
        correct = 0
        # get the inputs
        inputs_cnn, inputs, labels = data
        # 将inputs与labels装进Variable中
        inputs_cnn, inputs, labels = Variable(inputs_cnn).to(device),Variable(inputs).to(device), Variable(labels).to(device)
        inputs_cnn = torch.unsqueeze(inputs_cnn, dim=1)
        inputs = torch.unsqueeze(inputs, dim=1)
        optimizer.zero_grad()
        output = model(inputs_cnn,inputs)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.long().view_as(pred)).sum().item()
        loss = loss_func(labels.long(), output)

        loss.backward()
        optimizer.step()

        if i% log_interval==0:
            print("Train Epoch: {} [{}/{} ({:0f}%)]\ttrain_Loss: {:.6f}\ttrain_acc:{:.4}".format(
                epoch, i * len(inputs), len(train_loader.dataset),
                       100. * i / len(train_loader), loss.item(),correct/batch_size)
            )

epochs = 100
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)

