from torch.utils.data import DataLoader, Dataset
import torch
from torch.autograd import Variable

class DataHelper(Dataset):
    # DataHelper继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小
    def __init__(self, data_cnn_tensor,data_tensor, target_tensor):
        self.data_cnn_tensor = data_cnn_tensor
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        data_cnn = self.data_cnn_tensor[index]
        data=self.data_tensor[index]
        target=self.target_tensor[index]
       
        return data_cnn, data, target

    def __len__(self):
        return self.data_tensor.size(0)


def TensorDataset(data_cnn_tensor, data_tensor, target_tensor):
    return DataHelper(data_cnn_tensor, data_tensor, target_tensor)