class My_categorical_crossentropy(nn.Module):
    def __init__(self, device):
        super(My_categorical_crossentropy, self).__init__()
        self.dim = 1
        self.value = 1
        self.device = device
    def forward(self, target, output):
        target = torch.reshape(target, (target.shape[0], -1))
        target = torch.zeros(output.shape[0], output.shape[1], device=self.device).scatter_(self.dim, target, self.value)
        output = output / torch.sum(output, 1).unsqueeze(1)
        _epsilon = torch.tensor(1e-7,device=self.device)
        output = torch.clamp(output, _epsilon, 1. - _epsilon)
        target = torch.tensor(target, dtype=torch.float).to(device)
        return torch.mean(-torch.sum(torch.mul(target, torch.log(output)), -1))