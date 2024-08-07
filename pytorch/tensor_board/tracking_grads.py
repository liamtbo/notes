import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(8000, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

net = ConvNet()
net(torch.rand(5,1,28,28)).mean().backward()

parameters = net.conv1.parameters()
norm_type = 2
total_norm = torch.norm(
    torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)
writer = SummaryWriter() # will output to ./runs/ dir


for name, module in net.named_children():
    norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
    writer.add_scalar(f'check_info/{name}', norm, iter)