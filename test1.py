import torch

a = torch.tensor([1,2,3,4], dtype=torch.float, requires_grad=True)
b = torch.tensor([5,6,7,8], dtype=torch.float, requires_grad=True)
c = a * 5
e = b * 5
d = (c+e).sum()

d.backward()
print(a.grad)