# %matplotlib inline

import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
# 25 evenly spaced values in range [0,2pi]
a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
# print(a)

b = torch.sin(a)
## detach bc matplot expects nump, cant convert numpy
plt.plot(a.detach(), b.detach())
# print(b)

c = 2 * b
# print(c)
d = c + 1
# print(d)

out = d.sum()
# print(out)

# print('d:')
# print(d.grad_fn)
# print(d.grad_fn.next_functions)
# print(d.grad_fn.next_functions[0][0].next_functions)
# print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
# print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)
# print('\nc:')
# print(c.grad_fn)
# print('\nb:')
# print(b.grad_fn)
# print('\na:')
# print(a.grad_fn)

out.backward()
# print(a.grad)
plt.plot(a.detach(), a.grad.detach()) # (x axis, f(x) y-axis)
# plt.show()

# only leaf nodes gettheir gradients computer, i.e. a but not b,c,or d

# -----------------------------------------------------------------
"""Applying to a network"""
BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 10

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.layer1 = torch.nn.Linear(DIM_IN, HIDDEN_SIZE)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(HIDDEN_SIZE, DIM_OUT)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

model = TinyModel()

# print(model.layer2.weight[0][0:10]) # just a small slice
# print(model.layer2.weight.grad)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

prediction = model(some_input)

loss = (ideal_output - prediction).pow(2).sum()
# print(loss)

loss.backward()
# print(model.layer2.weight[0][0:10])
# print(model.layer2.weight.grad[0][0:10])

optimizer.step()
# print(model.layer2.weight[0][0:10])
# print(model.layer2.weight.grad[0][0:10])
optimizer.zero_grad()

# --------------------------------------
"""Turning Autograd off and on"""
a = torch.ones(2, 3, requires_grad=True)
# print(a)
b1 = 2 * a
# print(b1)
a.requires_grad = False
b2 = 2 * a
# print(b2)

a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3
c1 = a + b
# print(c1)
with torch.no_grad():
    c2 = a + b
# print(c2)
c3 = a * b
# print(c3)

## function decorator
def add_tensors1(x, y):
    return x + y
@torch.no_grad()
def add_tensors2(x, y):
    return x + y
a = torch.ones(2, 3, requires_grad=True) * 2
b = torch.ones(2, 3, requires_grad=True) * 3
c1 = add_tensors1(a, b)
# print(c1)
c2 = add_tensors2(a, b)
# print(c2)

## making copy that doesn't track gradient
x = torch.rand(5, requires_grad=True)
y = x.detach()
# print(x)
# print(y)

# --------------------------------------
"""Autograd and in-place operations"""
## autograd needs output to be stored to add function to computatinal graph
## this throws an error

# a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
# torch.sin_(a)

# --------------------------------------
"""Autograd Profiler"""
device = torch.device('cpu')
run_on_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda')
    run_on_gpu = True

x = torch.randn(2, 3, requires_grad=True)
y = torch.rand(2, 3, requires_grad=True)
z = torch.ones(2, 3, requires_grad=True)

with torch.autograd.profiler.profile(use_cuda=run_on_gpu) as prf:
    for _ in range(1000):
        z = (z / x) * y

print(prf.key_averages().table(sort_by='self_cpu_time_total'))