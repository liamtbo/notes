import torch

import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
# x.requires_grad_(True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# deriviate of functions for the backward propagation step
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward()
print(w.grad)
print(b.grad)

# disabling gradient tracking
## with grad, intermediate results and function are tracked, takes up memory
## gradients are actually calculated when loss.backward() is called
## reasons one would want to disable gradient tracking
## Dags are dynamic
    ### mark some parameters in your nn as frozen parameters
    ### speed up compuation when only doing forward pass
z = torch.matmul(x,w)+b
print(z.requires_grad) # tracks w, b, and y

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad) # False

## can also do this method for same result
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad) # False