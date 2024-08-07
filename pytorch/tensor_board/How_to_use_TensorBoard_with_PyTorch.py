import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter() # will output to ./runs/ dir

# add_scalar(tag, scalar_value, global_step=None, walltime=None)
x = torch.arange(-5, 5, 0.1).view(-1, 1) # creates tensor of range [-5,5) with 0.1 step
y = -5 * x + 0.1 * torch.randn(x.size())

model = torch.nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def train_model(iter):
    for epoch in range(iter):
        y1 = model(x)
        loss = criterion(y1, y)
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_model(10)
writer.flush() # all pending events get written to disk
writer.close()
# tensorboard --logdir=runs
# visit http://localhost:6006/ to see graphs