import torch
import torch.nn as nn # actin-value nn
import torch.optim as optim # optimizer
import torch.nn.functional as F # activates and loss functions

import random
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import namedtuple, deque

class tmp():
    def __init__(self, x):
        self.x = x

d = {"tmp": tmp}

tmp1 = d["tmp"](5)
tmp1.x = 99
print(d["tmp"].x)

