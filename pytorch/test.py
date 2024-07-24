import random as r
import numpy as np
import torch


alpha = torch.FloatTensor([[1,1,1,1],
                           [10,2,2,2],
                           [3,3,3,3],
                           [4,4,4,4]])

beta = torch.FloatTensor([[1,1,1,1],
                           [2,2,2,2],
                           [3,3,3,3],
                           [4,4,4,4]])

print(alpha[:,1])