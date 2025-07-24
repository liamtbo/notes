
import torch
import torch.nn as nn

# an Embedding module containing 10 tensors of size 3
# lookup table where each of the 10 possible indices corresponds to  a 3-dim vector
embedding = nn.Embedding(10, 3)
# a batch of 2 samples of 4 indices each
input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
embedding(input) # outputs shape (2,4), grabbing the embedding at each index froom input
"""
tensor([[[-0.0251, -1.6902,  0.7172],
         [-0.6431,  0.0748,  0.6969],
         [ 1.4970,  1.3448, -0.9685],
         [-0.3677, -2.7265, -0.1685]],

        [[ 1.4970,  1.3448, -0.9685],
         [ 0.4362, -0.4004,  0.9400],
         [-0.6431,  0.0748,  0.6969],
         [ 0.9124, -2.3616,  1.1151]]])
"""

# example with padding_idx
# the embedding for index 0 will be set to a zero vector - ensures that padding tokens do not contribute to model training
embedding = nn.Embedding(10, 3, padding_idx=0)
# every tensor needs to be the same size x for transformer
# padding is used when an input has less then size x, padding is added at the end
input = torch.LongTensor([[2, 5, 0, 0]])
embedding(input)
"""
tensor([[[-0.0757,  1.0842,  1.7720],
         [ 0.4868, -1.3341,  3.1784],
         [ 0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000]]], grad_fn=<EmbeddingBackward0>)
"""

# example of changing `pad` vector
# weights are just the embedding values
padding_idx = 0
embedding = nn.Embedding(3, 3, padding_idx=padding_idx)
print(embedding.weight) # prints the embedding layer
"""Parameter containing:
tensor([[ 0.0000,  0.0000,  0.0000],
        [-0.7895, -0.7089, -0.0364],
        [ 0.6778,  0.5803,  0.2678]], requires_grad=True)
"""
with torch.no_grad():
    embedding.weight[padding_idx] = torch.ones(3)
embedding.weight