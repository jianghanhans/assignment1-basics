import torch

X = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
print(X)
y = torch.einsum("ij->j", X)
print(y)   # tensor([ 6., 15.])