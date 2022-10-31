import torch
import torch.nn as nn
import torch.nn.functional as F


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn((8,384,4096),dtype = torch.float16, device=dev)

for iter in range(10):
    model = nn.Sequential(nn.GELU())
    model.to(dev)
    y = model(x)
