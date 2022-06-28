import torch
from collections import OrderedDict

def dni(net_A_name, net_B_name):
    net_A = torch.load(net_A_name)
    net_B = torch.load(net_B_name)
    net_interp = OrderedDict()
    for k, v_A in net_A.items():
        v_B = net_B[k]
        net_interp[k] = alpha * v_A + (1 - alpha) * v_B
    return net_interp
    
net_A_name = 'path_to_net_A.pth'
net_B_name = 'path_to_net_B.pth'
alpha = 0.3                              # interpolation coefficient
net_interp = dni(net_A_name, net_B_name)

