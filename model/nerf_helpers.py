
"""
        一些辅助完成nerf的功能
"""
import torch
import math
import numpy as np

# 高维映射编码
def Positional_encoding(x:torch.Tensor,frequency_L:int) -> torch.Tensor:
    encoded= []
    cal = []
    func = (torch.sin,torch.cos)
    for fre in range(frequency_L):
        tmp =(2. ** fre) * (math.pi)
        for f in func:
            cal.append(f(tmp * x))  
    encoded = torch.cat(cal, dim = -1)
       
    return encoded


def loss_fn():
    
# if __name__ == "__main__":
#     t = torch.rand([3,4,5])
#     test = Positional_encoding(t,10)
#     print(test.shape)
     