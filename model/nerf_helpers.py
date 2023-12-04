
"""
        一些辅助完成nerf的功能
"""
import torch
import math
import numpy as np

# 高维映射编码
#将[N,3]的tensor映射为[N,3*2*frequency_L]的tensor
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


def Generate_view(rays): 
    view = rays[:, :,3:6]  #维度[batch_size,n_per_rays,3]
    view = view / torch.norm(view, dim=-1, keepdim=True) #归一化
    view = torch.reshape(view, [-1,3]).float() #维度[batch_size*n_per_rays,3]

    return view 

# if __name__ == "__main__":
#     t = torch.rand([3,4,5])
#     test = Positional_encoding(t,10)
#     print(test.shape)
     