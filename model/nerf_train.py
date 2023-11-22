"""
    用于创建网络、运行网络
"""
import torch
from model.nerf_model import Nerf

# 创建网络的整个流程
def create_nerf():

    #初始化mlp
    mlp_model = Nerf()

    #模型的梯度变量
    grad_vars = 

    #运行网络：生成给定点的颜色和密度
    run_nerf()

    #创建优化器
    optimizer = torch.optim.Adam()

    #需要的返回值
    # 现在整体的初始化已经完成，我们需要对返回值进行一些处理
    nerf_trained_args = {
        'network_query_fn' : 
        'N_coarse' :  #粗采样的数量
        'network_coarse' :  #粗网络
        'N_fine' :     #细采样的数量
        'network_fine' :    #细网络
        'white_bkgd' :  
        'raw_noise_std' : #归一化密度 ,
    }

    


def run_nerf():
    # 将编码过的点以批处理的形式输入到 网络模型 中得到 输出（RGB,A）

    output = []

    return output