"""
    用于创建网络、运行网络
"""
import torch
from model.nerf_model import Nerf

# 创建网络的整个流程
def create_nerf(args):
     #初始化mlp
    """
    input:
        fre_position_L, #对位置坐标的映射维度
        fre_view_L, #对视角的映射维度
        network_depth = 8,
        hidden_unit_num = 256,
        output_features_dim = 256, #输出的特征值的维度
        output_dim = 128 #拼接层的输出

    output:

    """
    fre_position_L      = args.fre_position_L
    fre_view_L          = args.fre_view_L
    network_depth       = args.network_depth
    hidden_unit_num     = args.hidden_unit_num
    output_features_dim = args.output_features_dim
    output_dim          = args.output_dim
    netchunkNum         = args.netchunkNum

    mlp_model = Nerf(fre_position_L,fre_view_L,network_depth,hidden_unit_num,output_features_dim,output_dim)

    #模型的梯度变量
    grad_vars = list(mlp_model.parameters())

    """
    #作用：运行网络生成给定点的颜色和密度
    input:
        position_inputs:(x,y,z)position输入
        view_inputs: view输入
        mlp_network_fn: 网络model
        netchunkNum: 并行处理的输入数量
    """
    mlp_query_fn = lambda position_inputs,view_inputs,mlp_network_fn: run_nerf(position_inputs,view_inputs,mlp_network_fn,netchunkNum )

    
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

    return grad_vars

"""


"""
def run_nerf():
    # 将编码过的点以批处理的形式输入到 网络模型 中得到 输出（RGB,A）

    output = []

    return output