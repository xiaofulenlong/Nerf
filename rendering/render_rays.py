"""
渲染光线
"""

import torch 
from rendering.intergrateToAll import intregrateTo_RGB_density

"""
输入: 
    parallel_ray_count : 并行处理的光线的数量
    batch_selected_rays : 经由批处理我们挑选出的光线
    nerf_trained_args : 初始化模型代码 create_nerf() 返回的字典数据
输出: 光束对应的rgb, 视差图，不透明度。 
"""
def render():

    # 生成光线的远近端，用于确定边界框，并将其聚合到 rays 中

    # 视图方向聚合到光线中

    # 开始并行计算光线属性
    render_ray()

    return





"""
Args:
      ray:光线
      mlp_network: 训练网络
      N_samples:采样点 
      
    Returns:
      rgb: 光线的RGB [ray_nums,3]
      density:光线的密度 [ray_nums,1]
"""
def render_ray():
    #生成光线上每个采样点的位置


    #将光线上的每个点投入到 MLP网络中前向传播得到每个点对应的(RGB，A)

    # 对这些离散点进行体积渲染，即进行积分操作
	rgb_map, disp_map, acc_map, weights, depth_map = intregrateTo_RGB_density(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    return 