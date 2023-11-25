"""
渲染光线
"""

import torch 
from rendering.intergrateToAll import intregrateTo_RGB_density
from rendering.rays import get_rays
"""
输入: 
    render_chunk : 一次并行处理的光线的数量
    rays_selected_to_train : 预处理采样的训练所用光线, 维度:[2, batch_size, 3]
    nerf_trained_args : 初始化模型代码 create_nerf() 返回的字典数据
输出: 
    RGB_map:[batch_size,3]光束对应的rgb
    视差图
    不透明度 
"""
def render(img_H,img_W,K,c2w,render_chunk, 
            rays_selected_to_train,
            near=0., far=1.,
            **nerf_trained_args):

    # 确定光线的原点和方向
    if c2w is not None:
        rays_o,rays_d = get_rays(img_H,img_W,c2w,K) #渲染整个图像
    else:
        rays_o,rays_d = rays_selected_to_train

    # 归一化  rays_d为(H,W,3)
    rays_d_normalized = rays_d
    rays_d_normalized = rays_d_normalized / torch.norm(rays_d_normalized, dim=-1, keepdim=True)
    rays_d_normalized = torch.reshape(rays_d_normalized, [-1,3]).float() #降维：变成[W*H,3]

    rays_d_shape = rays_d.shape # 保存射线方向的形状，以备后续使用。

    # 生成光线的远近端，用于确定边界框，并将其聚合到 rays 中
    rays_o = torch.reshape(rays_o, [-1,3]).float()#降维：变成[W*H,3]
    rays_d = torch.reshape(rays_d, [-1,3]).float()#降维：变成[W*H,3]
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1]) #near、far维度：[H*W,1]
    # 视图方向聚合到光线中
    rays = torch.cat([rays_o, rays_d, near, far,rays_d_normalized], -1) #拼接：变成[W*H,3+3+1+1+3]维度
    
    # 开始并行计算光线属性
    all_color_density = render_ray(rays,render_chunk,**nerf_trained_args)

    return





"""
Args:
      ray:生成处理好的光线
      mlp_network: 训练网络
      coarse_num:粗采样点 
      
    Returns:
      rgb: 光线的RGB [ray_nums,3]
      density:光线的密度 [ray_nums,1]
"""
def render_ray(rays,render_chunk,coarse_num,mlp_query_fn,mlp_network_fn):
    
    # 从 ray 中分离出 rays_o, rays_d, viewdirs, near, far
    number_of_rays = rays.shape[0]
    rays_o, rays_d,  rays_view = rays[:, 0:3], rays[:, 3:6],rays[:,-3:] #维度[number_of_rays,3]
    bounds = torch.reshape(rays[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1]   #维度[number_of_rays,1]

    #生成光线上每个采样点的位置  【粗采样】
    #先确立z轴
    terminal = torch.linspace(0.,1.,steps= coarse_num) #维度:[coarse_num]
    z_vals = near + terminal*(far - near) 
    z_vals = z_vals.expand([number_of_rays, coarse_num]) #维度:[number_of_rays, coarse_num]
    
    #  某一点的位置 = 原点+距离*单位方向向量
    # sampled_points：每条光线上的每个粗采样点计算位置，维度为[number_of_rays, coarse_num, 3]
    sampled_points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]

    #将光线上的每个点投入到 MLP网络中前向传播得到每个点对应的(RGB，A),然后聚合在raw中
    raw = mlp_query_fn(sampled_points,rays_view,mlp_network_fn)

    # 对这些离散点进行体积渲染，即进行积分操作
    rgb_map, disp_map, acc_map, weights, depth_map = intregrateTo_RGB_density(raw, z_vals, rays_d, )

    return 