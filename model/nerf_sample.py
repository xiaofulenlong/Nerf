"""
采样 
"""
import torch


def Coarse_sampling(rays,coarse_num):

     # 从 ray 中分离出 rays_o, rays_d 
    number_of_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  #维度[number_of_rays,3]
    
    # near, far
    near=0.
    far=1.
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1]) #near、far维度：[number_of_rays,1]

    #生成光线上每个采样点的位置  【粗采样】
    #先确立z轴
    terminal = torch.linspace(0.,1.,steps= coarse_num) #维度:[coarse_num]
    z_vals = near + terminal*(far - near) 
    z_vals = z_vals.expand([number_of_rays, coarse_num]) #维度:[number_of_rays, coarse_num]
    
    #  某一点的位置 = 原点+距离*单位方向向量
    # sampled_points：每条光线上的每个粗采样点计算位置，维度为[number_of_rays, coarse_num, 3]
    sampled_points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]
    sampled_points = torch.reshape(sampled_points,(-1,3)) #维度为[number_of_rays*coarse_num, 3]

    return sampled_points

