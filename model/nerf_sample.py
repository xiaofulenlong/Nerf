"""
采样 
"""
import torch

"""
input:
     rays:tensor,[batch_size,chunk,6] n_rays是指每张图片所生成的光线数目
     coarse_num:int,粗采样的数目
"""
def Coarse_sampling(rays,coarse_num):
    device = torch.device("cpu")
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 从 ray 中分离出 rays_o, rays_d 
    batch_size = rays.shape[0]
    rays_o, rays_d = rays[:, :,0:3], rays[:, :,3:6]  #维度[batch_size,chunk,3]
    
    # near, far
    near=0.
    far=1.
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1]) 
    #near、far维度：[batch_size,chunk,1],其中1表示z
    
    #生成光线上每个采样点的位置  【粗采样】
    #先确立z轴
    terminal = torch.linspace(0.,1.,steps= coarse_num).to(device)#维度:[coarse_num]
    z_vals = near + terminal*(far - near)  #z_val维度为[batch_size,chunk,coarse_num]
    z_ret = z_vals.to(device) #记录返回值，用于后续体素渲染
    z_vals = z_vals.unsqueeze(-1)  #z_val维度为[batch_size,chunk,coarse_num,1]
    z_vals = z_vals.expand(-1,-1,-1,3)    #z_val维度为[batch_size,chunk,coarse_num,3]

    #生成采样点：某一点的位置 = 原点rays_o+距离z_vals*单位方向向量rays_d
    #得到每条光线上的每个粗采样点计算位置：[batch_size, chunk, coarse_num, 3]
    sampled_points = rays_o[...,None,:] + z_vals * rays_d[...,None,:]
    #将 sampled_points 重新形状为 [batch_size * chunk * coarse_num, 3]
    sampled_points = torch.reshape(sampled_points, ( -1, 3))


    return sampled_points,z_ret

