"""
根据:每个点对应的颜色RGB 以及 密度σ
再通过: 体积渲染公式对这条光线上的点进行累积积分
得到:光线的颜色。
"""
import torch
import torch.nn.functional as F

"""
Input:
    raw:[number_of_rays, coarse_num, 4]. 从mlp模型中采样得出的,4:[RGB,A].
    z_vals:#维度:[number_of_rays, coarse_num] 在z轴采样的位置
    rays_d:#维度:[number_of_rays, 3] 每个光线的方向

Output:
"""
def render(raw,z_vals):
    #以下代码根据体渲染离散公式得出
    
    #分离raw中的RGBA
    RGB_of_raw = raw[...,3] #维度：[number_of_rays, coarse_num, 3]

    #计算采样相邻样本之间的距离
    distance = z_vals[...,1:] - z_vals[...,:-1] #维度[number_of_rays, coarse_num-1] 
    distance = torch.cat([distance,torch.Tensor([1e10]).expand(distance[...,:1].shape)],-1) #扩展，维度为:[number_of_rays, coarse_num]

    #不透明度 alpha: 维度[number_of_rays,coarse_num]
    alpha = 1. - torch.exp(-F.relu(RGB_of_raw)*distance)

    #RGB：维度：[number_of_rays, N_samples, 3],这是相当于公式里的C(辐射度，不过在推导中可以用归一化后的color平替)
    C_rgb = torch.sigmoid(RGB_of_raw)
    
    t = torch.cat([torch.ones(alpha.shape[0], 1), 1. - alpha + 1e-10], -1)
    T = torch.cumprod(t, -1)  # 公式中的累乘
    weights = alpha * T[:, :-1]

    #最终公式得到的渲染的图像
    rgb_map = torch.sum(weights[...,None] * C_rgb, -2)  # [number_of_rays, 3]

    return rgb_map 