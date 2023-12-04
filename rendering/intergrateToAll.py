"""
根据:每个点对应的颜色RGB 以及 密度σ
再通过: 体积渲染公式对这条光线上的点进行累积积分
得到:光线的颜色。
"""
import torch
import torch.nn.functional as F

"""
Input:
    raw:tensor[N,4] = [batch_size*n_sampling,4] . 从mlp模型中得出的,4:[RGB,A].
    z_vals:tensor 采样间隔[batch_size,coarse_num]
    h,w:宽高

Output:
    output:tensor[batch_size,3,H,W] rgb通道
"""
def render(raw,z_vals,h,w):
    #以下代码根据体渲染离散公式得出
    
    #分离raw中的RGBA
    RGB_of_raw = raw[...,3] #维度：[N,3]

    #计算采样相邻样本之间的距离
    distance = z_vals[...,1:] - z_vals[...,:-1] #维度[batch_size, coarse_num-1] 
    distance = torch.cat([distance,torch.Tensor([1e10]).expand(distance[...,:1].shape)],-1) #扩展，维度为:[batch_size, coarse_num]

    #不透明度 alpha: 维度[N,coarse_num]
    alpha = 1. - torch.exp(-F.relu(RGB_of_raw)*distance)

    #RGB：维度：[N, 3],这是相当于公式里的C(辐射度，不过在推导中可以用归一化后的color平替)
    C_rgb = torch.sigmoid(RGB_of_raw)
    
    t = torch.cat([torch.ones(alpha.shape[0], 1), 1. - alpha + 1e-10], -1)
    T = torch.cumprod(t, -1)  # 公式中的累乘
    weights = alpha * T[:, :-1]

    #最终公式得到的渲染的rgb
    rgb_map = torch.sum(weights[...,None] * C_rgb, -2)  # [N1, 3]

    #变换，得到最终维度：[batch_size,3,H,W]  
    # N1 = batch_size*n_per_rays = batch_size*H*W 
    rgb_map = torch.reshape(rgb_map,(-1,h,w,3)) #[batch_size,H,W,3]  
    rgb_map = rgb_map.permute(0, 3, 1, 2) #[batch_size,3,H,W] 
    
    return rgb_map 