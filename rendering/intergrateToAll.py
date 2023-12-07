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
    rays_d: [batch_size,chunk,3]
    z_vals:tensor采样间隔 [batch_size,chunk,coarse_num]
    coarse_num:每条光线上粗采样的点数
    chunk:选中的光线数目

Output:
    output:tensor[batch_size,chunk,3]
"""
def render(raw,rays_d,z_vals,coarse_num,chunk):
    #以下代码根据体渲染离散公式得出
    device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #分离raw中的RGBA
    RGB_of_raw = raw[...,:3] #维度：[N,3], N=batch_size*n_sampling = batch_size*chunk*coarse_num
    RGB_of_raw = torch.reshape(RGB_of_raw,(-1,coarse_num,3)) #维度：[N_rays,N_coarse,3]

    #RGB：维度：[N_rays,N_coarse,3],这是相当于公式里的C(辐射度，不过在推导中可以用归一化后的color平替)
    C_rgb = torch.sigmoid(RGB_of_raw)
    
    #计算采样相邻样本之间的距离
    z_vals = torch.reshape(z_vals,(-1,coarse_num)) #[N_rays,N_coarse]
    rays_d =  torch.reshape(rays_d,(-1,3))  #[N_rays,3]
    distance = z_vals[...,1:] - z_vals[...,:-1]  
    distance = torch.cat([distance,torch.Tensor([1e10]).expand(distance[...,:1].shape).to(device)],-1) #扩展，维度为:[N_rays,N_coarse]
    distance = distance * torch.norm(rays_d[...,None,:],dim=-1) 

    #不透明度 alpha: 维度[N_rays,coarse_num]
    raws = torch.reshape(raw,(-1,coarse_num,4))
    sigma = -F.relu(raws[...,3])
    alpha = 1. - torch.exp(sigma*distance)
   

    t = torch.cat([torch.ones(alpha.shape[0], 1).to(device), 1. - alpha + 1e-10], -1)
    T = torch.cumprod(t, -1)  # 公式中的累乘
    weights = alpha * T[:, :-1] #[N_rays,N_sample]

    #最终公式得到的渲染的rgb
    rgb_map = torch.sum(weights[...,None]*C_rgb,-2)   #[N_rays,3]

    #变换，得到[batch_size,chunk,3]
    rgb_map = torch.reshape(rgb_map,(-1,chunk,3))
    
    return rgb_map 