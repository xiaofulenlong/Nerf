"""
获得生成的光线:
    要生成的是：每个方向下的像素点到光心的单位方向。
    （原理：有了这个单位方向就可以通过调整 Z 轴坐标生成空间中每一个点坐标，借此模拟一条光线。）
"""
import torch
"""
input: 
    img_H: 高
    img_W: 宽
    c2w:相机外参,由矩阵transform_matrix得出
    K: 相机内参
output:
    rays_o:
    rays_d:

"""
def getRaysFromImg(img_H,img_W,c2w:torch.Tensor,K) :
    #​step1 ：写出相机中心、像素点在相机坐标系下的坐标
    #生成网格 i:横,j:列
    i, j = torch.meshgrid(torch.linspace(0,img_W-1,img_W),torch.linspace(0,img_H-1,img_H))
    i = i.t()
    j = j.t()
    """
    根据内参计算射线的初始值:
    dirs维度是[H,W,3],dirs[:,:,0],dirs[:,:,1] 和 dirs[:,:,2] 分别表示 x,y 和 z 方向的分量。
    计算原理见我的博客:nerf-01【待上传】
    """
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)

	# step2 ：使用c2w矩阵变换到世界坐标系上去
    """
    c2w矩阵的知识点见我的博客:nerf-01,下面讲讲详细变换过程
    c2w是 3x4 的矩阵，其中包含旋转矩阵和平移向量[x,y,z,o]
    抽出c2w的旋转矩阵,左乘得到世界坐标,然后在最后一个轴上进行求和，即沿着最后一个轴将 3 个方向分量相加。
    得到: rays_d为(H,W,3) rays_o为(H,W,3)
    """
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3],dim=-1) 
    rays_o = c2w[:3, -1].view(1, 1, -1).expand_as(rays_d) #扩展后的维度为：(H,W,3) 

    rays_d = (torch.tensor(rays_d)).reshape(-1,3) #维度为(H*W,3)
    rays_o = rays_o.reshape(-1,3) #维度为(H*W,3)

    #组装，rays维度为(H*W,6),前3是对应的光心，后3是方向
    rays = torch.cat((rays_o,rays_d),dim=1)
    return rays 