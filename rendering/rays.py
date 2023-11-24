"""
获得生成的光线:
    要生成的是：每个方向下的像素点到光心的单位方向。
    （原理：有了这个单位方向就可以通过调整 Z 轴坐标生成空间中每一个点坐标，借此模拟一条光线。）
"""
import torch
import numpy as np
"""
input: 
    img_H: 高
    img_W: 宽
    c2w:相机外参,由矩阵transform_matrix得出
    train_focal:焦距,用来确定相机内参
output:
    rays_o:
    rays_d:

"""
def get_rays(img_H,img_W,c2w:torch.Tensor,train_focal ) :
    #相机内参：
    K = np.array([
            [train_focal, 0, 0.5*img_W],
            [0, train_focal, 0.5*img_H],
            [0, 0, 1]
        ])
    
    #​step1 ：写出相机中心、像素点在相机坐标系下的3D坐标
        #生成网格 i:横,j:列
    i, j = torch.meshgrid(torch.linspace(0,img_W-1,img_W),torch.linspace(0,img_H-1,img_H))
    i = i.t()
    j = j.t()
    #转置结果：(W,H)变为了(H,W),但是为什么捏，还没明白，之后看看后面运行流程，先mark了 
    """
    根据内参计算射线的初始值:
    dirs维度是[H,W,3],dirs[:,:,0],dirs[:,:,1] 和 dirs[:,:,2] 分别表示 x,y 和 z 方向的分量。
     【mark】计算思路还是没有很懂,明天再看看
    """
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)

	# step2 ：使用c2w矩阵变换到世界坐标系上去
    # 点乘，等同于：[c2w.dot(dir) for dir in dirs]
    """
    c2w矩阵的知识点见笔记【之后记得同步到github上】,下面讲讲变换过程
    c2w是 3x4 的矩阵，其中包含旋转矩阵和平移向量[x,y,z,o]

    """

    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1) 
    rays_o = c2w[:3,-1].expand(rays_d) #扩展后的维度为：(H,W,3) 

    return rays_o,rays_d