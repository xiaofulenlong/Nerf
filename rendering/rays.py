"""
获得生成的光线:
    要生成的是：每个方向下的像素点到光心的单位方向。
    （原理：有了这个单位方向就可以通过调整 Z 轴坐标生成空间中每一个点坐标，借此模拟一条光线。）
"""
import torch
def get_rays(img_H,img_W,K,pose:torch.Tensor ) :
    rays_o = []
    rays_d = []


    return rays_o,rays_d