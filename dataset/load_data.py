
"""
load data , resize data
"""

import os
import json
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils import data
from torchvision.transforms import functional as F, InterpolationMode
from dataset.get_Rays import getRaysFromImg

class BlenderDataSet(data.Dataset):
    """
    input: 
        json_dir:根目录地址
        dataset_label_type:取值范围[train,test,val]
        transform:变换函数,生成光线
    output:
        focal:焦距
        view_pos:相机位姿:tensor
        img_H:图像的高 
        img_w:图像的宽
        rotation:相机旋转角(?目前不知道有什么用)
    """
    def __init__(self,json_dir,dataset_label_type,transform):
        self.json_dir = json_dir
        self.dataset_label_type = dataset_label_type
        self.transform = transform
        self._input_json_data() #读取全部的json文件的内容
        #获得参数：焦距，相机位姿，图像的高、宽，相机旋转角
        self.focal, self.view_pos,self.img_H,self.img_w,self.rotation= self._get_getblenderDataParam() #获取要用的参数

    def __getitem__(self,index):
        #获取一张图片的位姿，总光线和RGB
        img_single_dir = os.path.join(self.json_dir,self.img_dirs[index]+ '.png')
        img = Image.open(img_single_dir,mode = 'r').convert("RGB")
        
        #pos,tensor:[3,4]
        pos = self.view_pos[index]
        #相机内参：
        K = np.array([
            [self.focal, 0, 0.5*self.img_w],
            [0, self.focal, 0.5*self.img_H],
            [0, 0, 1]
        ])
        #dataset:rays, tensor:[H*W,6]
        rays = getRaysFromImg(self.img_H,self.img_w,pos,K)
        

        #label:image的RGB,tensor:[H,W,3]
        img = self.transform(img)
        img = torch.reshape(img,(-1,3))

        return rays,img


    def __len__(self):
        #返回图像的总数目
        nume_of_rays =  self.view_pos.shape[0]
        return nume_of_rays   

    def _input_json_data(self):
        #读取全部的json文件的内容
        assert os.path.exists(self.json_dir)
        self.all_jsons_data = {}

        with open(os.path.join(self.json_dir,'transforms_{}.json'.format(self.dataset_label_type)),'r') as js:
            self.all_jsons_data  = json.load(js)
        
        #写入camera_angle_x
        self.camera_angle_x =  self.all_jsons_data['camera_angle_x']
        #将每个frame的 地址 、姿势 和 旋转(?有啥用) 读出来存起来
        self.img_dirs = [frame["file_path"] for frame in self.all_jsons_data['frames']]
        self.view_pos = np.stack([frame['transform_matrix'] for frame in self.all_jsons_data['frames']])
        self.rotation = [frame["rotation"] for frame in self.all_jsons_data['frames']]


    def _getCameraFocal(self,img_w):
        #焦距focal
        focal = .5 * img_w / np.tan(.5 * self.camera_angle_x)
        return focal


    #最终处理图片信息，确定最终的返回值
    def _get_getblenderDataParam(self):
        
        #高，宽
        img0_dir = self.img_dirs[0]
        img0_loc = os.path.join(self.json_dir,img0_dir+ '.png')
        img0 = Image.open(img0_loc,mode = 'r').convert("RGB")
        img0 = self.transform(img0)
        img_h, img_w = img0.shape[1], img0.shape[2]

        #焦距
        focal = self._getCameraFocal(img_w)

        #相机视角姿势:tensor[num_of_img,3,4]
        pos = torch.from_numpy(self.view_pos)[:,:3,:]

        #相机旋转角
        rotation = self.rotation

        return  focal,pos.float(),img_h, img_w,rotation


#裁切，预处理图像
class ResizeImg(nn.Module):
    def __init__(self, img_scale):
        super().__init__()
        self.scale = img_scale
        self.interpolation  = InterpolationMode.BILINEAR  # 插值方法，这里使用双线性插值
        self.max_size = None # 最大尺寸，如果不指定，则为 None
        self.antialias = None  # 是否使用抗锯齿，如果不指定，则为 None

    def forward(self,input:Image.Image):
        # 根据指定的比例计算调整后的尺寸
        #其中resize()中的size是(h,w),而PIL读取img.size是(w,h),所以需要调换位置
        size = (int(input.size[1] * self.scale), int(input.size[0] * self.scale))
        # 使用torchvision.transforms.functional中的resize函数进行调整大小：
        # 这里使用了构造函数中定义的插值方法、最大尺寸和抗锯齿选项
        return F.resize(input, size, self.interpolation, self.max_size, self.antialias)

        





# #test 
# if __name__ == "__main__":
#     transform_function = transforms.Compose([
#         ResizeImg(0.5),
#         transforms.ToTensor()
#         ]) 
#     test = BlenderDataSet("/home/hrr/my_code/nerf_pro/nerf_synthetic/lego/","train",transform_function)
#     print(   test[0 ])
 
   