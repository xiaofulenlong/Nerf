
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
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode

class BlenderDataSet(data.Dataset):
    """
    input: 
        json_dir:根目录地址
        dataset_label_type:取值范围[train,test,val]
        transform:变换函数
    output:
        focal:焦距
        img_dataset:全部的Image信息【已经经过transform变换】:tensor
        view_pos:相机姿势:tensor
        img_H:图像的高 
        img_w:图像的宽
        rotation:相机旋转角(?目前不知道有什么用)
    """
    def __init__(self,json_dir,dataset_label_type,transform=None):
        self.json_dir = json_dir
        self.dataset_label_type = dataset_label_type
        self.transform = transform
        self._input_json_data() #读取全部的json文件的内容
        #获得参数：焦距，全部的Image信息，相机姿势，图像的高、宽，相机旋转角
        self.focal,self.img_dataset,self.view_pos,self.img_H,self.img_w,self.rotation= self._get_getblenderDataParam() #获取要用的参数

    def __getitem__(self,index):
        #读取json中每一个frame的地址图片
        img_single_dir = os.path.join(self.json_dir,self.img_dirs[index]+ '.png')
        image_single = Image.open(img_single_dir,mode = 'r')
        if self.transform != None:
            image_single =self.transform(image_single)

        return image_single


    def __len__(self):
        return len(self.allimgs)    #具体返回什么长度之后再修改

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
        image_0 = self.__getitem__(0)
        img_h, img_w = image_0.shape[1], image_0.shape[2]

        #焦距
        focal = self._getCameraFocal(img_w)

        #全部的图像信息
        all_images = [] #全部的图片信息
        for img_dir in self.img_dirs:
            img_loc = os.path.join(self.json_dir,img_dir+ '.png')
            img = Image.open(img_loc).convert("RGB")
            all_images.append(self.transform(img))
        all_images_tensor = torch.stack(all_images, dim=0) #[n, channals, H, W,]
        all_images_tensor = np.transpose(all_images_tensor,[0,2,3,1]) #[n,  H, W,channals]
        #相机视角姿势
        view = torch.from_numpy(self.view_pos)[:,:3,:]

        #相机旋转角（？）
        rotation = self.rotation

        return  focal,all_images_tensor, view.float(),img_h, img_w,rotation


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
#    img_scale = 0.5
#    transform_function = transforms.Compose([
#         ResizeImg(img_scale),
#         transforms.ToTensor(),
#     ])
#    test = BlenderDataSet("/home/hrr/my_code/nerf_pro/nerf_synthetic/lego/","train",transform_function)
#    print(   test[0 ])
 
   