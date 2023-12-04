
"""
    Origin nerf
"""
from torch import nn
import torch
from typing import Optional,Tuple
from model.nerf_helpers import Positional_encoding

class Nerf(nn.Module):

    def __init__(self,
        fre_position_L, #对位置坐标的映射维度 10
        fre_view_L, #对视角的映射维度 4
        network_depth = 8,
        hidden_unit_num = 256,
        output_features_dim = 256, #输出的特征值的维度
        output_dim = 128 #拼接层的输出
        
    ) -> None:
        super(Nerf,self).__init__()
        self.fre_position_L = fre_position_L
        self.fre_view_L = fre_view_L
        self.nnetwork_depth = network_depth
        self.hidden_unit_num = hidden_unit_num
        self.output_features_dim = output_features_dim 
        self.skip = 4 #跳跃连接发生的层数
        self.input_position_dim = 2 * 3 * fre_position_L #60
        self.input_view_dim = 2 * 3 * fre_view_L #24

        # mlp构建：lineal
        self.lineal_position_input = nn.Linear( self.input_position_dim,hidden_unit_num) #[60,256]
        self.lineal_position_skip_input = nn.Linear(self.input_position_dim + hidden_unit_num,hidden_unit_num) #[316,256]
       
       
        self.lineal_hidden = nn.ModuleList([
            nn.Linear(hidden_unit_num, hidden_unit_num) #[256,256]
            if i != self.skip else self.lineal_position_skip_input
            for i in range(network_depth - 1)
        ])
        self.lineal_features  = nn.Linear(hidden_unit_num,output_features_dim) #输出256特征  [128,256]
        self.lineal_view_input = nn.Linear(self.input_view_dim + output_features_dim,output_dim) #输出的256特征+拼接的view维度，输出128维 [280,128]
        self.lineal_colorRGB = nn.Linear(output_dim,3) #[128,3]
        self.lineal_density = nn.Linear(hidden_unit_num,1) #[256,1]

        # 初始化权重
        self.apply(self._init_weights)


    """
        Input:
                Position:  tensor  [batch_size*n_sampling, 3]=[N,3]
                view_dir:  tensor [batch_size*n_rays_perImg, 3]=[M,3]
        Output: 
                RGB: emitted_color (r,g,b) 
                density: volume_density 
    """
    def forward(self, Position:torch.Tensor, view_dir: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        

        #高维映射
        encoded_position = Positional_encoding( Position , self.fre_position_L) #[N,60]
        encoded_view_direction = Positional_encoding( view_dir,self.fre_view_L) #[M,24]
        # encoded_position =  Position   
        # encoded_view_direction =  view_dir
        
        #开始输送数据并激活
        input_data = encoded_position #[N,60]
        for i,lineal_item in enumerate(self.lineal_hidden):
            input_data = self.lineal_hidden[i](input_data) #传入数据
            input_data = nn.relu(input_data) 
            if i == self.skips:
                input_data = torch.cat([encoded_position, input_data], -1) #跳跃连接：拼接一下输入的数据
       
        #输出density: volume_density 
        density = self.lineal_density(input_data)
       
        #RGB: emitted_color (r,g,b) 
        feature = self.self.lineal_features(input_data) #256维
        input_data = torch.cat([feature, encoded_view_direction], -1) 
        input_data = self.lineal_view_input(input_data)
        input_data = nn.relu(input_data)
        rgb = self.lineal_colorRGB(input_data)
        
        #
        ret = torch.cat([rgb,density],dim=-1)

        return   ret     
    

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)