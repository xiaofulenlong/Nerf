
"""
    Origin nerf
"""
from torch import nn
import torch
from typing import Optional,Tuple
from model.nerf_helpers import Positional_encoding

class Nerf(nn.Module):

    def __init__(self,
        fre_position_L, #对位置坐标的映射维度
        fre_view_L, #对视角的映射维度
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
        self.skip = 4 #跳跃连接发生的层数 【由论文的连接层示意图得出，不知道对不对】
        self.input_position_dim = 2 * 3 * fre_position_L
        self.input_view_dim = 2 * 3 * fre_view_L

        # mlp构建：lineal
        self.lineal_position_input = nn.Linear( self.input_position_dim,hidden_unit_num)
        self.lineal_position_skip_input = nn.Linear(self.input_position_dim + hidden_unit_num,hidden_unit_num)
       
        self.lineal_hidden = nn.ModuleList( #将输入层加入隐藏层的输入
            [self.lineal_position_input] + [nn.Linear(hidden_unit_num, hidden_unit_num) if i!=self.skip else  self.lineal_position_skip_input for i in range(network_depth-1)])
        self.lineal_features  = nn.Linear(hidden_unit_num,output_features_dim) #输出256特征 
        self.lineal_view_input = nn.Linear(self.input_view_dim + output_features_dim,output_dim) #输出的256特征+拼接的view维度，输出128维
        self.lineal_colorRGB = nn.Linear(output_dim,3)
        self.lineal_density = nn.Linear(hidden_unit_num,1)

        # 初始化权重
        self.apply(self._init_weights)


    """
        Input:
                Position: 3D (x,y,z)
                viewing_direction
        Output: 
                RGB: emitted_color (r,g,b) 
                density: volume_density 
    """
    def forward(self, Position:torch.Tensor, viewing_direction: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        #高维映射
        encoded_position = Positional_encoding( Position , self.fre_position_L)
        encoded_view_direction = Positional_encoding( viewing_direction,self.fre_view_L)
        
        #开始输送数据并激活
        input_data = encoded_position
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
        
        outputs = torch.cat([rgb, density], -1)
    
        return outputs    
    

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)