
from torchvision import transforms
from utils.public_tools import get_parser
from dataset.load_data import BlenderDataSet,ResizeImg
import numpy as np  
import torch
from torch.utils.data import DataLoader,RandomSampler
from tqdm import tqdm, trange
from model.nerf_model import Nerf
from model.nerf_helpers import loss_fn
from model.nerf_sample import Coarse_sampling

def main(args):

    # 传参  
    
    dataset_dir         = args.dataset_dir #json所在目录
    img_scale           = args.img_scale #图像比例：取值应为float,[0,1]
    if_use_batchs       = args.if_use_batchs #是否批处理，默认值为false
    render_chunk        = args.render_chunk #渲染批数
    lrate_decay         = args.lrate_decay  #超参数，用于控制学习率的衰减步数
    lrate               = args.lrate  #学习率
    coarse_num          = args.coarse_num
    fre_position_L      = args.fre_position_L #对位置坐标的映射维度
    fre_view_L          = args.fre_view_L #对视角的映射维度
    network_depth       = args.network_depth #8
    hidden_unit_num     = args.hidden_unit_num #256
    output_features_dim = args.output_features_dim #256
    output_dim          = args.output_dim #128
   
    # load dataset   
    transform_function = transforms.Compose([
        ResizeImg(img_scale),
        transforms.ToTensor()
        ])  
    #加载处理训练集和数据集
    train_dataset = BlenderDataSet(dataset_dir,'train',transform_function)
    #test_dataset =  BlenderDataSet(dataset_dir,'test',transform_function)
    #dataloader
    train_dataloader = DataLoader(train_dataset,batch_size=1024*32, shuffle=True,sampler=RandomSampler(train_dataset))

    #整理训练集的参数
    train_focal = train_dataset.focal #焦距：555.5555155968841
    train_pos = train_dataset.view_pos #相机位姿:tensor[100, 3, 4]
    train_img_h = train_dataset.img_H #400
    train_img_w = train_dataset.img_w #400
    #相机内参：
    K = np.array([
            [train_focal, 0, 0.5*train_img_w],
            [0, train_focal, 0.5*train_img_h],
            [0, 0, 1]
        ])
 
    #调用nerf，初始化模型 
    mlp_model = Nerf(fre_position_L,fre_view_L,network_depth,hidden_unit_num,output_features_dim,output_dim)
    #梯度与优化器
    grad_vars = list(mlp_model.parameters())
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
 
    # ========= train !!!! =============
    #采样

    epoch_num = 200000
    for i in trange(0,epoch_num):
      for rays,label in train_dataloader:
        pts =  Coarse_sampling(rays,coarse_num)  #pts: [n_sampling, 3]
        output = mlp_model(pts,train_pos)  
        #optimizer
        loss = loss_fn(output,label)
        # psnr =  -10. * torch.log(loss) / torch.log(torch.Tensor([10.]))
        optimizer.zero_grad()
        loss.backward()  # 损失反向传播
        optimizer.step()
        #更新学习率
        decay_rate = 0.1
        decay_steps = lrate_decay * 1000
        #新学习率，以指数衰减系数
        new_lrate = lrate * (decay_rate ** (i / decay_steps))
        for param in optimizer.param_groups:
          param['lr'] = new_lrate
    



 
#创建解析器，读取命令行参数
parser = get_parser()
args = parser.parse_args()
main(args)