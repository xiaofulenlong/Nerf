
from torchvision import transforms
from utils.public_tools import get_parser
from dataset.load_data import BlenderDataSet,ResizeImg
from model.nerf_train import create_nerf
from rendering.rays import get_rays
from rendering.render_rays import render
import numpy as np  
import torch
from tqdm import tqdm, trange

def main(args):

    #========= 传参 ==============
    
    dataset_dir       = args.dataset_dir #json所在目录
    img_scale         = args.img_scale #图像比例：取值应为float,[0,1]
    if_use_batchs     = args.if_use_batchs #是否批处理，默认值为false
    render_chunk      = args.render_chunk #渲染批数
    lrate_decay       = args.lrate_decay  #超参数，用于控制学习率的衰减步数
    lrate             = args.lrate  #学习率
    N_rand            = args.N_rand
    # ========= load dataset ==============  
       
    """
    加载数据，每个数据集上需要得到的数据：
      imgs : 根据 .json 文件加载到的所有图像数据。  
      view_poses : 转置矩阵,表示姿势。 
      height,width,focal: 图像的高、宽、焦距。
    """
    transform_function = transforms.Compose([
        ResizeImg(img_scale),
        transforms.ToTensor()
        ])  
    #加载处理训练集和数据集
    train_dataset = BlenderDataSet(dataset_dir,'train',transform_function)
    test_dataset =  BlenderDataSet(dataset_dir,'test',transform_function)
    
    #训练集的参数
    train_focal = train_dataset.focal #焦距：555.5555155968841
    train_view_pos = train_dataset.view_pos #相机位姿:tensor[100, 3, 4]
    train_img_h = train_dataset.img_H #400
    train_img_w = train_dataset.img_w #400
    train_img = train_dataset.img_dataset #图像信息：Tensor [100, 400, 400, 3 ]

    # =========  nerf ============== 
    
    #调用nerf，初始化模型 
    grad_vars,nerf_trained_args = create_nerf(args)
    

    # =========  create rays ============== 
    #生成所有图片的像素点对应的光线原点和方向
    #相机内参：
    K = np.array([
            [train_focal, 0, 0.5*train_img_w],
            [0, train_focal, 0.5*train_img_h],
            [0, 0, 1]
        ])
    """
    train_view_pos 的形状应该为 (num_of_poses, 3, 4),其中 num_of_poses 是位姿的数量。
    pos: 每个相机姿势是一个 3x4 的矩阵，其中包含旋转矩阵和平移向量[x,y,z,o]。
    get_rays: 列表推导式,for 每一个pos,使用get_rays获取相应的射线。
    堆叠起来output: rays:(num_of_poses, 2, H, W, 3)，其中 2 表示每个射线由原点和方向两个部分组成, 3 表示射线方向的三个分量x,y,z。
    """
    rays = np.stack([get_rays(train_img_h,train_img_w,pos,K) for pos in train_view_pos[:,:3,4]],0) 
    #将光线的原点、方向、以及这条光线对应的像素颜色结合到一起，便于后面的 shuffle 操作
    rays_with_RGB =  np.concatenate([rays, train_img[:, None]], 1)  # [num_of_poses, ro+rd+rgb, H, W, 3] 
    rays_with_RGB = np.transpose(rays_with_RGB, [0,2,3,1,4])  # [num_of_poses, H, W, ro+rd+rgb, 3]
    rays_with_RGB = np.reshape(rays_with_RGB, [-1,3,3])  # [(num_of_poses)*H*W, ro+rd+rgb, 3]
    rays_with_RGB = rays_with_RGB.astype(np.float)
    np.random.shuffle(rays_with_RGB) #shuffle

    # ========= train !!!! =============
    #采样
    global_step = 0
    i_batch = 0
    N_iters = 200000
    for i in trange(0,N_iters):

      #挑选img
      if if_use_batchs:
        #开始迭代训练：以批处理的形式对进行训练
        batch = rays_with_RGB[i_batch:i_batch+N_rand] #[N_rand, ro+rd+rgb, 3]
        batch = torch.transpose(batch,0,1)  #[ro+rd+rgb, N_rand, 3]
        rays_selected_to_train,rays_selected_color_target = batch[:2],batch[2]  ##[ro+rd, N_rand, 3],[N_rand, 3,] 将方向和RGB颜色分离
        i_batch += N_rand
      
        if i_batch >= rays_with_RGB.shape[0]:
          rand = torch.randperm(rays_with_RGB.shape[0]) #返回num_of_poses个随机打乱的数组
          rays_with_RGB = rays_with_RGB[rand]
          i_batch = 0
      
      #render:[ro+rd, N_rand, 3]
      rgb_map = render(train_img_h,train_img_w,K,train_view_pos,render_chunk,  rays_selected_to_train, **nerf_trained_args)

      #loss function 
      loss = torch.mean((rgb_map - rays_selected_color_target) ** 2)
      psnr =  -10. * torch.log(loss) / torch.log(torch.Tensor([10.]))

      #optimizer
      optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
      optimizer.zero_grad()

      loss.backward()  # 损失反向传播
      optimizer.step()

      #更新学习率
      decay_rate = 0.1
      decay_steps = lrate_decay * 1000
        #新学习率，以指数衰减系数
      new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
      for param in optimizer.param_groups:
        param['lr'] = new_lrate
      global_step += 1



# if __name__ == "__main__":
# #创建解析器，读取命令行参数
#     parser = get_parser()
#     args = parser.parse_args()
#     main(args)