 
from torchvision import transforms
from utils.public_tools import get_parser
from dataset.load_data import BlenderDataSet,ResizeImg
from model.nerf_train import create_nerf
from rendering.rays import get_rays
from rendering.render_rays import render
import numpy as np  

def main(args):

    #========= 传参 ==============
    #数据集地址
    dataset_dir       = args.dataset_dir  

    #图像比例：取值应为float,[0,1]
    img_scale         = args.img_scale


    # ========= load dataset ==============  
       
    """
    加载数据，每个数据集上需要得到的数据：
      imgs : 根据 .json 文件加载到的所有图像数据。  
      view_poses : 转置矩阵,表示姿势。 
      height,width,focal: 图像的高、宽、焦距。
    """
    transform_function = transforms.Compose(
        ResizeImg(img_scale),
        transforms.ToTensor(),
    )
    #加载处理训练集和数据集
    train_dataset = BlenderDataSet(f"../{dataset_dir}/",'train',transform_function)
    test_dataset =  BlenderDataSet(f"../{dataset_dir}/",'test',transform_function)
    
    #训练集的参数
    train_focal = train_dataset.focal #焦距
    train_view_pos = train_dataset.view_pos #相机姿势
    train_img_h = train_dataset.img_H
    train_img_w = train_dataset.img_w
    train_img = train_dataset.img_dataset #图像信息

    # =========  nerf ============== 
    
    #调用nerf，初始化模型 
    nerf_trained_args = create_nerf(args)
    
    #生成所有图片的像素点对应的光线原点和方向，并将光线对应的像素颜色与光线聚合到了一起构成 rays_rgb
    """
    train_view_pos 的形状应该为 (num_of_poses, 3, 4),其中 num_of_poses 是相机姿势的数量。
    pos: 每个相机姿势是一个 3x4 的矩阵，其中包含旋转矩阵和平移向量[x,y,z,o]。
    get_rays: 列表推导式,for 每一个pos,使用get_rays获取相应的射线。
    堆叠起来output: rays:(num_of_poses, 2, H, W, 3)
        其中 2 表示每个射线由原点和方向两个部分组成, 3 表示射线方向的三个分量x,y,z。
    """
    rays = np.stack([get_rays(train_img_h,train_img_w,pos,train_focal) for pos in train_view_pos[:,:3,4]],0) 
    
    #由得到的光线得到： 生成所有图片的光线
    if batchs:#开始迭代训练：以批处理的形式对进行训练
      for i in trange(start, N_iters):
    
    else: # 从所有的图像中随机选择一张图像用于训练

    #以上步骤将生成的光线和对应的像素点颜色分离，得到:batch_rays, target_colors

    # ========= rendering =============
    rgb, disp, acc, extras = render(并行处理的光线的数量,  batch_rays, **nerf_trained_args)

    # ========= loss function =============
    loss = 
    psnr = 

    # ======== optimizer and schedule ===========
    #创建优化器
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    loss.backward()  # 损失反向传播
    optimizer.step()

    #更新学习率
    decay_rate = 

 


# if __name__ == "__main__":
# #创建解析器，读取命令行参数
#     parser = get_parser()
#     args = parser.parse_args()
#     main(args)