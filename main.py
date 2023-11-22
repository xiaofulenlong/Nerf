 
from torchvision import transforms
from utils.public_tools import get_parser
from dataset.load_data import BlenderDataSet,ResizeImg
from model.nerf_train import create_nerf
from rendering.rays import get_rays
from rendering.render_rays import render


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
      poses : 转置矩阵,表示姿势。 
      height,width,focal: 图像的高、宽、焦距。
    """
    transform_function = transforms.Compose(
        ResizeImg(img_scale),
        transforms.ToTensor(),
    )

    train_dataset = BlenderDataSet(f"../{dataset_dir}/",'train',transform_function)


    # =========  nerf ============== 
    
    #调用nerf，初始化模型 
    nerf_trained_args = create_nerf()
    
    #生成了所有图片的像素点对应的光线原点和方向，并将光线对应的像素颜色与光线聚合到了一起构成 rays_rgb
    ray = get_rays()
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
    loss.backward()  # 损失反向传播
    optimizer.step()

    #更新学习率
    decay_rate = 

 


if __name__ == "__main__":
#创建解析器，读取命令行参数
    parser = get_parser()
    args = parser.parse_args()
    main(args)