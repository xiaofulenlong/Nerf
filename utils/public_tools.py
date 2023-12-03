import argparse

def get_parser():
    parser = argparse.ArgumentParser() 

    # dataset optionss
    parser.add_argument('--dataset_dir',type=str,default='./nerf_synthetic/lego',help="dataset json dir")
    parser.add_argument('--dataset_type',type=str,default='blender', help='options: blender / llff')
    parser.add_argument("--img_scale", type = float, default = 0.5, help = "Scale of the image")

    #mlp options
    parser.add_argument('--fre_position_L',type=int,default=10,help="the frequence of position for the encoding")
    parser.add_argument('--fre_view_L',type=int,default=4,help="the frequence of view for the encoding")
    parser.add_argument('--network_depth',type=int,default=8,help="the depth of mlp network")
    parser.add_argument('--hidden_unit_num',type=int,default=256,help="the hidden unit num of mlp network")
    parser.add_argument('--output_features_dim',type=int,default=256,help="the output features num of mlp network")
    parser.add_argument('--output_dim',type=int,default=128,help="the output dim of mlp network")
    parser.add_argument('--netchunkNum',type=int,default=1024*64,help="number of sent through network in parallel")

    #render options
    parser.add_argument("--coarse_num", type=int, default=32*32*4, 
                        help='coarse number')
    parser.add_argument('--if_use_batchs',action='store_true', 
                        help='whether only take random rays from 1 image at a time or n images')
    parser.add_argument("--render_chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')    

    #train options
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    return parser
