import argparse

def get_parser():
    parser = argparse.ArgumentParser() 

    # dataset optionss
    parser.add_argument('--dataset_dir',type=str,default='./data/nerf_synthetic/lego',help="dataset dir")
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


    return parser
