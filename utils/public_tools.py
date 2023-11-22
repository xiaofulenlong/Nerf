import argparse

def get_parser():
    parser = argparse.ArgumentParser() 

    parser.add_argument('--dataset_dir',type=str,default='./data/nerf_synthetic/lego',help="dataset dir")
    # dataset options
    parser.add_argument('--dataset_type',type=str,default='blender', help='options: blender / llff')

    parser.add_argument("--img_scale", type = float, default = 0.5, help = "Scale of the image")

    return parser
