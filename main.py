from collections import namedtuple

import torchvision.utils
import torch

# from glass_post.detail.ltm.neur_br.net import *
# from glass_post.detail.ltm.neur_br.net.downsampler import *
from glass_post.detail.ltm.neur_br.enhancement import lowlight_enhancer
from PIL import Image
import torch
import argparse
from glob import glob
import os
import json
from thop import profile


parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", type=str, default="data", help="test image folder")
parser.add_argument("--result", "-r", type=str, default="./result", help="result folder")
parser.add_argument("--gamma", "-gc", type=float, default=0.6, help="gamma correction factor")
parser.add_argument("--low_size", "-ls", type=int, default=128, help="gamma correction factor")
parser.add_argument("--noise_size", "-ns", type=int, default=128, help="input noise size")
parser.add_argument("--iter_num", "-iter", type=int, default=200, help="input noise size")
arg = parser.parse_args()


EnhancementResult = namedtuple("EnhancementResult", ["illumination"])

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.cuda.set_device(1)


# cv2.imwrite('output/result-{}.png'.format(step), self.best_result)


if __name__ == "__main__":
    print(arg)
    input_root = arg.input
    output_root = arg.result
    # datasets = ['DICM', 'ExDark','LIME', 'Fusion', 'NPEA', 'Nasa', 'VV']
    datasets = ["LIME"]
    for dataset in datasets:
        print("Dataset:{}".format(dataset))
        input_folder = os.path.join(input_root, dataset)
        output_folder = os.path.join(output_root, dataset)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # print(output_folder)
        path = glob(input_folder + "/*.*")
        path.sort()
        for i in range(len(path)):
            filename = os.path.basename(path[i])
            img_path = os.path.join(input_folder, filename)
            img_path_out = os.path.join(output_folder, filename)
            img = Image.open(img_path).convert("RGB")
            lowlight_enhancer(img_path_out, img)
