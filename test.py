import jittor as jt
from jittor import init
from jittor import nn
import argparse
import os
import numpy as np
import math

import cv2

from modelsA import GeneratorUNet
from datasets import *


import warnings
warnings.filterwarnings("ignore")
jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="./Data")
parser.add_argument("--output_path", type=str, default="./result")
parser.add_argument("--epoch", type=int, default=0, help="epoch of trained model")
parser.add_argument("--img_height", type=int, default=384, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")

opt = parser.parse_args()
print(opt)

os.makedirs(f"{opt.output_path}/images/", exist_ok=True)

transforms = [
    transform.Resize(size=(opt.img_height, opt.img_width), mode=Image.BICUBIC),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]
val_dataloader = ImageDataset(opt.input_path, mode="val", transforms=transforms).set_attrs(
     batch_size=10,
     shuffle=False,
     num_workers=1,
 )

generator = GeneratorUNet()


@jt.no_grad()
def eval():
    ckpt = jt.load(f"{opt.output_path}/saved_models/state_dict_{opt.epoch}.pkl")
    generator.load_state_dict(ckpt['generator'])
    generator.eval()
    cnt = 1
    os.makedirs(f"{opt.output_path}/images/result/epoch_{opt.epoch}", exist_ok=True)
    for i, (_, real_A, photo_id) in enumerate(val_dataloader):
        fake_B = generator(real_A)

        fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')
        for idx in range(fake_B.shape[0]):
            cv2.imwrite(f"{opt.output_path}/images/result/epoch_{opt.epoch}/{photo_id[idx]}.jpg",
                        fake_B[idx].transpose(1, 2, 0)[:, :, ::-1])
            cnt += 1


eval()
