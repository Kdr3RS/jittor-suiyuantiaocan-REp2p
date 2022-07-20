import random
import argparse
from PIL import Image, ImageEnhance
import os.path

parser = argparse.ArgumentParser()
parser.add_argument("--origin_path", type=str, default="./Data/train/imgs", help="images loaded from")
opt = parser.parse_args()
origin_path = opt.origin_path
new_path = opt.origin_path
filelist = os.listdir(origin_path)
totalnum = len(filelist)
print(totalnum, "images loaded from", origin_path)


def Flip(im):
    imgf = im.transpose(Image.FLIP_LEFT_RIGHT)
    return imgf


def mirrorL(im):
    imm = im
    width, height = imm.size
    half = int(width/2)
    for x in range(half):
        for y in range(height):
            origin = imm.getpixel((width-x-1, y))
            imm.putpixel((x, y), origin)
    return imm


def enhance_constract(im):
    imge = ImageEnhance.Contrast(im)
    r = round(random.uniform(0.7, 1.8), 3)
    imger = imge.enhance(r)
    return imger


def enhance_color(im):
    imgc = ImageEnhance.Color(im)
    rd = round(random.uniform(0.6, 1.8), 3)
    imgcl = imgc.enhance(rd)
    return imgcl


def enhance_brightness(im):
    imgb = ImageEnhance.Brightness(im)
    rb = round(random.uniform(0.7, 1.3), 3)
    imgbr = imgb.enhance(rb)
    return imgbr
"""
def rename():
    i = 0
    path = new_path
    arglist = os.listdir(path)

    for files in arglist:
        i=i+1
        old_dir = os.path.join(path, files)
        filetype = ".jpg"
        new_dir = os.path.join(path, str(i) + filetype)
        os.rename(old_dir, new_dir)
    return True
"""

os.makedirs(f"{new_path}", exist_ok=True)
i = 0
for subdir in filelist:
    sub_dir = origin_path + '/' + subdir

    im = Image.open(sub_dir)
    #im.save(new_path + '/' + subdir)
    fimg = Flip(im)
    fimg.save(new_path + '/' + 'fl-' + subdir)
    #eimg = enhance_constract(im)
    #eimg.save(new_path + '/' + 'co-' + subdir)
    #cimg = enhance_color(im)
    #cimg.save(new_path + '/' + "cl-" + subdir)
    #simg = enhance_brightness(im)
    #simg.save(new_path + '/' + 'br-' + subdir)
    #mimg = mirrorL(im)
    #mimg.save(new_path + '/' + 'mi-' + subdir)
    i +=1
    print(i, "of", totalnum, " processed")

#rename()
print("done")
