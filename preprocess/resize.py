import os
import numpy as np
import cv2
from PIL import Image

## resize higher resolution image to lower resolution

root_path = "/home1/yz2337/projects/2023/graph/IGNN/datasets/xray/test/HR"
save_path = "/home1/yz2337/projects/2023/graph/IGNN/datasets/xray/test/LR"


if not os.path.exists(save_path):
    os.makedirs(save_path)

for file in os.listdir(root_path):
    img_path = os.path.join(root_path, file)
    img = Image.open(img_path)
    w,h = img.size
    crop_img = img.resize((w//2,h//2))
    crop_img.save(os.path.join(save_path, file))
    print(crop_img.size)

    print(img.size)


