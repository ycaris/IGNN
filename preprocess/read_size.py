import os
import numpy as np
import cv2
from PIL import Image

## resize higher resolution image to lower resolution

root_path = "/home1/yz2337/projects/2023/graph/IGNN/datasets/xray_original_size/test/HR"

for file in os.listdir(root_path):
    img_path = os.path.join(root_path, file)
    img = Image.open(img_path)
    print(img.size)


