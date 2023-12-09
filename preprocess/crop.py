import os
import numpy as np
import cv2
from PIL import Image

## resize higher resolution image to lower resolution

root_path = "/home1/yz2337/projects/2023/graph/IGNN/datasets/xray_original_size/test/HR"
save_path = "/home1/yz2337/projects/2023/graph/IGNN/datasets/xray/test/HR"


def center_crop_image(img, crop_width, crop_height):
    """
    Crops the image at the center to the specified width and height.

    :param image_path: Path to the image file.
    :param crop_width: Desired width of the cropped image.
    :param crop_height: Desired height of the cropped image.
    :return: Cropped image.
    """
    img_width, img_height = img.size

    # Calculate the left, upper, right, and lower pixels for cropping
    left = (img_width - crop_width) / 2
    top = (img_height - crop_height) / 2
    right = (img_width + crop_width) / 2
    bottom = (img_height + crop_height) / 2

    # Crop the center of the image
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img


if not os.path.exists(save_path):
    os.makedirs(save_path)

for file in os.listdir(root_path):
    img_path = os.path.join(root_path, file)
    img = Image.open(img_path)
    w,h = img.size
    crop_img = center_crop_image(img, 900, 800)
    crop_img.save(os.path.join(save_path, file))
    print(crop_img.size)

    print(img.size)


