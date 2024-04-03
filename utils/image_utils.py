#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import cv2 
import matplotlib.pyplot as plt
import numpy as np

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def viridis_cmap(gray: np.ndarray) -> np.ndarray:
    """
    Visualize a single-channel image using matplotlib's viridis color map
    yellow is high value, blue is low
    :param gray: np.ndarray, (H, W) or (H, W, 1) unscaled
    :return: (H, W, 3) float32 in [0, 1]
    """
    colored = plt.cm.viridis(plt.Normalize()(gray.squeeze()))[..., :-1]
    return colored.astype(np.float32)

def erode(img_in, erode_size=4):
    img_out = np.copy(img_in)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    img_out = cv2.erode(img_out, kernel, iterations=1)

    return img_out