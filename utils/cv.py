
import torch
import torch.nn.functional as F
from torch import nn
import cv2
import numpy as np
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import pickle
import pathlib
import os


def rgb2gray(batch_img):
    device = batch_img.device
    weight = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(device)  # RGB order
    gray = (batch_img * weight).sum(dim=1, keepdim=True)
    return gray


def tensor2img(tensor):
    shape = tensor.shape
    show = tensor
    if len(shape) == 4:
        if shape[0] == 1:
            show = tensor.squeeze(0)
        else:
            show = make_grid(tensor, nrow=tensor.shape[0], padding=0)

    show = show.detach().permute(1, 2, 0).cpu().numpy()
    if show.max() > 1:
        show = show.astype('uint8')
    elif show.min()<0:
        show = (show*128+127.5).astype('uint8')
        #show = show*0.5+0.5
        #show = (show * 255).astype('uint8')
    else:
        show = (show*255).astype('uint8')
    return show


@torch.no_grad()
def save_result(batch_img_list, pth):
    batch_list = []
    B, C, H, W = batch_img_list[0].shape
    for batch_img in batch_img_list:
        b, c, h, w = batch_img.shape
        if c != 3:
            batch_img = batch_img.repeat(1, 3, 1, 1)
        if h!=H:
            batch_img = F.interpolate(batch_img, (H, W), mode='bilinear', align_corners=True)

        batch_list.append(batch_img)

    bz = batch_img_list[0].shape[0]
    total = torch.cat(batch_list, dim=0)
    # save_image(total, pth, nrow=bz, padding=2, normalize=True, scale_each=True)
    show = make_grid(total, nrow=bz, padding=2, normalize=True, scale_each=True)
    show = (show.detach().squeeze().permute(1, 2, 0).cpu().numpy()*255).astype('uint8')
    cv2.imwrite(pth, show)


def save_ckpt(ckpt_dict, pth):
    torch.save(ckpt_dict, pth)


def mkdir(name, format=None):
    pth = pathlib.Path(__file__).parent.parent.absolute()
    dir_pth = os.path.join(pth, name)
    if not os.path.exists(dir_pth):
        os.mkdir(dir_pth)
    if format is not None:
        dir_pth = os.path.join(dir_pth, format)
    return dir_pth


def img2tensor(img):
    if img.max() > 1:
        img = img.astype(np.float32) / 255.

    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img




if __name__ == '__main__':
    import matplotlib.pyplot as plt
