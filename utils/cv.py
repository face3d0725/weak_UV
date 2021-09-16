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


def ldmk2mask(ldmk, size=128):
    ldmk = ldmk.astype('int64')
    mask = np.ones((size, size), dtype='uint8')

    # face = cv2.convexHull(np.vstack((ldmk[0:17],
    #                                  ldmk[22:27][::-1],
    #                                  ldmk[17:22][::-1])))
    left_eye = cv2.convexHull(ldmk[36:42])
    right_eye = cv2.convexHull(ldmk[42:48])
    inner_mouth = ldmk[60:]
    # cv2.fillPoly(mask, [face], 1)
    cv2.fillPoly(mask, [left_eye, right_eye], 0)
    cv2.fillPoly(mask, [inner_mouth], 0)

    return mask


def draw_ldmk(img, ldmk):
    if type(img) is torch.Tensor:
        img = tensor2img(img)
    if type(ldmk) is torch.Tensor:
        ldmk = ldmk.squeeze().numpy()

    shape = img.shape
    if img.shape[0] < 512:
        scale = 512 // img.shape[0]
        dsize = (shape[1] * scale, shape[0] * scale)
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR)
        ldmk = ldmk * scale

    for idx in range(ldmk.shape[0]):
        point = tuple(ldmk[idx].astype('int'))
        cv2.circle(img, point, radius=1, color=(255, 0, 0), thickness=-1)
        cv2.putText(img, '{}'.format(idx), point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    return img


def show_img(img, mode=0):
    if mode == 0:
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.show()


def beauty_face(img):
    v1 = 5  # 磨皮程度
    v2 = 3  # 细节程度
    dx = v1 * 5  # 双边滤波参数之一
    fc = v1 * 12.5  # 双边滤波参数之一
    p = 0.1
    # 双边滤波
    # temp1 = cv2.bilateralFilter(img, dx, fc, fc)
    temp1 = cv2.GaussianBlur(img, (5, 5), 0)

    temp2 = cv2.subtract(temp1, img);
    temp2 = cv2.add(temp2, (10, 10, 10, 128))
    # 高斯模糊
    temp3 = cv2.GaussianBlur(temp2, (2 * v2 - 1, 2 * v2 - 1), 0)
    temp4 = cv2.add(img, temp3)
    dst = cv2.addWeighted(img, p, temp4, 1 - p, 0.0)
    dst = cv2.add(dst, (10, 10, 10, 255))
    return dst


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    I = cv2.imread('../weights/a.png')[:, :, ::-1]
    J = beauty_face(I)

    plt.figure()
    plt.imshow(I)

    plt.figure()
    plt.imshow(J)
    plt.show()