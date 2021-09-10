#!/usr/bin/python
# -*- encoding: utf-8 -*-

import random
import numpy as np
import cv2


class RandomAffine(object):
    def __init__(self, scale, angle, flip=0.5):
        self.scale = scale
        self.angle = np.deg2rad(angle)
        self.flip = flip  # probability of left-right flip

    def __call__(self, data):
        # data = {'img': img, 'uv': uv, 'mat_inv': mat_inverse, 'mask': mask}
        img, uv, mat_inv, mask, seg = data['img'], data['uv'], data['mat_inv'], data['mask'], data['seg']
        h, w = img.shape[:2]

        # flip flag
        flip = random.random() < self.flip

        # rotation matrix
        angle = 2 * self.angle * random.random() - self.angle
        cos = np.cos(angle)
        sin = np.sin(angle)

        s = 1 + 2 * self.scale * random.random() - self.scale

        M_rot = np.array([
            [s * cos, s * sin, s * (-sin * h / 2 - cos * w / 2) + w / 2],
            [-s * sin, s * cos, s * (sin * w / 2 - cos * h / 2) + h / 2]
        ])

        if flip:
            M_flip = np.array([[-1, 0, w],
                               [0, 1, 0],
                               [0, 0, 1]])
            M = M_rot @ M_flip
            uv = cv2.flip(uv, 1)
            flip = 1
        else:
            M = M_rot
            flip = 0

        image = cv2.warpAffine(img, M, (h, w))
        mask = cv2.warpAffine(mask, M, (h, w), flags=cv2.INTER_NEAREST)
        seg = cv2.warpAffine(seg, M, (h, w), flags=cv2.INTER_NEAREST)

        M = np.array([
            [M[0, 0], M[0, 1], 0, M[0, 2]],
            [M[1, 0], M[1, 1], 0, M[1, 2]],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        mat_inv = (M @ mat_inv).astype(np.float32)

        data = {'img': image, 'uv': uv, 'mat_inv': mat_inv, 'mask': mask, 'seg': seg, 'flip': flip}
        return data


if __name__ == '__main__':
    import os

    img_root = '/media/xn/SSD1T/CelebAMask-HQ/img256_mic'
    img_list = os.listdir(img_root)
    tf = RandomAffine(0.1, 30, 0.5)

    for idx in range(10):
        I = cv2.imread(os.path.join(img_root, img_list[idx]))
        J = tf(I)
        IJ = np.concatenate((I, J), axis=1)
        cv2.imshow("show", IJ)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
