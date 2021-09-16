from scipy.io import loadmat
import numpy as np
import random
import cv2
import pathlib
from PIL import Image
import os


def load_lm3d():
    pth = os.path.join(pathlib.Path(__file__).parent.parent, 'BFM/similarity_Lm3D_all.mat')
    Lm3D = loadmat(pth)
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(Lm3D[lm_idx[[3, 4]], :], 0),
                     Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D


def POS(xp, x):
    npts = xp.shape[0]
    if npts == 68:
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        xp = np.stack([xp[lm_idx[0], :], np.mean(xp[lm_idx[[1, 2]], :], 0), np.mean(xp[lm_idx[[3, 4]], :], 0),
                       xp[lm_idx[5], :], xp[lm_idx[6], :]], axis=0)
        xp = xp[[1, 2, 0, 3, 4], :]
        npts = 5

    A = np.zeros([2 * npts, 8])
    x = np.concatenate((x, np.ones((npts, 1))), axis=1)
    A[0:2 * npts - 1:2, 0:4] = x

    A[1:2 * npts:2, 4:] = x

    b = np.reshape(xp, [-1, 1])

    k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)

    return t, s


def get_inverse_mat(lm, src_sz=1024, microsoft_sz=224):
    lm_ = np.stack([lm[:, 0], src_sz - 1 - lm[:, 1]], axis=1)
    t, s = POS(lm_, lm3D)
    scale = 102. / s
    dx = -(t[0, 0] * scale - microsoft_sz / 2)
    dy = -((src_sz - t[1, 0]) * scale - microsoft_sz / 2)
    mat_inverse = np.array([[1 / scale, 0, 0, -dx / scale],
                            [0, 1 / scale, 0, -dy / scale],
                            [0, 0, 1 / scale, 0],
                            [0, 0, 0, 1]]).astype('float32')
    return mat_inverse


def get_inverse_mat_256(lm, src_sz=1024, mic_sz=256):
    lm_ = np.stack([lm[:, 0], src_sz - 1 - lm[:, 1]], axis=1)
    t, s = POS(lm_, lm3D)
    scale = 116. / s
    dx = -(t[0, 0] * scale - mic_sz / 2)
    dy = -((src_sz - t[1, 0]) * scale - mic_sz / 2)
    mat_inverse = np.array([[1 / scale, 0, -dx / scale],
                            [0, 1 / scale, -dy / scale]]).astype('float32')
    return mat_inverse


lm3D = load_lm3d()
