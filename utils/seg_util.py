import torch
from torch import nn
import torch.nn.functional as F


@torch.no_grad()
def seg_rgb2shape(seg_rgb, face_mask=None):
    # get the ground truth of training, no need to require grad
    # seg_rgb: Nx1xHxW or Nx13xHxW
    seg_rgb = seg_rgb.squeeze(dim=1)
    if seg_rgb.dim() == 4:
        seg_rgb = seg_rgb.argmax(dim=1)  # NxHxW
    assert seg_rgb.dim() == 3

    if face_mask is not None:
        face_mask = face_mask.squeeze(1)
        seg_shape = torch.zeros_like(face_mask)  # background 0
        seg_shape += face_mask  # skin 1
        seg_shape[seg_rgb == 5] = 2  # l_eye 2
        seg_shape[seg_rgb == 6] = 3  # r_eye 3
        seg_shape[seg_rgb == 7] = 4  # nose 4
        seg_shape[seg_rgb == 8] = 0  # mouth 0
        seg_shape[seg_rgb == 9] = 6  # u_lip 6
        seg_shape[seg_rgb == 10] = 7  # l_lip 7

    else:
        seg_shape = torch.zeros_like(seg_rgb)
        seg_shape[seg_rgb == 1] = 1  # skin
        seg_shape[seg_rgb == 11] = 1  # hair to skin
        seg_shape[seg_rgb == 2] = 1  # l_brow to skin
        seg_shape[seg_rgb == 3] = 1  # r_brow to skin
        seg_shape[seg_rgb == 4] = 1  # eye_g to skin
        seg_shape[seg_rgb == 5] = 2  # l_eye
        seg_shape[seg_rgb == 6] = 3  # r_eye
        seg_shape[seg_rgb == 7] = 4  # nose
        seg_shape[seg_rgb == 8] = 0  # mouth
        seg_shape[seg_rgb == 9] = 5  # u_lip
        seg_shape[seg_rgb == 10] = 6  # l_lip

    return seg_shape


@torch.no_grad()
def valid_seg_rgb(seg_rgb):
    # seg_rgb: output of data fetcher, Bx1xHxW
    # set border
    seg_rgb[:, :, 0:2, :] = 0
    seg_rgb[:, :, -2:, :] = 0
    seg_rgb[:, :, :, 0:2] = 0
    seg_rgb[:, :, :, -2:] = 0

    # convert label
    seg_rgb = seg_rgb.squeeze(dim=1)
    seg_valid = torch.zeros_like(seg_rgb)
    seg_valid[seg_rgb == 1] = 1  # skin
    seg_valid[seg_rgb == 2] = 1  # l_brow to skin
    seg_valid[seg_rgb == 3] = 1  # r_brow to skin
    seg_valid[seg_rgb == 4] = 0  # eye_g to bg
    seg_valid[seg_rgb == 5] = 2  # l_eye
    seg_valid[seg_rgb == 6] = 3  # r_eye
    seg_valid[seg_rgb == 7] = 4  # nose
    seg_valid[seg_rgb == 8] = 0  # mouth
    seg_valid[seg_rgb == 9] = 5  # u_lip
    seg_valid[seg_rgb == 10] = 6  # u_lip
    seg_valid[seg_rgb == 11] = 0  # hair to bg
    return seg_valid


@torch.no_grad()
def label2onehot(label):
    onehot = torch.eye(7).to(label.device)
    return onehot[label].permute(0, 3, 1, 2)


@torch.no_grad()
def get_face_mask(label):
    mask = (label > 0) * (label < 11) * (~(label == 4)) * (~(label == 8))  # when train sampler set comment label==8
    mask = mask.type(torch.float32)
    return mask


def tensor_erode(bin_img, ksize=5):
    # 首先加入 padding，防止腐蚀后图像尺寸缩小
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k
    eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded


def tensor_dilate(bin_img, ksize=5):
    # 首先加入 padding，防止腐蚀后图像尺寸缩小
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k
    eroded, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)
    return eroded


def tensor_close(bin_img, ksize=3):
    out = tensor_dilate(bin_img, ksize)
    out = tensor_erode(out, ksize)
    return out


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class OhemBCELoss(nn.Module):
    def __init__(self, thresh, n_min):
        super(OhemBCELoss, self).__init__()
        self.n_min = n_min
        self.criteria = nn.BCEWithLogitsLoss(reduction='none')
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()

    def forward(self, logits, labels):
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]

        else:
            loss = loss[: self.n_min]

        return torch.mean(loss)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        #self.activation = nn.Sigmoid()

    def forward(self, pr, gt, eps=1e-7):
        #pr = self.activation(pr)
        tp = torch.sum(gt * pr)
        fp = torch.sum(pr) - tp
        fn = torch.sum(gt) - tp
        score = (2 * tp + eps) / (2 * tp + fn + fp + eps)
        return 1 - score
