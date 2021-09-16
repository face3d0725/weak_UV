import torch
from torch import nn
import torch.nn.functional as F


@torch.no_grad()
def label2onehot(label):
    onehot = torch.eye(7).to(label.device)
    return onehot[label].permute(0, 3, 1, 2)


def tensor_erode(bin_img, ksize=5):
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k
    eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded


def tensor_dilate(bin_img, ksize=5):
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
