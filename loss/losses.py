import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class Loss(nn.Module):
    def __init__(self, device='cuda:0'):
        super(Loss, self).__init__()
        self.facenet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)  # vggface2
        landmark_weight = torch.cat((torch.ones(28), 20 * torch.ones(3), torch.ones(29), 20 * torch.ones(8)))
        self.landmark_weight = landmark_weight.view(1, 68).to(device)
        self.device = device
        self.to(device)

    def id_loss(self, x_hat, x):
        bz = x.shape[0]
        feat_x_hat = self.facenet(x_hat)
        with torch.no_grad():
            feat_x = self.facenet(x)

        cosine_loss = 1 - (feat_x * feat_x_hat).sum() / bz
        return cosine_loss

    @staticmethod
    def tv(x_hat):
        H, W = x_hat.shape[2:]
        dx = x_hat[:, :, 0:H - 1, :] - x_hat[:, :, 1:H, :]
        dy = x_hat[:, :, :, 0:W - 1] - x_hat[:, :, :, 1:W]
        total_loss = torch.mean(dx ** 2) + torch.mean(dy ** 2)
        return total_loss

    @staticmethod
    def adv_loss(logits, target):
        assert target in [1, 0]
        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss

    @staticmethod
    def l1_loss(pred, target):
        return F.l1_loss(pred, target)

    @staticmethod
    def Oheml1_loss(pred, target, thresh=0.5, n_min=256 ** 2):
        loss = F.l1_loss(pred, target, reduction='none').view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[n_min] > thresh:
            loss = loss[loss > thresh]
        else:
            loss = loss[:n_min]
        return torch.mean(loss)

    @staticmethod
    def l2_loss(pred, target):
        return F.mse_loss(pred, target)

    @staticmethod
    def symmetric(pred):
        pred_flip = torch.flip(pred, dims=(3,))
        return F.l1_loss(pred, pred_flip)

    @staticmethod
    def r1_reg(d_out, x_in):
        # zero-centered gradient penalty for real images
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
        return reg

