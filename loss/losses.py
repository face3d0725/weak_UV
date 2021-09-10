import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import vgg19
import torch.nn.functional as F
from loss.ssim import SSIM
from loss.vgg_loss import VGG_Activations, vgg_face
from loss.light_cnn import LightCNN_29Layers_v2
from facenet_pytorch import InceptionResnetV1
import pdb
import pickle


class Loss(nn.Module):
    def __init__(self, vgg=False, device='cuda:0'):
        super(Loss, self).__init__()
        if vgg:
            self.VGG_FACE_AC = VGG_Activations(vgg_face(pretrained=True), [1, 6, 11, 18, 25])
            self.VGG19_AC = VGG_Activations(vgg19(pretrained=True), [1, 6, 11, 20, 29])
        # self.lightcnn = LightCNN_29Layers_v2()
        self.facenet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)  # vggface2
        self.light_cnn = LightCNN_29Layers_v2().eval().to(device)
        # self.vgg_face = VGG_Activations(vgg_face(pretrained=True), [36])  # [33,36]
        landmark_weight = torch.cat((torch.ones(28), 20 * torch.ones(3), torch.ones(29), 20 * torch.ones(8)))
        self.landmark_weight = landmark_weight.view(1, 68).to(device)
        self.ssim_fun = SSIM(window_size=11, size_average=True)

        self.device = device
        self.to(device)

    def vgg(self, x, x_hat):
        # vgg Loss
        vgg19_x_hat = self.VGG19_AC(x_hat)
        vgg19_x = self.VGG19_AC(x)

        vgg19_loss = 0
        for i in range(0, len(vgg19_x)):
            vgg19_loss += F.l1_loss(vgg19_x_hat[i], vgg19_x[i])

        vgg_face_x_hat = self.VGG_FACE_AC(x_hat)
        vgg_face_x = self.VGG_FACE_AC(x)
        vgg_face_loss = 0
        for i in range(0, len(vgg_face_x)):
            vgg_face_loss += F.l1_loss(vgg_face_x_hat[i], vgg_face_x[i])

        return 1e-2 * vgg19_loss + 2e-3 * vgg_face_loss

    def vgg_19(self, x, x_hat):
        # VGG19 Loss
        vgg19_x_hat = self.VGG19_AC(x_hat)
        vgg19_x = self.VGG19_AC(x)

        vgg19_loss = 0
        for i in range(0, len(vgg19_x)):
            vgg19_loss += F.l1_loss(vgg19_x_hat[i], vgg19_x[i])

        return vgg19_loss * 1e-2

    def id_loss(self, x_hat, x):
        bz = x.shape[0]
        feat_x_hat = self.facenet(x_hat)
        with torch.no_grad():
            feat_x = self.facenet(x)

        cosine_loss = 1 - (feat_x * feat_x_hat).sum() / bz
        return cosine_loss

    def id_loss_vgg(self, x_hat, x):
        bz = x.shape[0]
        feat_x_hat = self.vgg_face(x_hat)[0]
        with torch.no_grad():
            feat_x = self.vgg_face(x)[0]

        feat_x_hat = F.normalize(feat_x_hat, p=2, dim=1)
        feat_x = F.normalize(feat_x, p=2, dim=1)
        cosine_loss = 1 - (feat_x * feat_x_hat).sum() / bz
        return cosine_loss

    def id_loss_light(self, x_hat, x):
        bz = x.shape[0]
        feat_x_hat = self.light_cnn(x_hat)
        with torch.no_grad():
            feat_x = self.light_cnn(x)

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
    def tv_edge(x_hat, mask_edge):
        H, W = x_hat.shape[2:]
        dx = x_hat[:, :, 0:H - 1, :] - x_hat[:, :, 1:H, :]
        dx = dx * mask_edge[:, :, :H - 1, :]

        dy = x_hat[:, :, :, 0:W - 1] - x_hat[:, :, :, 1:W]
        dy = dy * mask_edge[:, :, :, :W - 1]
        total_pix = torch.sum(mask_edge)
        total_loss = (torch.sum(dx**2)+torch.sum(dy**2))/total_pix
        return total_loss

    @staticmethod
    def dice(pred, target):
        union = torch.abs(pred * target).sum(dim=(2, 3), keepdim=True)
        num_pred = torch.abs(pred).sum(dim=(2, 3), keepdim=True)
        num_target = torch.abs(target).sum(dim=(2, 3), keepdim=True)
        loss = union * 2.0 / (num_pred + num_target + 1e-5)
        loss_dice = 1 - loss.mean()
        return loss_dice

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

    def ssim(self, x, x_hat):
        return -self.ssim_fun(x_hat, x)

    def photo_loss(self, x, x_hat, mask):
        loss = (((x - x_hat) ** 2).sum(dim=1, keepdim=True)) ** 0.5 * mask
        try:
            dbg = {'x': x, 'x_hat': x_hat, 'mask': mask}
            with open('debug.pkl', 'wb') as f:
                pickle.dump(dbg, f)
            loss = loss.sum() / torch.max(mask.sum(), torch.tensor(1.0).to(self.device))
        except:
            print('ENTER DEBUG')
            pdb.set_trace()
        return loss

    def landmark_loss(self, ldmk_p, ldmk_label):
        bz = ldmk_p.shape[0]
        weight = self.landmark_weight.repeat(bz, 1)
        loss = (((ldmk_p - ldmk_label) ** 2).sum(2) * weight).sum() / (68 * bz)
        # loss = ((torch.abs(ldmk_p - ldmk_label)).sum(2) * weight).sum() / (68 * bz)
        return loss

    @staticmethod
    def regulation_loss(id_coeff, ex_coeff, tex_coeff):
        # id_coeff: Nx80, ex_coeff: Nx64, tex_coeff: Nx80
        w_ex = 0.8
        w_tex = 1.7e-2
        bz = id_coeff.shape[0]
        loss = (id_coeff ** 2).sum() + w_ex * (ex_coeff ** 2).sum() + w_tex * (tex_coeff ** 2).sum()
        loss = loss / bz
        return loss

    @staticmethod
    def gamma_loss(gamma):
        gamma = gamma.view(-1, 3, 9)
        gamma_mean = torch.mean(gamma, dim=1, keepdim=True)
        gamma_loss = ((gamma - gamma_mean) ** 2).mean()
        return gamma_loss

    def reflectance_loss(self, face_texture, skin_mask):  # a batman mask, only top half front face w/o eyes
        # face_texture: Nx35709x3, skin_mask: 1x35709x1
        bz = face_texture.shape[0]
        texture_mean = (face_texture * skin_mask).sum(dim=(2, 3), keepdim=True) / (skin_mask[0].sum())
        loss = (((face_texture - texture_mean) * skin_mask) ** 2).sum() / (skin_mask.sum())
        return loss

    def forward(self, x, x_hat):
        return self.vgg(x, x_hat)


if __name__ == '__main__':
    criterion = Loss(vgg=False, device='cuda:0')
    # x = torch.rand(10, 3, 128, 128).cuda()
    # y = torch.rand(10, 3, 128, 128).cuda()
    ldmk_p, ldmk_label = torch.rand(10, 68, 2), torch.rand(10, 68, 2)

    loss = criterion.landmark_loss(ldmk_p, ldmk_label)
    print(loss.item())
