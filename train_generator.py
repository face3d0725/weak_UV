from models.sampler import Sampler
from models.face_decoder import Face3D
from models.networks import Generator, Discriminator, DiscriminatorUV
from dataset.face_dataset import InputFetcher
from dataset.dataset_aug import Fetcher
from utils.uv import Attr2Uv, Uv2Attr, ImgVertAttr
from utils.cv import tensor2img, mkdir
from utils.logger import setup_logger
from utils.simple_renderer import SimpleRenderer
from utils.seg_util import OhemBCELoss
import torch
import torch.nn.functional as F
from loss.losses import Loss
import random
import numpy as np
import cv2
import math
import tqdm
from contiguous_params import ContiguousParams
import pickle

import torch.backends.cudnn as cudnn

cudnn.benchmark = True


# grid gt doesn't consider the hair, only focus on the shape

class GenSolver(object):
    def __init__(self):
        result_folder = 'res_gen'
        ckpt_folder = 'ckpt_gen'
        self.result_pth = mkdir(result_folder, format='it_{:05}.png')
        self.ckpt_pth = mkdir(ckpt_folder, format='it_{:05}.pkl')
        self.ckpt_sampler_pth = './ckpt_sampler/it_70000.pkl'
        self.device = 'cuda:0'
        self.bz = 4
        self.size_uv = 256
        self.size_img = 256
        self.lr = 1e-4
        self.start_step = 0
        self.n_steps = 200000
        self.test_step = 500
        self.save_step = 10000
        self.device_id = [0, 1]

        self.gen, self.dis, self.dis_uv_c, self.dis_uv_p, self.sampler, self.face_decoder, \
        self.opt_gen, self.opt_dis, self.opt_sampler, self.opt_disuv_c, self.opt_disuv_p = self.init_model()

        self.sampler.eval()
        self.scheduler_gen = torch.optim.lr_scheduler.MultiStepLR(self.opt_gen, milestones=[100000], gamma=0.1)
        self.scheduler_dis = torch.optim.lr_scheduler.MultiStepLR(self.opt_dis, milestones=[100000], gamma=0.1)
        self.scheduler_disuv_c = torch.optim.lr_scheduler.MultiStepLR(self.opt_disuv_c, milestones=[100000], gamma=0.1)
        self.scheduler_disuv_p = torch.optim.lr_scheduler.MultiStepLR(self.opt_disuv_p, milestones=[100000], gamma=0.1)
        self.renderer = SimpleRenderer(self.device).to(self.device)
        tri = self.face_decoder.facemodel.tri
        self.tri_bch = tri.unsqueeze(0).repeat(self.bz, 1, 1)
        uv_coords = self.face_decoder.facemodel.uv
        self.img2vert = ImgVertAttr()
        self.uv_reader = Uv2Attr(uv_coords, size=self.size_uv)
        self.uv_writer = Attr2Uv(uv_coords, tri, self.bz, size=self.size_uv)
        self.datafetcher = InputFetcher('ffceleba', self.bz)
        uv_seg = torch.from_numpy(np.load('BFM/uv_seg.npy')).unsqueeze(0).to(self.device)
        self.uv_template_mask = uv_seg > 0

        uv_ctr_mask = torch.from_numpy((cv2.imread('BFM/center_uv_mask.png', 0) / 255.0).astype('float32'))
        self.uv_ctr_mask = uv_ctr_mask.unsqueeze(0).unsqueeze(0).repeat(self.bz, 1, 1, 1).to(self.device)

        self.bg_fetcher = Fetcher(name='bg', batch_size=self.bz, device=self.device)
        self.uv_ctr_fetcher = Fetcher(name='uvfront', batch_size=self.bz, device=self.device)
        self.uv_sd_fetcher = Fetcher(name='uvside', batch_size=2 * self.bz, device=self.device)
        self.uv_mask_fetcher = Fetcher(name='uvmask', batch_size=self.bz, device=self.device)

        self.criterion = torch.nn.DataParallel(Loss(device=self.device).cuda(), device_ids=self.device_id)
        self.seg_criterion = OhemBCELoss(thresh=0.7, n_min=256 ** 2)
        self.logger = setup_logger('./log_gen')

    def init_model(self):
        sampler = torch.nn.DataParallel(Sampler().cuda(), self.device_id)
        generator = torch.nn.DataParallel(Generator().cuda(), self.device_id)
        discriminator = torch.nn.DataParallel(Discriminator().cuda(), self.device_id)
        discriminator_uv_c = torch.nn.DataParallel(Discriminator().cuda(), self.device_id)
        discriminator_uv_p = torch.nn.DataParallel(DiscriminatorUV().cuda(), self.device_id)

        sampler_parameters = ContiguousParams(sampler.parameters())
        opt_sampler = torch.optim.Adam(sampler_parameters.contiguous(), lr=1e-5, betas=(0.5, 0.999))

        gen_params = ContiguousParams(generator.parameters())
        dis_params = ContiguousParams(discriminator.parameters())
        disuv_c_params = ContiguousParams(discriminator_uv_c.parameters())
        disuv_p_params = ContiguousParams(discriminator_uv_p.parameters())

        opt_gen = torch.optim.Adam(gen_params.contiguous(), lr=self.lr, betas=(0.5, 0.999))
        opt_dis = torch.optim.Adam(dis_params.contiguous(), lr=self.lr, betas=(0.5, 0.999))
        opt_disuv_c = torch.optim.Adam(disuv_c_params.contiguous(), lr=self.lr, betas=(0.5, 0.999))
        opt_disuv_p = torch.optim.Adam(disuv_p_params.contiguous(), lr=self.lr, betas=(0.5, 0.999))

        face_decoder = Face3D().cuda()
        if self.start_step != 0:
            state_dict = torch.load(self.ckpt_pth.format(self.start_step))
            generator.load_state_dict(state_dict['gen'])
            discriminator.load_state_dict(state_dict['dis'])
            sampler.load_state_dict(state_dict['sampelr'])
            discriminator_uv_c.load_state_dict(state_dict['dis_uv_c'])
            discriminator_uv_p.load_state_dict(state_dict['dis_uv_p'])
        else:
            state_dict = torch.load(self.ckpt_sampler_pth)
            sampler.load_state_dict(state_dict)

        return generator, discriminator, discriminator_uv_c, discriminator_uv_p, sampler, face_decoder, \
               opt_gen, opt_dis, opt_sampler, opt_disuv_c, opt_disuv_p

    @torch.no_grad()
    def get_3d_shape(self, coeff, mat_inverse):
        verts, _, _ = self.face_decoder(coeff, mat_inverse, src_sz=self.size_img)
        return verts

    @torch.no_grad()
    def sample_uvmap(self, rgb_img, verts, tri):
        out = self.renderer(verts, tri, self.size_img, colors=None)
        mask_eros, mask_verts = out['mask_eros'], out['mask_verts']
        verts_color = self.img2vert.nearest(verts, rgb_img * mask_eros).permute(0, 2, 1)
        verts_color = verts_color * mask_verts
        uv_map = self.uv_writer(verts_color, verts, cull=False)
        uv_map = torch.clamp(uv_map, 0.0, 1.0)
        return uv_map

    @torch.no_grad()
    def syn_profile(self, coeff, mat_inverse):
        angles = coeff[:, 224:227]
        pos_yaw_mask = angles[:, 1] > 0
        neg_yaw_mask = ~pos_yaw_mask
        pos_yaw = angles[pos_yaw_mask, 1]
        angles[pos_yaw_mask, 1] = math.pi / 6 + torch.rand_like(pos_yaw) * math.pi / 4
        neg_yaw = angles[neg_yaw_mask, 1]
        angles[neg_yaw_mask, 1] = -math.pi / 6 - torch.rand_like(neg_yaw) * math.pi / 4
        coeff[:, 224:227] = angles
        verts_profile = self.get_3d_shape(coeff, mat_inverse)
        return verts_profile, coeff[:, 225]

    @torch.no_grad()
    def syn_flip(self, coeff, mat_inverse):
        angles = coeff[:, 224:227]
        pos_yaw_mask = angles[:, 1] > 0
        big_yaw_mask = torch.abs(angles[:, 1]) > math.pi / 6.0
        extrem_yaw_mask = torch.abs(angles[:, 1]) > math.pi / 3.0
        angles[big_yaw_mask * (~extrem_yaw_mask), 1] *= -1
        big_yaw_shape = angles[big_yaw_mask * (~extrem_yaw_mask), 1]
        angles[big_yaw_mask * (~extrem_yaw_mask), 0] = torch.rand_like(big_yaw_shape) * math.pi / 12 - math.pi / 24
        angles[big_yaw_mask * (~extrem_yaw_mask), 2] = torch.rand_like(big_yaw_shape) * math.pi / 12 - math.pi / 24

        extrem_shape = angles[extrem_yaw_mask, 1]
        angles[extrem_yaw_mask, 1] = torch.rand_like(extrem_shape) * math.pi / 6 - math.pi / 12
        angles[extrem_yaw_mask, 0] = torch.rand_like(extrem_shape) * math.pi / 12 - math.pi / 24
        angles[extrem_yaw_mask, 2] = torch.rand_like(extrem_shape) * math.pi / 12 - math.pi / 24

        angles[(~big_yaw_mask) * pos_yaw_mask, 1] -= math.pi / 4
        angles[(~big_yaw_mask) * (~pos_yaw_mask), 1] += math.pi / 4

        coeff[:, 224:227] = angles
        verts_flip = self.get_3d_shape(coeff, mat_inverse)
        return verts_flip

    @torch.no_grad()
    def syn_data(self, coeff, mat_inverse, uv):
        colors = self.uv_reader(uv, bilinear=True).permute(0, 2, 1).contiguous()
        verts = self.syn_flip(coeff, mat_inverse)
        output = self.renderer(verts, self.tri_bch, size=self.size_img, colors=colors)
        syn_face = output['rgb']
        syn_mask = output['mask']
        bg_profile = next(self.bg_fetcher)
        syn = syn_face * syn_mask + bg_profile * (1 - syn_mask)
        return syn

    def split_uv(self, uv):
        uv_ctr = uv * self.uv_ctr_mask
        uv_l = uv[..., :self.size_uv // 2]
        uv_r = uv[..., self.size_uv // 2:]
        uv_sd = torch.cat((uv_l, uv_r), 0)
        return uv_ctr, uv_sd

    @torch.no_grad()
    def syn_front(self, coeff, mat_inverse):
        angles = coeff[:, 224:227]
        pos_yaw_mask = angles[:, 1] > 0
        neg_yaw_mask = ~pos_yaw_mask
        angles[pos_yaw_mask, 1] -= math.pi / 24
        angles[neg_yaw_mask, 1] += math.pi / 24
        coeff[:, 224:227] = angles
        verts_front = self.get_3d_shape(coeff, mat_inverse)
        return verts_front

    def noise_uvmap(self, uv_map):
        mask = uv_map == torch.zeros(1, 3, 1, 1).to(self.device)
        noise = ((torch.rand_like(uv_map) - 0.5) / 0.5) * mask
        uv_map_noise = uv_map + noise
        uv_map_flip = torch.flip(uv_map, dims=(3,))
        mask_flip = torch.flip(mask, dims=(3,))
        noise_flip = ((torch.rand_like(uv_map) - 0.5) / 0.5) * mask_flip
        uv_map_flip_noise = uv_map_flip + noise_flip
        uv_map_combine = torch.cat((uv_map_noise, uv_map_flip_noise), dim=1)
        return uv_map_combine

    def train_discriminator(self, dis_model, opt_dis, real, fake):
        real.requires_grad_()
        out_real = dis_model(real)
        loss_real = self.criterion.module.adv_loss(out_real, 1)
        loss_reg = self.criterion.module.r1_reg(out_real, real)
        out_fake = dis_model(fake.detach())
        loss_fake = self.criterion.module.adv_loss(out_fake, 0)
        loss_d = loss_real + loss_fake + loss_reg
        opt_dis.zero_grad()
        loss_d.backward()
        opt_dis.step()

    def train(self, step):
        img_init, seg, uv, coeff, mat_inverse = next(self.datafetcher)
        img = img_init * seg

        with torch.no_grad():
            verts = self.get_3d_shape(coeff, mat_inverse)

        # train real
        with torch.no_grad():
            grid, _ = self.sampler(img)
            uv_map_sample = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros',
                                          align_corners=True)
            uv_map_sample = uv_map_sample + ((uv_map_sample == 0) * torch.flip(uv_map_sample, dims=(3,)))

            if random.random() > 0.8:
                uv_mask = next(self.uv_mask_fetcher)
                uv_map_sample = uv_map_sample * (1 - uv_mask)

        uv_map_pred = torch.clamp(self.gen(uv_map_sample), 0., 1.)
        verts_colors_pred = self.uv_reader(uv_map_pred, bilinear=True).permute(0, 2, 1).contiguous()
        out = self.renderer(verts, self.tri_bch, size=self.size_img,
                            colors=verts_colors_pred)
        # train discriminator face
        with torch.no_grad():
            mask = out['mask']
            img_face = img * mask

        recon = out['rgb'] * seg
        self.train_discriminator(self.dis, self.opt_dis, img_face, recon)

        uv_ctr, uv_sd = self.split_uv(uv_map_pred)
        uv_ctr_gt = next(self.uv_ctr_fetcher)
        uv_sd_gt = next(self.uv_sd_fetcher)
        self.train_discriminator(self.dis_uv_c, self.opt_disuv_c, uv_ctr_gt, uv_ctr)
        self.train_discriminator(self.dis_uv_p, self.opt_disuv_p, uv_sd_gt, uv_sd)
        # train generator
        pred_fake = self.dis(recon)
        pred_fake_ctr = self.dis_uv_c(uv_ctr)
        pred_fake_sd = self.dis_uv_p(uv_sd)

        loss_adv = 0.01 * (
                self.criterion.module.adv_loss(pred_fake, 1) +
                self.criterion.module.adv_loss(pred_fake_ctr, 1) +
                self.criterion.module.adv_loss(pred_fake_sd, 1) * 0.5
        )

        loss_recon = 10 * F.l1_loss(recon, img_face)
        loss_id = 0.1 * self.criterion.module.id_loss(recon, img_face)
        loss_recon_uv = F.l1_loss(uv_map_pred, uv)
        loss_tv = 0.1 * self.criterion.module.tv(uv_map_pred)

        loss_sym = self.criterion.module.symmetric(uv_map_pred)
        loss = loss_recon + loss_recon_uv + loss_tv + loss_adv + loss_sym + loss_id
        self.opt_gen.zero_grad()
        loss.backward()
        self.opt_gen.step()

        # train syn
        syn = self.syn_data(coeff, mat_inverse, uv)
        with torch.no_grad():
            grid_s, _ = self.sampler(syn)
            uv_map_sample_s = F.grid_sample(syn, grid_s, mode='bilinear', padding_mode='zeros',
                                            align_corners=True)
            uv_map_sample_s = uv_map_sample_s + ((uv_map_sample_s == 0) * torch.flip(uv_map_sample_s, dims=(3,)))
            if random.random() > 0.8:
                uv_mask = next(self.uv_mask_fetcher)
                uv_map_sample_s = uv_map_sample_s * (1 - uv_mask)

        uv_map_pred_s = torch.clamp(self.gen(uv_map_sample_s), 0., 1.)
        uv_ctr_s, uv_sd_s = self.split_uv(uv_map_pred_s)

        uv_ctr_gt = next(self.uv_ctr_fetcher)
        uv_sd_gt = next(self.uv_sd_fetcher)
        self.train_discriminator(self.dis_uv_c, self.opt_disuv_c, uv_ctr_gt, uv_ctr_s)
        self.train_discriminator(self.dis_uv_p, self.opt_disuv_p, uv_sd_gt, uv_sd_s)

        pred_fake_ctr_s = self.dis_uv_c(uv_ctr_s)
        pred_fake_sd_s = self.dis_uv_p(uv_sd_s)

        loss_adv_s = 0.01 * (self.criterion.module.adv_loss(pred_fake_ctr_s, 1) +
                             self.criterion.module.adv_loss(pred_fake_sd_s, 1) * 0.5)

        loss_recon_uv_s = F.l1_loss(uv_map_pred_s, uv)
        loss_tv_s = 0.1 * self.criterion.module.tv(uv_map_pred_s)
        loss_sym_s = self.criterion.module.symmetric(uv_map_pred_s)
        loss_s = loss_recon_uv_s + loss_tv_s + loss_adv_s + loss_sym_s
        self.opt_gen.zero_grad()
        loss_s.backward()
        self.opt_gen.step()

        result = [img_init, recon, uv_map_sample, uv_map_pred, syn, uv_map_pred_s, uv]

        losses = [loss_recon.item(), loss_recon_uv.item(), loss_tv.item(), loss_sym.item(),
                  loss_id.item()]

        return result, losses

    def solve(self):
        pbar = tqdm.tqdm(range(self.start_step, self.n_steps))
        for step in pbar:
            result, losses = self.train(step)
            loss_recon, loss_recon_uv, loss_tv, loss_sym, loss_id = losses
            self.scheduler_dis.step()
            self.scheduler_disuv_c.step()
            self.scheduler_disuv_p.step()
            self.scheduler_gen.step()
            lr = self.scheduler_gen.get_last_lr()
            pbar.set_postfix({
                'recon_face': '{:.3f}'.format(loss_recon),
                'recon_uv': '{:.3f}'.format(loss_recon_uv),
                'tv': '{:.3f}'.format(loss_tv),
                'sym': '{:.3f}'.format(loss_sym),
                'id': '{:.3f}'.format(loss_id),
            })
            if (step + 1) % self.test_step == 0:
                result = list(map(tensor2img, result))
                show = np.concatenate(result, axis=0)
                cv2.imwrite(self.result_pth.format(step + 1), show[:, :, ::-1])

            if (step + 1) % self.save_step == 0:
                state_dict = {'gen': self.gen.state_dict(),
                              'dis': self.dis.state_dict(),
                              'sampelr': self.sampler.state_dict(),
                              'dis_uv_c': self.dis_uv_c.state_dict(),
                              'dis_uv_p': self.dis_uv_p.state_dict()}
                torch.save(state_dict, self.ckpt_pth.format(step + 1))

            if (step + 1) % 50 == 0:
                msg = 'step: {} recon_uv: {} recon_face: {} ' \
                      'tv: {} sym: {} id: {} lr: {}'.format(step + 1,
                                                            loss_recon_uv,
                                                            loss_recon,
                                                            loss_tv,
                                                            loss_sym,
                                                            loss_id,
                                                            lr[0])
                self.logger.info(msg)


if __name__ == '__main__':
    solver = GenSolver()
    solver.solve()
