from models.sampler import Sampler
from models.face_decoder import Face3D
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
import numpy as np
import cv2
import math
from tensorboardX import SummaryWriter
import tqdm
from contiguous_params import ContiguousParams

import torch.backends.cudnn as cudnn

cudnn.benchmark = True


class SamplerSolver(object):
    def __init__(self):
        result_folder = 'res_sampler'
        ckpt_folder = 'ckpt_sampler'
        self.result_pth = mkdir(result_folder, format='it_{:05}.png')
        self.ckpt_pth = mkdir(ckpt_folder, format='it_{:05}.pkl')
        self.device = 'cuda:0'
        self.bz = 4
        self.size_uv = 256
        self.size_img = 256
        self.lr = 1e-4
        self.start_step = 0
        self.syn_start_step = 0  # 100000
        self.n_steps = 100000
        self.test_step = 500
        self.save_step = 10000
        self.device_id = [0, 1]

        self.sampler, self.face_decoder, self.optimizer = self.init_model()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100000],
                                                              gamma=0.1)
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

        self.bgfetcher = Fetcher(name='bg',batch_size=self.bz, device=self.device)

        self.criterion = torch.nn.DataParallel(Loss(device=self.device).cuda(), device_ids=self.device_id)
        self.seg_criterion = OhemBCELoss(thresh=0.7, n_min=256 ** 2)
        self.logger = setup_logger('./log_sampler')
        self.writer = SummaryWriter()

    def init_model(self):
        sampler = torch.nn.DataParallel(Sampler().cuda(), self.device_id)
        parameters = ContiguousParams(sampler.parameters())
        opt_sampler = torch.optim.Adam(parameters.contiguous(), lr=self.lr, betas=(0.5, 0.999))

        face_decoder = Face3D().cuda()
        if self.start_step != 0:
            state_dict = torch.load(self.ckpt_pth.format(self.start_step))
            sampler.load_state_dict(state_dict)

        return sampler, face_decoder, opt_sampler

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
    def syn_front(self, coeff, mat_inverse):
        angles = coeff[:, 224:227]
        pos_yaw_mask = angles[:, 1] > 0
        neg_yaw_mask = ~pos_yaw_mask
        angles[pos_yaw_mask, 1] -= math.pi / 24
        angles[neg_yaw_mask, 1] += math.pi / 24
        coeff[:, 224:227] = angles
        verts_profile = self.get_3d_shape(coeff, mat_inverse)
        return verts_profile

    def train_sampler(self, step):
        img, mask, uv, coeff, mat_inverse = next(self.datafetcher)

        with torch.no_grad():
            verts = self.get_3d_shape(coeff, mat_inverse)
            out = self.renderer(verts, self.tri_bch, size=256, colors=None)
            verts_mask = out['vert_mask'].unsqueeze(2)
            face_mask_model = out['mask']

            yaw = coeff[:, 225]
            extrem_left_yaw = yaw < -math.pi / 4
            extrem_right_yaw = yaw > math.pi / 4
            large_yaw = torch.abs(yaw) > math.pi / 12
            if large_yaw.any():
                front_vert = self.syn_front(coeff, mat_inverse)
                verts_mask[large_yaw] *= self.img2vert.bilinear(front_vert, face_mask_model).permute(0, 2, 1)[large_yaw]
            verts_mask[~large_yaw] *= self.img2vert.bilinear(verts, face_mask_model).permute(0, 2, 1)[~large_yaw]
            grid_mask = self.uv_writer(torch.cat((verts[..., :2], verts_mask), dim=2), verts, cull=True)
            grid_mask[:, 1, ...] *= -1
            grid_gt, uv_mask_gt = torch.split(grid_mask, (2, 1), dim=1)
            uv_mask_gt = torch.round(uv_mask_gt) * self.uv_template_mask
            grid_gt = grid_gt * uv_mask_gt + (1 - uv_mask_gt) * torch.ones_like(grid_gt) * (-1.1)

            grid_gt[extrem_left_yaw, :, 50:, :self.size_uv // 2 - 20] = -1.1
            grid_gt[extrem_right_yaw, :, 50:, self.size_uv // 2 + 20:] = -1.1

            uv_gt = F.grid_sample(img * mask, grid_gt.permute(0, 2, 3, 1), 'nearest', 'zeros', True)

        # train real
        grid, seg_pred = self.sampler(img)
        uv_map_pred = F.grid_sample(img * (mask > 0), grid, mode='nearest', padding_mode='zeros',
                                    align_corners=True)

        grid = grid.permute(0, 3, 1, 2).contiguous()
        loss_tv_grid = self.criterion.module.tv(grid)
        loss_recon_grid = F.l1_loss(grid, grid_gt)
        loss_recon_uv = F.l1_loss(uv_map_pred, uv_gt)
        loss_tv = self.criterion.module.tv(uv_map_pred)
        loss_seg = self.seg_criterion(seg_pred, mask)
        loss = loss_recon_uv + loss_recon_grid + loss_seg + 0.1 * loss_tv + 0.1 * loss_tv_grid
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar('real/uv', loss_recon_uv.item(), step)
        self.writer.add_scalar('real/tv', loss_tv.item(), step)
        self.writer.add_scalar('real/tv_grid', loss_tv_grid.item(), step)

        if (step + 1) > self.syn_start_step:
            with torch.no_grad():
                # synthesiz profile image and corresponding uv maps
                verts_s, yaw = self.syn_profile(coeff, mat_inverse)
                extrem_left_yaw = yaw < -math.pi / 4
                extrem_right_yaw = yaw > math.pi / 4

                verts_color_s = self.uv_reader(uv).permute(0, 2, 1).contiguous()
                out_s = self.renderer(verts_s, self.tri_bch, size=self.size_img, colors=verts_color_s)
                face_mask_s = out_s['mask']
                verts_mask_s = self.img2vert.bilinear(self.syn_front(coeff, mat_inverse), face_mask_s).permute(0, 2, 1)
                verts_mask_s = out_s['vert_mask'].unsqueeze(2) * verts_mask_s

                grid_mask_s = self.uv_writer(torch.cat((verts_s[..., :2], verts_mask_s), dim=2), verts_s, cull=True)
                grid_mask_s[:, 1, ...] *= -1
                grid_gt_s, uv_mask_gt_s = torch.split(grid_mask_s, (2, 1), dim=1)
                uv_mask_gt_s = torch.round(uv_mask_gt_s) * self.uv_template_mask
                grid_gt_s = grid_gt_s * uv_mask_gt_s + (1 - uv_mask_gt_s) * torch.ones_like(grid_gt_s) * (-1.1)
                grid_gt_s[extrem_left_yaw, :, 50:, :self.size_uv // 2 - 20] = -1.1
                grid_gt_s[extrem_right_yaw, :, 50:, self.size_uv // 2 + 20:] = -1.1
                uv_gt_s = F.grid_sample(out_s['rgb'], grid_gt_s.permute(0, 2, 3, 1), 'nearest', 'zeros', True)
                bg_profile = next(self.bgfetcher)
                profile_s = out_s['rgb'] * face_mask_s + bg_profile * (1 - face_mask_s)

            # train syn profile
            grid_s, _ = self.sampler(profile_s)
            uv_map_pred_s = F.grid_sample(profile_s, grid_s, 'nearest', 'zeros', True)
            grid_s = grid_s.permute(0, 3, 1, 2).contiguous()
            loss_tv_grid_s = self.criterion.module.tv(grid_s)
            loss_recon_grid_s = F.l1_loss(grid_s, grid_gt_s)
            loss_recon_uv_s = F.l1_loss(uv_map_pred_s, uv_gt_s)
            loss_tv_s = self.criterion.module.tv(uv_map_pred_s)
            loss_s = loss_recon_uv_s + loss_recon_grid_s + 0.1 * loss_tv_s + 0.1 * loss_tv_grid_s
            self.optimizer.zero_grad()
            loss_s.backward()
            self.optimizer.step()
            self.writer.add_scalar('syn/uv', loss_recon_uv_s.item(), step)
            self.writer.add_scalar('syn/tv', loss_tv_s.item(), step)
            self.writer.add_scalar('syn/tv_grid', loss_tv_grid.item(), step)

        results = []

        if (step + 1) % self.test_step == 0:
            with torch.no_grad():
                grid, seg_pred = self.sampler(img)
                face_pred = img*(seg_pred>0)
                uv_map_pred = F.grid_sample(face_pred, grid, 'nearest', 'zeros', True)
                verts_color_pred = self.uv_reader(uv_map_pred).permute(0, 2, 1).contiguous()
                out = self.renderer(verts, self.tri_bch, size=self.size_img, colors=verts_color_pred)
                rgb_pred = out['rgb']
                results = [img, face_pred, rgb_pred, uv_map_pred, uv_gt]

                if (step + 1) > self.syn_start_step:
                    grid_s, _ = self.sampler(profile_s)
                    uv_map_pred_s = F.grid_sample(profile_s, grid_s, 'nearest', 'zeros', True)
                    verts_color_pred_s = self.uv_reader(uv_map_pred_s).permute(0, 2, 1).contiguous()
                    out_s = self.renderer(verts_s, self.tri_bch, size=self.size_img, colors=verts_color_pred_s)
                    rgb_pred_s = out_s['rgb']
                    results.extend([profile_s, rgb_pred_s, uv_map_pred_s, uv_gt_s])

        return results, [loss_recon_uv.item(), loss_recon_grid.item(), loss_tv.item(),
                         loss_tv_grid.item()]

    def solve_sampler(self):
        pbar = tqdm.tqdm(range(self.start_step, self.n_steps))

        for step in pbar:
            results, loss = self.train_sampler(step)
            loss_recon_uv, loss_recon_grid, loss_tv, loss_tv_grid = loss
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()
            pbar.set_postfix({
                'recon_uv_loss': '{:.3f}'.format(loss_recon_uv),
                'recon_grid_loss': '{:.3f}'.format(loss_recon_grid),
                'tv_loss': '{:.3f}'.format(loss_tv),
                'tv_grid_loss': '{:.3f}'.format(loss_tv_grid),
                'lr': '{}'.format(lr)
            })
            if (step + 1) % self.test_step == 0:
                results = list(map(tensor2img, results))
                show = np.concatenate(results, axis=0)
                cv2.imwrite(self.result_pth.format(step + 1), show[:, :, ::-1])

            if (step + 1) % self.save_step == 0:
                state_dict = self.sampler.state_dict()
                torch.save(state_dict, self.ckpt_pth.format(step + 1))

            if (step + 1) % 10 == 0:
                msg = 'step: {} loss_recon_uv: {:.3f} loss_tv: {:.3f} ' \
                      'loss_tv_grid: {:.3f} lr: {} '.format(step + 1, loss_recon_uv,
                                                            loss_tv, loss_tv_grid,
                                                            lr[0])
                self.logger.info(msg)


if __name__ == '__main__':
    solver = SamplerSolver()
    solver.solve_sampler()
