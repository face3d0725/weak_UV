from pytorch3d.structures import Meshes
from pytorch3d.renderer import rasterize_meshes, PointsRenderer
from pytorch3d.ops import interpolate_face_attributes
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np
import math as m
from abc import ABC
from .light_position import light_point, light_direction
from utils.seg_util import tensor_erode, tensor_dilate


class SimpleRenderer(nn.Module, ABC):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = torch.device(device)

    @staticmethod
    def rasterize(verts, faces, image_size=128):
        with torch.no_grad():
            rot_mat = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32).expand(verts.shape[0], 3,
                                                                                                    3).to(verts.device)
            trans = torch.zeros_like(verts)
            trans[:, :, 2] = 2

        verts = torch.bmm(verts, rot_mat)
        verts = verts + trans
        meshes = Meshes(verts, faces)
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes,
            image_size,
            blur_radius=0,
            faces_per_pixel=1,
            clip_barycentric_coords=False,
            cull_backfaces=False
        )
        return pix_to_face, bary_coords, meshes.faces_packed(), zbuf

    @staticmethod
    def compute_normal(verts_packed, faces_packed):
        faces_verts = verts_packed[faces_packed]
        verts_normals = torch.zeros_like(verts_packed)
        # face_normals = torch.cross(faces_verts[:, 1] - faces_verts[:, 0],
        #                            faces_verts[:, 2] - faces_verts[:, 0], dim=1) # when use microsoft tri

        face_normals = torch.cross(faces_verts[:, 2] - faces_verts[:, 0],
                                   faces_verts[:, 1] - faces_verts[:, 0], dim=1)  # when use tddfa tri

        for idx in range(3):
            verts_normals = verts_normals.index_add(0, faces_packed[:, idx], face_normals)

        verts_normals_packed = torch.nn.functional.normalize(verts_normals, eps=1e-6, dim=1)
        return verts_normals_packed

    def lighting(self, verts, faces_packed, pix_to_face, bary_coords, point=False, random=0.7, specular=False):
        verts_packed = verts.view(-1, 3)
        verts_normals_packed = self.compute_normal(verts_packed, faces_packed)
        faces_normals = verts_normals_packed[faces_packed]
        pixel_normals = interpolate_face_attributes(pix_to_face, bary_coords, faces_normals)
        pixel_coords = None
        if point:
            light_loc = light_point(random=random, device=self.device)
            faces_verts = verts_packed[faces_packed]
            pixel_coords = interpolate_face_attributes(pix_to_face, bary_coords, faces_verts)  # to calc light direction
            direction = light_loc - pixel_coords
        else:
            direction = light_direction(random=random, device=self.device)

        direction = F.normalize(direction, p=2, dim=-1, eps=1e-6)

        # color of the light
        color = torch.tensor([1.0, 1.0, 1.0]).to(self.device)

        normals = -F.normalize(pixel_normals, p=2, dim=-1, eps=1e-6)

        # diffuse
        cos_angle = torch.sum(normals * direction, dim=-1)
        diffuse = color * F.relu(cos_angle)[..., None]

        # specular
        if specular:
            camera_position = torch.tensor([[0., 0., -1000.]], dtype=torch.float32).to(self.device)
            mask = (cos_angle > 0).type(torch.float32)
            reflect_direction = -direction + 2 * (cos_angle[..., None] * normals)
            if pixel_coords is None:
                faces_verts = verts_packed[faces_packed]
                pixel_coords = interpolate_face_attributes(pix_to_face, bary_coords, faces_verts)

            view_direction = camera_position - pixel_coords
            view_direction = F.normalize(view_direction, p=2, dim=-1, eps=1e-6)
            alpha = F.relu(torch.sum(view_direction * reflect_direction, dim=-1)) * mask
            specular = color * torch.pow(alpha, 1e3)[..., None]  # shininess of the material:5
            return diffuse, specular, normals, verts_normals_packed

        else:
            return diffuse, 0, normals, verts_normals_packed

    @staticmethod
    @torch.no_grad()
    def visible_face_mask(verts, faces_packed, pix_to_face, bary_coords, size, mask_edge=None):
        verts_packed = verts.view(-1, 3)
        faces_verts = verts_packed[faces_packed]
        pixel_coords = interpolate_face_attributes(pix_to_face, bary_coords, faces_verts)  # # 3x128x128x1x3
        depth = pixel_coords.squeeze(3)[..., 2]  # 3x128x128x1
        if mask_edge is not None:
            # edge_left = mask_edge[...,0:128]
            # edge_right = mask_edge[...,128:]
            # sum_l = edge_left.sum(dim=(1,2,3))
            # sum_r = edge_right.sum(dim=(1,2,3))
            # bin = (sum_r-sum_l)>100
            # edge_left[bin] *= 10
            # edge_right[~bin] *= -10
            # mask_edge = torch.cat((edge_left, edge_right),dim=3)
            depth += -10 * mask_edge.squeeze(1)

        x = (verts[..., 0] * 0.5 + 0.5) * (size - 1)
        y = (1 - (verts[..., 1] * 0.5 + 0.5)) * (size - 1)
        outlier = (x < 0) + (x > (size - 1)) + (y < 0) + (y > (size - 1))
        x = torch.clamp(torch.round(x), 0, size - 1)
        y = torch.clamp(torch.round(y), 0, size - 1)
        vert_depth = verts[..., 2]
        idx = (size * y + x).type(torch.long)
        depth = depth.view(-1, size * size)
        vert_buffer = depth.gather(1, idx)
        mask = (vert_depth >= (vert_buffer - 0.03))
        mask = mask * ~outlier
        # mask_packed = mask.view(-1, 1)
        # face_mask = mask_packed[faces_packed]
        # face_mask = face_mask.sum(1)>0
        return mask.type(torch.float32)

    def mask_erosion(self, batch_mask, mask_kernel):  # bz*H*W
        bz = batch_mask.shape[0]
        masks = torch.chunk(batch_mask, bz)
        erode = lambda m: torch.from_numpy(
            cv2.erode(m.detach().squeeze().cpu().numpy().astype('uint8'), mask_kernel, iterations=1)).type(
            torch.float32)
        masks = list(map(erode, masks))
        batch_mask_ero = torch.stack(masks, dim=0).to(self.device)
        return batch_mask_ero

    @staticmethod
    def illumination(face_texture, norm_r, gamma):
        n_data = gamma.shape[0]
        n_point = norm_r.shape[1]
        device = face_texture.device
        gamma = torch.reshape(gamma, (-1, 3, 9)).permute(0, 2, 1)
        init_lit = torch.zeros_like(gamma)
        init_lit[:, 0] = 0.8
        gamma = gamma + init_lit

        a0 = m.pi
        a1 = 2 * m.pi / m.sqrt(3.0)
        a2 = 2 * m.pi / m.sqrt(8.0)
        c0 = 1 / m.sqrt(4 * m.pi)
        c1 = m.sqrt(3.0) / m.sqrt(4 * m.pi)
        c2 = 3 * m.sqrt(5.0) / m.sqrt(12 * m.pi)

        Y = torch.cat((a0 * c0 * torch.ones(n_data, n_point, 1).to(device),
                       -a1 * c1 * norm_r[:, :, 1:2],
                       a1 * c1 * norm_r[:, :, 2:3],
                       -a1 * c1 * norm_r[:, :, 0:1],
                       a2 * c2 * norm_r[:, :, 0:1] * norm_r[:, :, 1:2],
                       -a2 * c2 * norm_r[:, :, 1:2] * norm_r[:, :, 2:3],
                       a2 * c2 * 0.5 / m.sqrt(3.0) * (3 * norm_r[:, :, 2:3] ** 2 - 1),
                       -a2 * c2 * norm_r[:, :, 0:1] * norm_r[:, :, 2:3],
                       a2 * c2 * 0.5 * (norm_r[:, :, 0:1] ** 2 - norm_r[:, :, 1:2] ** 2)
                       ), dim=2).to(device)

        color = torch.bmm(Y, gamma)
        face_color = face_texture * color
        return face_color

    @staticmethod
    def forward_color(colors, faces_packed, pix_to_face, bary_coords):
        colors_packed = colors.reshape(-1, colors.shape[2])
        faces_colors = colors_packed[faces_packed]
        pixel_colors = interpolate_face_attributes(pix_to_face, bary_coords, faces_colors)
        return pixel_colors

    def get_normal(self, verts, faces_packed, pix_to_face, bary_coords):
        verts_packed = verts.view(-1, 3)
        verts_normals_packed = self.compute_normal(verts_packed, faces_packed)
        faces_normals = verts_normals_packed[faces_packed]
        pixel_normals = interpolate_face_attributes(pix_to_face, bary_coords, faces_normals)
        pixel_normals = pixel_normals.squeeze(3).permute(0, 3, 1, 2).contiguous()
        pixel_normals = F.normalize(pixel_normals, dim=1)
        return pixel_normals

    @torch.no_grad()
    def get_mask(self, pix_to_face):
        bz = pix_to_face.shape[0]
        face_idx = torch.arange(1, bz + 1) * 70903  # 70903 is ttl number of tri with mouth
        face_idx_no_mouth = (face_idx - 114).view(bz, 1, 1, 1).to(pix_to_face.device)  # 114 mouth triangles
        mask = (pix_to_face > -1) * (pix_to_face < face_idx_no_mouth)
        mask = mask.type(torch.float32)
        return mask.permute(0, 3, 1, 2)

    def forward(self, verts, faces, size=256, colors=None, gamma=None):
        pix_to_face, bary_coords, faces_packed, zbuf = self.rasterize(verts, faces, size)
        diffuse, specular, normals, verts_normals_packed = self.lighting(verts, faces_packed, pix_to_face, bary_coords)
        verts_normals = verts_normals_packed.view(*verts.shape)
        if gamma is not None:
            colors = self.illumination(colors, verts_normals, gamma)

        light = torch.clamp(diffuse, 0.0, 1.0)
        light = light.squeeze(3).permute(0, 3, 1, 2)
        # mask = (zbuf > 0).permute(0, 3, 1, 2).type(torch.float32)
        mask = self.get_mask(pix_to_face)
        mask_edge = mask - tensor_erode(mask, 15)
        # pixel_normals = self.get_normal(verts, faces_packed, pix_to_face, bary_coords)
        # mask_edge = torch.abs(pixel_normals[:,0:1,...])>0.95
        vert_mask = self.visible_face_mask(verts, faces_packed, pix_to_face, bary_coords, size, mask_edge=mask_edge)

        out = {'rgb': None, 'mask': None, 'vert_mask': vert_mask, 'light': light, 'colors': colors}

        if colors is not None:
            if isinstance(colors, list):
                rgb_list = []
                for color in colors:
                    rgb = self.forward_color(color, faces_packed, pix_to_face, bary_coords)
                    rgb = rgb.squeeze(dim=3).permute(0, 3, 1, 2).contiguous()
                    rgb = torch.clamp(rgb, 0.0, 1.0)
                    rgb_list.append(rgb)
                out['rgb'] = rgb_list
            else:
                rgb = self.forward_color(colors, faces_packed, pix_to_face, bary_coords)
                rgb = rgb.squeeze(dim=3).permute(0, 3, 1, 2).contiguous()
                rgb = torch.clamp(rgb, 0.0, 1.0)
                out['rgb'] = rgb

        out['mask'] = mask
        return out
