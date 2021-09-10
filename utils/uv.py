import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import rasterize_meshes
from torch import nn


class Attr2Uv(nn.Module):
    def __init__(self, uv_coords, faces, batch_size=8, size=128):
        super().__init__()
        self.size = size
        self.uv_coords = uv_coords
        pix_to_face, bary_coords, faces_packed = self.rasterize_uv(uv_coords, faces, batch_size)
        self.pix_to_face = pix_to_face
        self.bary_coords = bary_coords
        self.faces_packed = faces_packed

    def process_uv(self, uv_coords):
        bz, n_vert, _ = uv_coords.shape  # data is in batch form
        uv_coords = (uv_coords - uv_coords.min(dim=1).values) / (
                uv_coords.max(dim=1).values - uv_coords.min(dim=1).values)
        uv_coords = (uv_coords - 0.5) * 2
        uv_coords[:, :, 0] *= -1  # prevent lr-flip of uv map
        ones = torch.ones(bz, n_vert, 1).type(torch.float32).to(uv_coords.device)
        uv_coords = torch.cat((uv_coords, ones), dim=2)
        return uv_coords

    def rasterize_uv(self, uv_coords, faces, bz):
        uv_coords = self.process_uv(uv_coords)
        uv_coords = uv_coords.repeat(bz, 1, 1)
        faces = faces.repeat(bz, 1, 1)
        meshes = Meshes(uv_coords, faces)
        pix_to_face, z_buf, bary_coords, dist = rasterize_meshes(meshes,
                                                                 self.size,
                                                                 blur_radius=0,
                                                                 faces_per_pixel=1,
                                                                 clip_barycentric_coords=False)
        faces_packed = meshes.faces_packed()
        return pix_to_face, bary_coords, faces_packed

    def backface_culling(self, faces_verts, faces_attrs):
        v0, v1, v2 = faces_verts.unbind(dim=1)
        areas = (v0[:, 0] - v1[:, 0]) * (v2[:, 1] - v1[:, 1]) - (v0[:, 1] - v1[:, 1]) * (v2[:, 0] - v1[:, 0])
        mask = areas <= 0
        faces_attrs[mask] = torch.zeros_like(faces_attrs[0])
        return faces_attrs

    def forward(self, vert_attr, vert, cull=True):
        c = vert_attr.shape[2]
        vert_attr_packed = vert_attr.reshape(-1, c)
        faces_attrs = vert_attr_packed[self.faces_packed]
        if cull:
            vert_packed = vert.reshape(-1, 3)
            faces_verts = vert_packed[self.faces_packed]
            faces_attrs = self.backface_culling(faces_verts, faces_attrs)

        uv_tex = interpolate_face_attributes(self.pix_to_face, self.bary_coords, faces_attrs)
        return uv_tex.squeeze(dim=3).permute(0, 3, 1, 2).contiguous()


class Uv2Attr(nn.Module):
    def __init__(self, uv_coords, faces_packed=None, size=128):
        super().__init__()
        # uv_coords 1*nvert*2
        uv_coords = (uv_coords - uv_coords.min(dim=1).values) / (
                uv_coords.max(dim=1).values - uv_coords.min(dim=1).values)
        x, y = torch.chunk(uv_coords, 2, dim=2)
        x = x * (size - 1)
        y = size - 1 - y * (size - 1)
        self.x = x.squeeze(2)
        self.y = y.squeeze(2)
        self.faces_packed = faces_packed

    @staticmethod
    def index_vert_attr(x, y, uvmap):
        B, C, H, W = uvmap.shape
        x = torch.clamp(x, 0, W - 1)
        y = torch.clamp(y, 0, H - 1)
        idx = (W * y + x).type(torch.long)
        uvmap = uvmap.view(-1, C, H * W)
        idx = idx.unsqueeze(1).repeat(B, C, 1)
        vert_attr = uvmap.gather(2, idx)
        return vert_attr

    def nearest(self, uvmap):
        # not work. too much points overlap, show nothing
        x = torch.round(self.x)
        y = torch.round(self.y)
        vert_attr = self.index_vert_attr(x, y, uvmap)
        return vert_attr

    def bilinear(self, uvmap):
        x = self.x
        y = self.y
        x0 = torch.floor(x)
        x1 = x0 + 1
        y0 = torch.floor(y)
        y1 = y0 + 1
        ia = self.index_vert_attr(x0, y0, uvmap)  # 5x3x38365
        ib = self.index_vert_attr(x0, y1, uvmap)
        ic = self.index_vert_attr(x1, y0, uvmap)
        id = self.index_vert_attr(x1, y1, uvmap)

        wa = (x1 - x) * (y1 - y)  # 1x38365
        wb = (x1 - x) * (y - y0)

        wc = (x - x0) * (y1 - y)

        wd = (x - x0) * (y - y0)

        vert_attr = wa.unsqueeze(1).repeat(1, 3, 1) * ia + wb.unsqueeze(1).repeat(1, 3, 1) * ib \
                    + wc.unsqueeze(1).repeat(1, 3, 1) * ic + wd.unsqueeze(1).repeat(1, 3, 1) * id
        return vert_attr

    def get_edge_tri_idx(self, vert_attr):
        vert_packed = vert_attr.permute(0, 2, 1).reshape(-1, 3)
        faces_verts = vert_packed[self.faces_packed]
        # background vert: [0.0, 0.0, 0.0]
        mask = (faces_verts == torch.zeros_like(faces_verts[0, 0])).all(dim=2).any(dim=1)
        return mask

    def forward(self, batch_uv, bilinear=False):
        if bilinear:
            vert_attr = self.bilinear(batch_uv)
        else:
            vert_attr = self.nearest(batch_uv)
        # mask = self.get_edge_tri_idx(vert_attr)
        return vert_attr  # , mask


class ImgVertAttr(object):
    @staticmethod
    def index_vert_attr(x, y, batch_img):
        B, C, H, W = batch_img.shape
        outlier = (x < 0) + (x > (W - 1)) + (y < 0) + (y > (H - 1))
        outlier = outlier.unsqueeze(1).repeat(1, C, 1)
        x = torch.clamp(x, 0, W - 1)
        y = torch.clamp(y, 0, H - 1)
        idx = (W * y + x).type(torch.long)
        batch_img = batch_img.view(-1, C, H * W)
        idx = idx.unsqueeze(1).repeat(1, C, 1)
        vert_attr = batch_img.gather(2, idx)
        vert_attr[outlier] = 0

        # black = -torch.ones(1,3,1).to(batch_img.device)*127.5/128
        # edge_mask = vert_attr==black
        # vert_attr[edge_mask] = 0
        return vert_attr

    def nearest(self, verts, batch_img):
        B, C, H, W = batch_img.shape
        x = (verts[..., 0] * 0.5 + 0.5) * (W - 1)
        y = (1 - (verts[..., 1] * 0.5 + 0.5)) * (H - 1)
        x = torch.round(x)
        y = torch.round(y)
        vert_attr = self.index_vert_attr(x, y, batch_img)
        return vert_attr

    def bilinear(self, verts, batch_img):
        B, C, H, W = batch_img.shape
        x = (verts[..., 0] * 0.5 + 0.5) * (W - 1)
        y = (1 - (verts[..., 1] * 0.5 + 0.5)) * (H - 1)
        x0 = torch.floor(x)
        x1 = x0 + 1
        y0 = torch.floor(y)
        y1 = y0 + 1
        ia = self.index_vert_attr(x0, y0, batch_img)  # 5x3x38365
        ib = self.index_vert_attr(x0, y1, batch_img)
        ic = self.index_vert_attr(x1, y0, batch_img)
        id = self.index_vert_attr(x1, y1, batch_img)

        wa = (x1 - x) * (y1 - y)  # 1x38365
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)
        vert_attr = wa.unsqueeze(1).repeat(1, C, 1) * ia + wb.unsqueeze(1).repeat(1, C, 1) * ib \
                    + wc.unsqueeze(1).repeat(1, C, 1) * ic + wd.unsqueeze(1).repeat(1, C, 1) * id
        return vert_attr


if __name__ == '__main__':
    from models.bfm_layer import BFM
    from utils.generate_face_texture import FaceTexture
    from utils.simple_renderer import SimpleRenderer
    import cv2
    import numpy as np


    def normalize_verts(verts):
        min_xyz = verts.min(dim=1)[0]
        max_xyz = verts.max(dim=1)[0]
        diff = (max_xyz - min_xyz).squeeze()
        size = max(diff)
        verts = (verts - min_xyz) / size
        ctr = verts.mean(dim=1)
        verts = 2 * (verts - ctr)
        verts[:, :, 2] -= 1.
        return verts


    @torch.no_grad()
    def rotate(verts, angles):
        angles = list(map(np.deg2rad, angles))
        sx, sy, sz = map(lambda t: np.sin(t), angles)
        cx, cy, cz = map(lambda t: np.cos(t), angles)
        R = np.array([[cy * cz, cy * sz, -sy],
                      [-cx * sz + sx * sy * cz, cx * cz + sx * sy * sz, sx * cy],
                      [sx * sz + cx * sy * cz, -sx * cz + cx * sy * sz, cx * cy]])
        R = torch.from_numpy(R.T).type(torch.float32).to(verts.device)

        rotated_verts = verts @ R
        return rotated_verts


    device = 'cuda:1'

    bfm = BFM().to(device)
    texture = FaceTexture(device)
    tex = texture(bz=1, random=False)
    verts = bfm.bfm.bias.data.reshape(-1, 3).unsqueeze(0)
    faces = bfm.tri.unsqueeze(0)
    uv_coords = bfm.uv.unsqueeze(0)
    renderer = SimpleRenderer(device)

    uv_renderer = Attr2Uv(uv_coords, faces, batch_size=1, size=128)
    uv_reader = Uv2Attr(uv_coords, size=128)
    verts = normalize_verts(verts)
    angles = [0, 45, 0]
    verts = rotate(verts, angles)
    verts = verts - verts.mean(1, keepdim=True)
    sz = 128

    output = renderer(verts, faces, size=sz, colors=tex)
    out_mask = output['mask']
    show_img = output['rgb'].detach().squeeze().permute(1, 2, 0).cpu().numpy()
    show_img = (show_img * 255).astype('uint8').copy()

    uv_map = uv_renderer(tex, verts, cull=True)
    show_uv = uv_map.detach().squeeze().permute(1, 2, 0).cpu().numpy()
    show_uv = (show_uv * 255).astype('uint8').copy()

    uv_map_dpth = uv_renderer(tex * out_mask, verts, cull=False)
    show_uv_dpth = uv_map_dpth.detach().squeeze().permute(1, 2, 0).cpu().numpy()
    show_uv_dpth = (show_uv_dpth * 255).astype('uint8').copy()

    uv_map_combine = uv_renderer(tex * out_mask, verts, cull=True)
    show_uv_combine = uv_map_combine.detach().squeeze().permute(1, 2, 0).cpu().numpy()
    show_uv_combine = (show_uv_combine * 255).astype('uint8').copy()

    angles = [0, 0, 0]
    while True:
        cv2.imshow('image', show_img)
        # cv2.imshow('uv_cull', out_uv)
        # cv2.imshow('uv_dpth', out_uv_dpth)
        cv2.imshow('uv_combine', show_uv_combine)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('j'):
            angles[1] += 1
        elif key == ord('l'):
            angles[1] -= 1
        elif key == ord('i'):
            angles[0] += 1

        elif key == ord('k'):
            angles[0] -= 1

        elif key == ord('u'):
            angles[2] -= 1

        elif key == ord('o'):
            angles[2] += 1

        if not angles == [0, 0, 0]:
            verts = rotate(verts, angles)
            verts = verts - verts.mean(1, keepdims=True)
            # output = renderer(verts, faces, size=sz, colors=tex)
            output = renderer(verts, faces, size=sz, colors=verts)
            out_img = output['rgb']
            out_mask = output['mask']
            show_img = out_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
            show_img = (show_img * 255).astype('uint8').copy()

            # uv_map = uv_renderer(tex, verts, cull=True)
            # show_uv = uv_map.detach().squeeze().permute(1, 2, 0).cpu().numpy()
            # show_uv = (show_uv * 255).astype('uint8').copy()
            #
            # uv_map_dpth = uv_renderer(tex * out_mask, verts, cull=False)
            # show_uv_dpth = uv_map_dpth.detach().squeeze().permute(1, 2, 0).cpu().numpy()
            # show_uv_dpth = (show_uv_dpth * 255).astype('uint8').copy()

            # uv_map_combine = uv_renderer(tex * out_mask, verts, cull=True)
            uv_map_combine = uv_renderer(verts * out_mask, verts, cull=True)
            show_uv_combine = uv_map_combine.detach().squeeze().permute(1, 2, 0).cpu().numpy()
            show_uv_combine = (show_uv_combine * 255).astype('uint8').copy()

            tex_from_uv = uv_reader(uv_map_combine).permute(0, 2, 1).contiguous()
            output = renderer(verts, faces, size=sz, colors=tex_from_uv)
            show_img_recon = output['rgb'].detach().cpu().squeeze().permute(1, 2, 0).numpy()
            show_img_recon = (show_img_recon * 255).astype('uint8').copy()
            cv2.imshow('img_recon', show_img_recon)

            angles = [0, 0, 0]

    cv2.destroyAllWindows()

    print('END')
