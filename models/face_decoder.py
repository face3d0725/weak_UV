from models.bfm import BfmExtend
import torch
from torch import nn
from utils.simple_renderer import SimpleRenderer


class Face3D(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device
        self.facemodel = BfmExtend().to(device)

    def Split_coeff(self, coeff):
        shp_coeff = coeff[:, :144]  # id_coeff :80, ex_coeff 80:144
        tex_coeff = coeff[:, 144:224]
        angles = coeff[:, 224:227]
        gamma = coeff[:, 227:254]
        translation = coeff[:, 254:257]

        return shp_coeff, tex_coeff, angles, translation, gamma

    def Compute_rotation_matrix(self, angles):
        N = angles.shape[0]
        device = angles.device
        x = angles[:, 0]
        y = angles[:, 1]
        z = angles[:, 2]
        cx, cy, cz = torch.cos(x), torch.cos(y), torch.cos(z)
        sx, sy, sz = torch.sin(x), torch.sin(y), torch.sin(z)
        rotation = torch.zeros(N, 3, 3).to(device)
        rotation[:, 0, 0] = cz * cy
        rotation[:, 0, 1] = sx * sy * cz - cx * sz
        rotation[:, 0, 2] = cx * sy * cz + sx * sz
        rotation[:, 1, 0] = cy * sz
        rotation[:, 1, 1] = sx * sy * sz + cx * cz
        rotation[:, 1, 2] = cx * sy * sz - sx * cz
        rotation[:, 2, 0] = -sy
        rotation[:, 2, 1] = sx * cy
        rotation[:, 2, 2] = cx * cy
        rotation = torch.transpose(rotation, 1, 2)
        return rotation

    def Rigid_transform_block(self, face_shape, rotation, translation):
        face_shape_r = torch.bmm(face_shape, rotation)
        face_shape_t = face_shape_r + translation.unsqueeze(1)
        return face_shape_t

    def Orthogonal_projection_block(self, face_shape, focal=1015.0):
        # the reconstructed coordinates are from -112 to 112
        div = torch.ones_like(face_shape)  # *10
        div[:, :, 0] = 10 - face_shape[:, :, 2]
        div[:, :, 1] = 10 - face_shape[:, :, 2]
        div[:, :, 2] = 10
        return face_shape * focal / div

    def forward(self, coeff, mat_inverse, src_sz=1024):
        # mat_reverse = None
        shp_coeff, tex_coeff, angles, translation, gamma = self.Split_coeff(coeff)
        shape, tex = self.facemodel(shp_coeff, params_tex=tex_coeff) # tex_coeff
        rotation = self.Compute_rotation_matrix(angles)
        verts = self.Rigid_transform_block(shape, rotation, translation)
        verts = self.Orthogonal_projection_block(verts)

        # to image coordinates
        bz = mat_inverse.shape[0]
        mat_img = torch.Tensor([[1, 0, 0, 112],
                                [0, -1, 0, 112],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]]).to(self.device).repeat(bz, 1, 1)

        mat_inverse = torch.bmm(mat_inverse, mat_img)
        verts = torch.bmm(torch.cat((verts, torch.ones_like(verts)[:, :, 0:1]), dim=2), mat_inverse.transpose(1,2))

        mat_ndc = torch.Tensor([[2/src_sz, 0, 0, -1],
                                [0, -2/src_sz, 0, 1-2/src_sz],
                                [0, 0, 2/src_sz, 0]]).to(self.device).repeat(bz, 1, 1)

        verts = torch.bmm(verts, mat_ndc.transpose(1, 2))
        return verts, tex, gamma



