from scipy.io import loadmat
import torch
import torch.nn as nn
import pickle
import pathlib
import os


class BfmExtend(nn.Module):
    def __init__(self, n_shp=80, n_exp=64, n_tex=80):
        super(BfmExtend, self).__init__()
        self.shape = nn.Linear(n_shp + n_exp, 107127)
        self.tex = nn.Linear(n_tex, 107127)
        self.load_weight()

    def load_weight(self):
        pth = pathlib.Path(__file__).parent.parent.absolute()
        weight_pth = os.path.join(pth, 'BFM/BFM_model_front_uv_stretch.mat')
        bfm = loadmat(weight_pth)
        meanshape = torch.from_numpy(bfm['meanshape']).squeeze()
        mean_ctr = meanshape.view(-1, 3).mean(dim=0).view(1, 1, 3)

        idBase = torch.from_numpy(bfm['idBase'])
        exBase = torch.from_numpy(bfm['exBase'])
        meantex = torch.from_numpy(bfm['meantex']).squeeze() / 255.
        texBase = torch.from_numpy(bfm['texBase']) / 255.
        # 1
        tri = torch.from_numpy(bfm['tri']).type(torch.long) - 1
        tri = torch.mm(tri, torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))

        # 2
        # with open(os.path.join(pth, 'BFM/microsoft_tri_mouth.pkl'), 'rb') as f:
        #     tri = pickle.load(f)
        # tri = torch.from_numpy(tri)

        uv = torch.from_numpy(bfm['uv_stretch']).type(torch.float32).unsqueeze(0)
        keypoints = torch.from_numpy(bfm['keypoints'][0].astype('int64')) - 1
        w = torch.cat((idBase, exBase), dim=1).type(torch.float32)
        skin_mask = torch.from_numpy(bfm['skinmask']).type(torch.float32).unsqueeze(-1)

        self.shape.weight.data = w
        self.shape.bias.data = meanshape
        self.tex.weight.data = texBase
        self.tex.bias.data = meantex
        self.register_buffer('tri', tri)
        self.register_buffer('uv', uv)
        self.register_buffer('mean_ctr', mean_ctr)
        self.register_buffer('keypoints', keypoints)
        self.register_buffer('skin_mask', skin_mask)

    def forward(self, params_shp, params_tex=None):
        bz = params_shp.shape[0]
        shape = self.shape(params_shp)
        shape = shape.view(bz, -1, 3)
        shape = shape - self.mean_ctr
        tex = None
        if params_tex is not None:
            tex = self.tex(params_tex)
            tex = tex.view(bz, -1, 3)

        return shape, tex


if __name__ == '__main__':
    from utils.simple_renderer import SimpleRenderer
    import cv2
    from utils.cv import tensor2img

    device = 'cuda:0'

    bfm = BfmExtend().to(device)
    param_shp = torch.rand(10, 80 + 64).to(device)
    param_tex = torch.rand(10, 80).to(device)
    out = bfm(param_shp, param_tex)
    print(out[0].shape)
    print(out[1].shape)

    renderer = SimpleRenderer(device).to(device)
    shape = bfm.shape.bias
    tex = bfm.tex.bias
    tri = bfm.tri.unsqueeze(0)

    shape = shape.view(-1, 3).unsqueeze(0)
    tex = tex.view(-1, 3).unsqueeze(0)

    out = renderer(shape, tri, size=512, colors=tex, gamma=None, eros=7)

    I = out['rgb']

    show = tensor2img(I)
    cv2.imshow('show', show[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
