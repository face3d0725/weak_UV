from abc import ABC, abstractmethod
from torchvision import transforms as TF
import torch.utils.data as Data
from dataset.transform import RandomAffine
import os
import pickle
import random
import numpy as np
import cv2
import pathlib
from utils.microsoft_align import get_inverse_mat

imgPath = {'ffceleba': ['/media/xn/SSD1T/ffhq/img256_mic',
                        '/media/xn/SSD1T/CelebAMask-HQ/img256_mic'],
           'illinois': ['/media/xn/SSD1T/illinois/front_img256_mic',
                        '/media/xn/SSD1T/illinois/side_img256_mic']}

coeffPath = {'ffceleba': ['/media/xn/SSD1T/ffhq/microsoft_coeff',
                          '/media/xn/SSD1T/CelebAMask-HQ/microsoft_coeff'],
             'illinois': ['/media/xn/SSD1T/illinois/front_microsoft_coeff',
                          '/media/xn/SSD1T/illinois/side_microsoft_coeff']}

ldmkPath = {'ffceleba': ['/media/xn/SSD1T/ffhq/ldmk_256_mic',
                         '/media/xn/SSD1T/CelebAMask-HQ/ldmk_256_mic'],
            'illinois': ['/media/xn/SSD1T/illinois/front_ldmk256_mic',
                         '/media/xn/SSD1T/illinois/side_ldmk256_mic']}

uvPath = {'ffceleba': ['/media/xn/SSD1T/ffhq/uv_merge',
                       '/media/xn/SSD1T/CelebAMask-HQ/uv_merge'],
          'illinois': ['/media/xn/SSD1T/illinois/uv_merge',
                       '/media/xn/SSD1T/illinois/uv_merge']}  # illinois has two same uv root

maskPath = {'ffceleba': ['/media/xn/SSD1T/ffhq/mask256_mic_slim',
                         '/media/xn/SSD1T/CelebAMask-HQ/mask256_mic_slim'],
            'illinois': ['/media/xn/SSD1T/illinois/front_mask256_mic_slim']}


class BaseDataset(Data.Dataset, ABC):
    def __init__(self, name, mode='train'):
        self.img_root = imgPath[name]
        self.coeff_root = coeffPath[name]
        self.ldmk_root = ldmkPath[name]
        self.mask_root = maskPath[name]
        self.uv_root = uvPath[name]
        self.mode = mode

        self.image_list = self.get_image_list()
        self.transform = RandomAffine(0.1, 0, 0.0)
        self.to_tensor = TF.ToTensor()

    def __len__(self):
        return len(self.image_list)

    @abstractmethod
    def get_image_list(self):
        pass

    def get_data(self, data_id, name):
        img_pth = os.path.join(self.img_root[data_id], name)
        img = cv2.imread(img_pth)[:, :, ::-1].copy()
        coeff_name = name.split('.')[0] + '.pkl'
        coeff_pth = os.path.join(self.coeff_root[data_id], coeff_name)
        with open(coeff_pth, 'rb') as f:
            coeff = pickle.load(f).squeeze()

        ldmk_pth = os.path.join(self.ldmk_root[data_id], coeff_name)
        with open(ldmk_pth, 'rb') as f:
            ldmk = pickle.load(f)

        mat_inverse = get_inverse_mat(ldmk, src_sz=256, microsoft_sz=224)

        mask_pth = os.path.join(self.mask_root[data_id], coeff_name)
        with open(mask_pth, 'rb') as f:
            mask = pickle.load(f).astype('float32')

        uv_pth = os.path.join(self.uv_root[data_id], coeff_name)
        with open(uv_pth, 'rb') as f:
            uv = pickle.load(f)

        # flip = 0
        # if self.mode == 'train':
        #     data = {'img': img, 'uv': uv, 'mat_inv': mat_inverse, 'mask': mask}
        #     data = self.transform(data)
        #     img = data['img']
        #     uv = data['uv']
        #     mat_inverse = data['mat_inv']
        #     mask = data['mask']
        #     flip = data['flip']

        img = self.to_tensor(img)
        mask = self.to_tensor(mask)
        uv = self.to_tensor(uv)
        return img, mask, uv, coeff, mat_inverse

    @abstractmethod
    def __getitem__(self, idx):
        pass


class FFCelebA(BaseDataset):
    def __init__(self, mode='train'):
        super().__init__('ffceleba', mode)

    def get_image_list(self):
        imgs = os.listdir(self.img_root[0]) + os.listdir(self.img_root[1])
        pth = pathlib.Path(__file__).parent.parent.absolute()
        pth = os.path.join(pth, 'dataset/ffceleba_glass.pkl')
        with open(pth, 'rb') as f:
            glasses = pickle.load(f)

        return list(set(imgs) - set(glasses))

    def __getitem__(self, idx):
        name = self.image_list[idx]
        if name.endswith('png'):
            id = 0
        else:
            id = 1

        img, mask, uv, coeff, mat_inverse = self.get_data(id, name)
        return img, mask, uv, coeff, mat_inverse


class Illinois(BaseDataset):
    def __init__(self, mode='train'):
        super().__init__('illinois', mode)

    def get_image_list(self):
        images = os.listdir(self.uv_root[0])
        images = [im.split('.')[0] + '.jpg' for im in images]
        pth = pathlib.Path(__file__).parent.parent.absolute()
        pth = os.path.join(pth, 'dataset/illinois_glass.pkl')
        with open(pth, 'rb') as f:
            glasses = pickle.load(f)

        return list(set(images) - set(glasses))

    def __getitem__(self, idx):
        name = self.image_list[idx]
        img, mask, uv, coeff, mat_inverse = self.get_data(0, name)
        return img, mask, uv, coeff, mat_inverse


class InputFetcher(object):
    def __init__(self, name='ffceleba', batch_size=8, device='cuda:0'):
        self.device = device
        self.name = name
        if name == 'ffceleba':
            dataset = FFCelebA()
        elif name == 'illinois':
            dataset = Illinois()
        else:
            raise ValueError('Dataset not implemented')
        self.dataset = dataset
        self.batch_size = batch_size
        self.iter = self.get_iter()

    def get_iter(self):
        loader = Data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True,
                                 num_workers=4, drop_last=True)
        return iter(loader)

    def __next__(self):
        try:
            out = next(self.iter)
        except StopIteration:
            self.iter = self.get_iter()
            out = next(self.iter)

        out = [o.to(self.device) for o in out]
        return out[0], out[1], out[2], out[3], out[4]


if __name__ == '__main__':
    from utils.cv import tensor2img
    from utils.visualization import vis_parsing_maps
    from torchvision.utils import make_grid
    import numpy as np
    from models.face_decoder import Face3D
    from utils.simple_renderer import SimpleRenderer
    import torch
    import matplotlib.pyplot as plt

    bz = 4
    fetcher = InputFetcher(name='illinois', batch_size=bz)
    face_decoder = Face3D(device='cuda:0').cuda()
    tri_bch = face_decoder.facemodel.tri.unsqueeze(0).repeat(bz, 1, 1)

    renderer = SimpleRenderer('cuda:0').cuda()

    for idx in range(100):
        img, mask, uv, coeff, mat_inverse = next(fetcher)
        verts, tex, gamma = face_decoder(coeff, mat_inverse, src_sz=256)
        # out = renderer(verts, tri_bch, size=256, colors=tex, gamma=gamma, eros=7)
        # bfm_rgb = out['rgb']
        # bfm_rgb_show = tensor2img(bfm_rgb)

        img_show = tensor2img(img)
        # merge = (bfm_rgb_show * 0.5 + img_show * 0.5).astype('uint8')
        uv_show = tensor2img(uv)
        face_mask = (mask > 0) * (mask < 11) * (~(mask == 4)) * (~(mask == 8))
        face = img * face_mask
        face_show = tensor2img(face)
        mask = make_grid(mask, nrow=8, padding=0)
        mask = mask.permute(1, 2, 0).detach().cpu().numpy()
        parse_show = vis_parsing_maps(img_show, mask, 1)

        show = np.concatenate((img_show, face_show, parse_show), axis=0)
        cv2.imshow('show', show[:, :, ::-1])
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
