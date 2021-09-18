from torchvision import transforms as TF
import torch.utils.data as Data
import os
import pickle
import random
from PIL import Image
import numpy as np
import cv2

uvmask_root = '/media/xn/SSD1T/uv_mask/'
background_root = '/media/xn/1TDisk/MIT_indoor_subset'


class UVMask(Data.Dataset):
    def __init__(self):
        self.root = uvmask_root
        self.name_list = os.listdir(self.root)
        self.transform = TF.Compose([TF.RandomHorizontalFlip(), TF.ToTensor()])

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        mask = cv2.imread(os.path.join(self.root, name), 0)
        mask = self.transform(mask)
        return mask


class BackGround(Data.Dataset):
    def __init__(self):
        self.root = background_root
        self.image_list = os.listdir(background_root)
        self.transform = TF.ToTensor()
        self.sz_img = 256

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        name = self.image_list[idx]
        img = cv2.imread(os.path.join(self.root, name))
        H, W, _ = img.shape
        if min(H, W) < 256:
            img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_NEAREST)
        H, W, _ = img.shape
        x, y = random.randint(0, H - self.sz_img), random.randint(0, W - self.sz_img)
        img = img[x:x + self.sz_img, y:y + self.sz_img, :]
        img = cv2.GaussianBlur(img, (11, 11), 0)[:, :, ::-1].copy()
        img = Image.fromarray(img)
        img = self.transform(img)
        return img


class UV(Data.Dataset):
    def __init__(self, side='front'):
        self.root = '/media/xn/SSD1T/ffhq/'
        self.side = side
        self.image_list = self.get_image_list()
        self.transform = TF.Compose([TF.RandomHorizontalFlip(), TF.ToTensor()])

    def get_image_list(self):
        if self.side == 'front':
            self.root = os.path.join(self.root, 'uv_front')
        elif self.side == 'side':
            self.root = os.path.join(self.root, 'uv_side')
        else:
            raise NotImplementedError('only support uv_front and uv_side')
        image_list = os.listdir(self.root)

        return image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        im_name = self.image_list[item]
        pth = os.path.join(self.root, im_name)
        with open(pth, 'rb') as f:
            uv = pickle.load(f)
        uv = Image.fromarray(uv)
        uv = self.transform(uv)
        return uv


class Fetcher(object):
    def __init__(self, name='bg', batch_size=4, device='cuda:0'):
        if name == 'bg':
            dataset = BackGround()
        elif name == 'uvmask':
            dataset = UVMask()
        elif name == 'uvfront':
            dataset = UV('front')
        elif name == 'uvside':
            dataset = UV('side')
        else:
            raise NotImplementedError('Dataset not implemented')
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.iter = self.get_iter()

    def get_iter(self):
        loader = Data.DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                 shuffle=True, num_workers=4,
                                 pin_memory=True, drop_last=True)

        return iter(loader)

    def __next__(self):
        try:
            out = next(self.iter)
        except StopIteration:
            self.iter = self.get_iter()
            out = next(self.iter)

        return out.to(self.device)


if __name__ == '__main__':
    from utils.cv import tensor2img

    fetcher = Fetcher('uvside')
    for idx in range(20):
        data = next(fetcher)
        data = tensor2img(data)
        cv2.imshow('data', data[:,:,::-1])
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
