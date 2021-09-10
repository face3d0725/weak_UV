import torch.utils.data as Data
import torchvision.transforms as TF
import os
import cv2
import random

background_pth = '/media/xn/1TDisk/MIT_indoor_subset'


class BackGround(Data.Dataset):
    def __init__(self):
        self.root = background_pth
        self.image_list = os.listdir(background_pth)
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
        img = self.transform(img)
        return img


class BGFetcher(object):
    def __init__(self, batch_size=4, device='cuda:0'):
        self.device = device
        self.dataset = BackGround()
        self.batch_size = batch_size
        self.iter = self.get_iter()

    def get_iter(self):
        loader = Data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                 drop_last=True)
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

    fetcher = BGFetcher()
    for _ in range(100):
        out = next(fetcher)
        out = tensor2img(out)
        cv2.imshow('background', out)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
