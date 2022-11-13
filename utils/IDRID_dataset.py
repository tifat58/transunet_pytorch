import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Additional Scripts
from config import cfg


class IDRIDDataset(Dataset):
    output_size = cfg.transunet.img_dim
    # output_size = 224

    def __init__(self, path, transform, lesion_type, train=True):
        super().__init__()

        self.transform = transform

        # img_folder = os.path.join(path, 'img')
        # mask_folder = os.path.join(path, 'mask')

        img_folder = os.path.join(path, 'images')
        mask_folder = os.path.join(path, 'masks', lesion_type)


        self.img_paths = []
        self.mask_paths = []
        for p in os.listdir(img_folder):
            name = p.split('.')[0]

            self.img_paths.append(os.path.join(img_folder, name + '.jpg'))
            self.mask_paths.append(os.path.join(mask_folder, name + '_' + lesion_type + '.tif'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.img_paths[idx]
        mask = self.mask_paths[idx]

        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.output_size, self.output_size))

        mask = cv2.imread(mask, 0)
        # print(mask.shape)
        mask = cv2.resize(mask, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=-1)

        sample = {'img': img, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        img, mask = sample['img'], sample['mask']

        img = img / 255.
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.astype('float32'))

        mask = mask / 255.
        mask = mask.transpose((2, 0, 1))
        mask = torch.from_numpy(mask.astype('float32'))

        return {'img': img, 'mask': mask}

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    import torchvision.transforms as transforms
    import transforms as T

    transform = None
    # transform = T.Compose([T.BGR2RGB(),
    #                                 # T.Rescale(cfg.input_size),
    #                                 T.Rescale(224),
    #                                 T.RandomAugmentation(2),
    #                                 T.Normalize(),
    #                                 T.ToTensor()])

    # md = DentalDataset('/home/kara/Downloads/UFBA_UESC_DENTAL_IMAGES_DEEP/dataset_and_code/test/set/train', transform)
    md = IDRIDDataset('/Users/hasan/Myprojects/DFKI/DR_Grading_Projects/data/IDRID_segmentation/Segmentation',
                      transform, lesion_type='EX')


    for sample in md:
        print(sample['img'].shape)
        print(sample['mask'].shape)
        '''cv2.imshow('img', sample['img'])
        cv2.imshow('mask', sample['mask'])
        cv2.waitKey()'''

        break
