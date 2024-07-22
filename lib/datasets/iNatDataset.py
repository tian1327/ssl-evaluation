import torch
import torch.utils.data as data
import numpy as np
import os
from torchvision.datasets import folder as dataset_parser
import json

def make_dataset(dataset_root, split, task='All', retrieved=None, pl_list=None):
    if retrieved is not None and 'l_train' in split:
        dataset_root = '/'+ os.path.join(*dataset_root.split('/')[:4])
        print(f"Root path for split file {split} is {dataset_root}")
    split_file_path = os.path.join('data', task, split+'.txt')
    print(f"Reading split file from {split_file_path}")

    with open(split_file_path, 'r') as f:
        img = f.readlines()

    if task == 'semi_fungi':
        img = [x.strip('\n').rsplit('.JPG ') for x in img]
    # elif task[:9] == 'semi_aves':
    else:
        img = [x.strip('\n').rsplit() for x in img]

    ## Use PL + l_train
    if pl_list is not None:
        if task == 'semi_fungi':
            pl_list = [x.strip('\n').rsplit('.JPG ') for x in pl_list]
        # elif task[:9] == 'semi_aves':
        else:
            pl_list = [x.strip('\n').rsplit() for x in pl_list]
        img += pl_list

    flag = True
    for idx, x in enumerate(img):
        if task == 'semi_fungi':
            img[idx][0] = os.path.join(dataset_root, x[0] + '.JPG')
        else:
            original_path_part = x[0].lstrip('/')
            constructed_path = os.path.join(dataset_root, original_path_part)
            img[idx][0] = constructed_path
        img[idx][1] = int(x[1])
        if not os.path.exists(img[idx][0]):
            print(f"     Warning: File {img[idx][0]} does not exist")
            flag = False
    if flag == True:
        print(f"All good for split {split}")


    classes = [x[1] for x in img]
    num_classes = len(set(classes)) 
    print('# images in {}: {}'.format(split,len(img)))
    return img, num_classes


class iNatDataset(data.Dataset):
    def __init__(self, dataset_root, split, task='All', transform=None, retrieved=None, 
            loader=dataset_parser.default_loader, pl_list=None, return_name=False):
        self.loader = loader
        self.dataset_root = dataset_root
        self.task = task

        self.imgs, self.num_classes = make_dataset(self.dataset_root, 
                    split, self.task, pl_list=pl_list, retrieved=retrieved)

        self.transform = transform

        self.return_name = return_name

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # print("Loading image from path:", path)
        img_original = self.loader(path)

        if self.transform is not None:
            img = self.transform(img_original)
        else:
            img = img_original.copy()

        if self.return_name:
            return img, target, path
        else:
            return img, target

    def __len__(self):
        return len(self.imgs)

    def get_num_classes(self):
        return self.num_classes
