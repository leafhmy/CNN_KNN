from __future__ import print_function, division
import torch
from torchvision import datasets, models, transforms
import os
from PIL import Image
# import cv2
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
# from albumentations import (Compose, Resize, OneOf, RandomBrightness, RandomContrast, MotionBlur,
#                             MedianBlur, GaussianBlur, VerticalFlip, HorizontalFlip, ShiftScaleRotate,
#                             Normalize, RandomBrightnessContrast, RandomResizedCrop, CenterCrop)
# from albumentations.pytorch import ToTensorV2


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def save_file_path_v2(args, labeled_ratio=0.2, test_ratio=0.2):
    """
    labeled_ratio: class with minimun samples * ratio
    """
    fpath_x = []
    labels_x = []
    fpath_u = []
    labels_u = []

    data_num = []
    all_data = {}
    real_cls_labels = []
    root = args.data_dir
    for index, d in enumerate(os.listdir(root)):
        fd = os.path.join(root, d)
        real_cls_labels.append(str(index) + ' ' + d)
        data_num.append(len(os.listdir(fd)))
        label = index
        for i in os.listdir(fd):
            fp = os.path.join(fd, i)
            all_data.setdefault(label, []).append(fp)

    labeled_num = int(min(data_num)*labeled_ratio)
    for label, path_list in all_data.items():
        for path in path_list[:labeled_num]:
            fpath_x.append(path)
            labels_x.append(label)

    for label, path_list in all_data.items():
        for path in path_list[labeled_num:]:
            fpath_u.append(path)
            labels_u.append(label)

    with open(args.anno + 'real_cls_labels' + '.txt', 'w') as f:
        for item in real_cls_labels:
            f.write(item + '\n')

    with open(args.anno + 'labeled' + '.txt', 'w') as f:
        for fn, l in zip(fpath_x, labels_x):
            f.write('{} {}\n'.format(fn, l))

    with open(args.anno + 'unlabeled' + '.txt', 'w') as f:
        for fn, l in zip(fpath_u, labels_u):
            f.write('{} {}\n'.format(fn, l))

    print('[INFO ] labeled: {} unlabeled: {}'.format(len(fpath_x), len(fpath_u)))

    x_train, x_val, y_train, y_val = train_test_split(fpath_x, labels_x, random_state=999, test_size=test_ratio)
    print('[INFO ] x_train: {} x_val: {}'.format(len(x_train), len(x_val)))

    with open(args.anno + 'train.txt', 'w')as f:
        for fn, l in zip(x_train, y_train):
            f.write('{} {}\n'.format(fn, l))

    with open(args.anno + 'val.txt', 'w')as f:
        for fn, l in zip(x_val, y_val):
            f.write('{} {}\n'.format(fn, l))


def save_file_path(args, split=False, test_size=0.2, phase=None):
    if phase is not None:
        root = args.train_dir if phase == 'train' else args.val_dir
    else:
        root = args.data_dir
    if phase == 'ordered':
        root = args.data_dir
    fpath = []
    labels = []
    real_cls_labels = []
    for index, d in enumerate(os.listdir(root)):
        fd = os.path.join(root, d)
        real_cls_labels.append(str(index)+' '+d)
        label = index
        for i in os.listdir(fd):  # 每类白细胞取80张图像[:80]
            fp = os.path.join(fd, i)
            fpath.append(fp)
            labels.append(label)

    if phase == 'ordered':
        with open(args.anno + 'ordered' + '.txt', 'w') as f:
            for fn, l in zip(fpath, labels):
                f.write('{} {}\n'.format(fn, l))

    with open(args.anno+'real_cls_labels'+'.txt', 'w') as f:
        for item in real_cls_labels:
            f.write(item+'\n')
    if phase is not None:
        print('{}:{}, {}'.format(phase, len(fpath), len(labels)))

        with open(args.anno + phase + '.txt', 'w')as f:
            for fn, l in zip(fpath, labels):
                f.write('{} {}\n'.format(fn, l))
    if split:
        x_train, x_val, y_train, y_val = train_test_split(fpath, labels, random_state=999, test_size=test_size)
        print(len(x_train), len(x_val))

        with open(args.anno + 'train.txt', 'w')as f:
            for fn, l in zip(x_train, y_train):
                f.write('{} {}\n'.format(fn, l))

        with open(args.anno + 'val.txt', 'w')as f:
            for fn, l in zip(x_val, y_val):
                f.write('{} {}\n'.format(fn, l))


# class AlbumentationsLoader:
#     def __init__(self, args):
#         self.args = args
#         self.data_transforms = {
#             'train': Compose([
#                               Resize(height=224, width=224),
#                               Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                               ToTensorV2(),
#                              ]),
#             'val': Compose([
#                             Resize(height=224, width=224),
#                             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                             ToTensorV2()
#                             ])}
#         self.__get_label_dict()
#
#     def __get_label_dict(self):
#         with open(self.args.anno+'real_cls_labels'+'.txt', 'r') as f:
#             index_label = f.readlines()
#             label = {int(l.split(' ')[0]): l.split(' ')[1] for l in index_label}
#         self.label_dict = label
#
#     def get_data_loader(self):
#         image_datasets = {'train': DataLoaderX(
#                                    AlbumentationsDatasetLoader(self.args.anno,
#                                    'train',
#                                    self.data_transforms['train']),
#                                    batch_size=self.args.batch_size,
#                                    num_workers=self.args.num_workers,
#                                    shuffle=True),
#                          'val': DataLoaderX(
#                                    AlbumentationsDatasetLoader(self.args.anno,
#                                    'val',
#                                    self.data_transforms['val']),
#                                    batch_size=self.args.val_batch_size,
#                                    num_workers=self.args.num_workers,
#                                    shuffle=False),
#                         'label_dict': self.label_dict}
#         return image_datasets
#
#
# class AlbumentationsDatasetLoader:
#     def __init__(self, root_dir, phase, data_transforms):
#         self.root_dir = root_dir
#         self.data_transforms = data_transforms
#         self.data = []
#         with open(self.root_dir+phase+'.txt', 'r') as f:
#             for item in f.readlines():
#                 img, label = item.split(' ')
#                 self.data.append((img, label.strip()))
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, item):
#         img_path, label = self.data[item]
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         if self.data_transforms is not None:
#             img = self.data_transforms(image=img)  # !!!一定要加image=
#         return img['image'], int(label), img_path


class Loader:
    def __init__(self, args):
        self.args = args
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            'val': transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        self.__get_label_dict()

    def __get_label_dict(self):
        with open(self.args.anno + 'real_cls_labels' + '.txt', 'r') as f:
            index_label = f.readlines()
            label = {int(l.split(' ')[0]): l.split(' ')[1] for l in index_label}
        self.label_dict = label

    def get_data_loader(self):
        image_datasets = {'train': DataLoaderX(
                                   DatasetLoader(self.args.anno,
                                   'train',
                                   self.data_transforms['train']),
                                   batch_size=self.args.batch_size,
                                   num_workers=self.args.num_workers,

                                   shuffle=True),
                         'val': DataLoaderX(DatasetLoader(self.args.anno,
                                   'val',
                                   self.data_transforms['val']),
                                   batch_size=self.args.val_batch_size,
                                   num_workers=self.args.num_workers,
                                   shuffle=False),
                        'labeled': DataLoaderX(DatasetLoader(self.args.anno,
                                   'labeled',
                                   self.data_transforms['train']),
                                   batch_size=self.args.batch_size,
                                   num_workers=self.args.num_workers,
                                   shuffle=False),
                        'unlabeled': DataLoaderX(DatasetLoader(self.args.anno,
                                   'unlabeled',
                                   self.data_transforms['train']),
                                   batch_size=self.args.batch_size,
                                   num_workers=self.args.num_workers,
                                   shuffle=True),
                        'label_dict': self.label_dict}
        return image_datasets


class DatasetLoader:
    def __init__(self, root_dir, phase, data_transforms):
        self.root_dir = root_dir
        self.data_transforms = data_transforms
        self.data = []
        with open(self.root_dir+phase+'.txt', 'r') as f:
            for item in f.readlines():
                img, label = item.split(' ')
                self.data.append((img, label.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path, label = self.data[item]
        img = Image.open(img_path).convert('RGB')
        if self.data_transforms is not None:
            img = self.data_transforms(img)
        return img, int(label), img_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='D:\PythonProjects\DLCV_P1\pyimagesearch\datasets\\flower_photos/')
    parser.add_argument('--anno', type=str,
                        default='../anno/flowers/')
    args = parser.parse_args()

    save_file_path(args, split=True)
