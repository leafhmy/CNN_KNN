import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection


class WBCDataset:
    def __init__(self, args, simclr=False):
        self.root = args.data_dir
        self.n_views = args.n_views
        self.anno = './anno/'
        if not os.path.exists(self.anno):
            os.mkdir(self.anno)
        self._save_file_path()
        if simclr:
            self.trans = ContrastiveLearningViewGenerator(self._get_simclr_pipeline_transform(224), self.n_views)
        else:
            self.trans = transforms.Compose([transforms.RandomResizedCrop(size=224),
                                              transforms.ToTensor()])

    def _save_file_path(self):
        fpath = []
        labels = []

        for index, d in enumerate(os.listdir(self.root)):
            fd = os.path.join(self.root, d)
            for i in os.listdir(fd):
                fp = os.path.join(fd, i)
                fpath.append(fp)
                labels.append(index)

        with open(self.anno + 'train.txt', 'w')as f:
            for fn, l in zip(fpath, labels):
                f.write('{} {}\n'.format(fn, l))


    def getDataLoader(self):
        train_loader = DatasetLoader(self.anno, 'train', self.trans)
        return train_loader


    def _get_simclr_pipeline_transform(self, size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms


class DatasetLoader:
    def __init__(self, anno_dir, phase, data_transforms):
        self.anno = anno_dir
        self.data_transforms = data_transforms
        self.data = []
        with open(self.anno+phase+'.txt', 'r') as f:
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
        return img, int(label)

class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
