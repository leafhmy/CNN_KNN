import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import math
import numpy as np
import time
import argparse
import os
import copy
from net.resnet import resnet18
from torchvision import models
from data_aug.contrastive_learning_dataset import WBCDataset
from models.resnet_simclr import ResNetSimCLR
# from load_data.data_loader import Loader
# from utils.ema import EMA
from matplotlib import pyplot as plt
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample, distance_metric, type_metric

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'


def distance(feature, center):
    """
    feature: tensor shape (batch_size, 512)
    center tensor shape (5, 512)
    return : (5, batch_size) each value denotes a distance
    """
    def cosine_sim(x, y):
        """
        余弦相似度
        相似性范围从-1到1
        -1意味着两个向量指向的方向正好截然相反，1表示它们的指向是完全相同的，0通常表示它们之间是独立的
        而在这之间的值则表示中间的相似性或相异性。
        """
        d = torch.cosine_similarity(x, y)
        return (-1)*(d - 1)

    def Jensen_Shannon(x, y):
        M = (x + y) / 2
        # 方法一：根据公式求解
        d = 0.5 * torch.sum(x * torch.log(x / M)) + 0.5 * torch.sum(y * torch.log(y / M))
        return 1 / d

    distances = torch.empty(size=(center.shape[0], feature.shape[0]), requires_grad=True).cuda()
    for i in range(center.shape[0]):
        for j in range(feature.shape[0]):
            distances[i, j] = cosine_sim(feature[j:j+1, :], center[i:i+1, :])
            # distances[i, j] = Jensen_Shannon(feature[j:j + 1, :], center[i:i + 1, :])

    return distances


class ExpLoss(nn.Module):
    def __init__(self):
        super(ExpLoss, self).__init__()

    def forward(self, x):
        min_distances = torch.min(x, dim=0).values  # (1, batch_size)
        loss = torch.sum(torch.exp(min_distances) - 1) / x.shape[1]
        return loss


class CNNCluster:
    def __init__(self, model, loader, num_cls, feature_dim=128, threshold=1.00):
        self.model = model
        self.loader = loader
        self.num_cls = num_cls
        self.dim = feature_dim
        self.threshold = threshold
        self.cluster_center_ = self._initialize_center()

    def _cosine_sim(self, x, y):
        y = y.unsqueeze(dim=0)  # (1, 128)
        x = torch.from_numpy(x).cuda()
        x = x.reshape(y.shape)
        d = torch.cosine_similarity(x, y)
        return (-1)*(d - 1)


    def _labels_to_original(self, labels, forclusterlist):
        """
        https://blog.csdn.net/weixin_43483381/article/details/90183968
        # labels为聚类结果的标签值
        # forclusterlist为聚类所使用的样本集
        # 函数的功能是将forclusterlist中的样本集按照labels中的标签值重新排序，得到按照类簇排列好的输出结果
        return [[cls0], [cls1], ...]
        """
        assert len(labels) == len(forclusterlist)
        maxlabel = max(labels)
        numberlabel = [i for i in range(0, maxlabel + 1, 1)]
        numberlabel.append(-1)
        result = [[] for i in range(len(numberlabel))]
        for i in range(len(labels)):
            index = numberlabel.index(labels[i])
            result[index].append(forclusterlist[i])
        return result

    def _initialize_center(self):
        self.model.eval()
        feature_matrix = torch.empty(size=(1, self.dim), requires_grad=True).cuda()
        # use top 5 data to initialize cluster center
        assert args.batch_size >= args.num_cls
        with torch.no_grad():
            for data, _ in self.loader:
                data = data.cuda()
                data = data[:self.num_cls]
                output = model(data)
                feature_matrix = torch.cat((feature_matrix, output), dim=0)  # (num_cls, dim)

        return feature_matrix[1:]

    def fit(self):
        self.model.train()
        self.targets = None
        all_data = torch.empty(size=(1, self.dim)).cuda()
        excep = False
        for row, (data, target) in enumerate(self.loader):
            data, target = data.cuda(), target.cuda()
            try:
                output = self.model(data)  # (batch_size ,512) batch_size
            except Exception:
                if not excep:
                    print('[EXCEPTION] memory full')
                    excep = True
                continue

            if self.targets is None:
                self.targets = target.cpu().numpy()
            else:
                self.targets = np.hstack((self.targets, target.cpu().numpy()))
            all_data = torch.cat((all_data, output), dim=0)
        all_data = all_data[1:].detach().cpu().numpy()

        # Prepare initial centers using K-Means++ method.
        initial_centers = kmeans_plusplus_initializer(all_data, 5).initialize()

        def cos_sim(x, y):
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
            x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
            d = torch.cosine_similarity(x, y)
            return d.detach().cpu().numpy()

        # def Jensen_Shannon(x, y):
        #     M = (x + y) / 2
        #     # 方法一：根据公式求解
        #     d = 0.5 * np.sum(x * np.log(x / M)) + 0.5 * np.sum(y * np.log(y / M))
        #     return 1 / d

        my_metric = distance_metric(type_metric.USER_DEFINED, func=cos_sim)
        # my_metric = distance_metric(type_metric.USER_DEFINED, func=Jensen_Shannon)

        # Create instance of K-Means algorithm with prepared centers.
        kmeans_instance = kmeans(all_data, initial_centers, metric=my_metric)

        # Run cluster analysis and obtain results.
        kmeans_instance.process()
        self.clusters = kmeans_instance.get_clusters()

    def score(self):
        print(self.targets.shape)
        sample_num = sum([len(s) for s in self.clusters])
        assert sample_num == len(self.targets)

        clus = np.array(self.clusters)

        cluster = []
        for i in range(clus.shape[0]):
            cluster.append(self.targets[clus[i]])
            print(self.targets[clus[i]])












# class CNN_Cluster:
#     def __init__(self, cnn_model, loader, ema, threshold, num_cls, ln):
#         self.model = cnn_model
#         self.loader = loader
#         self.num_cls = num_cls
#         self.ln = ln
#         self.cluster_center_ = self._prepare_data()
#         self.ema = ema
#         self.threshold = threshold
#
#     def _cosine_sim(self, x, y):
#         y = y.unsqueeze(dim=0)  # (1, 512)
#         x = torch.from_numpy(x).cuda()
#         x = x.reshape(y.shape)
#         d = torch.cosine_similarity(x, y)
#         return (-1)*(d - 1)
#
#     def _prepare_data(self):
#         self.model.eval()
#         labeled_loader = self.loader['labeled']
#         feature_matrix = torch.empty(size=(1, 512), requires_grad=True).cuda()
#         # use labeled data to initialize cluster center
#         with torch.no_grad():
#             for data, target, _ in labeled_loader:
#                 data, target = data.cuda(), target.cuda()
#                 output = model(data)
#                 feature_matrix = torch.cat((feature_matrix, output), dim=0)  # (num_labeled, 512)
#
#         feature_matrix = feature_matrix[1:, :]  # (80, 512)
#         assert feature_matrix.shape[0] == self.ln * self.num_cls and feature_matrix.shape[1] == 512
#
#         cluster_mean = torch.empty(size=(self.num_cls, 512), requires_grad=True).cuda()
#         for row in range(self.num_cls):
#             cluster_mean[row:, :] = torch.mean(feature_matrix[row * self.ln: (row + 1) * self.ln, :], dim=0)  # (5, 512)
#         return cluster_mean
#
#     def fit(self, optimizer, criterion, checkpoint=10, epoch=100, prt_interval=10, save_dir=''):
#         self.model.train()
#         best_acc = 0
#         for e in range(epoch):
#             correct = 0
#             batch_loss = 0
#             all_predict = torch.empty(size=(1, 1)).cuda()
#             all_data = torch.empty(size=(1, 512)).cuda()
#             for row, (data, target, _) in enumerate(self.loader['unlabeled']):
#                 data, target = data.cuda(), target.cuda()
#                 optimizer.zero_grad()
#                 output = self.model(data)  # (batch_size ,512) batch_size
#                 distances = distance(output, self.cluster_center_)  # (5, batch_size)
#                 loss = criterion(distances)
#                 loss.backward(retain_graph=True)
#                 ema.update_params()
#
#                 predict = torch.argmin(distances, dim=0).reshape(-1, 1)  # (batch_size, 1)
#                 target = target.view(-1, 1)
#                 correct += torch.eq(predict, target).sum().float().item()
#                 batch_loss += loss
#
#                 all_predict = torch.cat((all_predict, predict), dim=0)
#                 all_data = torch.cat((all_data, output), dim=0)
#             # update cluster center
#             # num is the unlabeled dataset size
#             # 对于一类中距离偏离比较远的数据，特征不参与均值计算，用上一次该类中心点的均值代替
#             all_predict = all_predict[1:, :].cpu().detach().numpy()
#             all_predict = all_predict.squeeze()  # (num, )
#             all_data = all_data[1:, :].cpu().detach().numpy()  # (5, num)
#             new_center = np.empty((self.num_cls, 512))
#             for row in range(self.num_cls):
#                 index = np.argwhere(all_predict == row)
#                 # print('[DEGUB] class {} {}'.format(row, len(index)))
#                 if len(index) == 0:
#                     history_center = self.cluster_center_[row:row + 1, :]
#                     history_center = history_center.cpu().detach().numpy()
#                     new_center[row:row + 1, :] = history_center
#                 else:
#                     # cls_data = all_data[index]  # 一个类别中的所有数据的预测距离
#                     for row_i in range(all_data[index].shape[0]):
#                         # 距离偏离比较远的数据，特征不参与均值计算，用上一次该类中心点的均值代替
#                         # threshold [0, 2] 0相同 2相反
#                         dis = self._cosine_sim(all_data[index][row_i, :], self.cluster_center_[row, :])
#                         if dis > self.threshold:
#                             all_data[index][row_i, :] = self.cluster_center_[row_i, :] * dis
#
#                     new_center[row:row+1, :] = np.mean(all_data[index], axis=0)
#             self.cluster_center_ = torch.from_numpy(new_center).cuda()
#
#             acc = correct / len(self.loader['unlabeled'].dataset)
#             if acc > best_acc:
#                 best_acc = acc
#             if (e + 1) % prt_interval == 0:
#                 print('[INFO] Epoch {}/{} acc: {:.6f} loss: {:.6f}'.format(e+1, epoch, acc, batch_loss / len(self.loader['unlabeled'].dataset)))
#
#             if save_dir and (e + 1) % checkpoint == 0:
#                 if acc > best_acc:
#                     torch.save(self.model.state_dict(), save_dir+'cnn_cluster_'+str(round(acc, 4))+'.pth')
#                     print('[INFO ] model saved')


if __name__ == '__main__':
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--gpus', type=str, default='0,1,2')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--checkpoint', type=str,
                        default='/home/st003/hmy/SimCLR_Cluster/runs/Mar27_21-54-40_gpu01/checkpoint_0100.pth.tar')
    parser.add_argument('--data_dir', type=str,
                        default='/home/st003/hmy/dataset/BloodCellSigned/segmentation_datasets/images/')
    parser.add_argument('--anno', type=str,
                        default='./anno/')
    parser.add_argument('--num_cls', type=int,
                        default=5)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--ema-alpha', type=float, default=0.8,
                        help='decay rate for ema module')
    parser.add_argument('--threshold', type=float, default=1.00)
    parser.add_argument('--n-views', default=2, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet50)')
    parser.add_argument('--out_dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('[INFO ] use_gpu: {}'.format(use_cuda))

    from functools import partial
    import pickle

    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

    data_loader = WBCDataset(args)
    loader = data_loader.getDataLoader()

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim).to(device)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])
    model = model.module

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    print('[INFO ] load from checkpoint: {}'.format(args.checkpoint))
    # state_dict = checkpoint['state_dict']


    # for k in list(state_dict.keys()):
    #     if k.startswith('backbone.'):
    #         if k.startswith('backbone') and not k.startswith('backbone.fc'):
    #             # remove prefix
    #             state_dict[k[len("backbone."):]] = state_dict[k]
    #     del state_dict[k]
    #
    # log = model.load_state_dict(state_dict, strict=False)
    # assert log.missing_keys == ['fc.weight', 'fc.bias']

    loader = WBCDataset(args)
    train_dataset = loader.getDataLoader()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               num_workers=10, drop_last=False, shuffle=True)

    # for name, param in model.named_parameters():
    #     if name not in ['fc.weight', 'fc.bias']:
    #         param.requires_grad = False
    #
    # parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # assert len(parameters) == 2  # fc.weight, fc.bias


    criterion = ExpLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    cnn_cluster = CNNCluster(model, train_loader, args.num_cls)
    cnn_cluster.fit()
    cnn_cluster.score()
    # cluster_mean = cnn_cluster.cluster_center_.cpu().detach().numpy()

    # label = {0: 'Basophil',
    #          1: 'Eosinophil',
    #          2: 'Lymphocyte',
    #          3: 'Monocyte',
    #          4: 'Neutrophil'}
    # for row in range(cluster_mean.shape[0]):
    #     plt.plot([x for x in range(cluster_mean.shape[1])], cluster_mean[row:row+1, :].reshape(512, -1).squeeze(),
    #                 label=label[row])
    # plt.legend()
    # plt.show()
    # cnn_cluster.fit(optimizer, criterion, save_dir='./models/')

