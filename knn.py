import torch
import numpy as np
import time
import argparse
import os
from load_data.data_loader import Loader
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from scipy.stats import mode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--data_dir', type=str,
                        # default='E:\dataset\segmentation_datasets\images/')
                        default='/tmp/pycharm_project_0/datasets/WBC_images/')
    parser.add_argument('--anno', type=str,
                        default='./anno/')
    parser.add_argument('--num_cls', type=int,
                        default=5)
    args = parser.parse_args()

    data_loader = Loader(args)
    loader = data_loader.get_data_loader()
    num = len(loader['unlabeled'].dataset)

    train_x = torch.empty(size=(1, 3, 224, 224))
    train_y = torch.tensor([])
    test_x = torch.empty(size=(1, 3, 224, 224))
    test_y = torch.tensor([])

    for (data, target, _) in loader['labeled']:
        train_x = torch.cat((train_x, data))
        train_y = torch.cat((train_y, target))

    for (data, target, _) in loader['unlabeled']:
        test_x = torch.cat((test_x, data))
        test_y = torch.cat((test_y, target))

    train_x = train_x[1:, :, :, :]
    train_x = torch.mean(train_x, dim=1)
    train_x = torch.flatten(train_x, start_dim=1).cpu().detach().numpy()
    train_y = train_y.cpu().detach().numpy()

    test_x = test_x[1:, :, :, :]
    test_x = torch.mean(test_x, dim=1)
    test_x = torch.flatten(test_x, start_dim=1).cpu().detach().numpy()
    test_y = test_y.cpu().detach().numpy()

    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)

    kNN_classifier = KNeighborsClassifier(weights='distance', n_neighbors=8, p=1)
    kNN_classifier.fit(train_x, train_y)
    score = kNN_classifier.score(test_x, test_y)
    print(score)

    # param_grid = [
    #     {
    #         'weights': ['uniform'],
    #         'n_neighbors': [i for i in range(1, 11)]
    #     },
    #     {
    #         'weights': ['distance'],
    #         'n_neighbors': [i for i in range(1, 11)],
    #         'p': [i for i in range(1, 6)]
    #     }
    # ]
    #
    # grid_search = GridSearchCV(kNN_classifier, param_grid)
    # grid_search.fit(train_x, train_y)
    # print(grid_search.best_estimator_)
    # print(grid_search.best_score_)
    #
    # score = grid_search.score(test_x, test_y)
    # print(score)


