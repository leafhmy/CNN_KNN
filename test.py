import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import argparse
from utils.label_smoothing import LabelSmoothingCrossEntropy
from load_data.data_loader import Loader, save_file_path_v2
# from utils.conf_mat import *
# from utils.draw_cam import DrawCam
from net.resnet import resnet18
# from utils.save_error import SaveError
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def predict(args, model, device, loader, criterion, layer=None):
    test_loader = loader['val']
    label_dict = loader['label_dict']  # key: int, value: cls
    model.eval()
    test_loss = 0
    correct = 0
    correct_top3 = 0
    error = []  # 'img_path pred target' type: str
    if args.conf_mat:
        conf_matrix = torch.zeros(args.num_cls, args.num_cls)

    with torch.no_grad():
        for data, target, img_path in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            if args.save_error:
                pred_sq = pred.squeeze()
                assert len(pred_sq) == len(target)
                for i in range(len(pred_sq)):
                    if pred_sq[i] != target[i]:
                        error.append(img_path[i]+' '+label_dict[pred_sq[i].item()]+' '+label_dict[target[i].item()])

            if args.conf_mat:
                conf_matrix = confusion_matrix(pred, labels=target, conf_matrix=conf_matrix)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # top3 acc
            target_resize = target.view(-1, 1)
            _, pred = output.topk(3, 1, True, True)
            correct_top3 += torch.eq(pred, target_resize).sum().float().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    acc_top3 = 100. * correct_top3 / len(test_loader.dataset)
    print('[Test ] set: average loss: {:.4f}, acc: {:.6f}% acc_top3: {:.6f}%'
          .format(test_loss, acc, acc_top3))

    if args.conf_mat:
        print('[INFO ] ploting confusion matrix...')
        plot_confusion_matrix(conf_matrix.numpy(), classes=list(label_dict.values()), normalize=False,
                              title='Normalized confusion matrix')
        print('[INFO ] save confusion matrix to conf_mat_pic')

    if args.save_error:
        print('[INFO ] error: {}'.format(len(error)))
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        with open(args.save_dir+'error.txt', 'w') as f:
            f.write('img_path predict true\n')
            for m in error:
                f.write(m+'\n')
            print('[INFO ] save error log to error/error.txt')
            print('[INFO ] save error images...')

            saver = SaveError(error, save_dir=args.save_dir+'pic/',
                              show_cam=args.show_cam, model=model, size=(224, 224), num_cls=args.num_cls, layer=layer)
            saver.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--resume', type=str,
                        # default='E:\\resnet_knn\models\model_resnet18_85.714286_0.4613.pth')
                        default='/tmp/pycharm_project_0/models/model_resnet18_85.714286_0.4613.pth')
    parser.add_argument('--data_dir', type=str,
                        # default='E:\dataset\segmentation_datasets\images/')
                        default='/tmp/pycharm_project_0/datasets/WBC_images/')
    parser.add_argument('--anno', type=str,
                        default='./anno/')
    parser.add_argument('--save_dir', type=str, default='./error/')
    parser.add_argument('--num_cls', type=int,
                        default=5)
    parser.add_argument('--conf_mat', action='store_true')
    parser.add_argument('--save_error', action='store_true')
    parser.add_argument('--show_cam', action='store_true')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('[INFO ] use_gpu: {}'.format(use_cuda))
    print('[INFO ] conf_mat: {} save_error: {}'.format(args.conf_mat, args.save_error))

    data_loader = Loader(args)
    loader = data_loader.get_data_loader()
    # data_loader = AlbumentationsLoader(args)
    # loader = data_loader.get_data_loader()

    # model = torchvision.models.resnet18(pretrained=True)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, args.num_cls)
    model = resnet18(num_classes=5)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model = model.module
    if args.resume:
        pretrained_dict = torch.load(args.resume)
        model.load_state_dict(pretrained_dict, strict=False)
        print('[INFO ] load from checkpoint: {}'.format(args.resume))

    # criterion = LabelSmoothingCrossEntropy()
    criterion = F.cross_entropy

    predict(args, model, device, loader, criterion)






