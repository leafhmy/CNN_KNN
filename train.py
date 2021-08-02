import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import argparse
import os
import copy
# from net.nets import *
from net.resnet import resnet18
from utils.mixup import mixup_data, mixup_criterion
from utils.label_smoothing import LabelSmoothingCrossEntropy
from load_data.data_loader import Loader, save_file_path_v2
from tensorboardX import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(args, model, device, loader, optimizer, criterion, lr_scheduler, mixup=False, model_name=''):
    train_loader = loader['train']
    test_loader = loader['val']
    # best_model = copy.deepcopy(model)
    best_model_dict = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    writer = SummaryWriter('./log')

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target, _) in enumerate(train_loader):
            # data, target = torch.tensor(data), torch.tensor(target)
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            if mixup:
                inputs, targets_a, targets_b, lam = mixup_data(data, target,
                                                               args.alpha, use_cuda=True)  # 对数据集进行mixup操作
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)  # 对loss#函数进行mixup操作
            else:
                output = model(data)
                loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if batch_idx % args.log_interval == 0:
                print('[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

                writer.add_scalar('loss', loss, epoch)
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().data.cpu().numpy(), epoch)

        if epoch % args.checkpoint == 0:
            model.eval()
            test_loss = 0
            correct = 0
            correct_top3 = 0

            with torch.no_grad():
                for data, target, _ in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += criterion(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

                    # top3 acc
                    target_resize = target.view(-1, 1)
                    _, pred = output.topk(3, 1, True, True)
                    correct_top3 += torch.eq(pred, target_resize).sum().float().item()

            test_loss /= len(test_loader.dataset)
            acc = 100. * correct / len(test_loader.dataset)
            acc_top3 = 100. * correct_top3 / len(test_loader.dataset)
            writer.add_scalar('accuracy', acc, epoch)
            print('[Test ] set: average loss: {:.4f}, current_lr: {}, acc: {:.6f}% acc_top3: {:.6f}%'
                  .format(test_loss, lr_scheduler.get_last_lr(), acc, acc_top3))

            if acc > best_acc:
                best_acc = acc
                best_model_dict = copy.deepcopy(model.state_dict())
                if args.save_model:
                    torch.save(best_model_dict, "./models/model_{}_{:.6f}_{:.4f}.pth".format(model_name, best_acc, test_loss))
                    print("[INFO ] Save checkpoint model_{}_{:.6f}_{:.4f}.pth".format(model_name, best_acc, test_loss))

    writer.close()
    print("[INFO ] train completely best acc: {}".format(best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--gpus', type=str, default='0'),
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--checkpoint', type=int, default=20)
    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--resume', type=str, default='/tmp/pycharm_project_0/models/model_resnet18_69.696970_0.6705.pth')
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/pycharm_project_0/datasets/WBC_images/')
    parser.add_argument('--anno', type=str,
                        default='./anno/')
    parser.add_argument('--num_cls', type=int,
                        default=5)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('[INFO ] use_gpu: {}'.format(use_cuda))

    if args.data_dir:
        save_file_path_v2(args, labeled_ratio=0.6, test_ratio=0.2)

    # load data
    data_loader = Loader(args)
    loader = data_loader.get_data_loader()
    # data_loader = AlbumentationsLoader(args)
    # loader = data_loader.get_data_loader()

    # define model
    # model = torchvision.models.resnet18(pretrained=True)
    # model = torchvision.models.resnet50(pretrained=True)
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, args.num_cls)

    model = resnet18(num_classes=5)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])
    model = model.module
    if args.resume:
        pretrained_dict = torch.load(args.resume)
        model.load_state_dict(pretrained_dict, strict=False)
        print('[INFO ] load from checkpoint: {}'.format(args.resume))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.5)
    criterion = F.cross_entropy
    # criterion = LabelSmoothingCrossEntropy()

    train(args, model, device, loader, optimizer, criterion, lr_scheduler, mixup=False, model_name='resnet18')