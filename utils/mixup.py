"""
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       args.alpha, use_cuda)    # 对数据集进行mixup操作
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)    #对loss#函数进行mixup操作
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
"""
import numpy as np
import torch


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    # 对数据的mixup 操作 x = lambda*x_i+(1-lamdda)*x_j
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]  # 此处是对数据x_i 进行操作
    y_a, y_b = y, y[index]  # 记录下y_i 和y_j
    return mixed_x, y_a, y_b, lam  # 返回y_i 和y_j 以及lambda


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    # 对loss函数进行混合，criterion是crossEntropy函数
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
