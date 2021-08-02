# coding: utf-8
"""
通过实现Grad-CAM学习module中的forward_hook和backward_hook函数
"""
import cv2
import os
import numpy as np
from PIL import Image
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
os.environ['CUDA_VISIBLE_DEVICES'] = '10'


class DrawCam:
    def __init__(self, model, path_img, img_size, num_cls, layer, output_dir='', show=True):
        assert isinstance(img_size, tuple) and img_size[0] == img_size[1]
        self.img_size = img_size
        self.layer = layer  # model.layer resnet18.layer4 for example
        self.num_cls = num_cls
        self.fmap_block = list()
        self.grad_block = list()
        self.show = show
        self.output_dir = output_dir
        img = cv2.imread(path_img, 1)  # H*W*C
        img_input = self.__img_preprocess(img)

        net = model
        # 注册hook
        self.layer.register_forward_hook(self.__farward_hook)
        self.layer.register_backward_hook(self.__backward_hook)

        # forward
        self.output = net(img_input)

        # backward
        net.zero_grad()
        class_loss = self.__comp_class_vec(self.output)
        class_loss.backward()

        # 生成cam
        grads_val = self.grad_block[0].cpu().data.numpy().squeeze()
        fmap = self.fmap_block[0].cpu().data.numpy().squeeze()
        self.cam = self.__gen_cam(fmap, grads_val)

        # 保存cam图片
        self.img_show = np.float32(cv2.resize(img, self.img_size)) / 255

    def show_cam_raw(self):
        self.__show_cam_on_image(self.img_show, self.cam, self.output_dir)

    def get_cam(self):
        cam = self.__show_cam_on_image(self.img_show, self.cam, self.output_dir, get=True)
        return cam

    def __show_cam_on_image(self, img, mask, out_dir, get=False):
        heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        raw = np.uint8(255 * img)
        cam = np.uint8(255 * cam)
        if get:
            return cam
        if out_dir:
            path_cam_img = os.path.join(out_dir, "cam.jpg")
            path_raw_img = os.path.join(out_dir, "raw.jpg")
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            cv2.imwrite(path_cam_img, cam)
            cv2.imwrite(path_raw_img, raw)

        if self.show:
            cam_raw_img = np.hstack((cam, raw))
            cv2.imshow('cam_raw_img', cam_raw_img)
            cv2.waitKey()

    def __img_transform(self, img_in, transform):
        """
        将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
        :param img_roi: np.array
        :return:
        """
        img = img_in.copy()
        img = Image.fromarray(np.uint8(img))
        img = transform(img)
        img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
        return img.cuda()

    def __img_preprocess(self, img_in):
        """
        读取图片，转为模型可读的形式
        :param img_in: ndarray, [H, W, C]
        :return: PIL.image
        """
        img = img_in.copy()
        img = cv2.resize(img, self.img_size)
        img = img[:, :, ::-1]   # BGR --> RGB
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_input = self.__img_transform(img, transform)
        return img_input

    def __backward_hook(self, module, grad_in, grad_out):
        self.grad_block.append(grad_out[0].detach())

    def __farward_hook(self, module, input, output):
        self.fmap_block.append(output)

    def __comp_class_vec(self, ouput_vec, index=None):
        """
        计算类向量
        :param ouput_vec: tensor
        :param index: int，指定类别
        :return: tensor
        """
        if not index:
            index = np.argmax(ouput_vec.cpu().data.numpy())
        else:
            index = np.array(index)
        index = index[np.newaxis, np.newaxis]
        index = torch.from_numpy(index)
        one_hot = torch.zeros(1, self.num_cls).scatter_(1, index, 1)

        one_hot.requires_grad = True
        class_vec = torch.sum(one_hot.cuda() * self.output.cuda())  # one_hot = 11.8605

        return class_vec


    def __gen_cam(self, feature_map, grads):
        """
        依据梯度和特征图，生成cam
        :param feature_map: np.array， in [C, H, W]
        :param grads: np.array， in [C, H, W]
        :return: np.array, [H, W]
        """
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

        weights = np.mean(grads, axis=(1, 2))  #

        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, self.img_size)
        cam -= np.min(cam)
        cam /= np.max(cam)

        return cam


if __name__ == '__main__':
    print('ok')
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--pretrained_model', type=str,
                        default='/home/leaf/PycharmProjects/Test/models/model_94.005450_0.2417.pth')
    parser.add_argument('--path_img', type=str,
                        default='/home/leaf/dataset/flower_photos/roses/5897035797_e67bf68124_n.jpg')
    parser.add_argument('--output_dir', type=str,
                        default='../cam/')
    parser.add_argument('--size', type=tuple,
                        default=(224, 224))
    parser.add_argument('--num_cls', type=int,
                        default=4)
    parser.add_argument('--show', type=bool,
                        default=True)
    args = parser.parse_args()
    net = models.resnet18(pretrained=True)
    num_features = net.fc.in_features
    net.fc = nn.Linear(num_features, args.num_cls)
    net.load_state_dict(torch.load(args.pretrained_model), strict=False)
    net = net.cuda()

    drawer = DrawCam(net, args.path_img, args.size, args.num_cls, net.layer4,
                     output_dir=args.output_dir, show=args.show)
    print('sava to {}'.format(args.output_dir))
