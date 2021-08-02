import numpy as np
from utils.draw_cam import DrawCam
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import os


class SaveError:
    def __init__(self, error: list, save_dir='./error/pics/',
                 show_cam=False, model=None, size=(224, 224), num_cls=10, layer=None):
        assert isinstance(size, tuple) and size[0] == size[1]
        self.error = error
        self.show_cam = show_cam
        self.model = model
        self.save_dir = save_dir
        self.size = size
        self.num_cls = num_cls
        self.layer = layer
        print('[INFO ] total: {} show_cam: {} save_dir: {}'.format(len(error), show_cam, save_dir))

    def save(self):
        with tqdm(total=len(self.error)) as pbar:
            for index, data in enumerate(self.error):
                data = [d.strip() for d in data.split(' ')]
                img = cv2.imread(data[0])
                img = cv2.resize(img, self.size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.uint8)
                """
                drawer = DrawCam(net, args.path_img, args.size, args.num_cls, net.layer4,
                                     output_dir=args.output_dir, show=args.show)
                """
                cam = None
                if self.show_cam:
                    drawer = DrawCam(self.model, data[0], self.size, self.num_cls, self.layer, show=False)
                    cam = drawer.get_cam()  # numpy
                    del drawer
                if cam is not None:
                    img = np.hstack((img, cam))

                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)
                cv2.putText(img, 'pred: {} true: {}'.format(data[1], data[2]),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.imwrite(self.save_dir+str(index)+'_pred_'+str(data[1])+'_true_'+str(data[2]+'.jpg'), img)
                pbar.update(1)
