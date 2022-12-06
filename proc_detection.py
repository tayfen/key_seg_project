import os
import torchvision.transforms as tt
import torch
import torch.nn as nn
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import os.path as osp
import mmcv
import numpy as np
import logging

from mmdet.apis import set_random_seed
from mmcv import Config


MMDETECTION_DIR = os.environ["MMDETECTION_DIR"]
MODEL_PATH = os.environ["DETECTION_MODEL_PATH"]


logger = logging.getLogger(__name__)


class key_detection:
    def __init__(self):

        self.log = logger.info
        self.cfg = Config.fromfile(
            os.path.join(MMDETECTION_DIR, "configs", "ssd", "ssd512_coco.py")
        )
        dataset_type = "CocoDataset"
        # Modify dataset type and path
        self.cfg.dataset_type = dataset_type
        self.cfg.model.bbox_head.num_classes = 1
        # Set up working dir to save files and logs.
        self.cfg.work_dir = ".\tutorial_exps"
        # The original learning rate (LR) is set for 8-GPU training.
        # We divide it by 8 since we only use one GPU.
        # ssd300_coco and ssd300_voc with 4gpu, 24 imgs per gpu and lr of 3e-3
        self.cfg.optimizer.lr = 0.02 / 8.0  # 3e-3
        # cfg.lr_config.warmup = None
        self.cfg.lr_config.policy = "step"
        self.cfg.lr_config.warmup = "linear"
        self.cfg.lr_config.warmup_ratio = 1.0 / 1e10
        self.cfg.log_config.interval = 10
        # Change the evaluation metric since we use customized dataset.
        self.cfg.evaluation.metric = "mAP"
        # We can set the evaluation interval to reduce the evaluation times
        self.cfg.evaluation.interval = 12
        # We can set the checkpoint saving interval to reduce the storage cost
        self.cfg.checkpoint_config.interval = 12
        # Set seed thus the results are more reproducible
        self.cfg.seed = 0
        set_random_seed(0, deterministic=False)
        self.cfg.gpu_ids = range(1)
        self.cfg.load_from = MODEL_PATH
        self.cfg.runner.max_epochs = 5

        self.model = init_detector(self.cfg, self.cfg.load_from, device="cpu")

    def proc_img(self, path):
        # read image
        img = mmcv.imread(path)

        # scale
        scaled_img = self.scale_img(img, 400)

        # inference scaled img
        self.result = inference_detector(self.model, scaled_img)

        # scale bbox back
        self.scaled_result = self.scale_back_bbox(
            self.result,
            (img.shape[0] / scaled_img.shape[0], img.shape[1] / scaled_img.shape[1]),
        )

        # filter low probabilities and get the highest one
        print(self.scaled_result)
        bbox = self.filter_low_prob(self.scaled_result)
        print(bbox)
        if len(bbox[0][0]) != 5:
            self.log("key not found, check image - {path}")
            return None

        # crop img according to bbox
        cropped = self.crop_img(img, bbox[0][0])

        return cropped

    def crop_img(self, img, bbox):
        add = 0.15
        i_h, i_w, _ = img.shape
        h = int(bbox[3] - bbox[1])
        w = int(bbox[2] - bbox[0])
        y = int(bbox[1] - add * h)
        x = int(bbox[0] - add * w)
        if y < 0:
            y = 0
        if x < 0:
            x = 0
        h = int((1 + 2 * add) * h)
        w = int((1 + 2 * add) * w)
        if h + y > i_h:
            h = i_h - y
        if w + x > i_w:
            w = i_w - x
        return img[y : y + h, x : x + w]

    def filter_low_prob(self, res):
        filtered = []
        for bbox in res[0]:
            if len(bbox) == 0 or bbox[4] > 0.2:
                filtered.append(bbox)

        if len(filtered) > 1:
            filtered = [sorted(filtered, key=lambda x: x[4], reverse=True)[0]]

        if len(filtered) == 0:
            filtered = [filtered]

        return [np.array(filtered)]

    def scale_img(self, img, dim):
        y = img.shape[0]
        x = img.shape[1]
        ratio = y / x
        return mmcv.imresize(img, (dim, int(ratio * dim)))

    def scale_back_bbox(self, result, ratio):
        y_ratio, x_ratio = ratio

        scaled_result = []
        for bbox in result:
            if len(bbox) > 0:
                bbox[0][0] *= y_ratio
                bbox[0][2] *= y_ratio
                bbox[0][1] *= x_ratio
                bbox[0][3] *= x_ratio
            else:
                bbox = [[]]
            scaled_result.append(bbox)
        return scaled_result
