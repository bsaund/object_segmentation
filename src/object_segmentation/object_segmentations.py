#! /usr/bin/env python
import os

import numpy as np

from PIL import Image


from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.config import cfg
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.dataset import TestDataset

import torch
import torch.nn as nn
from torchvision import transforms
import rospkg
from scipy.io import loadmat, savemat
import csv

import time
from pathlib import Path


class Segmenter:
    def __init__(self,
                 cfg="config/fat-mobilenetv2dilated-c1_deepsup.yaml",
                 gpu=0):
        self.colors = None
        self.names = {}
        self.model = None

        self._load_params(cfg, gpu)

    def _load_params(self, cfg, gpu):
        self._load_model(cfg, gpu)
        basepath = Path(rospkg.RosPack().get_path('object_segmentation'))
        self.colors = loadmat((basepath / 'data/color150.mat').as_posix())['colors']
        with open((basepath / 'data/object_info.csv').as_posix()) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.names[int(row[0])] = row[1]

    def _load_model(self, cfg_file, gpu):
        torch.cuda.set_device(0)
        basepath = rospkg.RosPack().get_path('object_segmentation')
        cfg.merge_from_file(basepath + "/" + cfg_file)

        logger = setup_logger(distributed_rank=0)
        # logger.info("Loaded configuration file {}".format(args.cfg))
        logger.info("Running with config:\n{}".format(cfg))

        cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
        cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

        # absolute paths of model weights
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

        assert os.path.exists(cfg.MODEL.weights_encoder) and \
               os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

        # Network Builders
        net_encoder = ModelBuilder.build_encoder(
            arch=cfg.MODEL.arch_encoder,
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_encoder)
        net_decoder = ModelBuilder.build_decoder(
            arch=cfg.MODEL.arch_decoder,
            fc_dim=cfg.MODEL.fc_dim,
            num_class=cfg.DATASET.num_class,
            weights=cfg.MODEL.weights_decoder,
            use_softmax=True)

        crit = nn.NLLLoss(ignore_index=-1)

        segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
        segmentation_module.cuda()
        segmentation_module.eval()
        self.model = segmentation_module

    def visualize_result(self, data, pred, overlay=True, concat=False):
        (img, info) = data

        # print predictions in descending order
        pred = np.int32(pred)
        pixs = pred.size
        uniques, counts = np.unique(pred, return_counts=True)
        # print("Predictions in [{}]:".format(info))
        for idx in np.argsort(counts)[::-1]:
            name = self.names[uniques[idx] + 1]
            ratio = counts[idx] / pixs * 100
            if ratio > 0.1:
                print("  {:20}: {:.2f}%".format(name, ratio))

        # colorize prediction
        pred_color = colorEncode(pred, self.colors).astype(np.uint8)

        # aggregate images and save
        # im_vis = np.concatenate((img, pred_color), axis=1)
        if not overlay and not concat:
            return np.repeat(np.expand_dims(pred.astype(np.uint8), 2), 3, 2)

        if overlay:
            img = (img.astype('float') + pred_color.astype('float')).clip(0, 255).astype('uint8')
        if concat:
            img = np.concatenate((img, pred_color), axis=1)
        return img

    def run_inference_for_single_image(self, image, gpu=0, overlay=True, concat=False):
        preproc_data = preprocess_image(image)

        batch_data = preproc_data

        seg_size = (batch_data['img_ori'].shape[0],
                    batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']
        t0 = time.time()
        with torch.no_grad():
            # scores = torch.zeros(1, cfg.DATASET.num_class, seg_size[0], seg_size[1])
            scores = torch.cuda.FloatTensor(1, cfg.DATASET.num_class, seg_size[0], seg_size[1]).fill_(0)
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = {'img_data': img}
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass

                pred_tmp = self.model(feed_dict, segSize=seg_size)

                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)
            _, pred = torch.max(scores, dim=1)
            pred = pred.squeeze(0).cpu().numpy()

        img_vis = self.visualize_result((batch_data['img_ori'], None), pred, overlay, concat)
        print("This took {}".format(time.time() - t0))
        return img_vis


def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


def img_transform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img.copy()))
    return img


def preprocess_image(image):
    img = Image.fromarray(image)
    ori_width, ori_height = img.size
    img_resized_list = []

    for this_short_size in cfg.DATASET.imgSizes:
        # calculate target height and width
        scale = min(this_short_size / float(min(ori_height, ori_width)),
                    cfg.DATASET.imgMaxSize / float(max(ori_height, ori_width)))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)

        # to avoid rounding in network
        target_width = round2nearest_multiple(target_width, cfg.DATASET.padding_constant)
        target_height = round2nearest_multiple(target_height, cfg.DATASET.padding_constant)

        # resize images
        img_resized = imresize(img, (target_width, target_height), interp='bilinear')

        # image transform, to torch float tensor 3xHxW
        img_resized = img_transform(img_resized)
        img_resized = torch.unsqueeze(img_resized, 0)
        img_resized_list.append(img_resized)
    output = dict()
    output['img_ori'] = np.array(img)
    output['img_data'] = [x.contiguous() for x in img_resized_list]
    # output['info'] = this_record['fpath_img']
    return output








