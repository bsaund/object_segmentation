#! /usr/bin/env python
import argparse
import os
import pathlib

import numpy as np
# import tensorflow as tf

from PIL import Image

# from object_detection.utils import ops as utils_ops
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util


from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.config import cfg
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.dataset import TestDataset

import torch
import torch.nn as nn
from torchvision import transforms


import cv2

import rospy
import rospkg
from amazon_ros_speech import talker
from sensor_msgs.msg import CompressedImage
from pointing_detection import img_utils
from scipy.io import loadmat
import csv

import time

colors = loadmat('data/color150.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

already_processing = False

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/bradsaund/research/tensorflow_model_zoo/models/research/object_detection/data/mscoco_label_map.pbtxt'
# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'

GREET_DELAY_TIME_SEC = 60 * 10  # seconds
LAST_HUMAN_TIME = time.time() - GREET_DELAY_TIME_SEC


def load_model(cfg_file, gpu):
    torch.cuda.set_device(0)
    basepath = rospkg.RosPack().get_path('object_segmentation')
    cfg.merge_from_file(basepath + "/" + cfg_file)

    logger = setup_logger(distributed_rank=0)  # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
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
    return segmentation_module
    print(basepath)

    # base_url = 'http://download.tensorflow.org/models/object_detection/'
    # model_file = model_name + '.tar.gz'
    # model_dir = tf.keras.utils.get_file(
    #     fname=model_name,
    #     origin=base_url + model_file,
    #     untar=True)
    #
    # model_dir = pathlib.Path(model_dir) / "saved_model"
    #
    # model = tf.saved_model.load(str(model_dir))
    #
    # return model


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


def run_inference_for_single_image(model, image, gpu=0):
    # image = np.asarray(image)
    # image = torch.tensor(image)
    # feed_dict = {'img_data': image}
    # feed_dict = async_copy_to(feed_dict, gpu)
    # seg_size = (image.shape[0], image.shape[1])

    # with torch.no_grad():
    #     scores = torch.zeros(1, cfg.DATASET.num_class, seg_size[0], seg_size[1])
    #     scores = async_copy_to(scores, gpu)
    #     pred_tmp = model(feed_dict, segSize=seg_size)
    # for batch_data in loader_test:
    # batch_data = next(loader_test.__iter__())
    # process data
    # batch_data = batch_data[0]


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

            pred_tmp = model(feed_dict, segSize=seg_size)

            scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)


        _, pred = torch.max(scores, dim=1)
        pred = pred.squeeze(0).cpu().numpy()


    img_vis = visualize_result((batch_data['img_ori'], None), pred, cfg)
    print("This took {}".format(time.time() - t0))
    return img_vis


def visualize_result(data, pred, cfg):
    (img, info) = data

    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    # print("Predictions in [{}]:".format(info))
    # for idx in np.argsort(counts)[::-1]:
    #     name = names[uniques[idx] + 1]
    #     ratio = counts[idx] / pixs * 100
        # if ratio > 0.1:
        #     print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)
    return im_vis


def img_callback(img_msg):
    global already_processing
    if already_processing:
        print("skipping this call")
        return

    dt = (rospy.get_rostime() - img_msg.header.stamp)
    delay = dt.secs + dt.nsecs * 1e-9
    if delay > 0.05:
        # print("Too far behind, skipping this call")
        return

    already_processing = True
    decompressed = img_utils.decompress_img(img_msg)
    decompressed = cv2.flip(decompressed, 1)
    t0 = time.time()
    prediction = run_inference_for_single_image(detection_model, decompressed)
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     decompressed,
    #     output_dict['detection_boxes'],
    #     output_dict['detection_classes'],
    #     output_dict['detection_scores'],
    #     category_index,
    #     instance_masks=output_dict.get('detection_masks_reframed', None),
    #     use_normalized_coordinates=True,
    #     line_thickness=8)
    img_msg.data = img_utils.compress_img(prediction)
    # img_msg.data = img_utils.compress_img(decompressed)
    marked_pub.publish(img_msg)
    # greet_new_people(output_dict)
    print("Inference took {} seconds".format(time.time() - t0))

    already_processing = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    parser.add_argument(
        "--cfg",
        # default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        default="config/ade20k-mobilenetv2dilated-c1_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu id for evaluation"
    )
    # parser.add_argument(
    #     "--imgs",
    #     required=True,
    #     type=str,
    #     help="an image path, or a directory name"
    # )

    args = parser.parse_args()

    # generate testing image list
    # if os.path.isdir(args.imgs):
    #     imgs = find_recursive(args.imgs)
    # else:
    #     imgs = [args.imgs]
    # assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    # cfg.list_test = [{'fpath_img': x} for x in imgs]

    # dataset_test = TestDataset(
    #     cfg.list_test,
    #     cfg.DATASET)
    # loader_test = torch.utils.data.DataLoader(
    #     dataset_test,
    #     batch_size=cfg.TEST.batch_size,
    #     shuffle=False,
    #     collate_fn=user_scattered_collate,
    #     num_workers=5,
    #     drop_last=True)

    rospy.init_node("object_segmentation")
    detection_model = load_model(args.cfg, args.gpu)
    talker.init()
    img_sub = rospy.Subscriber("/kinect2_victor_head/qhd/image_color/compressed", CompressedImage, img_callback,
                               queue_size=1)
    marked_pub = rospy.Publisher("/marked_image/compressed", CompressedImage, queue_size=1)
    rospy.spin()
