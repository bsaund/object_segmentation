#! /usr/bin/env python
import argparse
import os
import pathlib

import numpy as np
# import tensorflow as tf

from PIL import Image
import message_filters

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

import rospy
import rospkg

from sensor_msgs.msg import CompressedImage
# from pointing_detection import img_utils
from scipy.io import loadmat
import csv

import time

from object_segmentation import object_segmentations as obseg

already_processing = False
segmenter = None


def img_callback(img_msg):
    global already_processing
    if already_processing:
        print("skipping this call")
        return
    # print("Image recieved")

    dt = (rospy.get_rostime() - img_msg.header.stamp)
    delay = dt.secs + dt.nsecs * 1e-9
    if delay > 0.3:
        print("Delay of {:2.3f} is too far behind, skipping this call".format(delay))
        return
    # print("We are {} seconds behind".format(delay))

    already_processing = True

    t0 = time.time()
    decompressed = obseg.decompress_img(img_msg)
    # decompressed = cv2.flip(decompressed, 1)
    t0 = time.time()
    prediction = segmenter.run_inference_for_single_image(decompressed)

    img_vis = segmenter.visualize_result((decompressed, None), prediction,
                                         overlay=True, concat=False)

    img_msg.data = obseg.compress_img(img_vis)
    # img_msg.data = obseg.compress_img(decompressed)
    marked_pub.publish(img_msg)

    img_mask = segmenter.visualize_result((decompressed, None), prediction,
                                          overlay=False, concat=False)
    img_msg.data = obseg.compress_img(img_mask)
    mask_pub.publish(img_msg)

    print("This took {}".format(time.time() - t0))

    print("Inference took {} seconds".format(time.time() - t0))

    already_processing = False


if __name__ == "__main__":
    rospy.init_node("object_segmentation")
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    parser.add_argument(
        "--cfg",
        # default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        # default="config/ade20k-mobilenetv2dilated-c1_deepsup.yaml",
        default="config/fat-mobilenetv2dilated-c1_deepsup.yaml",
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

    args = parser.parse_args()

    segmenter = obseg.Segmenter(args.cfg, args.gpu)

    marked_pub = rospy.Publisher("marked_image/compressed", CompressedImage, queue_size=1)
    mask_pub = rospy.Publisher("segmentation_mask/compressed", CompressedImage, queue_size=1)

    # image_sub = message_filters.Subscriber("/kinect2_victor_head/hd/image_color/compressed", CompressedImage)
    # image_rect_sub = message_filters.Subscriber("/kinect2_victor_head/qhd/image_color_rect/compressed", CompressedImage)
    # depth_image_sub = message_filters.Subscriber("/kinect2_victor_head/qhd/image_depth_rect/compressed", CompressedImage)
    #
    # time_sync = message_filters.TimeSynchronizer([image_rect_sub, depth_image_sub], 10)
    # time_sync.registerCallback(kinect_callback)

    img_sub = rospy.Subscriber("/kinect2_victor_head/qhd/image_color/compressed", CompressedImage, img_callback,
                               queue_size=1)

    rospy.spin()
