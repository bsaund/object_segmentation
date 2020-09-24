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

from object_segmentation import object_segmentations as obseg

already_processing = False
segmenter = None


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
    prediction = segmenter.run_inference_for_single_image(decompressed)
    # prediction = run_inference_for_single_image(detection_model, decompressed)
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

    talker.init()

    marked_pub = rospy.Publisher("/marked_image/compressed", CompressedImage, queue_size=1)
    img_sub = rospy.Subscriber("/kinect2_victor_head/qhd/image_color/compressed", CompressedImage, img_callback,
                               queue_size=1)

    rospy.spin()
