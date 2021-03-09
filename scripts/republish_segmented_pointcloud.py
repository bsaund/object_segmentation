#!/usr/bin/env python
from __future__ import print_function

import argparse
import threading
import rospy

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CompressedImage
import message_filters
from object_segmentation import pointcloud_utils as utils
import sys


def filter_pointcloud_worker():
    while not rospy.is_shutdown():
        if pt_republisher.filter_pointcloud() is not None:
            pt_republisher.prepare_for_next_callback()


if __name__ == "__main__":
    rospy.init_node("segmented_pointcloud_republisher")
    parser = argparse.ArgumentParser(
        description="Ros Semantic Segmentation"
    )
    parser.add_argument(
        "--objects",
        default=[x for x in range(1, 25)],
        nargs='*',
        type=int,
        help="list of object categories to include in segmentation"
    )

    args = parser.parse_args(args=rospy.myargv(argv=sys.argv)[1:])
    print(f"Matching categories: {args.objects}")
    pt_republisher = utils.PointcloudCreator(args.objects)

    pt_thread = threading.Thread(target=filter_pointcloud_worker)
    pt_thread.start()


    rospy.spin()
