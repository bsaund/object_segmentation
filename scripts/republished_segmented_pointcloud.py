#!/usr/bin/env python
from __future__ import print_function

import threading
import rospy

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CompressedImage
from shape_completion_training.model import utils
import message_filters
from object_segmentation import pointcloud_utils as utils

img_msgs_to_process = None


def kinect_callback(img_mask_msg, img_rect_msg, depth_rect_msg):
    global img_msgs_to_process
    if img_msgs_to_process is not None:
        return
    img_msgs_to_process = (img_mask_msg, img_rect_msg, depth_rect_msg)


def filter_pointcloud_worker():
    global img_msgs_to_process
    while not rospy.is_shutdown():
        if img_msgs_to_process is None:
            rospy.sleep(0.01)
            continue
        filter_pointcloud(*img_msgs_to_process)
        img_msgs_to_process = None


def filter_pointcloud(img_mask_msg, img_rect_msg, depth_rect_msg):
    if camera_model.camera_model is None:
        print("Waiting for camera model")
        return
    img_mask = utils.decompress_img(img_mask_msg)
    img_rect = utils.decompress_img(img_rect_msg)
    depth_rect = utils.decompress_depth(depth_rect_msg)

    cheezeit_box_categories = [2, 15]

    pts = utils.convert_masked_depth_img_to_pointcloud(depth_rect, img_rect, img_mask,
                                                       camera_model.camera_model,
                                                       categories=cheezeit_box_categories)

    pt_msg = utils.pts_to_ptmsg(pts, img_rect_msg.header.frame_id)
    cloud_pub.publish(pt_msg)
    print("Pointcloud published")

    return pt_msg, pt_msg


if __name__ == "__main__":
    rospy.init_node("segmented_pointcloud_republisher")

    cloud_pub = rospy.Publisher("/segmented_pointcloud", PointCloud2, queue_size=1)

    pt_thread = threading.Thread(target=filter_pointcloud_worker)
    pt_thread.start()

    last_update = rospy.get_rostime()
    camera_model = utils.CameraModel("/kinect2_victor_head/qhd/camera_info")

    image_mask_sub = message_filters.Subscriber("/segmentation_mask/compressed", CompressedImage)
    image_rect_sub = message_filters.Subscriber("/kinect2_victor_head/qhd/image_color_rect/compressed", CompressedImage)
    depth_image_sub = message_filters.Subscriber("/kinect2_victor_head/qhd/image_depth_rect/compressed",
                                                 CompressedImage)

    time_sync = message_filters.TimeSynchronizer([image_mask_sub, image_rect_sub, depth_image_sub], 10)
    time_sync.registerCallback(kinect_callback)

    rospy.spin()
