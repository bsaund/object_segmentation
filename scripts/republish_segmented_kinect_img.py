#! /usr/bin/env python
import argparse
import threading
import rospy

from sensor_msgs.msg import CompressedImage
import time
import sys
from object_segmentation import object_segmentations as obseg

segmenter = None
img_msg_to_process = None


def img_callback(img_msg):
    global img_msg_to_process
    if img_msg_to_process is not None:
        # print("already processing an image, skipping this call")
        return
    img_msg_to_process = img_msg
    # print("Image received")

    # dt = (rospy.get_rostime() - img_msg.header.stamp)
    # delay = dt.secs + dt.nsecs * 1e-9
    # if delay > 0.3:
    #     print("Delay of {:2.3f} is too far behind, skipping this call".format(delay))
    #     return
    # print("We are {} seconds behind".format(delay))


def segment_thread_worker():
    global img_msg_to_process
    while not rospy.is_shutdown():
        if img_msg_to_process is None:
            rospy.sleep(0.01)
            continue
        segment_and_republish(img_msg_to_process)
        img_msg_to_process = None


def segment_and_republish(img_msg):
    t0 = time.time()
    decompressed = obseg.decompress_img(img_msg)
    # decompressed = cv2.flip(decompressed, 1)
    t0 = time.time()
    prediction = segmenter.run_inference_for_single_image(decompressed)

    img_vis = segmenter.visualize_result((decompressed, None), prediction,
                                         overlay=True, concat=False, verbose=True)

    img_msg.data = obseg.compress_img(img_vis)
    # img_msg.data = obseg.compress_img(decompressed)
    marked_pub.publish(img_msg)

    img_mask = segmenter.visualize_result((decompressed, None), prediction,
                                          overlay=False, concat=False)
    img_msg.data = obseg.compress_img(img_mask)
    mask_pub.publish(img_msg)

    rospy.loginfo("Inference took {} seconds".format(time.time() - t0))

    already_processing = False


if __name__ == "__main__":
    rospy.init_node("object_segmentation")
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    parser.add_argument(
        "--cfg",
        default="config/ycbvideo-mobilenetv2dilated-c1_deepsup.yaml",
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

    # print(rospy.myargv(argv=sys.argv))
    args = parser.parse_args(args=rospy.myargv(argv=sys.argv)[1:])

    # cfg = default="config/ycbvideo-mobilenetv2dilated-c1_deepsup.yaml"
    # gpu = 0
    segmenter = obseg.Segmenter(args.cfg, args.gpu)

    marked_pub = rospy.Publisher("marked_image/compressed", CompressedImage, queue_size=1)
    mask_pub = rospy.Publisher("segmentation_mask/compressed", CompressedImage, queue_size=1)

    segment_thread = threading.Thread(target=segment_thread_worker)
    segment_thread.start()

    img_sub = rospy.Subscriber("/kinect2_victor_head/qhd/image_color/compressed", CompressedImage, img_callback,
                               queue_size=1)

    rospy.spin()
