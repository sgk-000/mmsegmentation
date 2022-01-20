import cv2
import numpy as np
import os
import PIL
import rospy
import sys
import tempfile
import threading
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from six.moves import urllib

sys.path.append("/home/digital/anaconda3/envs/mmseg2/lib/python3.8/site-packages")

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette


class MmsegNode(object):
    def __init__(self):
        self._cv_bridge = CvBridge()
        self._last_msg = None
        self._msg_lock = threading.Lock()

        self._publish_rate = rospy.get_param('~publish_rate', 10)
        self._visualize = rospy.get_param('~visualize', True)
        self.config = rospy.get_param('~config')
        self.checkpoint = rospy.get_param('~checkpoint')
        self.device = rospy.get_param('~device', 'cuda:0')
        self.pallete = rospy.get_param('~pallete')
        self.input_height = rospy.get_param('~input_height', 480)
        self.input_width = rospy.get_param('~input_width', 480)
        self.output_height = rospy.get_param('~output_height', 480)
        self.output_width = rospy.get_param('~output_width', 480)
        self.opacity = rospy.get_param('~opacity', 0.5)

        self.palette = [
            [0, 0, 0],
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [153, 153, 153],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [0, 0, 142],
            [119, 11, 32],
            [0, 0, 230],
            [250, 170, 30],
            [220, 220, 0],
        ]

        rgb_input = rospy.get_param('~rgb_input', '/camera/rgb/image_color')

        rospy.Subscriber(rgb_input, Image, self._image_callback, queue_size=1)

        self.label_pub = rospy.Publisher('~segmentation', Image, queue_size=1)
        self.vis_pub = rospy.Publisher('~segmentation_viz', Image, queue_size=1)

        self.model = init_segmentor(self.config, self.checkpoint, device=self.device)
    def run(self):
        rate = rospy.Rate(self._publish_rate)

        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if msg is not None:
                rgb_image = self._cv_bridge.imgmsg_to_cv2(msg, "passthrough")
                rgb_image = cv2.resize(rgb_image, (self.input_width, self.input_height))
                # Run detection.
                seg_map = self.detect(rgb_image, self.model)
                # seg_map = cv2.resize(seg_map, (self.output_width, self.output_height))
                rospy.logdebug("Publishing semantic labels.")
                # label_msg = self._cv_bridge.cv2_to_imgmsg(seg_map, 'mono16')
                # label_msg.header = msg.header
                # self.label_pub.publish(label_msg)

                if self._visualize:
                    # Overlay segmentation on RGB image.
                    image = self.visualize(self.model, rgb_image, seg_map, self.palette, self.opacity)
                    image = cv2.resize(image, (self.output_width, self.output_height))
                    label_color_msg = self._cv_bridge.cv2_to_imgmsg(image, 'bgr8')
                    label_color_msg.header = msg.header
                    self.vis_pub.publish(label_color_msg)


            rate.sleep()

    def detect(self, rgb_image, model):
        # test a single image
        seg_map = inference_segmentor(model, rgb_image)
        # print(seg_map)
        # print(len(seg_map[0]))
        # print(len(seg_map[0][0]))
        return seg_map

    def visualize(self, model, rgb_image, seg_map, palette, opacity):
        draw_img = model.show_result(
            rgb_image,
            seg_map,
            palette=palette,
            show=False,
            opacity=opacity)
        return draw_img


    def _image_callback(self, msg):
        rospy.logdebug("Got an image.")

        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()


def main():
    rospy.init_node('mmseg_ros')

    node = MmsegNode()
    node.run()


if __name__ == '__main__':
    main()
