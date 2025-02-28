import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class ImageMatcher(Node):
    def __init__(self):
        super().__init__('image_matcher')
        self.bridge = CvBridge()
        self.template_image = None
        self.sift = cv2.SIFT_create()
        self.max_dist = 0.5
        self.image_sub = self.create_subscription(
            CompressedImage, '/oakd/rgb/preview/image_raw/compressed', self.image_callback, 10)

    def image_callback(self, msg):
        try:
            cam_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("Feature Matching", cam_image)
            cv2.waitKey(1)
            # cv2.destroyAllWindows()
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")


def main():
    rclpy.init()
    node = ImageMatcher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
