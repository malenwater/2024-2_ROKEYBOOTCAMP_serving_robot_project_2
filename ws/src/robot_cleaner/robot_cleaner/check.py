import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class ImageMatcher(Node):
    def __init__(self, template_path):
        super().__init__('image_matcher')
        self.bridge = CvBridge()
        self.template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        self.sift = cv2.SIFT_create()
        self.kp1, self.des1 = self.sift.detectAndCompute(self.template_image, None)
        self.image_sub = self.create_subscription(
            CompressedImage, '/oakd/rgb/preview/image_raw/compressed', self.image_callback, 10)

    def image_callback(self, msg):
        try:
            cam_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2GRAY)
            kp2, des2 = self.sift.detectAndCompute(cam_image, None)
            
            if des2 is None or self.des1 is None or len(kp2) < 2:
                self.get_logger().info("Not enough keypoints detected.")
                return
            
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(self.des1, des2, k=2)
            
            good_matches = [m for m, n in matches if m.distance < 0.4 * n.distance]
            if len(good_matches) < 4:
                self.get_logger().info("Not enough good matches found.")
                return
            
            matched_image = cv2.drawMatches(self.template_image, self.kp1, cam_image, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Feature Matching", matched_image)
            cv2.waitKey(500)
            # cv2.destroyAllWindows()
            
            self.get_logger().info(f"Matched {len(good_matches)} keypoints!")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")


def main():
    rclpy.init()
    # template_path = "/home/sunwolee/Downloads/ext_orig.png"
    template_path = "/home/sunwolee/Downloads/man_orig.png"
    node = ImageMatcher(template_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
