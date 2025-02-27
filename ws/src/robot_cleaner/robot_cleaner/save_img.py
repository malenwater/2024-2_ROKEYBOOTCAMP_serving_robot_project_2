import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import os

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.image_sub = self.create_subscription(
            CompressedImage, 
            '/oakd/rgb/preview/image_raw/compressed', 
            self.image_callback, 
            10
        )
        
        # 저장할 디렉토리 설정
        self.save_path = "./images"
        os.makedirs(self.save_path, exist_ok=True)
        
        # 이미지 저장을 위한 카운터
        self.image_counter = 0
        self.get_logger().info(f"Saved image: {self.image_counter}")

    def image_callback(self, msg):
        # CompressedImage -> OpenCV 이미지 변환
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 파일명 생성 및 저장
        filename = os.path.join(self.save_path, f"image_{self.image_counter:04d}.jpg")
        cv2.imwrite(filename, image)
        self.get_logger().info(f"Saved image: {filename}")

        # 이미지 카운터 증가
        self.image_counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
