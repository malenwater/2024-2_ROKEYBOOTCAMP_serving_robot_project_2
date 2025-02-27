import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry

class SIFT_PNP_Calculator(Node):
    def __init__(self, template_path):
        super().__init__('sift_pnp_calculator')
        self.bridge = CvBridge()
        
        # 템플릿 이미지 (3D 모델과 대응되는 이미지)
        self.template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        self.target_size = (500, 500)
        self.template_image = cv2.resize(self.template_image, self.target_size)
        
        self.K = None  # 카메라 내적 행렬
        self.D = None  # 카메라 왜곡 계수
        self.object_points = None  # 3D 모델의 객체 점 (예: 객체의 3D 좌표)
        
        # ROS2 구독자 설정
        self.image_sub = self.create_subscription(
            CompressedImage, '/camera/image_raw/compressed', self.image_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, 10)
        
        self.sift = cv2.SIFT_create()

    def info_callback(self, msg):
        self.K = np.array(msg.k).reshape((3, 3))
        self.D = np.array(msg.d[:5]).reshape((1, 5))

        # 3D 모델의 객체 점 (예시: 정육면체, 마커 등)
        # 객체의 3D 좌표를 예로 들면:
        self.object_points = np.array([
            [0.0, 0.0, 0.0],   # 객체의 (0, 0, 0)
            [1.0, 0.0, 0.0],   # 객체의 (1, 0, 0)
            [0.0, 1.0, 0.0],   # 객체의 (0, 1, 0)
            [1.0, 1.0, 0.0],   # 객체의 (1, 1, 0)
        ], dtype=np.float32)

    def image_callback(self, msg):
        try:
            # 이미지 변환
            cam_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            cam_image_gray = cv2.cvtColor(cam_image, cv2.COLOR_BGR2GRAY)
            cam_image_gray = cv2.resize(cam_image_gray, self.target_size)

            # SIFT 특징점 탐지
            kp1, des1 = self.sift.detectAndCompute(self.template_image, None)
            kp2, des2 = self.sift.detectAndCompute(cam_image_gray, None)
            
            # 매칭 (FLANN을 사용하여 특징점 매칭)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # 좋은 매칭만 선택 (비율 테스트)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) > 4:
                # 객체의 2D 점과 매칭된 3D 점 추출
                image_points = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32)
                
                # PnP 알고리즘을 사용하여 카메라의 위치 추정 (RANSAC 사용)
                success, rvec, tvec, inliers = cv2.solvePnPRansac(self.object_points, image_points, self.K, self.D)
                
                if success:
                    # 변환 행렬 계산 (R, T를 통해)
                    rot_mat, _ = cv2.Rodrigues(rvec)
                    transform_matrix = np.hstack((rot_mat, tvec))
                    transform_matrix = np.vstack((transform_matrix, np.array([0, 0, 0, 1])))
                    
                    self.get_logger().info(f"✅ Transformation Matrix: \n{transform_matrix}")
                    self.get_logger().info(f"Inliers: {inliers}")
                else:
                    self.get_logger().info("❌ PnP RANSAC failed.")
            else:
                self.get_logger().info("❌ Not enough good matches to compute PnP.")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main():
    rclpy.init()
    template_path = "/path/to/your/template_image.png"  # 템플릿 이미지 경로
    node = SIFT_PNP_Calculator(template_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
