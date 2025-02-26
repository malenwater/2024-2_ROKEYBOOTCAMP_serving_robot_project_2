import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage  
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np

BASELINK_TO_CAMERA = np.array([ 
    [0.000, 0.000, 1.000, -0.060],
    [-1.000, 0.000, 0.000, 0.000],
    [0.000, -1.000, 0.000, 0.244],
    [0.000, 0.000, 0.000, 1.000]
])
CAMERA_K = [[202.39749146,   0.,         125.49773407],
            [  0.,         202.39749146, 125.75233459],
            [  0.,           0.,           1.        ]]
CAMERA_D = [[-3.51905060e+00, -2.84767342e+01, -3.02788394e-04,  1.01520610e-03,
        2.35221481e+02, -3.68542147e+00, -2.67263298e+01,  2.28351166e+02]]

class SIFTDetector():
    def __init__(self, ori_img, cap_img, types: int):
        self.ori_img = ori_img
        self.cap_img = cap_img
        self.result = False
        self.types = types
        self.sift = cv2.SIFT_create()
        self.kp1, self.des1 = self.sift.detectAndCompute(self.ori_img, None)
        self.detect()

    def detect(self):
        self.kp2, self.des2 = self.sift.detectAndCompute(self.cap_img, None)

        if self.des1 is None or self.des2 is None or len(self.kp1) < 2 or len(self.kp2) < 2:
            return

        self.good_matches = self.match_features(self.des1, self.des2)
        n = 65 if self.types == 1 else 100 if self.types == 2 else 0

        if len(self.good_matches) > n:
            self.result = True

    def match_features(self, des1, des2, threshold=0.85):
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        return [m for m, n in matches if m.distance < threshold * n.distance]

class ImageSubscriber(Node):
    def __init__(self, template_path):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        self.target_size = (500, 500)
        self.template_image = cv2.resize(self.template_image, self.target_size)
        self.detector = None

        self.image_sub = self.create_subscription(
            CompressedImage, '/oakd/rgb/preview/image_raw/compressed', self.image_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/oakd/rgb/preview/camera_info', self.info_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/pose',
            self.pose_callback,
            10
        )
        # self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.K = None  # 카메라 내적 행렬
        self.D = None  # 왜곡 계수
        self.tf_map_camera = None

    def pose_callback(self, msg):
        # quaternion 값을 받아옴
        self.get_logger().info(f'스타틍')
        translation = msg.pose.pose.position
        rotation = msg.pose.pose.orientation
        map_to_odom = np.array([
                [rotation.x, rotation.y, rotation.z, translation.x],
                [rotation.y, rotation.x, rotation.z, translation.y],
                [rotation.z, rotation.z, rotation.x, translation.z],
                [0, 0, 0, 1]
            ])
        # 변환된 회전 행렬을 로그로 출력
        quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
        # scipy의 Rotation 클래스를 이용하여 quaternion을 회전 행렬로 변환
        r = R.from_quat(quaternion)
        rotation_matrix_3x3 = r.as_matrix()
        map_to_odom[:3, :3] = rotation_matrix_3x3 
        self.get_logger().info(f'Rotation Matrix (4x4):\n{map_to_odom}')
        
        self.tf_map_camera = np.dot(map_to_odom, BASELINK_TO_CAMERA)
            

    # def odom_callback(self, msg):
    #     translation = msg.pose.pose.position
    #     rotation = msg.pose.pose.orientation
    #     odom_to_base_link = np.array([
    #         [rotation.x, rotation.y, rotation.z, translation.x],
    #         [rotation.y, rotation.x, rotation.z, translation.y],
    #         [rotation.z, rotation.z, rotation.x, translation.z],
    #         [0, 0, 0, 1]
    #     ])
    #     if self.tf_map_camera is not None:
    #         self.tf_map_camera = np.dot(self.tf_map_camera, odom_to_base_link)
    #         self.get_logger().info(f"")

    def info_callback(self, msg):
        self.K = np.array(msg.k).reshape((3,3))
        self.D = np.array(msg.d).reshape((1,8))
        print(f"self.D {self.D} , self.K { self.K}")

    def convert_to_map_coordinates(self, pixel_x, pixel_y, depth=1.0):
        if self.K is None or self.tf_map_camera is None:
            return None

        inv_K = np.linalg.inv(self.K)
        pixel_coords = np.array([pixel_x, pixel_y, 1])
        camera_coords = depth * inv_K @ pixel_coords
        camera_coords = np.append(camera_coords, 1)
        map_coords = self.tf_map_camera @ camera_coords
        return map_coords[:3]

    def image_callback(self, msg):
        try:
            cam_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2GRAY)
            cam_image = cv2.resize(cam_image, self.target_size)
            self.detector = SIFTDetector(self.template_image, cam_image, types=1)
            
            if self.detector.result and self.tf_map_camera is not None:
                print("✅ Object detected and mapped!")
                print(f"TF Map to Camera: \n{self.tf_map_camera}")
                
                detected_x, detected_y = 250, 250  # 임시 객체 위치 (중앙)
                map_coords = self.convert_to_map_coordinates(detected_x, detected_y, depth=1.0)
                if map_coords is not None:
                    print(f"🌍 Object in Map Coordinates: {map_coords}")
            else:
                print("❌ No valid detection or missing TF data.")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")


def main():
    rclpy.init()
    template_path = "/home/sunwolee/Downloads/ext_orig.png"
    node = ImageSubscriber(template_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
