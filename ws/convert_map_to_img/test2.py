import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage  
from nav_msgs.msg import Odometry

'''

'''
# ** <- 문제 
# tf_static을 1회 가져와서 저장해놓음 ! 
# tf2_view_frame
BASELINK_TO_CAMERA = np.array([ 
    [0.000, 1.000, 0.000, 0.000],
    [-1.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 1.000, 0.244],
    [0.000, 0.000, 0.000, 1.000]
])

# sift기반 객체 탐지
# ori_img - cap_img 간 특징점 매칭을 수행함.
class SIFTDetector():
    def __init__(self, ori_img, cap_img, types: int):
        self.ori_img = ori_img
        self.cap_img = cap_img
        self.result = False
        self.types = types
        self.detect()

    def detect(self):
        self.sift = cv2.SIFT_create()
        # detectAndCompute_ 두 이미지에서 특징점, 설명자를 추가한다.
        self.kp1, self.des1 = self.sift.detectAndCompute(self.ori_img, None)
        self.kp2, self.des2 = self.sift.detectAndCompute(self.cap_img, None)

        if self.des1 is None or self.des2 is None or len(self.kp1) < 2 or len(self.kp2) < 2:
            return
        # 특징점을 호출, 매칭개수가 n을 넘으면 객체 탐지 성공
        self.good_matches = self.match_features(self.des1, self.des2)
        n = 65 if self.types == 1 else 100 if self.types == 2 else 0

        if len(self.good_matches) > n:
            self.result = True

    def match_features(self, des1, des2, threshold=0.85):
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        # fann을 이용하여, 두이미지의 특징점을 매칭한다
        # 최근접 이웃 매칭 수행하여 유사도 매칭점 반환
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
        self.tf_sub = self.create_subscription(TFMessage, '/tf', self.tf_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.K = None  # 카메라 내적 행렬
        self.D = None  # 왜곡 계수
        self.tf_map_camera = None


    # tf 데이터 수신
    def tf_callback(self, msg):
        map_to_odom = None 
        for transform in msg.transforms:
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            map_to_odom = np.array([
                [rotation.x, rotation.y, rotation.z, translation.x],
                [rotation.y, rotation.x, rotation.z, translation.y],
                [rotation.z, rotation.z, rotation.x, translation.z],
                [0, 0, 0, 1]
            ])
        if map_to_odom is not None:
            self.tf_map_camera = np.dot(map_to_odom, BASELINK_TO_CAMERA)

    # odom 데이터를 받아 odom -> base_link변환 수행
    # tf_map_camera에 반영
    def odom_callback(self, msg):
        translation = msg.pose.pose.position
        rotation = msg.pose.pose.orientation
        odom_to_base_link = np.array([
            [rotation.x, rotation.y, rotation.z, translation.x],
            [rotation.y, rotation.x, rotation.z, translation.y],
            [rotation.z, rotation.z, rotation.x, translation.z],
            [0, 0, 0, 1]
        ])
        if self.tf_map_camera is not None:
            self.tf_map_camera = np.dot(self.tf_map_camera, odom_to_base_link)

    def info_callback(self, msg):
        self.K = np.array(msg.k).reshape((3,3))
        self.D = np.array(msg.d[:5]).reshape((1,5))

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
                #print(f"TF Map to Camera: \n{self.tf_map_camera}")
                
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
    template_path = "/home/jsy/week7/template.png"
    node = ImageSubscriber(template_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()