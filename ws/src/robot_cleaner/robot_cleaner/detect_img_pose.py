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
CAMERA_K =np.array([[202.39749146,   0.,         125.49773407],
            [  0.,         202.39749146, 125.75233459],
            [  0.,           0.,           1.        ]]) 
CAMERA_D = np.array([[-3.51905060e+00, -2.84767342e+01, -3.02788394e-04,  1.01520610e-03,
        2.35221481e+02, -3.68542147e+00, -2.67263298e+01,  2.28351166e+02]]) 

class SIFTDetector():
    def __init__(self, ori_img, cap_img, types: int):
        self.ori_img = ori_img
        self.cap_img = cap_img
        self.result = False
        self.result_img = None
        self.types = types
        self.EXT = 0.18
        self.EXT_PIXEL = 680
        self.sift = cv2.SIFT_create()
        self.kp1, self.des1 = self.sift.detectAndCompute(self.ori_img, None)
        
        # 모든 pt에 대해 변환된 값을 리스트로 저장
        self.transformed_pts = [(kp.pt[0] / self.EXT_PIXEL * self.EXT, kp.pt[1] / self.EXT_PIXEL * self.EXT,0) for kp in self.kp1]

        # 변환된 pt 리스트 출력
        # print(self.transformed_pts)
        print(len(self.transformed_pts))
        # print()
        # print(self.des1.shape)
        self.detect()

    def detect(self):
        self.kp2, self.des2 = self.sift.detectAndCompute(self.cap_img, None)

        if self.des1 is None or self.des2 is None or len(self.kp1) < 2 or len(self.kp2) < 2:
            return

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(self.des1, self.des2, k=2)
        # print(matches)
        print(len(matches))
        if len(matches) < 4 :
            return
        self.result = True
        # good_matches에서 매칭된 각 특징점에 대해 transformed_pts와 kp2의 좌표 짝지기
        object_points = []
        image_points = []
        for match, n in matches:
            # match의 첫 번째 요소(m)는 self.kp1에서, 두 번째 요소(n)는 self.kp2에서 매칭된 특징점
            # self.transformed_pts는 self.kp1에서 추출된 좌표들로부터 계산된 변환된 좌표들
            pt1 = self.transformed_pts[match.queryIdx]  # self.kp1에서 매칭된 transformed_pts
            pt2 = self.kp2[match.trainIdx].pt         # self.kp2에서 매칭된 원본 이미지의 좌표
            
            # matched_pts에 변환된 pt1과 원본 pt2 좌표를 추가
            object_points.append(pt1)
            image_points.append(pt2)
        object_points = np.array(object_points)
        image_points = np.array(image_points)
        # print((object_points))
        # print()
        # print((image_points))
        print(len(object_points))
        print(len(image_points))
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, CAMERA_K, CAMERA_D, flags=cv2.SOLVEPNP_ITERATIVE)
        if success:
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()

            print("✅ Rotation Vector (rvec):\n", rvec)
            print("✅ Translation Vector (tvec):\n", tvec)
            print("✅ Transformation Matrix (T):\n", T)
            self.result_img =   T[:, 3].reshape(4, 1) 

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

    def info_callback(self, msg):
        self.K = np.array(msg.k).reshape((3,3))
        self.D = np.array(msg.d).reshape((1,8))
        # print(f"self.D {self.D} , self.K { self.K}")

    def image_callback(self, msg):
        try:
            cam_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2GRAY)
            cam_image = cv2.resize(cam_image, self.target_size)
            self.detector = SIFTDetector(self.template_image, cam_image, types=1)
            print(f"man man man: {self.tf_map_camera}")
            print(f"man man man: {self.detector.result_img}")
            
            if self.detector.result and self.tf_map_camera is not None:
                print("✅ Object detected and mapped!")
                print(f"TF Map to Camera: \n{self.tf_map_camera}")
                map_coords = np.dot(self.tf_map_camera, self.detector.result_img)
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
