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
from visualization_msgs.msg import Marker
import threading
from geometry_msgs.msg import Point
BASELINK_TO_CAMERA = np.array([ 
    [0.000, 0.000, 1.000, -0.060],
    [-1.000, 0.000, 0.000, 0.000],
    [0.000, -1.000, 0.000, 0.244],
    [0.000, 0.000, 0.000, 1.000]
])
  
# CAMERA_K =np.array([[202.39749146,   0.,         125.49773407],
#             [  0.,         202.39749146, 125.75233459],
#             [  0.,           0.,           1.        ]]) 
# CAMERA_D = np.array([[-3.51905060e+00, -2.84767342e+01, -3.02788394e-04,  1.01520610e-03,
#         2.35221481e+02, -3.68542147e+00, -2.67263298e+01,  2.28351166e+02]]) 

class SIFTDetector():
    def __init__(self, ori_img, cap_img, types: int, CAMERA_K, CAMERA_D):
        self.ori_img = ori_img
        self.cap_img = cap_img
        self.result = False
        self.result_img = None
        self.types = types
        self.EXT = 0.18
        self.EXT_PIXEL = 680
        
        self.MAN_HEIGHT = 0.18
        self.MAN_WIGHT = 0.23
        self.MAN_HEIGHT_PIXEL = 680
        self.MAN_WIGHT_PIXEL = 869
        self.sift = cv2.SIFT_create()
        
        # print(ori_img.shape)
        # print(cap_img.shape)
        
        self.kp1, self.des1 = self.sift.detectAndCompute(self.ori_img, None)
        self.CAMERA_K = CAMERA_K
        self.CAMERA_D = CAMERA_D
        # 모든 pt에 대해 변환된 값을 리스트로 저장
        # self.transformed_pts = [(kp.pt[0] / self.EXT_PIXEL * self.EXT, kp.pt[1] / self.EXT_PIXEL * self.EXT, 0) for kp in self.kp1]
        self.transformed_pts = [(kp.pt[0] / self.MAN_WIGHT_PIXEL * self.MAN_WIGHT, kp.pt[1] / self.MAN_HEIGHT_PIXEL * self.MAN_HEIGHT, 0) for kp in self.kp1]
        # print(f'self.cap_img.shape : {self.cap_img.shape}')

        # 변환된 pt 리스트 출력
        # print(self.transformed_pts)
        # print(len(self.transformed_pts))
        # print()
        # print(self.des1.shape)
        self.detect()

    def detect(self):
        if self.CAMERA_K is None:
            return
        self.kp2, self.des2 = self.sift.detectAndCompute(self.cap_img, None)

        if self.des1 is None or self.des2 is None or len(self.kp1) < 2 or len(self.kp2) < 2:
            return

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(self.des1, self.des2, k=2)
        # print(matches)
        # print(len(matches))
        # good_matches에서 매칭된 각 특징점에 대해 transformed_pts와 kp2의 좌표 짝지기
        object_points = []
        good_matches = []
        image_points = []
        for match, n in matches:
            # match의 첫 번째 요소(m)는 self.kp1에서, 두 번째 요소(n)는 self.kp2에서 매칭된 특징점
            # self.transformed_pts는 self.kp1에서 추출된 좌표들로부터 계산된 변환된 좌표들
            if match.distance < 0.4 * n.distance:
                pt1 = self.transformed_pts[match.queryIdx]  # self.kp1에서 매칭된 transformed_pts
                pt2 = self.kp2[match.trainIdx].pt         # self.kp2에서 매칭된 원본 이미지의 좌표
                good_matches.append(match)
                # matched_pts에 변환된 pt1과 원본 pt2 좌표를 추가
                object_points.append(pt1)
                image_points.append(pt2)
        # print(f'good_matches {good_matches}')
        if len(good_matches) < 4 :
            return
        self.result = True
        
        object_points = np.array(object_points)
        image_points = np.array(image_points)
        # print((object_points))
        # print()
        # print((image_points))
        # print(len(object_points))
        # print(len(image_points))
        # print(f'object_points : {object_points[:10]}')
        # print(f'image_points : {image_points[:10]}')
        # print(f'camera K : {self.CAMERA_K}')
        # print(f'camera K : {self.CAMERA_D}')
        initial_rotation = np.array([0, 0, 0], dtype=np.float64)  # 회전 벡터
        initial_translation = np.array([0, 0.3, 0.6], dtype=np.float64)  # 변환 벡터 (z축으로 0.35m)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, self.CAMERA_K, self.CAMERA_D,
                                                        # useExtrinsicGuess=True, 
                                                        # rvec=initial_rotation, 
                                                        # tvec=initial_translation
                                                        )
        # success, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, self.CAMERA_K, self.CAMERA_D,reprojectionError=2.0,iterationsCount=1000)
        if success:
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()

            # print("✅ Rotation Vector (rvec):\n", rvec)
            # print("✅ Translation Vector (tvec):\n", tvec)
            # print("✅ Transformation Matrix (T):\n", T)
            z = (T[:, 3].reshape(4, 1))
            # z[2] = z[2] * 2
            self.result_img =  z
            
            # 🔹 특징점 매칭 이미지 생성 및 출력
            # matched_image = cv2.drawMatches(self.ori_img, self.kp1, self.cap_img, self.kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # cv2.imshow("Feature Matching", matched_image)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()
            
        
class ImageSubscriber(Node):
    def __init__(self, template_path):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        self.detector = None
        self.K = None
        self.D = None
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
        self.publisher = self.create_publisher(Marker, 'visualization_marker', 10)
        self.timer = self.create_timer(1.0, self.publish_marker)  # 1초마다 실행
        self.map_coords = None
        self.K = None  # 카메라 내적 행렬
        self.D = None  # 왜곡 계수
        self.tf_map_camera = None

    def pose_callback(self, msg):
        # quaternion 값을 받아옴
        # self.get_logger().info(f'스타틍')
        translation = msg.pose.pose.position
        rotation = msg.pose.pose.orientation
        map_to_baselink = np.array([
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
        map_to_baselink[:3, :3] = rotation_matrix_3x3 
        # self.get_logger().info(f'Rotation Matrix (4x4):\n{map_to_baselink}')
        self.map_coords = None
        self.tf_map_camera = np.dot(map_to_baselink, BASELINK_TO_CAMERA)

    def info_callback(self, msg):
        self.K = np.array(msg.k).reshape((3,3))
        self.D = np.array(msg.d).reshape((1,8))
        # print(f"self.D {self.D} , self.K { self.K}")


    def image_callback(self, msg):
        try:
            self.get_logger().info(f"Received image with format: {msg.format}")
            cam_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2GRAY)
            self.detector = SIFTDetector(self.template_image, cam_image, 1, self.K, self.D)
            # print(f"man man 로봇 좌표계: {self.tf_map_camera}")
            # print(f"man man 로봇 감지: {self.detector.result}")
            print(f"man man 카메라로부터 이미지 거리: \n {self.detector.result_img}")
            
            if self.detector.result and self.tf_map_camera is not None:
                # print("✅ Object detected and mapped!")
                # print(f"TF Map to Camera: \n{self.tf_map_camera}")
                self.map_coords = np.dot(self.tf_map_camera, self.detector.result_img)
                if self.map_coords is not None:
                    pass
                    # print(f"🌍 Object in Map Coordinates: \n{self.map_coords}")
            else:
                print("❌ No valid detection or missing TF data.")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
            

    
    def publish_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"  # RViz2의 기준 좌표계
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "my_namespace"
        marker.id = 0
        marker.type = Marker.POINTS  # 포인트 타입 마커
        marker.action = Marker.ADD

        # 마커 색상 및 크기 설정
        marker.scale.x = 0.2  # 포인트 크기
        marker.scale.y = 0.2
        marker.color.a = 1.0  # 투명도
        marker.color.r = 0.0  # 빨간색
        marker.color.g = 1.0
        marker.color.b = 0.0
        if self.map_coords is None:
            return
        # 마커 위치 추가
        point = Point()
        # point.x = self.map_coords[0][3]
        # point.y = self.map_coords[1][3]
        # point.z = self.map_coords[2][3]
        point.x = self.map_coords[0][0]
        point.y = self.map_coords[1][0]
        point.z = self.map_coords[2][0]
        marker.points.append(point)

        self.publisher.publish(marker)
        # self.get_logger().info(f'Published marker at {point.x} {point.y} {point.z}')

def main():
    rclpy.init()
    # template_path = "/home/sunwolee/Downloads/ext_orig.png"
    template_path = "/home/sunwolee/Downloads/man_orig.png"
    node = ImageSubscriber(template_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
