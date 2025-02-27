import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry

BASELINK_TO_CAMERA = np.array([
    [0.000, 0.000, 1.000, -0.059],
    [-1.000, 0.000, 0.000, 0.000],
    [0.000, -1.000, 0.000, 0.149],
    [0.000, 0.000, 0.000, 1.000]
])

class SIFT_PNP_Calculator(Node):
    def __init__(self, template_path):
        super().__init__('sift_pnp_calculator')
        self.bridge = CvBridge()
        
        # 템플릿 이미지 로딩
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
        
        # TF 리스너 설정
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # RViz2로 pose를 전송하기 위한 퍼블리셔 설정
        self.pose_pub = self.create_publisher(PoseStamped, '/image_pose', 10)
        
        self.sift = cv2.SIFT_create()

    def info_callback(self, msg):
        self.K = np.array(msg.k).reshape((3, 3))
        self.D = np.array(msg.d[:5]).reshape((1, 5))

        # 3D 모델의 객체 점 (예시: 정육면체, 마커 등)
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

                    # 카메라 좌표계를 로봇의 base_link 좌표계로 변환
                    camera_to_base_link = np.dot(BASELINK_TO_CAMERA, transform_matrix)

                    # 맵 좌표계와 로봇의 base_link 좌표계 사이의 변환을 TF로 가져오기
                    try:
                        transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
                        base_to_map = np.array([
                            [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.translation.x],
                            [transform.transform.rotation.y, transform.transform.rotation.x, transform.transform.rotation.z, transform.transform.translation.y],
                            [transform.transform.rotation.z, transform.transform.rotation.z, transform.transform.rotation.x, transform.transform.translation.z],
                            [0, 0, 0, 1]
                        ])

                        # 최종적으로 맵 좌표계에서 객체의 pose 계산
                        map_coordinates = np.dot(base_to_map, camera_to_base_link)
                        
                        # RViz에 표시할 pose 메시지 생성
                        pose_msg = PoseStamped()
                        pose_msg.header.stamp = self.get_clock().now().to_msg()
                        pose_msg.header.frame_id = 'map'
                        pose_msg.pose.position.x = map_coordinates[0, 3]
                        pose_msg.pose.position.y = map_coordinates[1, 3]
                        pose_msg.pose.position.z = map_coordinates[2, 3]

                        # 회전 행렬을 쿼터니언으로 변환
                        q = self.euler_to_quaternion(map_coordinates[:3, :3])
                        pose_msg.pose.orientation.x = q[0]
                        pose_msg.pose.orientation.y = q[1]
                        pose_msg.pose.orientation.z = q[2]
                        pose_msg.pose.orientation.w = q[3]

                        self.pose_pub.publish(pose_msg)
                        self.get_logger().info(f"Pose published to RViz: {pose_msg.pose}")

                    except Exception as e:
                        self.get_logger().error(f"Error in TF lookup: {e}")
                else:
                    self.get_logger().info("❌ PnP RANSAC failed.")
            else:
                self.get_logger().info("❌ Not enough good matches to compute PnP.")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def euler_to_quaternion(self, rotation_matrix):
        """회전 행렬을 쿼터니언으로 변환"""
        tr = np.trace(rotation_matrix)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2  # S=4*qw
            qw = 0.25 * S
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
        else:
            i = 0
            if rotation_matrix[1, 1] > rotation_matrix[0, 0]:
                i = 1
            if rotation_matrix[2, 2] > rotation_matrix[i, i]:
                i = 2
            j = (i + 1) % 3
            k = (i + 2) % 3
            S = np.sqrt(rotation_matrix[i, i] - rotation_matrix[j, j] - rotation_matrix[k, k] + 1.0) * 2
            qw = (rotation_matrix[k, j] - rotation_matrix[j, k]) / S
            qx = 0.25 * S
            qy = (rotation_matrix[j, i] + rotation_matrix[i, j]) / S
            qz = (rotation_matrix[k, i] + rotation_matrix[i, k]) / S
        return [qx, qy, qz, qw]

def main():
    rclpy.init()
    template_path = "/path/to/your/template_image.png"  # 템플릿 이미지 경로
    node = SIFT_PNP_Calculator(template_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
