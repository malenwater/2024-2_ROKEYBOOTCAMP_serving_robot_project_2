import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

class PnPProcessor(Node):
    def __init__(self):
        super().__init__('pnp_processor')
        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        
        # 카메라 이미지 및 오도메트리 구독
        self.image_sub = self.create_subscription(
            CompressedImage, '/oakd/rgb/preview/image_raw/compressed', self.image_callback, 10)
        self.odom_subscription = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile)

        self.bridge = CvBridge()
        self.robot_position = (0.0, 0.0, 0.0)  # 초기 로봇 위치 (x, y, z)

        # 실제 객체 이미지 파일 로드
        self.real_image_paths = [
            # "/home/jsy/week7/img/ext_orig.png",
            # "/home/jsy/week7/img/human_orig.png",
            "/home/kante/Downloads/poster_reference.png"
        ]

        # SIFT 생성
        self.sift = cv2.SIFT_create()
        self.real_kp_des = self.load_and_process_real_images(self.real_image_paths)

        # BFMatcher 생성
        self.bf = cv2.BFMatcher()

        # 카메라 내부 파라미터
        self.K = np.array([
            [202.6661376953125, 0.0, 123.86566162109375],
            [0.0, 202.6661376953125, 124.75257873535156],
            [0.0, 0.0, 1.0]
        ])
        self.D = np.array([
            -2.6201, -38.5899, -0.0011, 0.0002,
            262.2898, -2.7987, -36.7088, 255.1872
        ])

    def load_and_process_real_images(self, paths):
        keypoints_descriptors = {}
        for path in paths:
            real_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if real_img is None:
                self.get_logger().error(f"Failed to load real image: {path}")
                continue
            kp, des = self.sift.detectAndCompute(real_img, None)
            keypoints_descriptors[path] = (kp, des)
        return keypoints_descriptors

    def odom_callback(self, msg):
        self.robot_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z  # Z 좌표 포함
        )

    def image_callback(self, msg):
        start_time = time.time()
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 카메라 왜곡 보정
            image_size = (gray_frame.shape[1], gray_frame.shape[0])
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, image_size, 1, image_size)
            undistorted_image = cv2.undistort(gray_frame, self.K, self.D, None, new_camera_matrix)

            # SIFT 특징 추출
            kp2, des2 = self.sift.detectAndCompute(undistorted_image, None)

            for real_path, (kp1, des1) in self.real_kp_des.items():
                matches = self.bf.knnMatch(des1, des2, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

                if len(good_matches) < 10:
                    continue

                obj_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                img_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # 왜곡 보정된 2D 포인트 변환
                undistorted_img_pts = cv2.undistortPoints(img_pts, self.K, self.D, P=new_camera_matrix)

                # 실제 물체의 크기 설정 (미터 단위)
                real_sizes = {"human": (0.18, 0.23), "ext": (0.18, 0.18)}
                object_type = "human" if "human" in real_path else "ext"

                obj_pts_3d = np.hstack((obj_pts.reshape(-1, 2), np.zeros((obj_pts.shape[0], 1)))).astype(np.float32)

                # PnP 적용
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_pts_3d, undistorted_img_pts, new_camera_matrix, None,
                    flags=cv2.SOLVEPNP_AP3P, confidence=0.99, reprojectionError=8.0
                )

                if not success:
                    continue

                R, _ = cv2.Rodrigues(rvec)

                # Z 축 기준 변환 적용
                world_coords = np.dot(R, obj_pts_3d.T).T + tvec.T
                world_coords = world_coords.mean(axis=0)  # 평균 좌표 계산

                adjusted_coords = (
                    world_coords[0],  # X
                    world_coords[1],  # Y
                    world_coords[2]   # Z
                )

                distance = np.linalg.norm(np.array(adjusted_coords) - np.array(self.robot_position))
                if distance >= 5:
                    continue

                # ROS 메시지 전송
                #publisher_pose = self.create_publisher(String, '/imagepose', qos_profile)
                publisher_pose = self.create_publisher(String, '/imagepose', 10)
                pose_message = f"{object_type},x:{adjusted_coords[0]:.2f},y:{adjusted_coords[1]:.2f},z:{adjusted_coords[2]:.2f}"
                publisher_pose.publish(String(data=pose_message))
                self.get_logger().info(f"Published Pose: {pose_message}")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

        finally:
            elapsed_time = time.time() - start_time
            time.sleep(max(0.5 - elapsed_time, 0))

def main(args=None):
    rclpy.init(args=args)
    pnp_processor = PnPProcessor()

    try:
        rclpy.spin(pnp_processor)
    except KeyboardInterrupt:
        pass

    pnp_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
