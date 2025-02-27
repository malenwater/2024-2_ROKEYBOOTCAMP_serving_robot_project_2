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
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.kp1, self.des1 = self.sift.detectAndCompute(self.ori_img, None)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # 모든 pt에 대해 변환된 값을 리스트로 저장
        self.transformed_pts = [(kp.pt[0] / self.EXT_PIXEL * self.EXT, kp.pt[1] / self.EXT_PIXEL * self.EXT,0) for kp in self.kp1]
        print(self.transformed_pts[0])
        print(self.kp1[0].pt)
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


        matches = self.flann.knnMatch(self.des1, self.des2, k=2)
        # print(matches)
        # print(len(matches))
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
        # print(len(object_points))
        # print(len(image_points))
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, CAMERA_K, CAMERA_D)
        if success:
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()

            print("✅ Rotation Vector (rvec):\n", rvec)
            print("✅ Translation Vector (tvec):\n", tvec)
            print("✅ Transformation Matrix (T):\n", T)
            self.result_img = T


template_image = cv2.imread("/home/sunwolee/Downloads/ext_orig.png", cv2.IMREAD_GRAYSCALE)
cam_image = cv2.imread('/home/sunwolee/Downloads/20250226_194848.jpg', cv2.IMREAD_GRAYSCALE)
detector = SIFTDetector(template_image, cam_image, types=1)










