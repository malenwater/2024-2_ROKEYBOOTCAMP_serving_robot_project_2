'''
 MAP → BASE → CAMERA → IMG 좌표 변환을 하나의 파일에서 실행할 수 있도록 통합한 전체 코드입니다.
이 코드에서는 SIFT를 사용한 특징점 매칭, Homography 계산, 그리고 PnP를 이용한 3D-2D 변환을 수행합니다.
'''

import cv2
import numpy as np

# ---------------------- 1. 이미지 로드 및 SIFT 특징점 검출 ----------------------
img1 = cv2.imread("image_from_camera.jpg", cv2.IMREAD_GRAYSCALE)  # Camera에서 찍힌 이미지
img2 = cv2.imread("reference_image.jpg", cv2.IMREAD_GRAYSCALE)  # 기존 참조 이미지

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 특징점 매칭 (BFMatcher 사용)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 상위 20개 매칭된 특징점 시각화
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None)

cv2.imshow("Feature Matches", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------- 2. Homography 행렬 계산 ----------------------
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
print("Homography Matrix:\n", H)

# ---------------------- 3. PnP를 이용한 3D → 2D 변환 ----------------------
# 3D 좌표 (월드 좌표계) - 실제 물체의 위치
object_points = np.array([
    [100, 200, 0],  # X, Y, Z 좌표 (예제값)
    [150, 220, 0],
    [120, 250, 0],
    [130, 270, 0]
], dtype=np.float32)

# 2D 이미지 좌표 (카메라 이미지 내 대응점)
image_points = np.array([
    [320, 240],  # u, v 좌표 (예제값)
    [340, 260],
    [360, 280],
    [380, 300]
], dtype=np.float32)

# Camera 내부 파라미터 (fx, fy, cx, cy 설정 필요)
camera_matrix = np.array([
    [800, 0, 320],  # fx, 0, cx
    [0, 800, 240],  # 0, fy, cy
    [0,  0,   1]
], dtype=np.float32)

dist_coeffs = np.zeros((4,1))  # 왜곡 계수

# PnP로 3D-2D 변환을 위한 Rotation, Translation 벡터 계산
success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
R_matrix, _ = cv2.Rodrigues(rvec)

print("Rotation Matrix:\n", R_matrix)
print("Translation Vector:\n", tvec)

# ---------------------- 4. MAP → BASE → CAMERA → IMG 변환 ----------------------
# (가정) MAP → BASE 변환 행렬
T_map_to_base = np.array([
    [1, 0, 0, 500],  # X축 이동
    [0, 1, 0, 300],  # Y축 이동
    [0, 0, 1, 0],    # Z축 이동
    [0, 0, 0, 1]
], dtype=np.float32)

# (가정) BASE → CAMERA 변환 행렬
T_base_to_camera = np.array([
    [1, 0, 0, 100],  # X축 이동
    [0, 1, 0, 50],   # Y축 이동
    [0, 0, 1, 20],   # Z축 이동
    [0, 0, 0, 1]
], dtype=np.float32)

# 변환 대상 물체의 MAP 좌표
P_map = np.array([200, 300, 0, 1], dtype=np.float32)  # Homogeneous 좌표계 사용

# 변환 적용
P_base = np.dot(T_map_to_base, P_map)  # MAP → BASE
P_camera = np.dot(T_base_to_camera, P_base)  # BASE → CAMERA

# CAMERA 좌표 → 이미지 픽셀 좌표 변환
P_img = np.dot(H, np.dot(R_matrix, P_camera[:3]) + tvec)
P_img /= P_img[-1]  # Homogeneous 좌표 정규화

print("Final Image Coordinates (u, v):", P_img[:2])
