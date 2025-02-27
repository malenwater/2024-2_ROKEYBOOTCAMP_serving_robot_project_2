import cv2
import numpy as np

# === 1. 포스터의 미리 정의된 3D 월드 좌표 (m 단위) ===
object_points = np.array([
    [0.0, 0.0, 0.0],   # 포스터 왼쪽 상단 (기준점)
    [0.5, 0.0, 0.0],   # 포스터 오른쪽 상단
    [0.0, 0.7, 0.0],   # 포스터 왼쪽 하단
    [0.25, 0.35, 0.0], # 포스터 중앙 특정 패턴
    [0.1, 0.2, 0.0],   # 내부 SIFT 특징점 1
    [0.3, 0.5, 0.0]    # 내부 SIFT 특징점 2
], dtype=np.float32)

# === 2. SIFT 특징점 검출 ===
def extract_sift_features(image):
    if image is None:
        print("❌ 이미지 로드 실패! 파일 경로를 확인하세요.")
        return None, None
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# 기준 포스터 이미지 로드
poster_img = cv2.imread('/home/minho/week7_ws/KDT/KDT/poster_reference.png', cv2.IMREAD_GRAYSCALE)
if poster_img is None:
    print("❌ 포스터 이미지 파일을 찾을 수 없습니다. 'poster_reference.jpg' 경로 확인!")
    exit()

poster_kp, poster_des = extract_sift_features(poster_img)

# === 3. 실시간 웹캠 캡처 ===
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ 웹캠에서 프레임을 가져오지 못했습니다.")
        break

    # 웹캠 프레임을 흑백 이미지로 변환
    camera_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    camera_kp, camera_des = extract_sift_features(camera_img)

    if camera_des is not None and poster_des is not None:
        # === 4. 특징점 매칭 (KNN 매칭) ===
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(poster_des, camera_des, k=2)

        # 좋은 매칭점 선별 (Lowe's Ratio Test)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.85 * n.distance:
                good_matches.append(m)

        print(f"🔹 검출된 매칭된 특징점 개수: {len(good_matches)}")

        # object_points 개수(6개)에 맞춰 image_points도 6개만 선택
        if len(good_matches) >= len(object_points):
            selected_matches = good_matches[:len(object_points)]  # 상위 6개만 선택
            image_points = np.array([camera_kp[m.trainIdx].pt for m in selected_matches], dtype=np.float32)

            print(f"🔹 solvePnP 실행 - 대응점 개수: {len(image_points)}")

            # === 5. PnP 수행하여 변환 행렬(R, t) 계산 ===
            K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.zeros(4)

            success, rvec, tvec = cv2.solvePnP(object_points, image_points, K, dist_coeffs)

            if success:
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = tvec.flatten()

                print("✅ Rotation Vector (rvec):\n", rvec)
                print("✅ Translation Vector (tvec):\n", tvec)
                print("✅ Transformation Matrix (T):\n", T)

            # === 6. 매칭된 특징점 시각화 ===
            matched_img = cv2.drawMatches(poster_img, poster_kp, frame, camera_kp, good_matches, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Feature Matching", matched_img)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()