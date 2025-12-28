import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = cap.read()

    if not success:
        print("无法读取摄像头画面")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # 修正：使用FACEMESH_TESSELATION或FACEMESH_CONTOURS
            # 方法1：只绘制脸部轮廓
            # mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
            #                       drawSpec, drawSpec)

            # 方法2：绘制完整的脸部网格（包括三角形）
            mpDraw.draw_landmarks(
                img,
                faceLms,
                mpFaceMesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawSpec,
                connection_drawing_spec=drawSpec
            )

            # 方法3：同时绘制轮廓和网格
            # mpDraw.draw_landmarks(
            #     img,
            #     faceLms,
            #     mpFaceMesh.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=drawSpec,
            #     connection_drawing_spec=drawSpec
            # )
            # mpDraw.draw_landmarks(
            #     img,
            #     faceLms,
            #     mpFaceMesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=drawSpec
            # )

            # 打印每个特征点的坐标
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                # 可以限制打印的数量，避免输出太多
                if id % 50 == 0:  # 每50个点打印一次
                    print(f"点 {id}: ({x}, {y})")

    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Face Mesh", img)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()