import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    success, img = cap.read()

    if not success:
        print("无法读取摄像头画面")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # 使用draw_detection方法（可选，如果你想要MediaPipe的默认绘制）
            # mpDraw.draw_detection(img, detection)

            # 获取边界框信息
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape

            # 修正bbox构建：移除多余的逗号
            # 错误示例：bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),
            # 正确应该是：
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)

            # 或者更清晰地使用元组：
            # bbox = (
            #     int(bboxC.xmin * iw),
            #     int(bboxC.ymin * ih),
            #     int(bboxC.width * iw),
            #     int(bboxC.height * ih)
            # )

            # 绘制矩形
            cv2.rectangle(img, (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          (255, 0, 255), 2)

            # 添加置信度文本
            cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 2)

    cv2.imshow("Image", img)

    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()