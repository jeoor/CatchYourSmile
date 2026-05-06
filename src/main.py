"""CatchYourSmile — AI-powered gesture and smile detection camera."""

import cv2
import os
import time

from camera import SmartCameraUltimate


def on_mouse_click(event, x, y, flags, param):
    """Open captured photo when clicking the preview thumbnail."""
    detector = param['det']
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = detector.preview_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            if detector.last_photo_path and os.path.exists(detector.last_photo_path):
                try:
                    os.startfile(os.path.abspath(detector.last_photo_path))
                    detector.add_status_message(
                        f"已打开照片: {os.path.basename(detector.last_photo_path)}")
                except AttributeError:
                    import subprocess
                    subprocess.call(('open' if os.name == 'posix' else 'xdg-open',
                                     detector.last_photo_path))
                    detector.add_status_message(
                        f"已打开照片: {os.path.basename(detector.last_photo_path)}")


def main():
    detector = SmartCameraUltimate()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ret, frame = cap.read()
    if not ret:
        print("无法连接摄像头")
        return

    window_name = "AI智能相机"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse_click, param={'det': detector})

    print("=" * 60)
    print("AI智能相机已启动!")
    print(f"照片保存目录: {detector.save_dir}")
    print("=" * 60)
    print("操作说明:")
    print("  剪刀手手势 - 拍照")
    print("  微笑 - 自动拍照")
    print("  OK手势 - 解锁系统")
    print("  握拳手势 - 退出程序")
    print("  点击预览图 - 查看照片")
    print("  快捷键:")
    print("    's' - 快速拍照")
    print("    '1' - 增加微笑检测灵敏度")
    print("    '2' - 降低微笑检测灵敏度")
    print("    'o' - 打开保存目录")
    print("    'h' - 显示操作提示")
    print("    'q' - 退出程序")
    print("=" * 60)

    detector.add_status_message("欢迎使用AI智能相机! 微笑检测已开启")
    last_gesture_state = "NONE"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        clean, display, gesture = detector.process_frame(frame)

        if detector.app_state == "RUNNING":
            if (detector.exit_confirm_start != 0 and
                    time.time() - detector.exit_confirm_start >= 5.0):
                detector.add_status_message("正在退出程序...")
                print("程序退出")
                time.sleep(0.5)
                break

            current_time = time.time()
            if (gesture == "V" and last_gesture_state != "V" and
                    not detector.is_counting_down and
                    current_time - detector.last_v_photo_time >= 1.0):
                detector.is_counting_down = True
                detector.countdown_start_time = current_time
                detector.smile_trigger_enabled = False
                detector.last_v_photo_time = current_time
                detector.add_status_message("剪刀手检测，开始倒计时拍照...")

            last_gesture_state = gesture if gesture is not None else "NONE"

            if detector.is_counting_down:
                if gesture == "OPEN":
                    detector.is_counting_down = False
                    detector.smile_trigger_enabled = False
                    detector.add_status_message("拍照已取消")
                elif time.time() - detector.countdown_start_time >= 3:
                    detector._handle_photo_capture(clean)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            detector.add_status_message("正在退出程序...")
            print("手动退出")
            time.sleep(0.5)
            break
        elif key == ord('s') and detector.app_state == "RUNNING":
            detector._handle_photo_capture(clean, shortcut=True)
        elif key == ord('h'):
            detector.add_status_message("剪刀手拍照 | 微笑拍照 | 握拳退出 | 点击预览")
        elif key == ord('o'):
            try:
                os.startfile(os.path.abspath(detector.save_dir))
                detector.add_status_message("已打开保存目录")
            except Exception:
                detector.add_status_message("无法打开保存目录")
        elif key == ord('1'):
            detector.smile_sensitivity = min(1.0, detector.smile_sensitivity + 0.1)
            detector.add_status_message(f"微笑检测灵敏度: {detector.smile_sensitivity:.1f}")
        elif key == ord('2'):
            detector.smile_sensitivity = max(0.1, detector.smile_sensitivity - 0.1)
            detector.add_status_message(f"微笑检测灵敏度: {detector.smile_sensitivity:.1f}")

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("窗口已关闭，程序退出")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
