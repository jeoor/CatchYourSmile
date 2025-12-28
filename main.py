import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from collections import deque


class SmartCameraUltimate:
    def __init__(self):
        # 1. 初始化模型
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

        # 2. 状态控制
        self.app_state = "START_SCREEN"
        self.is_counting_down = False
        self.countdown_start_time = 0
        self.flash_alpha = 0

        # --- 核心逻辑优化变量 ---
        self.waiting_for_release = False  # 锁定后是否正在等待手势释放
        self.gesture_buffer = deque(maxlen=5)  # [新增] 手势缓冲区，用于防抖 (存最近5帧)

        # 3. 计时器与阈值
        self.last_lock_time = 0  # 上次锁定的时间戳
        self.unlock_cooldown = 5.0  # 锁定后冷却时间 (秒)

        self.exit_confirm_start = 0
        self.exit_threshold = 5.0  # 握拳退出阈值

        self.lock_confirm_start = 0
        self.lock_threshold = 2.0  # OK锁定阈值 (防抖后可以设短一点，体验更灵敏)

        # 4. 照片预览变量
        self.last_photo_path = ""
        self.preview_img = None
        self.preview_display_time = 0
        self.preview_rect = (0, 0, 0, 0)

        # 5. 环境配置
        # 尝试查找中文字体，如果没有则回退
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",  # Win10/11 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "/System/Library/Fonts/PingFang.ttc"  # Mac
        ]
        self.zh_font = None
        for path in font_paths:
            if os.path.exists(path):
                self.zh_font = path
                break

        self.save_dir = "captured_photos"
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)

    def draw_text_safe(self, img, text, pos, size=22, color=(255, 255, 255)):
        """ 绘制中文文本的辅助函数 """
        if self.zh_font:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            try:
                font = ImageFont.truetype(self.zh_font, size)
                draw.text(pos, text, font=font, fill=color[::-1])
                return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            except Exception:
                pass
                # 如果没有中文字体或出错，回退到 OpenCV 默认字体 (不支持中文)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img

    def check_gestures(self, lm):
        """ 基础手势几何判断 """
        # 获取指尖和指节坐标
        idx_up = lm[8].y < lm[6].y
        mid_up = lm[12].y < lm[10].y
        rng_up = lm[16].y < lm[14].y
        pnk_up = lm[20].y < lm[18].y

        # 1. 握拳 (FIST)
        if not (idx_up or mid_up or rng_up or pnk_up):
            return "FIST"

        # 2. 剪刀手 (V)
        if idx_up and mid_up and not rng_up and not pnk_up:
            # 检查食指和中指是否分开
            dist = np.hypot(lm[8].x - lm[12].x, lm[8].y - lm[12].y)
            return "V" if dist > 0.03 else None

        # 3. OK 手势
        # 拇指指尖(4)与食指指尖(8)距离很近
        if np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) < 0.04:
            return "OK"

        # 4. 张开手掌 (OPEN)
        if idx_up and mid_up and rng_up and pnk_up:
            return "OPEN"

        return None

    def get_stable_gesture(self, raw_gesture):
        """ [核心] 手势防抖逻辑 """
        self.gesture_buffer.append(raw_gesture)
        if len(self.gesture_buffer) < 3:
            return None

        # 统计缓冲区中出现次数最多的手势 (忽略 None)
        valid_gestures = [g for g in self.gesture_buffer if g is not None]
        if not valid_gestures:
            return None

        # 只有当某个手势在缓冲区占比超过 60% 才认为是有效手势
        most_common = max(set(valid_gestures), key=valid_gestures.count)
        if valid_gestures.count(most_common) >= 3:
            return most_common
        return None

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_res = self.hands.process(img_rgb)

        # 创建带边框的大画布 (h+160, w+80)
        display = np.zeros((h + 160, w + 80, 3), dtype=np.uint8)
        display[:] = (20, 20, 20)  # 深灰色背景
        vx, vy = 40, 80
        display[vy:vy + h, vx:vx + w] = frame
        clean_save = display.copy()

        # --- 手势检测与平滑 ---
        raw_gesture = None
        if hand_res.multi_hand_landmarks:
            lm = hand_res.multi_hand_landmarks[0].landmark
            raw_gesture = self.check_gestures(lm)

            # 绘制关键点反馈
            px, py = int(lm[9].x * w) + vx, int(lm[9].y * h) + vy
            cv2.circle(display, (px, py), 45, (0, 255, 127) if raw_gesture else (100, 100, 100), 2)

        # 获取平滑后的手势
        gesture = self.get_stable_gesture(raw_gesture)

        # --- 绘制四周轨道文字提示 ---
        display = self._draw_border_info(display)

        # --- 状态机分发 ---
        if self.app_state == "START_SCREEN":
            display = self._draw_start_ui(display, gesture)
        else:
            display = self._draw_running_ui(display, gesture, (vx, vy, w, h))
            display = self._draw_photo_preview(display)

        return clean_save, display, gesture

    def _draw_border_info(self, canvas):
        """在四周黑色边框添加文字指示"""
        ch, cw = canvas.shape[:2]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 左上：标题
        canvas = self.draw_text_safe(canvas, "AI SMART CAMERA ULT", (20, 20), 18, (150, 150, 150))
        # 右上：时间
        canvas = self.draw_text_safe(canvas, now, (cw - 220, 20), 16, (100, 100, 100))

        # 底部：根据状态变色的操作指南
        if self.app_state == "RUNNING":
            guide_text = "操作指南: [剪刀手] 拍照 | [OK] 锁定系统 | [握拳] 退出程序"
            guide_color = (0, 255, 255)
        else:
            guide_text = "系统锁定中 - 请保持 [OK] 手势以解锁"
            guide_color = (100, 100, 100)

        canvas = self.draw_text_safe(canvas, guide_text, (40, ch - 40), 18, guide_color)
        canvas = self.draw_text_safe(canvas, f"DIR: {self.save_dir}/", (cw - 250, ch - 40), 14, (60, 60, 60))

        return canvas

    def _draw_start_ui(self, canvas, gesture):
        """ 绘制锁屏界面 (美化版) """
        ch, cw = canvas.shape[:2]
        cx, cy = cw // 2, ch // 2

        # 1. 背景暗化与模糊感
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (cw, ch), (10, 10, 10), -1)
        canvas = cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0)

        # 2. 状态判定
        elapsed_since_lock = time.time() - self.last_lock_time
        in_cooldown = elapsed_since_lock < self.unlock_cooldown

        # 3. 颜色定义
        if in_cooldown:
            color = (80, 80, 80)  # 冷却灰
            glow_color = (40, 40, 40)
            status_msg = f"COOLING DOWN ({int(self.unlock_cooldown - elapsed_since_lock) + 1}s)"
        elif self.waiting_for_release:
            # 如果还在等待释放手势
            if gesture is None:
                self.waiting_for_release = False  # 检测到释放，状态归位
            color = (0, 140, 255)  # 警示橙
            glow_color = (0, 80, 150)
            status_msg = "RELEASE HAND"
        else:
            color = (0, 255, 255)  # 激活青
            glow_color = (0, 100, 100)
            status_msg = "SYSTEM LOCKED"

        # 4. 绘制科技感边角 (Corner Brackets)
        length, t = 40, 2
        corners = [
            ((cx - 150, cy - 150), (1, 1)),  # 左上
            ((cx + 150, cy - 150), (-1, 1)),  # 右上
            ((cx - 150, cy + 150), (1, -1)),  # 左下
            ((cx + 150, cy + 150), (-1, -1))  # 右下
        ]
        for (x, y), (dx, dy) in corners:
            cv2.line(canvas, (x, y), (x + length * dx, y), color, t)
            cv2.line(canvas, (x, y), (x, y + length * dy), color, t)

        # 5. 中心圆环动画
        # 静态外环
        cv2.circle(canvas, (cx, cy), 115, glow_color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 105, color, 2, cv2.LINE_AA)

        # 动态旋转卫星点 (仅在就绪状态显示)
        if not in_cooldown and not self.waiting_for_release:
            angle = (time.time() * 3) % (2 * np.pi)
            px = int(cx + 105 * np.cos(angle))
            py = int(cy + 105 * np.sin(angle))
            cv2.circle(canvas, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)

        # 6. 文字绘制
        canvas = self.draw_text_safe(canvas, status_msg, (cx - 70, cy - 110), 16, color)
        canvas = self.draw_text_safe(canvas, "系统已锁定", (cx - 55, cy - 20), 24, (255, 255, 255))

        if in_cooldown:
            sub_msg = "安全冷却中..."
        elif self.waiting_for_release:
            sub_msg = "请收回手势"
        else:
            sub_msg = "比出 OK 解锁"

        canvas = self.draw_text_safe(canvas, sub_msg, (cx - 60, cy + 30), 20, color)

        # 7. 解锁触发
        if not in_cooldown and not self.waiting_for_release and gesture == "OK":
            self.app_state = "RUNNING"
            self.lock_confirm_start = 0

        return canvas

    def _draw_running_ui(self, canvas, gesture, v_rect):
        vx, vy, vw, vh = v_rect

        # 1. 握拳退出逻辑
        if gesture == "FIST":
            if self.exit_confirm_start == 0: self.exit_confirm_start = time.time()
            elapsed = time.time() - self.exit_confirm_start
            progress = min(elapsed / self.exit_threshold, 1.0)

            # 绘制进度条
            bar_w = int(progress * vw)
            cv2.rectangle(canvas, (vx, vy + vh + 90), (vx + vw, vy + vh + 105), (60, 60, 60), -1)
            cv2.rectangle(canvas, (vx, vy + vh + 90), (vx + bar_w, vy + vh + 105), (50, 50, 255), -1)
            canvas = self.draw_text_safe(canvas, f"关闭程序... {int(progress * 100)}%", (vx, vy + vh + 55), 18,
                                         (50, 50, 255))
        else:
            self.exit_confirm_start = 0

        # 2. OK锁定逻辑 (使用平滑后的手势，这里会非常稳定)
        if gesture == "OK" and not self.is_counting_down:
            if self.lock_confirm_start == 0: self.lock_confirm_start = time.time()
            elapsed = time.time() - self.lock_confirm_start
            progress = min(elapsed / self.lock_threshold, 1.0)

            # 绘制进度条
            bar_w = int(progress * vw)
            cv2.rectangle(canvas, (vx, vy + vh + 90), (vx + vw, vy + vh + 105), (60, 60, 60), -1)
            cv2.rectangle(canvas, (vx, vy + vh + 90), (vx + bar_w, vy + vh + 105), (255, 255, 0), -1)
            canvas = self.draw_text_safe(canvas, f"锁定系统中... {int(progress * 100)}%", (vx, vy + vh + 55), 18,
                                         (255, 255, 0))

            # 锁定触发
            if progress >= 1.0:
                self.app_state = "START_SCREEN"
                self.lock_confirm_start = 0
                self.waiting_for_release = True  # 标记：需要放手
                self.last_lock_time = time.time()  # 记录时间：开始冷却
        else:
            self.lock_confirm_start = 0

        # 3. 拍照倒计时
        if self.is_counting_down:
            rem = 3 - int(time.time() - self.countdown_start_time)
            cv2.putText(canvas, str(max(1, rem)), (vx + vw // 2 - 25, vy + vh // 2), cv2.FONT_HERSHEY_TRIPLEX, 4,
                        (0, 255, 127), 5)

        # 4. 闪光灯特效
        if self.flash_alpha > 0:
            flash = canvas.copy()
            flash[:] = (255, 255, 255)
            canvas = cv2.addWeighted(flash, self.flash_alpha, canvas, 1 - self.flash_alpha, 0)
            self.flash_alpha -= 0.15

        return canvas

    def _draw_photo_preview(self, canvas):
        if self.preview_img is not None and time.time() - self.preview_display_time < 5.0:
            ph, pw = self.preview_img.shape[:2]
            ch, cw = canvas.shape[:2]
            # 预览框位置 (右下角上方)
            x1, y1 = cw - pw - 25, ch - ph - 85
            x2, y2 = cw - 25, ch - 85
            self.preview_rect = (x1, y1, x2, y2)

            cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)
            canvas[y1:y2, x1:x2] = self.preview_img
            canvas = self.draw_text_safe(canvas, "点击预览", (x1, y1 - 30), 16)
        return canvas


def on_mouse_click(event, x, y, flags, param):
    """ 鼠标交互：点击预览图打开文件 """
    det = param['det']
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = det.preview_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            if det.last_photo_path and os.path.exists(det.last_photo_path):
                try:
                    os.startfile(os.path.abspath(det.last_photo_path))
                except AttributeError:
                    # MacOS/Linux 兼容
                    import subprocess
                    subprocess.call(('open' if os.name == 'posix' else 'xdg-open', det.last_photo_path))


def main():
    detector = SmartCameraUltimate()
    cap = cv2.VideoCapture(0)

    # --- 1. 尝试提升摄像头清晰度 (可选) ---
    # 许多摄像头默认是 640x480，设置为 1280x720 会更清晰且比例更自然
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # --- 2. 动态计算正确的窗口比例 ---
    ret, frame = cap.read()
    if not ret:
        print("无法连接摄像头")
        return

    # 获取原始画面尺寸
    h, w = frame.shape[:2]
    # 计算加上边框后的总尺寸 (对应 process_frame 中的逻辑: h+160, w+80)
    final_h, final_w = h + 160, w + 80

    # 设定一个合适的目标高度 (比如屏幕高度的 80%)，反推宽度，保持比例
    target_h = 800
    target_w = int(target_h * (final_w / final_h))

    win_name = "AI Smart Camera Ultimate"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # 使用计算出的宽高，而不是写死的 1000x750
    cv2.resizeWindow(win_name, target_w, target_h)

    cv2.setMouseCallback(win_name, on_mouse_click, param={'det': detector})

    while True:
        ret, frame = cap.read()
        if not ret: break

        clean, display, gesture = detector.process_frame(frame)

        if detector.app_state == "RUNNING":
            if detector.exit_confirm_start != 0 and (
                    time.time() - detector.exit_confirm_start >= detector.exit_threshold):
                break

            if gesture == "V" and not detector.is_counting_down:
                detector.is_counting_down = True
                detector.countdown_start_time = time.time()

            if detector.is_counting_down:
                if gesture == "OPEN":
                    detector.is_counting_down = False
                elif time.time() - detector.countdown_start_time >= 3:
                    detector.is_counting_down = False
                    detector.flash_alpha = 1.0

                    filename = f"Photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    save_path = os.path.join(detector.save_dir, filename)
                    cv2.imwrite(save_path, clean)

                    detector.last_photo_path = save_path
                    detector.preview_display_time = time.time()
                    detector.preview_img = cv2.resize(clean, (160, 120))

        cv2.imshow(win_name, display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()