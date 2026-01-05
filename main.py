import cv2
import mediapipe as mp
import numpy as np
import time
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from collections import deque
import math

'''CatchYourSmile'''

class SmartCameraUltimate:
    def __init__(self):
        # 初始化模型
        self._init_models()

        # 状态控制变量
        self._init_state_variables()

        # UI动画变量
        self._init_ui_variables()

        # 计时器与阈值
        self._init_timer_thresholds()

        # 照片预览相关
        self._init_preview_variables()

        # 环境配置
        self._init_environment()

        # 状态提示系统
        self._init_status_system()

    def _init_models(self):
        """初始化AI模型"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _init_state_variables(self):
        """初始化状态控制变量"""
        self.app_state = "START_SCREEN"
        self.is_counting_down = False
        self.countdown_start_time = 0
        self.flash_alpha = 0
        self.flash_duration = 0.5
        self.flash_start_time = 0

        # 微笑检测相关
        self.smile_detection_mode = True
        self.last_smile_time = 0
        self.smile_cooldown = 5.0
        self.smile_threshold = 0.5
        self.smile_sensitivity = 0.5
        self.smile_counter = 0
        self.smile_frames_needed = 12
        self.smile_trigger_enabled = False
        self.last_no_smile_time = 0
        self.last_smile_state = False

    def _init_ui_variables(self):
        """初始化UI动画变量"""
        self.ui_anim_time = 0
        self.preview_anim_scale = 0
        self.button_pulse = 0
        self.gesture_buffer = deque(maxlen=5)
        self.waiting_for_release = False

    def _init_timer_thresholds(self):
        """初始化计时器和阈值"""
        self.last_lock_time = 0
        self.unlock_cooldown = 5.0
        self.exit_confirm_start = 0
        self.exit_threshold = 5.0
        self.last_v_photo_time = 0
        self.v_photo_cooldown = 1.0

    def _init_preview_variables(self):
        """初始化照片预览变量"""
        self.last_photo_path = ""
        self.preview_img = None
        self.preview_display_time = 0
        self.preview_rect = (0, 0, 0, 0)
        self.preview_border_color = (0, 180, 255)

    def _init_environment(self):
        """初始化环境配置"""
        # 字体配置
        self.zh_font = self._find_chinese_font()

        # 创建保存目录
        self.save_dir = "captured_photos"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 颜色主题
        self.theme = {
            'primary': (0, 180, 255),
            'secondary': (0, 255, 200),
            'success': (0, 255, 127),
            'danger': (255, 80, 80),
            'warning': (255, 200, 50),
            'dark': (15, 20, 30),
            'light': (240, 245, 255),
            'gray': (60, 70, 85),
            'accent': (180, 100, 255)
        }

    def _init_status_system(self):
        """初始化状态提示系统"""
        self.status_messages = []
        self.status_duration = 3.0

    def _find_chinese_font(self):
        """查找系统中文字体"""
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/simkai.ttf",
            "C:/Windows/Fonts/msjh.ttc",
        ]

        for path in font_paths:
            if os.path.exists(path):
                print(f"找到字体: {path}")
                return path
        return None

    def add_timestamp_to_image(self, image):
        """在图片上添加时间戳和顶部文字"""
        h, w = image.shape[:2]
        current_time = datetime.now()

        # 文件名格式
        filename_format = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # 显示格式
        display_format = current_time.strftime("%Y-%m-%d %H:%M:%S")

        # 添加顶部文字
        image = self._add_top_text(image, w, h)

        # 添加时间戳水印
        image = self._add_timestamp_watermark(image, w, h, display_format)

        return image, filename_format

    def _add_top_text(self, image, width, height):
        """添加顶部文字"""
        top_text = "Keep smile everyday!"
        font_scale = 1.2
        thickness = 3
        padding = 20

        text_size = cv2.getTextSize(top_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = text_size[1] + padding

        # 添加半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay,
                      (text_x - padding // 2, text_y - text_size[1] - padding // 2),
                      (text_x + text_size[0] + padding // 2, text_y + padding // 2),
                      (0, 0, 0), -1)

        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

        # 添加文字
        cv2.putText(image, top_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return image

    def _add_timestamp_watermark(self, image, width, height, timestamp):
        """添加时间戳水印"""
        font_scale = 0.6
        thickness = 2
        padding = 10

        text_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        pos_x = width - text_size[0] - padding
        pos_y = height - padding

        # 添加半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay,
                      (pos_x - padding // 2, pos_y - text_size[1] - padding // 2),
                      (pos_x + text_size[0] + padding // 2, pos_y + padding // 2),
                      (0, 0, 0), -1)

        image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

        # 添加文字
        cv2.putText(image, timestamp, (pos_x, pos_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return image

    def detect_smile(self, face_landmarks, img_width, img_height):
        """检测微笑"""
        try:
            # 获取嘴巴关键点
            landmarks = self._get_mouth_landmarks(face_landmarks, img_width, img_height)

            # 计算嘴巴特征
            features = self._calculate_mouth_features(landmarks)

            # 计算微笑分数
            smile_score = self._calculate_smile_score(features)

            # 调整阈值
            adjusted_threshold = self.smile_threshold * (1.5 - self.smile_sensitivity)

            # 判断微笑
            is_smiling = smile_score > adjusted_threshold and features['inner_height'] > 5

            return is_smiling, smile_score, landmarks['points']

        except Exception:
            return False, 0, (0, 0, 0, 0, 0, 0, 0, 0)

    def _get_mouth_landmarks(self, face_landmarks, img_width, img_height):
        """获取嘴巴关键点坐标"""
        # 关键点索引
        points_idx = {
            'left_corner': 61, 'right_corner': 291,
            'top_center': 0, 'bottom_center': 17,
            'inner_left': 78, 'inner_right': 308,
            'inner_top': 13, 'inner_bottom': 14
        }

        # 转换为像素坐标
        points = {}
        for name, idx in points_idx.items():
            landmark = face_landmarks.landmark[idx]
            points[name] = (
                int(landmark.x * img_width),
                int(landmark.y * img_height)
            )

        # 转换为元组格式
        points_tuple = (
            points['left_corner'][0], points['left_corner'][1],
            points['right_corner'][0], points['right_corner'][1],
            points['top_center'][0], points['top_center'][1],
            points['bottom_center'][0], points['bottom_center'][1]
        )

        return {'points': points_tuple, 'coords': points}

    def _calculate_mouth_features(self, landmarks):
        """计算嘴巴特征"""
        coords = landmarks['coords']

        # 计算外嘴唇尺寸
        outer_width = math.sqrt(
            (coords['right_corner'][0] - coords['left_corner'][0]) ** 2 +
            (coords['right_corner'][1] - coords['left_corner'][1]) ** 2
        )
        outer_height = math.sqrt(
            (coords['bottom_center'][0] - coords['top_center'][0]) ** 2 +
            (coords['bottom_center'][1] - coords['top_center'][1]) ** 2
        )

        # 计算内嘴唇尺寸
        inner_width = math.sqrt(
            (coords['inner_right'][0] - coords['inner_left'][0]) ** 2 +
            (coords['inner_right'][1] - coords['inner_left'][1]) ** 2
        )
        inner_height = math.sqrt(
            (coords['inner_bottom'][0] - coords['inner_top'][0]) ** 2 +
            (coords['inner_bottom'][1] - coords['inner_top'][1]) ** 2
        )

        # 计算面部宽度参考
        face_width = self._calculate_face_width(landmarks)

        return {
            'outer_width': max(outer_width, 1),
            'outer_height': max(outer_height, 1),
            'inner_width': max(inner_width, 1),
            'inner_height': max(inner_height, 1),
            'face_width': max(face_width, 1)
        }

    def _calculate_face_width(self, landmarks):
        """计算面部宽度"""
        # 使用脸颊点估算面部宽度
        left_cheek = landmarks['coords']['left_corner'][0] * 0.9
        right_cheek = landmarks['coords']['right_corner'][0] * 1.1
        return abs(right_cheek - left_cheek)

    def _calculate_smile_score(self, features):
        """计算微笑分数"""
        outer_ratio = features['outer_width'] / features['outer_height']
        inner_ratio = features['inner_width'] / features['inner_height']
        mouth_to_face = features['outer_width'] / features['face_width']

        # 综合评分
        smile_score = (
                outer_ratio * 0.4 +
                inner_ratio * 0.3 +
                mouth_to_face * 10 * 0.3
        )

        return smile_score

    def draw_smile_indicator(self, canvas, mouth_points, is_smiling, smile_score, vx, vy, w, h):
        """绘制微笑指示器"""
        if not is_smiling:
            return canvas

        left_x, left_y, right_x, right_y, top_x, top_y, bottom_x, bottom_y = mouth_points

        # 计算嘴巴中心
        mouth_center_x = (left_x + right_x) // 2 + vx
        mouth_center_y = (top_y + bottom_y) // 2 + vy

        # 脉动效果
        pulse = 0.8 + 0.2 * abs(math.sin(time.time() * 3))
        circle_color = tuple(int(c * pulse) for c in self.theme['accent'])

        # 绘制指示圈
        cv2.circle(canvas, (mouth_center_x, mouth_center_y), 25, circle_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (mouth_center_x, mouth_center_y), 15,
                   tuple(int(c * 0.6) for c in circle_color), 1, cv2.LINE_AA)

        return canvas

    def add_status_message(self, message):
        """添加状态提示消息"""
        self.status_messages.append({
            'text': message,
            'time': time.time(),
            'duration': self.status_duration
        })
        print(f"状态: {message}")

    def draw_text_safe(self, img, text, pos, size=22, color=(255, 255, 255),
                       shadow=False, shadow_color=(0, 0, 0), align="left"):
        """安全绘制文本（支持中英文）"""
        x, y = pos

        # 检查是否为中文
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)

        if has_chinese and self.zh_font:
            return self._draw_chinese_text(img, text, (x, y), size, color, shadow, shadow_color, align)
        else:
            return self._draw_english_text(img, text, (x, y), size, color, shadow, shadow_color)

    def _draw_chinese_text(self, img, text, pos, size, color, shadow, shadow_color, align):
        """使用PIL绘制中文"""
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype(self.zh_font, size)

            # 计算文本尺寸和对齐
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]

            x, y = pos
            if align == "center":
                x = x - text_width // 2
            elif align == "right":
                x = x - text_width

            # 绘制阴影
            if shadow:
                draw.text((x + 2, y + 2 - text_bbox[3]), text,
                          fill=shadow_color, font=font)

            # 绘制文本
            draw.text((x, y - text_bbox[3]), text, fill=color, font=font)

            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        except Exception:
            # 降级到英文绘制
            return self._draw_english_text(img, text, pos, size, color, shadow, shadow_color)

    def _draw_english_text(self, img, text, pos, size, color, shadow, shadow_color):
        """使用OpenCV绘制英文"""
        font_scale = size / 30

        if shadow:
            cv2.putText(img, text, (pos[0] + 2, pos[1] + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow_color, 2)

        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 2)
        return img

    def draw_rounded_rect(self, img, pt1, pt2, color, radius=10, thickness=-1):
        """绘制圆角矩形"""
        x1, y1 = pt1
        x2, y2 = pt2

        if thickness == -1:
            # 填充模式
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)

            # 绘制四个圆角
            corners = [
                (x1 + radius, y1 + radius),
                (x2 - radius, y1 + radius),
                (x1 + radius, y2 - radius),
                (x2 - radius, y2 - radius)
            ]
            for cx, cy in corners:
                cv2.circle(img, (cx, cy), radius, color, -1)
        else:
            # 边框模式
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

            # 绘制四个圆角
            angles = [(180, 270), (270, 360), (90, 180), (0, 90)]
            corners = [
                (x1 + radius, y1 + radius),
                (x2 - radius, y1 + radius),
                (x1 + radius, y2 - radius),
                (x2 - radius, y2 - radius)
            ]
            for (cx, cy), (start_angle, end_angle) in zip(corners, angles):
                cv2.ellipse(img, (cx, cy), (radius, radius), 0,
                            start_angle, end_angle, color, thickness)

    def create_gradient_background(self, width, height):
        """创建渐变背景"""
        gradient = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            color_factor = 1.0 - (y / height) * 0.3
            color = np.array(self.theme['dark']) * color_factor
            gradient[y, :] = color.astype(np.uint8)

        return gradient

    def check_gestures(self, landmarks):
        """检测基础手势"""
        # 获取指尖和指节坐标
        idx_up = landmarks[8].y < landmarks[6].y
        mid_up = landmarks[12].y < landmarks[10].y
        rng_up = landmarks[16].y < landmarks[14].y
        pnk_up = landmarks[20].y < landmarks[18].y

        # 手势判断
        if not (idx_up or mid_up or rng_up or pnk_up):
            return "FIST"

        if idx_up and mid_up and not rng_up and not pnk_up:
            dist = np.hypot(landmarks[8].x - landmarks[12].x,
                            landmarks[8].y - landmarks[12].y)
            return "V" if dist > 0.03 else None

        if np.hypot(landmarks[4].x - landmarks[8].x,
                    landmarks[4].y - landmarks[8].y) < 0.04:
            return "OK"

        if idx_up and mid_up and rng_up and pnk_up:
            return "OPEN"

        return None

    def get_stable_gesture(self, raw_gesture):
        """获取稳定的手势（防抖）"""
        self.gesture_buffer.append(raw_gesture)
        if len(self.gesture_buffer) < 3:
            return None

        valid_gestures = [g for g in self.gesture_buffer if g is not None]
        if not valid_gestures:
            return None

        most_common = max(set(valid_gestures), key=valid_gestures.count)
        return most_common if valid_gestures.count(most_common) >= 3 else None

    def process_frame(self, frame):
        """处理单帧图像"""
        # 预处理
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 检测手势和面部
        hand_res = self.hands.process(img_rgb)
        face_res = self.face_mesh.process(img_rgb) if self.smile_detection_mode else None

        # 创建显示画布
        display, video_info = self._create_display_canvas(frame, h, w)
        vx, vy, video_w, video_h = video_info

        # 手势检测
        gesture = self._detect_gesture(hand_res, display, vx, vy, video_w, video_h)

        # 微笑检测
        self._detect_smile_trigger(face_res, video_w, video_h, vx, vy, display)

        # 更新UI
        self.ui_anim_time += 0.05
        self.button_pulse = abs(math.sin(self.ui_anim_time)) * 0.3 + 0.7

        # 绘制UI组件
        display = self._draw_border_info(display, display.shape[1], display.shape[0])

        # 状态机分发
        if self.app_state == "START_SCREEN":
            display = self._draw_start_ui(display, gesture,
                                          display.shape[1], display.shape[0])
        else:
            display = self._draw_running_ui(display, gesture,
                                            (vx, vy, w, h),
                                            display.shape[1], display.shape[0])
            display = self._draw_photo_preview(display,
                                               display.shape[1], display.shape[0])

        # 特效和状态消息
        display = self._apply_flash_effect(display)
        display = self._draw_status_messages(display,
                                             display.shape[1], display.shape[0])

        return frame, display, gesture

    def _create_display_canvas(self, frame, h, w):
        """创建显示画布"""
        display_h, display_w = h + 160, w + 80
        display = self.create_gradient_background(display_w, display_h)

        vx, vy = 40, 80
        border_size = 2

        # 调整视频帧大小
        video_h = h - 2 * border_size
        video_w = w - 2 * border_size

        if video_h > 0 and video_w > 0:
            frame_resized = cv2.resize(frame, (video_w, video_h))
            cv2.rectangle(display, (vx, vy), (vx + w, vy + h), self.theme['gray'], border_size)
            display[vy + border_size:vy + border_size + video_h,
            vx + border_size:vx + border_size + video_w] = frame_resized
        else:
            display[vy:vy + h, vx:vx + w] = frame

        return display, (vx, vy, video_w, video_h)

    def _detect_gesture(self, hand_res, display, vx, vy, video_w, video_h):
        """检测手势并绘制指示器"""
        raw_gesture = None

        if hand_res.multi_hand_landmarks:
            landmarks = hand_res.multi_hand_landmarks[0].landmark
            raw_gesture = self.check_gestures(landmarks)
            self._draw_gesture_indicator(display, landmarks, vx, vy, video_w, video_h, raw_gesture)

        return self.get_stable_gesture(raw_gesture)

    def _draw_gesture_indicator(self, display, landmarks, vx, vy, video_w, video_h, gesture):
        """绘制手势指示器"""
        px = int(landmarks[9].x * video_w) + vx + 2
        py = int(landmarks[9].y * video_h) + vy + 2

        status_color = self.theme['success'] if gesture else self.theme['gray']
        pulse = abs(math.sin(time.time() * 3)) * 0.3 + 0.7 if gesture else 0.5

        # 绘制指示圈
        alpha = 0.6
        main_color = tuple(int(c * pulse * alpha) for c in status_color)
        cv2.circle(display, (px, py), 35, main_color, 2, cv2.LINE_AA)
        cv2.circle(display, (px, py), 25, tuple(int(c * 0.8) for c in status_color), 1, cv2.LINE_AA)

        # 显示手势名称
        if gesture:
            text_size = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = px - text_size[0] // 2
            text_y = py - 50

            cv2.rectangle(display,
                          (text_x - 5, text_y - text_size[1] - 5),
                          (text_x + text_size[0] + 5, text_y + 5),
                          (0, 0, 0), -1)

            cv2.putText(display, gesture, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    def _detect_smile_trigger(self, face_res, video_w, video_h, vx, vy, display):
        """检测微笑并处理触发"""
        if not self.smile_detection_mode or not face_res or not face_res.multi_face_landmarks:
            return

        current_time = time.time()
        in_cooldown = current_time - self.last_smile_time < self.smile_cooldown

        for face_landmarks in face_res.multi_face_landmarks:
            is_smiling, smile_score, mouth_points = self.detect_smile(
                face_landmarks, video_w, video_h
            )

            # 更新微笑状态
            self._update_smile_state(is_smiling, current_time, in_cooldown)

            # 绘制微笑指示器
            if is_smiling:
                display = self.draw_smile_indicator(
                    display, mouth_points, is_smiling, smile_score,
                    vx + 2, vy + 2, video_w, video_h
                )

    def _update_smile_state(self, is_smiling, current_time, in_cooldown):
        """更新微笑检测状态"""
        if is_smiling:
            self.smile_counter += 1
            self.last_no_smile_time = 0

            if (self.smile_counter >= self.smile_frames_needed and
                    not in_cooldown and
                    not self.is_counting_down and
                    self.app_state == "RUNNING" and
                    not self.last_smile_state):
                self.is_counting_down = True
                self.countdown_start_time = current_time
                self.smile_trigger_enabled = True
                self.add_status_message("检测到微笑，开始倒计时拍照...")
                self.last_smile_time = current_time
                self.smile_counter = 0
        else:
            if self.last_no_smile_time == 0:
                self.last_no_smile_time = current_time

            if current_time - self.last_no_smile_time > 0.5:
                self.smile_counter = max(0, self.smile_counter - 2)
            else:
                self.smile_counter = max(0, self.smile_counter - 1)

        self.last_smile_state = is_smiling

    def _handle_photo_capture(self, clean, shortcut=False):
        """处理照片捕获"""
        self.flash_alpha = 1.0
        self.flash_start_time = time.time()

        # 添加时间戳
        clean_with_timestamp, filename = self.add_timestamp_to_image(clean)
        filename = f"{filename}.jpg"
        save_path = os.path.join(self.save_dir, filename)

        # 保存照片
        cv2.imwrite(save_path, clean_with_timestamp)
        self.last_photo_path = save_path
        self.preview_display_time = time.time()

        # 创建预览图
        preview_h, preview_w = 120, 160
        self.preview_img = cv2.resize(clean_with_timestamp, (preview_w, preview_h))
        self.preview_anim_scale = 0

        # 状态消息
        if shortcut:
            message = f"快捷键拍照: {filename}"
        elif self.smile_trigger_enabled:
            message = f"微笑拍照已保存: {filename}"
        else:
            message = f"照片已保存: {filename}"

        self.add_status_message(message)
        self.smile_trigger_enabled = False
        self.is_counting_down = False

    def _draw_border_info(self, canvas, width, height):
        """绘制边框信息"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 左上角标题
        canvas = self.draw_text_safe(canvas, "AI智能相机", (30, 40),
                                     24, self.theme['primary'], shadow=True)

        # 右上角时间
        canvas = self.draw_text_safe(canvas, now, (width - 230, 35),
                                     16, self.theme['light'], shadow=True)

        # 底部指南
        if self.app_state == "RUNNING":
            guide_text = "剪刀手拍照 | 微笑拍照 | 握拳退出"
            guide_color = self.theme['secondary']
        else:
            guide_text = "系统已锁定 - 请比出OK手势解锁"
            guide_color = self.theme['gray']

        guide_x = (width - 400) // 2 if width > 400 else 20
        canvas = self.draw_text_safe(canvas, guide_text, (guide_x, height - 30),
                                     20, guide_color, shadow=True)

        # 右下角目录信息
        canvas = self.draw_text_safe(canvas, f"目录: {self.save_dir}/",
                                     (width - 250, height - 55),
                                     14, self.theme['gray'], shadow=True)

        return canvas

    def _draw_start_ui(self, canvas, gesture, width, height):
        """绘制启动界面"""
        cx, cy = width // 2, height // 2

        # 创建半透明覆盖层
        overlay = canvas.copy()
        dark_layer = np.zeros_like(overlay)
        dark_layer[:] = tuple(int(c * 0.5) for c in self.theme['dark'])
        canvas = cv2.addWeighted(canvas, 0.4, dark_layer, 0.6, 0)

        # 状态判断
        elapsed = time.time() - self.last_lock_time
        in_cooldown = elapsed < self.unlock_cooldown

        # 确定颜色和状态
        if in_cooldown:
            color = self.theme['gray']
            status_msg = f"冷却中 ({int(self.unlock_cooldown - elapsed) + 1}秒)"
            pulse_factor = 0.5
        elif self.waiting_for_release:
            color = self.theme['warning']
            status_msg = "请收回手势"
            pulse_factor = 0.7 + 0.3 * abs(math.sin(self.ui_anim_time * 2))
        else:
            color = self.theme['primary']
            status_msg = "系统已锁定"
            pulse_factor = 0.8 + 0.2 * abs(math.sin(self.ui_anim_time))

        # 绘制中心卡片
        card_width, card_height = min(400, width - 40), min(300, height - 40)
        card_x, card_y = cx - card_width // 2, cy - card_height // 2

        if (0 <= card_x < width and 0 <= card_y < height and
                card_x + card_width <= width and card_y + card_height <= height):
            # 绘制卡片
            self._draw_card(canvas, card_x, card_y, card_width, card_height,
                            color, pulse_factor, cx, cy, in_cooldown)

            # 状态文本
            canvas = self.draw_text_safe(canvas, status_msg,
                                         (cx - 60, cy - 20),
                                         24, tuple(int(c * pulse_factor) for c in color),
                                         shadow=True)

            # 说明文本
            sub_msg = "请稍候..." if in_cooldown else \
                "检测到手势已释放" if self.waiting_for_release else \
                    "比出OK手势解锁系统"

            canvas = self.draw_text_safe(canvas, sub_msg, (cx - 100, cy + 60),
                                         18, self.theme['light'], shadow=True)

        # 解锁触发
        if not in_cooldown and not self.waiting_for_release and gesture == "OK":
            self.app_state = "RUNNING"
            self.add_status_message("系统已解锁")

        return canvas

    def _draw_card(self, canvas, x, y, width, height, color, pulse_factor, cx, cy, in_cooldown):
        """绘制卡片"""
        # 卡片背景
        card_bg = np.zeros((height, width, 3), dtype=np.uint8)
        card_bg[:] = tuple(int(c * 0.8) for c in self.theme['dark'])
        canvas[y:y + height, x:x + width] = \
            cv2.addWeighted(canvas[y:y + height, x:x + width],
                            0.3, card_bg, 0.7, 0)

        # 卡片边框
        border_color = tuple(int(c * pulse_factor) for c in color)
        self.draw_rounded_rect(canvas, (x, y), (x + width, y + height),
                               border_color, 15, 2)

        # 解锁环
        ring_radius = min(100, width // 4, height // 4)
        ring_thickness = 8

        if in_cooldown:
            progress = 1 - ((time.time() - self.last_lock_time) / self.unlock_cooldown)
            end_angle = int(360 * progress)
            cv2.ellipse(canvas, (cx, cy), (ring_radius, ring_radius),
                        0, 0, end_angle, color, ring_thickness, cv2.LINE_AA)
        elif not self.waiting_for_release:
            angle = int(self.ui_anim_time * 50) % 360
            cv2.ellipse(canvas, (cx, cy), (ring_radius, ring_radius),
                        angle, 0, 270, border_color, ring_thickness, cv2.LINE_AA)

            cv2.circle(canvas, (cx, cy), ring_radius - ring_thickness * 2,
                       tuple(int(c * 0.7) for c in color), 2, cv2.LINE_AA)

    def _draw_running_ui(self, canvas, gesture, video_rect, width, height):
        """绘制运行界面"""
        vx, vy, vw, vh = video_rect
        current_time = time.time()

        # 握拳退出逻辑
        if gesture == "FIST":
            self._draw_exit_progress(canvas, vx, vy, vw, vh, current_time)
        else:
            self.exit_confirm_start = 0

        # 拍照倒计时
        if self.is_counting_down:
            self._draw_countdown(canvas, vx, vy, vw, vh, current_time)

        return canvas

    def _draw_exit_progress(self, canvas, vx, vy, vw, vh, current_time):
        """绘制退出进度条（透明背景）"""
        if self.exit_confirm_start == 0:
            self.exit_confirm_start = current_time

        elapsed = current_time - self.exit_confirm_start
        progress = min(elapsed / self.exit_threshold, 1.0)

        # 中心位置
        center_x = vx + vw // 2
        center_y = vy + vh // 2

        # 进度环参数
        outer_radius = min(100, vw // 4, vh // 4)
        inner_radius = outer_radius - 20

        # 不再绘制黑色背景，改为透明背景

        if progress > 0:
            # 绘制进度环背景（透明的灰色圆环）
            background_thickness = 20
            background_color = tuple(int(c * 0.2) for c in self.theme['dark'])
            cv2.ellipse(canvas, (center_x, center_y),
                        (outer_radius, outer_radius), 0, 0, 360,
                        background_color, background_thickness, cv2.LINE_AA)

            # 绘制彩色进度环（动态填充）
            end_angle = int(360 * progress)
            pulse = 0.8 + 0.2 * abs(math.sin(current_time * 5))
            progress_color = tuple(int(c * pulse) for c in self.theme['danger'])

            # 绘制进度环
            cv2.ellipse(canvas, (center_x, center_y),
                        (outer_radius, outer_radius), 0, 0, end_angle,
                        progress_color, background_thickness, cv2.LINE_AA)

            # 进度文本
            if progress < 1.0:
                percentage = int(progress * 100)
                cv2.putText(canvas, f"{percentage}%",
                            (center_x - 50, center_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.theme['danger'],
                            3, cv2.LINE_AA)

                remaining = self.exit_threshold - elapsed
                cv2.putText(canvas, f"{remaining:.1f}s",
                            (center_x - 30, center_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.theme['light'],
                            2, cv2.LINE_AA)

                # 拳头图标指示
                fist_radius = 15
                fist_x = center_x
                fist_y = center_y + 65

                cv2.circle(canvas, (fist_x, fist_y), fist_radius,
                           self.theme['danger'], 2, cv2.LINE_AA)
                cv2.circle(canvas, (fist_x, fist_y), 5,
                           self.theme['danger'], -1, cv2.LINE_AA)
            else:
                # 进度完成，显示准备退出
                cv2.putText(canvas, "EXITING...",
                            (center_x - 80, center_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.theme['danger'],
                            3, cv2.LINE_AA)

        # 绘制外圈边框（可选）
        border_color = tuple(int(c * 0.5) for c in self.theme['dark'])
        cv2.circle(canvas, (center_x, center_y), outer_radius + 10,
                   border_color, 1, cv2.LINE_AA)

    def _draw_countdown(self, canvas, vx, vy, vw, vh, current_time):
        """绘制拍照倒计时"""
        remaining = 3 - int(current_time - self.countdown_start_time)

        if remaining > 0:
            # 倒计时圆圈参数
            radius = min(80, vw // 2, vh // 2)
            center_x, center_y = vx + vw // 2, vy + vh // 2

            # 背景圆
            cv2.circle(canvas, (center_x, center_y), radius,
                       tuple(int(c * 0.2) for c in self.theme['dark']), -1, cv2.LINE_AA)

            # 进度环
            ring_progress = 1 - (current_time - self.countdown_start_time) / 3
            end_angle = int(360 * ring_progress)

            # 根据触发方式选择颜色
            if self.smile_trigger_enabled:
                ring_color = tuple(int(c * (0.8 + 0.2 * abs(math.sin(self.ui_anim_time * 5))))
                                   for c in self.theme['accent'])
                text_color = self.theme['accent']
            else:
                ring_color = tuple(int(c * (0.8 + 0.2 * abs(math.sin(self.ui_anim_time * 5))))
                                   for c in self.theme['success'])
                text_color = self.theme['success']

            cv2.ellipse(canvas, (center_x, center_y), (radius, radius),
                        0, 0, end_angle, ring_color, 8, cv2.LINE_AA)

            # 倒计时数字
            text = str(remaining)
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_TRIPLEX, 4, 5)

            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2

            cv2.putText(canvas, text, (text_x + 4, text_y + 4),
                        cv2.FONT_HERSHEY_TRIPLEX, 4, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(canvas, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_TRIPLEX, 4, text_color, 5, cv2.LINE_AA)

    def _draw_photo_preview(self, canvas, width, height):
        """绘制照片预览"""
        if (self.preview_img is None or
                time.time() - self.preview_display_time >= 5.0):
            self.preview_anim_scale = 0
            return canvas

        # 预览图尺寸和位置
        preview_h, preview_w = 120, 160
        target_x1, target_y1 = width - preview_w - 25, height - preview_h - 85
        target_x2, target_y2 = width - 25, height - 85

        if not (0 <= target_x1 < width and 0 <= target_y1 < height and
                target_x2 <= width and target_y2 <= height):
            return canvas

        # 缩放动画
        if self.preview_anim_scale < 1.0:
            self.preview_anim_scale = min(1.0, self.preview_anim_scale + 0.1)
            return self._draw_preview_animation(canvas, preview_w, preview_h,
                                                target_x1, target_y1, target_x2, target_y2)
        else:
            return self._draw_preview_static(canvas, preview_w, preview_h,
                                             target_x1, target_y1, target_x2, target_y2)

    def _draw_preview_animation(self, canvas, pw, ph, x1, y1, x2, y2):
        """绘制预览图动画"""
        scale = self.preview_anim_scale
        anim_width = int(pw * scale)
        anim_height = int(ph * scale)

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        ax1 = center_x - anim_width // 2
        ay1 = center_y - anim_height // 2
        ax2 = ax1 + anim_width
        ay2 = ay1 + anim_height

        if anim_width > 0 and anim_height > 0:
            preview_resized = cv2.resize(self.preview_img, (anim_width, anim_height))
            canvas[ay1:ay2, ax1:ax2] = preview_resized

            border_color = tuple(int(c * (0.8 + 0.2 * abs(math.sin(self.ui_anim_time * 2))))
                                 for c in self.preview_border_color)
            cv2.rectangle(canvas, (ax1, ay1), (ax2, ay2), border_color, 3)

        return canvas

    def _draw_preview_static(self, canvas, pw, ph, x1, y1, x2, y2):
        """绘制静态预览图"""
        self.preview_rect = (x1, y1, x2, y2)
        canvas[y1:y2, x1:x2] = self.preview_img

        # 边框
        border_color = tuple(int(c * (0.8 + 0.2 * abs(math.sin(self.ui_anim_time * 2))))
                             for c in self.preview_border_color)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 3)

        # 角装饰
        corner_length = 15
        lines = [
            ((x1, y1), (x1 + corner_length, y1)),
            ((x1, y1), (x1, y1 + corner_length)),
            ((x2, y1), (x2 - corner_length, y1)),
            ((x2, y1), (x2, y1 + corner_length)),
            ((x1, y2), (x1 + corner_length, y2)),
            ((x1, y2), (x1, y2 - corner_length)),
            ((x2, y2), (x2 - corner_length, y2)),
            ((x2, y2), (x2, y2 - corner_length))
        ]
        for pt1, pt2 in lines:
            cv2.line(canvas, pt1, pt2, border_color, 2)

        # 提示文字
        if y1 - 35 >= 0:
            canvas = self.draw_text_safe(canvas, "点击预览", (x1, y1 - 35),
                                         18, self.theme['light'], shadow=True)

        return canvas

    def _apply_flash_effect(self, canvas):
        """应用闪光灯效果"""
        if self.flash_alpha > 0:
            flash_layer = np.ones_like(canvas) * 255
            elapsed = time.time() - self.flash_start_time

            if elapsed < self.flash_duration:
                self.flash_alpha = 1.0 - (elapsed / self.flash_duration)
                canvas = cv2.addWeighted(flash_layer, self.flash_alpha,
                                         canvas, 1 - self.flash_alpha, 0)
            else:
                self.flash_alpha = 0

        return canvas

    def _draw_status_messages(self, canvas, width, height):
        """绘制状态消息"""
        current_time = time.time()

        # 清理过期消息
        self.status_messages = [
            msg for msg in self.status_messages
            if current_time - msg['time'] < msg['duration']
        ]

        # 绘制消息
        message_y = 80
        for msg in reversed(self.status_messages[-3:]):
            elapsed = current_time - msg['time']
            alpha = 1.0 - (elapsed / msg['duration'])

            if alpha > 0:
                bg_y = message_y - 25
                cv2.rectangle(canvas, (20, bg_y), (width - 20, bg_y + 35),
                              tuple(int(c * 0.7) for c in self.theme['dark']), -1)

                # 根据消息类型选择颜色
                text = msg['text']
                if "保存" in text or "解锁" in text:
                    text_color = self.theme['success']
                elif "锁定" in text or "取消" in text:
                    text_color = self.theme['warning']
                elif "退出" in text:
                    text_color = self.theme['danger']
                elif "微笑" in text:
                    text_color = self.theme['accent']
                else:
                    text_color = self.theme['light']

                text_x = (width - 400) // 2 if width > 400 else 40
                canvas = self.draw_text_safe(canvas, text, (text_x, message_y),
                                             18, text_color, shadow=True)

                message_y += 40

        return canvas


def on_mouse_click(event, x, y, flags, param):
    """鼠标点击事件处理"""
    detector = param['det']

    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = detector.preview_rect

        if x1 <= x <= x2 and y1 <= y <= y2:
            if detector.last_photo_path and os.path.exists(detector.last_photo_path):
                try:
                    os.startfile(os.path.abspath(detector.last_photo_path))
                    detector.add_status_message(f"已打开照片: {os.path.basename(detector.last_photo_path)}")
                except AttributeError:
                    import subprocess
                    subprocess.call(('open' if os.name == 'posix' else 'xdg-open',
                                     detector.last_photo_path))
                    detector.add_status_message(f"已打开照片: {os.path.basename(detector.last_photo_path)}")


def main():
    """主程序"""
    detector = SmartCameraUltimate()
    cap = cv2.VideoCapture(0)

    # 摄像头设置
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 获取分辨率
    ret, frame = cap.read()
    if not ret:
        print("无法连接摄像头")
        return

    h, w = frame.shape[:2]
    final_h, final_w = h + 160, w + 80

    # 窗口设置
    window_name = "AI智能相机"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 设置鼠标回调
    cv2.setMouseCallback(window_name, on_mouse_click, param={'det': detector})

    # 打印启动信息
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

    # 主循环
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 处理帧
        clean, display, gesture = detector.process_frame(frame)

        # 状态处理
        if detector.app_state == "RUNNING":
            # 退出检测
            if (detector.exit_confirm_start != 0 and
                    time.time() - detector.exit_confirm_start >= detector.exit_threshold):
                detector.add_status_message("正在退出程序...")
                print("程序退出")
                time.sleep(0.5)
                break

            # 手势触发拍照
            current_time = time.time()
            if (gesture == "V" and last_gesture_state != "V" and
                    not detector.is_counting_down and
                    current_time - detector.last_v_photo_time >= detector.v_photo_cooldown):
                detector.is_counting_down = True
                detector.countdown_start_time = current_time
                detector.smile_trigger_enabled = False
                detector.last_v_photo_time = current_time
                detector.add_status_message("剪刀手检测，开始倒计时拍照...")

            # 更新手势状态
            last_gesture_state = gesture if gesture is not None else "NONE"

            # 倒计时处理
            if detector.is_counting_down:
                if gesture == "OPEN":
                    detector.is_counting_down = False
                    detector.smile_trigger_enabled = False
                    detector.add_status_message("拍照已取消")
                elif time.time() - detector.countdown_start_time >= 3:
                    detector._handle_photo_capture(clean)

        # 显示画面
        cv2.imshow(window_name, display)

        # 按键处理
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
            except:
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

    # 清理资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()