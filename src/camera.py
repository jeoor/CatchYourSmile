"""SmartCameraUltimate — core orchestrator for the CatchYourSmile camera app."""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import math
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

import config
from gesture import check_gestures, GestureStabilizer
from smile_detector import detect_smile
from image_utils import (
    find_chinese_font, draw_rounded_rect, create_gradient_background,
    add_timestamp_to_image,
)


class SmartCameraUltimate:
    def __init__(self):
        self._init_models()
        self._init_state_variables()
        self._init_ui_variables()
        self._init_timer_thresholds()
        self._init_preview_variables()
        self._init_environment()
        self._init_status_system()

    def _init_models(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=config.HAND_MAX_NUM,
            min_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE,
        )
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=config.FACE_MAX_NUM,
            min_detection_confidence=config.FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_TRACKING_CONFIDENCE,
        )

    def _init_state_variables(self):
        self.app_state = "START_SCREEN"
        self.is_counting_down = False
        self.countdown_start_time = 0
        self.flash_alpha = 0
        self.flash_start_time = 0
        self.smile_detection_mode = True
        self.last_smile_time = 0
        self.smile_threshold = 0.5
        self.smile_sensitivity = 0.5
        self.smile_counter = 0
        self.smile_trigger_enabled = False
        self.last_no_smile_time = 0
        self.last_smile_state = False

    def _init_ui_variables(self):
        self.ui_anim_time = 0
        self.preview_anim_scale = 0
        self.button_pulse = 0
        self.gesture_stabilizer = GestureStabilizer(
            config.GESTURE_BUFFER_SIZE, config.GESTURE_MIN_CONSENSUS
        )
        self.waiting_for_release = False

    def _init_timer_thresholds(self):
        self.last_lock_time = 0
        self.exit_confirm_start = 0
        self.last_v_photo_time = 0

    def _init_preview_variables(self):
        self.last_photo_path = ""
        self.preview_img = None
        self.preview_display_time = 0
        self.preview_rect = (0, 0, 0, 0)
        self.preview_border_color = (0, 180, 255)

    def _init_environment(self):
        self.zh_font = find_chinese_font(config.FONT_PATHS)
        self.save_dir = config.SAVE_DIR
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.theme = config.THEME

    def _init_status_system(self):
        self.status_messages = []

    def add_status_message(self, message):
        self.status_messages.append({'text': message, 'time': time.time()})
        print(f"状态: {message}")

    def draw_text_safe(self, img, text, pos, size=22, color=(255, 255, 255),
                       shadow=False, shadow_color=(0, 0, 0), align="left"):
        x, y = pos
        has_chinese = any('一' <= c <= '鿿' for c in text)
        if has_chinese and self.zh_font:
            return self._draw_chinese_text(img, text, (x, y), size, color,
                                           shadow, shadow_color, align)
        else:
            return self._draw_english_text(img, text, (x, y), size, color,
                                           shadow, shadow_color)

    def _draw_chinese_text(self, img, text, pos, size, color, shadow,
                           shadow_color, align):
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype(self.zh_font, size)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            x, y = pos
            if align == "center":
                x = x - tw // 2
            elif align == "right":
                x = x - tw
            if shadow:
                draw.text((x + 2, y + 2 - bbox[3]), text, fill=shadow_color, font=font)
            draw.text((x, y - bbox[3]), text, fill=color, font=font)
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return self._draw_english_text(img, text, pos, size, color, shadow, shadow_color)

    def _draw_english_text(self, img, text, pos, size, color, shadow, shadow_color):
        font_scale = size / 30
        if shadow:
            cv2.putText(img, text, (pos[0] + 2, pos[1] + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow_color, 2)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        return img

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_res = self.hands.process(img_rgb)
        face_res = (self.face_mesh.process(img_rgb) if self.smile_detection_mode else None)
        display, video_info = self._create_display_canvas(frame, h, w)
        vx, vy, video_w, video_h = video_info
        gesture = self._detect_gesture(hand_res, display, vx, vy, video_w, video_h)
        self._detect_smile_trigger(face_res, video_w, video_h, vx, vy, display)
        self.ui_anim_time += 0.05
        self.button_pulse = abs(math.sin(self.ui_anim_time)) * 0.3 + 0.7
        display = self._draw_border_info(display, display.shape[1], display.shape[0])
        if self.app_state == "START_SCREEN":
            display = self._draw_start_ui(display, gesture, display.shape[1], display.shape[0])
        else:
            display = self._draw_running_ui(display, gesture, (vx, vy, w, h),
                                            display.shape[1], display.shape[0])
            display = self._draw_photo_preview(display, display.shape[1], display.shape[0])
        display = self._apply_flash_effect(display)
        display = self._draw_status_messages(display, display.shape[1], display.shape[0])
        return frame, display, gesture

    def _create_display_canvas(self, frame, h, w):
        display_h, display_w = h + 160, w + 80
        display = create_gradient_background(display_w, display_h, self.theme['dark'])
        vx, vy = 40, 80
        border_size = 2
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
        raw_gesture = None
        if hand_res.multi_hand_landmarks:
            landmarks = hand_res.multi_hand_landmarks[0].landmark
            raw_gesture = check_gestures(landmarks)
            self._draw_gesture_indicator(display, landmarks, vx, vy, video_w, video_h, raw_gesture)
        return self.gesture_stabilizer.update(raw_gesture)

    def _draw_gesture_indicator(self, display, landmarks, vx, vy, video_w, video_h, gesture):
        px = int(landmarks[9].x * video_w) + vx + 2
        py = int(landmarks[9].y * video_h) + vy + 2
        status_color = self.theme['success'] if gesture else self.theme['gray']
        pulse = abs(math.sin(time.time() * 3)) * 0.3 + 0.7 if gesture else 0.5
        main_color = tuple(int(c * pulse * 0.6) for c in status_color)
        cv2.circle(display, (px, py), 35, main_color, 2, cv2.LINE_AA)
        cv2.circle(display, (px, py), 25, tuple(int(c * 0.8) for c in status_color), 1, cv2.LINE_AA)
        if gesture:
            (tw, th), _ = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            tx = px - tw // 2
            ty = py - 50
            cv2.rectangle(display, (tx - 5, ty - th - 5), (tx + tw + 5, ty + 5), (0, 0, 0), -1)
            cv2.putText(display, gesture, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    def _detect_smile_trigger(self, face_res, video_w, video_h, vx, vy, display):
        if (not self.smile_detection_mode or not face_res or
                not face_res.multi_face_landmarks):
            return
        current_time = time.time()
        in_cooldown = (current_time - self.last_smile_time < config.SMILE_COOLDOWN)
        for face_landmarks in face_res.multi_face_landmarks:
            is_smiling, smile_score, mouth_points = detect_smile(
                face_landmarks, video_w, video_h, self.smile_threshold, self.smile_sensitivity)
            self._update_smile_state(is_smiling, current_time, in_cooldown)
            if is_smiling:
                self.draw_smile_indicator(display, mouth_points, vx + 2, vy + 2, video_w, video_h)

    def _update_smile_state(self, is_smiling, current_time, in_cooldown):
        if is_smiling:
            self.smile_counter += 1
            self.last_no_smile_time = 0
            if (self.smile_counter >= config.SMILE_FRAMES_NEEDED and not in_cooldown and
                    not self.is_counting_down and self.app_state == "RUNNING" and
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
            if current_time - self.last_no_smile_time > config.NO_SMILE_RESET_DELAY:
                self.smile_counter = max(0, self.smile_counter - 2)
            else:
                self.smile_counter = max(0, self.smile_counter - 1)
        self.last_smile_state = is_smiling

    def draw_smile_indicator(self, canvas, mouth_points, vx, vy, w, h):
        lx, ly, rx, ry, tx, ty, bx, by = mouth_points
        mouth_center_x = (lx + rx) // 2 + vx
        mouth_center_y = (ty + by) // 2 + vy
        pulse = 0.8 + 0.2 * abs(math.sin(time.time() * 3))
        circle_color = tuple(int(c * pulse) for c in self.theme['accent'])
        cv2.circle(canvas, (mouth_center_x, mouth_center_y), 25, circle_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (mouth_center_x, mouth_center_y), 15,
                   tuple(int(c * 0.6) for c in circle_color), 1, cv2.LINE_AA)
        return canvas

    def _handle_photo_capture(self, clean, shortcut=False):
        self.flash_alpha = 1.0
        self.flash_start_time = time.time()
        clean_with_timestamp, filename = add_timestamp_to_image(clean)
        filename = f"{filename}.jpg"
        save_path = os.path.join(self.save_dir, filename)
        cv2.imwrite(save_path, clean_with_timestamp)
        self.last_photo_path = save_path
        self.preview_display_time = time.time()
        preview_h, preview_w = 120, 160
        self.preview_img = cv2.resize(clean_with_timestamp, (preview_w, preview_h))
        self.preview_anim_scale = 0
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
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        canvas = self.draw_text_safe(canvas, "AI智能相机", (30, 40),
                                     24, self.theme['primary'], shadow=True)
        canvas = self.draw_text_safe(canvas, now, (width - 230, 35),
                                     16, self.theme['light'], shadow=True)
        if self.app_state == "RUNNING":
            guide_text = "剪刀手拍照 | 微笑拍照 | 握拳退出"
            guide_color = self.theme['secondary']
        else:
            guide_text = "系统已锁定 - 请比出OK手势解锁"
            guide_color = self.theme['gray']
        guide_x = (width - 400) // 2 if width > 400 else 20
        canvas = self.draw_text_safe(canvas, guide_text, (guide_x, height - 30),
                                     20, guide_color, shadow=True)
        canvas = self.draw_text_safe(canvas, f"目录: {self.save_dir}/",
                                     (width - 250, height - 55),
                                     14, self.theme['gray'], shadow=True)
        return canvas

    def _draw_start_ui(self, canvas, gesture, width, height):
        cx, cy = width // 2, height // 2
        overlay = canvas.copy()
        dark_layer = np.zeros_like(overlay)
        dark_layer[:] = tuple(int(c * 0.5) for c in self.theme['dark'])
        canvas = cv2.addWeighted(canvas, 0.4, dark_layer, 0.6, 0)
        elapsed = time.time() - self.last_lock_time
        in_cooldown = elapsed < config.UNLOCK_COOLDOWN
        if in_cooldown:
            color = self.theme['gray']
            status_msg = f"冷却中 ({int(config.UNLOCK_COOLDOWN - elapsed) + 1}秒)"
            pulse_factor = 0.5
        elif self.waiting_for_release:
            color = self.theme['warning']
            status_msg = "请收回手势"
            pulse_factor = 0.7 + 0.3 * abs(math.sin(self.ui_anim_time * 2))
        else:
            color = self.theme['primary']
            status_msg = "系统已锁定"
            pulse_factor = 0.8 + 0.2 * abs(math.sin(self.ui_anim_time))
        card_width = min(400, width - 40)
        card_height = min(300, height - 40)
        card_x = cx - card_width // 2
        card_y = cy - card_height // 2
        if (0 <= card_x < width and 0 <= card_y < height and
                card_x + card_width <= width and card_y + card_height <= height):
            self._draw_card(canvas, card_x, card_y, card_width, card_height,
                            color, pulse_factor, cx, cy, in_cooldown)
            canvas = self.draw_text_safe(canvas, status_msg, (cx - 60, cy - 20),
                                         24, tuple(int(c * pulse_factor) for c in color),
                                         shadow=True)
            if in_cooldown:
                sub_msg = "请稍候..."
            elif self.waiting_for_release:
                sub_msg = "检测到手势已释放"
            else:
                sub_msg = "比出OK手势解锁系统"
            canvas = self.draw_text_safe(canvas, sub_msg, (cx - 100, cy + 60),
                                         18, self.theme['light'], shadow=True)
        if not in_cooldown and not self.waiting_for_release and gesture == "OK":
            self.app_state = "RUNNING"
            self.add_status_message("系统已解锁")
        return canvas

    def _draw_card(self, canvas, x, y, w, h, color, pulse_factor, cx, cy, in_cooldown):
        card_bg = np.zeros((h, w, 3), dtype=np.uint8)
        card_bg[:] = tuple(int(c * 0.8) for c in self.theme['dark'])
        canvas[y:y + h, x:x + w] = cv2.addWeighted(canvas[y:y + h, x:x + w], 0.3, card_bg, 0.7, 0)
        border_color = tuple(int(c * pulse_factor) for c in color)
        draw_rounded_rect(canvas, (x, y), (x + w, y + h), border_color, 15, 2)
        ring_radius = min(100, w // 4, h // 4)
        ring_thickness = 8
        if in_cooldown:
            progress = 1 - ((time.time() - self.last_lock_time) / config.UNLOCK_COOLDOWN)
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
        vx, vy, vw, vh = video_rect
        current_time = time.time()
        if gesture == "FIST":
            self._draw_exit_progress(canvas, vx, vy, vw, vh, current_time)
        else:
            self.exit_confirm_start = 0
        if self.is_counting_down:
            self._draw_countdown(canvas, vx, vy, vw, vh, current_time)
        return canvas

    def _draw_exit_progress(self, canvas, vx, vy, vw, vh, current_time):
        if self.exit_confirm_start == 0:
            self.exit_confirm_start = current_time
        elapsed = current_time - self.exit_confirm_start
        progress = min(elapsed / config.EXIT_THRESHOLD, 1.0)
        cx = vx + vw // 2
        cy = vy + vh // 2
        outer_radius = min(100, vw // 4, vh // 4)
        if progress > 0:
            bg_thickness = 20
            bg_color = tuple(int(c * 0.2) for c in self.theme['dark'])
            cv2.ellipse(canvas, (cx, cy), (outer_radius, outer_radius),
                        0, 0, 360, bg_color, bg_thickness, cv2.LINE_AA)
            end_angle = int(360 * progress)
            pulse = 0.8 + 0.2 * abs(math.sin(current_time * 5))
            progress_color = tuple(int(c * pulse) for c in self.theme['danger'])
            cv2.ellipse(canvas, (cx, cy), (outer_radius, outer_radius),
                        0, 0, end_angle, progress_color, bg_thickness, cv2.LINE_AA)
            if progress < 1.0:
                cv2.putText(canvas, f"{int(progress * 100)}%",
                            (cx - 50, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            self.theme['danger'], 3, cv2.LINE_AA)
                remaining = config.EXIT_THRESHOLD - elapsed
                cv2.putText(canvas, f"{remaining:.1f}s",
                            (cx - 30, cy + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            self.theme['light'], 2, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy + 65), 15, self.theme['danger'], 2, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy + 65), 5, self.theme['danger'], -1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, "EXITING...", (cx - 80, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.theme['danger'], 3, cv2.LINE_AA)
        border_color = tuple(int(c * 0.5) for c in self.theme['dark'])
        cv2.circle(canvas, (cx, cy), outer_radius + 10, border_color, 1, cv2.LINE_AA)

    def _draw_countdown(self, canvas, vx, vy, vw, vh, current_time):
        remaining = 3 - int(current_time - self.countdown_start_time)
        if remaining <= 0:
            return
        radius = min(80, vw // 2, vh // 2)
        cx, cy = vx + vw // 2, vy + vh // 2
        cv2.circle(canvas, (cx, cy), radius,
                   tuple(int(c * 0.2) for c in self.theme['dark']), -1, cv2.LINE_AA)
        ring_progress = 1 - (current_time - self.countdown_start_time) / 3
        end_angle = int(360 * ring_progress)
        if self.smile_trigger_enabled:
            ring_color = tuple(int(c * (0.8 + 0.2 * abs(math.sin(self.ui_anim_time * 5))))
                              for c in self.theme['accent'])
            text_color = self.theme['accent']
        else:
            ring_color = tuple(int(c * (0.8 + 0.2 * abs(math.sin(self.ui_anim_time * 5))))
                              for c in self.theme['success'])
            text_color = self.theme['success']
        cv2.ellipse(canvas, (cx, cy), (radius, radius), 0, 0, end_angle, ring_color, 8, cv2.LINE_AA)
        text = str(remaining)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 4, 5)
        tx = cx - tw // 2
        ty = cy + th // 2
        cv2.putText(canvas, text, (tx + 4, ty + 4), cv2.FONT_HERSHEY_TRIPLEX, 4, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_TRIPLEX, 4, text_color, 5, cv2.LINE_AA)

    def _draw_photo_preview(self, canvas, width, height):
        if (self.preview_img is None or
                time.time() - self.preview_display_time >= config.PREVIEW_DISPLAY_DURATION):
            self.preview_anim_scale = 0
            return canvas
        preview_h, preview_w = 120, 160
        x1, y1 = width - preview_w - 25, height - preview_h - 85
        x2, y2 = width - 25, height - 85
        if not (0 <= x1 < width and 0 <= y1 < height and x2 <= width and y2 <= height):
            return canvas
        if self.preview_anim_scale < 1.0:
            self.preview_anim_scale = min(1.0, self.preview_anim_scale + 0.1)
            return self._draw_preview_animation(canvas, preview_w, preview_h, x1, y1, x2, y2)
        else:
            return self._draw_preview_static(canvas, preview_w, preview_h, x1, y1, x2, y2)

    def _draw_preview_animation(self, canvas, pw, ph, x1, y1, x2, y2):
        scale = self.preview_anim_scale
        aw = int(pw * scale)
        ah = int(ph * scale)
        mcx = (x1 + x2) // 2
        mcy = (y1 + y2) // 2
        ax1 = mcx - aw // 2
        ay1 = mcy - ah // 2
        ax2 = ax1 + aw
        ay2 = ay1 + ah
        if aw > 0 and ah > 0:
            preview_resized = cv2.resize(self.preview_img, (aw, ah))
            canvas[ay1:ay2, ax1:ax2] = preview_resized
            border_color = tuple(int(c * (0.8 + 0.2 * abs(math.sin(self.ui_anim_time * 2))))
                                for c in self.preview_border_color)
            cv2.rectangle(canvas, (ax1, ay1), (ax2, ay2), border_color, 3)
        return canvas

    def _draw_preview_static(self, canvas, pw, ph, x1, y1, x2, y2):
        self.preview_rect = (x1, y1, x2, y2)
        canvas[y1:y2, x1:x2] = self.preview_img
        border_color = tuple(int(c * (0.8 + 0.2 * abs(math.sin(self.ui_anim_time * 2))))
                            for c in self.preview_border_color)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 3)
        cl = 15
        for pt1, pt2 in [((x1, y1), (x1 + cl, y1)), ((x1, y1), (x1, y1 + cl)),
                         ((x2, y1), (x2 - cl, y1)), ((x2, y1), (x2, y1 + cl)),
                         ((x1, y2), (x1 + cl, y2)), ((x1, y2), (x1, y2 - cl)),
                         ((x2, y2), (x2 - cl, y2)), ((x2, y2), (x2, y2 - cl))]:
            cv2.line(canvas, pt1, pt2, border_color, 2)
        if y1 - 35 >= 0:
            canvas = self.draw_text_safe(canvas, "点击预览", (x1, y1 - 35),
                                         18, self.theme['light'], shadow=True)
        return canvas

    def _apply_flash_effect(self, canvas):
        if self.flash_alpha > 0:
            flash_layer = np.ones_like(canvas) * 255
            elapsed = time.time() - self.flash_start_time
            if elapsed < config.FLASH_DURATION:
                self.flash_alpha = 1.0 - (elapsed / config.FLASH_DURATION)
                canvas = cv2.addWeighted(flash_layer, self.flash_alpha, canvas, 1 - self.flash_alpha, 0)
            else:
                self.flash_alpha = 0
        return canvas

    def _draw_status_messages(self, canvas, width, height):
        current_time = time.time()
        self.status_messages = [m for m in self.status_messages
                                if current_time - m['time'] < config.STATUS_DURATION]
        msg_y = 80
        for msg in reversed(self.status_messages[-3:]):
            elapsed = current_time - msg['time']
            alpha = 1.0 - (elapsed / config.STATUS_DURATION)
            if alpha > 0:
                bg_y = msg_y - 25
                cv2.rectangle(canvas, (20, bg_y), (width - 20, bg_y + 35),
                              tuple(int(c * 0.7) for c in self.theme['dark']), -1)
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
                tx = (width - 400) // 2 if width > 400 else 40
                canvas = self.draw_text_safe(canvas, text, (tx, msg_y), 18, text_color, shadow=True)
                msg_y += 40
        return canvas
