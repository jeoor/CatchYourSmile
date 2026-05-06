# CatchYourSmile

基于 MediaPipe + OpenCV 的 AI 智能相机，通过手势识别和微笑检测触发拍照。

## 功能

- **手势拍照**：剪刀手手势触发 3 秒倒计时拍照
- **微笑自动拍照**：检测到微笑后自动倒计时拍照
- **手势解锁**：OK 手势解锁系统
- **握拳退出**：握拳 5 秒退出程序
- **快捷键拍照**：按 `s` 键快速拍照
- **预览查看**：点击右下角预览缩略图打开照片

## 环境要求

- Python 3.11
- 摄像头

## 安装依赖

```bash
# 创建虚拟环境
conda create -n CatchYourSmile python=3.11
conda activate CatchYourSmile

# 安装依赖
pip install -r requirements.txt
```

## 运行

```bash
python src/main.py
```

## 操作说明

| 操作 | 说明 |
|------|------|
| 剪刀手手势 | 倒计时拍照 |
| 微笑 | 自动倒计时拍照 |
| OK 手势 | 解锁系统 |
| 握拳 5 秒 | 退出程序 |
| `s` | 快速拍照 |
| `1` / `2` | 增加/降低微笑检测灵敏度 |
| `o` | 打开照片保存目录 |
| `h` | 显示操作提示 |
| `q` | 退出程序 |

## 项目结构

```
├── src/
│   ├── main.py              # 入口，摄像头初始化，主循环
│   ├── camera.py            # SmartCameraUltimate 类，核心编排
│   ├── config.py            # 颜色主题、阈值、模型参数等常量
│   ├── gesture.py           # 手势识别（check_gestures + GestureStabilizer）
│   ├── smile_detector.py    # 微笑检测（嘴部关键点提取 + 微笑分数计算）
│   └── image_utils.py       # 图像处理工具（圆角矩形、渐变背景、水印）
├── requirements.txt         # Python 依赖
├── .gitignore
└── captured_photos/         # 照片保存目录（自动创建）
```

## 照片

拍照后照片保存在 `captured_photos/` 目录，文件名格式为 `YYYYMMDD_HHMMSS_mmm.jpg`。照片包含 "Keep smile everyday!" 标题和时间戳水印。
