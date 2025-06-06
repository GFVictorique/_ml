import logging.config 
import sys
import os
sys.path.append("C:/YOLO_anime_project/yolov5")  # 匯入 yolov5 模型
BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
YOLOV5_DIR = os.path.join(BASE_DIR, 'yolov5')

if YOLOV5_DIR not in sys.path:
    sys.path.append(YOLOV5_DIR)
import time
import torch
import numpy as np
from collections import defaultdict
from PyQt5.QtCore import Qt, QTimer, QRect
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QGuiApplication
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
import cv2
from mss import mss
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from ultralytics.utils.plotting import colors

# YOLO 模型設定
BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
MODEL_PATH = os.path.join(BASE_DIR, "best_anime.pt")
CLASSES = ['Mutsumi', 'Soyo', 'Anon', 'Taki', 'Tomori', 'Rana', 'Sakiko', 'Umiri', 'Uika', 'Nyamu']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend(MODEL_PATH, device=device)
model.eval()
model.warmup(imgsz=(1, 3, 960, 1280))

# 統計資料
appear_time = defaultdict(list)
active = {}

def make_divisible(x, divisor=32):
    return int(np.ceil(x / divisor) * divisor)

class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anime Overlay")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 螢幕 DPI 與觀測
        self.sct = mss()
        screen = QGuiApplication.primaryScreen()
        geometry = screen.availableGeometry()
        dpr = screen.devicePixelRatio()

        self.screen_width = int(geometry.width() * dpr)
        self.screen_height = int(geometry.height() * dpr)

        self.setGeometry(geometry)

        self.capture_region = {
            "top":    int(geometry.top() * dpr),
            "left":   int(geometry.left() * dpr),
            "width":  self.screen_width,
            "height": self.screen_height
        }

        scale = 1/3
        self.model_w = make_divisible(int(self.screen_width * scale))
        self.model_h = make_divisible(int(self.model_w * self.screen_height / self.screen_width))

        self.prev_boxes = []
        self.prev_labels = []

        # 選單
        self.menu = QWidget(self)
        menu_width = int(self.screen_width * 0.08)
        menu_height = int(self.screen_height * 0.12)
        self.menu.setGeometry(self.screen_width - menu_width - 10, 10, menu_width, menu_height)
        self.menu.setStyleSheet("background-color: rgba(50, 50, 50, 180);")

        layout = QVBoxLayout()
        self.toggle_btn = QPushButton("顯示/隱藏")
        self.toggle_btn.setStyleSheet("background-color: black; color: white;")
        self.toggle_btn.clicked.connect(self.toggle_stats)
        layout.addWidget(self.toggle_btn)
        self.quit_btn = QPushButton("關閉程式")
        self.quit_btn.setStyleSheet("background-color: black; color: white;")
        self.quit_btn.clicked.connect(QApplication.quit)
        layout.addWidget(self.quit_btn)
        self.menu.setLayout(layout)

        font_size = max(10, int(self.screen_height * 0.015))  # 縮小字體
        self.toggle_btn.setFont(QFont("Arial", font_size))
        self.quit_btn.setFont(QFont("Arial", font_size))

        self.show_stats = True
        self.last_frame = None
        self.boxes = []
        self.labels = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_overlay)
        self.timer.start(int(1000 / 60))

    def toggle_stats(self):
        self.show_stats = not self.show_stats
        self.update()

    def update_overlay(self):
        now = time.time()
        img = np.array(self.sct.grab(self.capture_region))[:, :, :3]
        img_resized = img
        img_model_input = cv2.resize(img_resized, (self.model_w, self.model_h))

        img_tensor = torch.from_numpy(img_model_input[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            pred = model(img_tensor)
            pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45)[0]

        self.boxes = []
        self.labels = []
        seen_now = set()

        if pred is not None and len(pred):
            pred[:, :4] = scale_boxes((self.model_h, self.model_w), pred[:, :4], img_resized.shape[:2]).round()
            for *xyxy, conf, cls_id in pred:
                class_id = int(cls_id.item())
                name = CLASSES[class_id]
                seen_now.add(name)
                if name not in active:
                    active[name] = now
                label = f"{name} {conf:.2f}"
                self.boxes.append(xyxy)
                self.labels.append((label, class_id))

        for name in list(active.keys()):
            if name not in seen_now:
                appear_time[name].append((active.pop(name), now))

        self.last_frame = img_resized
        if self.boxes != self.prev_boxes or self.labels != self.prev_labels:
            self.prev_boxes = self.boxes.copy()
            self.prev_labels = self.labels.copy()
            self.update()

    def paintEvent(self, event):
        if self.last_frame is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        for (xyxy, (label, cls_id)) in zip(self.boxes, self.labels):
            x1, y1, x2, y2 = map(int, xyxy)
            pen = QPen(QColor(*colors(cls_id, True)))
            pen.setWidth(6)
            painter.setPen(pen)

            width = x2 - x1
            height = y2 - y1
            size = min(width, height)
            x_center = (x1 + x2) // 2
            y_center = (y1 + y2) // 2
            x1_square = x_center - size // 2
            y1_square = y_center - size // 2

            painter.drawRect(QRect(x1_square, y1_square, size, size))
            # ✅ 加在這裡調整字體大小
            font_size = max(10, int(self.screen_height * 0.02))  # 這裡控制框旁文字大小
            painter.setFont(QFont("Arial", font_size))
            painter.drawText(x1_square, y1_square - 5, label)

        if self.show_stats:
            font_size = max(10, int(self.screen_height * 0.015))  # 縮小統計字體
            painter.setFont(QFont("Arial", font_size))
            line_spacing = int(self.screen_height * 0.05)  # 縮小間距
            y = line_spacing
            now = time.time()
            for cls in CLASSES:
                total = sum(end - start for start, end in appear_time[cls])
                if cls in active:
                    total += now - active[cls]
                text = f"{cls}: {total:.1f} 秒"
                font_metrics = painter.fontMetrics()
                text_width = font_metrics.width(text)
                text_height = font_metrics.height()

                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, 0, 0, 180))
                painter.drawRect(15, y - text_height + 8, text_width + 10, text_height)

                painter.setPen(QColor(255, 255, 255))
                painter.drawText(20, y, text)
                y += line_spacing

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OverlayWindow()
    window.showFullScreen()
    sys.exit(app.exec_())