"""
app.py — 手写数字识别可视化小程序（基于训练好的 CNN）

功能：
    1. 运行后弹出一个手写面板，鼠标左键在白色画板上书写 0~9 的数字；
    2. 松开鼠标后自动把笔画裁剪、等比缩放并居中到 28×28 灰度图（MNIST 格式）；
    3. 使用项目里训练好的 best_mnist_cnn.pth 进行预测，并实时显示结果与置信度。

说明：
    - 本文件为独立新增，不修改项目的其它任何文件。
    - 为保证启动速度，这里内联了与 model.py 中 MNIST_CNN 完全一致的结构，
      只读取权重文件，不会触发 utils.py 里加载整个数据集的耗时逻辑。
    - 关键：训练时用了 transforms.Normalize(mean=0.1307, std=0.3081)，
      因此送入模型前必须做同样的标准化，否则预测会失准。
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# 基础配置
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "best_mnist_cnn.pth")

# 与训练时保持一致的 MNIST 灰度标准化参数
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

# 画板绘制参数
CANVAS_SIZE = 280          # 画板边长（= 28 × 10，便于高分辨率书写）
BRUSH_RADIUS = 11          # 画笔半径（像素）
INK_THRESHOLD = 0.05       # 判断"有笔画"的阈值

# 配色（浅色风格）
COLOR_BG = "#f5f6f8"
COLOR_PANEL = "#ffffff"
COLOR_BORDER = "#d0d4dc"
COLOR_TEXT = "#22262e"
COLOR_MUTED = "#7a828f"
COLOR_ACCENT = "#2f6feb"
COLOR_BAR_BG = "#eceef2"


# ---------------------------------------------------------------------------
# 与 model.py 中 MNIST_CNN 完全一致的结构（仅用于加载权重）
# ---------------------------------------------------------------------------

class MNIST_CNN(nn.Module):
    def __init__(self, dropout_p=0.25):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),      # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),      # 14 -> 7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_trained_model(ckpt_path=CKPT_PATH, device="cpu"):
    """加载 best_mnist_cnn.pth，返回处于 eval 模式的模型。"""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"未找到权重文件：{ckpt_path}\n请先运行 train.py 生成 best_mnist_cnn.pth。"
        )

    ckpt = torch.load(ckpt_path, map_location=device)

    # 兼容 {"model_state_dict": ...} 与纯 state_dict 两种保存格式
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        dropout_p = ckpt.get("dropout_p", 0.25)
    else:
        state_dict = ckpt
        dropout_p = 0.25

    model = MNIST_CNN(dropout_p=dropout_p).to(device)
    model.load_state_dict(state_dict)
    model.eval()                     # 关闭 Dropout / BatchNorm 更新
    return model


# ---------------------------------------------------------------------------
# 主界面
# ---------------------------------------------------------------------------

class MNISTPanelApp:

    def __init__(self, model):
        self.model = model

        # 每个像素的"墨水量"，取值范围 0~1
        self.image = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
        self.last_point = None        # 上一个采样点，用于连线
        self.delay_id = None          # 自动预测的防抖计时器

        self._build_brush()
        self._build_ui()

    # ---------------------------------------------------------------- 画笔
    def _build_brush(self):
        """预生成一个带柔边的圆形笔刷，写入时更接近真实笔迹。"""
        r = BRUSH_RADIUS
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        dist = np.sqrt(xx ** 2 + yy ** 2)
        # 中心为 1，边缘在 2 像素内平滑衰减
        self.brush = np.clip((r - dist) / 2.0 + 0.5, 0.0, 1.0).astype(np.float32)
        self.brush_r = r

    def _stamp(self, x, y):
        """在 (x, y) 处盖上笔刷。"""
        r = self.brush_r
        x0, x1 = max(0, x - r), min(CANVAS_SIZE, x + r + 1)
        y0, y1 = max(0, y - r), min(CANVAS_SIZE, y + r + 1)
        if x0 >= x1 or y0 >= y1:
            return

        bx0, bx1 = x0 - (x - r), x1 - (x - r)
        by0, by1 = y0 - (y - r), y1 - (y - r)
        region = self.image[y0:y1, x0:x1]
        np.maximum(region, self.brush[by0:by1, bx0:bx1], out=region)

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("MNIST 手写数字识别 · CNN")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(False, False)

        title_font = tkfont.Font(family="Microsoft YaHei UI", size=15, weight="bold")
        normal_font = tkfont.Font(family="Microsoft YaHei UI", size=10)
        digit_font = tkfont.Font(family="Microsoft YaHei UI", size=40, weight="bold")

        # ---- 顶部标题 ----
        header = tk.Frame(self.root, bg=COLOR_BG)
        header.pack(fill="x", padx=18, pady=(16, 6))
        tk.Label(header, text="手写数字识别", font=title_font,
                 bg=COLOR_BG, fg=COLOR_TEXT).pack(anchor="w")
        tk.Label(header, text="在左侧白色画板上写一个数字（0~9），松开鼠标即自动识别",
                 font=normal_font, bg=COLOR_BG, fg=COLOR_MUTED).pack(anchor="w", pady=(2, 0))

        body = tk.Frame(self.root, bg=COLOR_BG)
        body.pack(padx=18, pady=8)

        # ---- 左侧：画板 ----
        left = tk.Frame(body, bg=COLOR_BG)
        left.pack(side="left", anchor="n")

        canvas_border = tk.Frame(left, bg=COLOR_BORDER, bd=0)
        canvas_border.pack()
        self.canvas = tk.Canvas(canvas_border, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="white", highlightthickness=0, cursor="pencil")
        self.canvas.pack(padx=1, pady=1)
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        btns = tk.Frame(left, bg=COLOR_BG)
        btns.pack(fill="x", pady=(10, 0))
        tk.Button(btns, text="重写", width=12, font=normal_font,
                  command=self.clear).pack(side="left")
        tk.Button(btns, text="识别", width=12, font=normal_font,
                  command=self.predict).pack(side="right")

        # ---- 右侧：结果区 ----
        right = tk.Frame(body, bg=COLOR_BG)
        right.pack(side="left", anchor="n", padx=(18, 0))

        result_card = tk.Frame(right, bg=COLOR_PANEL, highlightthickness=1,
                               highlightbackground=COLOR_BORDER)
        result_card.pack(fill="x")

        tk.Label(result_card, text="识别结果", font=normal_font,
                 bg=COLOR_PANEL, fg=COLOR_MUTED).pack(anchor="w", padx=14, pady=(10, 0))
        self.result_digit = tk.Label(result_card, text="—", font=digit_font,
                                     bg=COLOR_PANEL, fg=COLOR_ACCENT)
        self.result_digit.pack(anchor="w", padx=14)
        self.result_conf = tk.Label(result_card, text="置信度 —",
                                    font=normal_font, bg=COLOR_PANEL, fg=COLOR_MUTED)
        self.result_conf.pack(anchor="w", padx=14, pady=(0, 12))

        # 概率分布柱状图
        tk.Label(right, text="各类别概率", font=normal_font,
                 bg=COLOR_BG, fg=COLOR_MUTED).pack(anchor="w", pady=(12, 4))
        self.prob_canvas = tk.Canvas(right, width=250, height=200, bg=COLOR_PANEL,
                                     highlightthickness=1, highlightbackground=COLOR_BORDER)
        self.prob_canvas.pack()
        self._draw_probabilities(None)

        # 模型实际看到的 28×28 输入预览
        tk.Label(right, text="模型输入 (28×28)", font=normal_font,
                 bg=COLOR_BG, fg=COLOR_MUTED).pack(anchor="w", pady=(12, 4))
        self.preview = tk.Canvas(right, width=112, height=112, bg="white",
                                 highlightthickness=1, highlightbackground=COLOR_BORDER)
        self.preview.pack()
        self._preview_photo = None

    # ---------------------------------------------------------------- 绘制事件
    def _on_press(self, event):
        self._cancel_pending()
        self.last_point = (event.x, event.y)
        self._paint_segment(event.x, event.y, event.x, event.y)
        self._reset_result()

    def _on_drag(self, event):
        if self.last_point is None:
            self.last_point = (event.x, event.y)
        x0, y0 = self.last_point
        self._paint_segment(x0, y0, event.x, event.y)
        self.last_point = (event.x, event.y)

    def _on_release(self, event):
        self.last_point = None
        # 松开后稍作停顿再识别，避免笔画间断触发
        self._schedule_predict(320)

    def _paint_segment(self, x0, y0, x1, y1):
        """在两点之间插值绘制，保证快速划动时笔迹连续。"""
        dx, dy = x1 - x0, y1 - y0
        dist = max(abs(dx), abs(dy), 1)
        step = max(1, self.brush_r // 2)
        n = int(dist // step) + 1
        for i in range(n + 1):
            t = i / n
            x = int(round(x0 + dx * t))
            y = int(round(y0 + dy * t))
            self._stamp(x, y)

        # 界面同步绘制（粗线 + 圆头，视觉更顺滑）
        self.canvas.create_line(x0, y0, x1, y1, fill="black",
                                width=self.brush_r * 2,
                                capstyle="round", joinstyle="round")
        self.canvas.create_oval(x1 - self.brush_r, y1 - self.brush_r,
                                x1 + self.brush_r, y1 + self.brush_r,
                                fill="black", outline="black")

    # ---------------------------------------------------------------- 预处理
    def preprocess(self):
        """把画板内容裁剪、等比缩放至长边 20 像素，并居中放入 28×28 灰度图。"""
        ys, xs = np.where(self.image > INK_THRESHOLD)
        if len(xs) == 0:
            return None

        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        digit = self.image[y0:y1, x0:x1]

        h, w = digit.shape
        scale = 20.0 / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        resample = getattr(Image, "Resampling", Image).LANCZOS
        im = Image.fromarray((digit * 255).astype(np.uint8), mode="L")
        im = im.resize((new_w, new_h), resample)

        small = np.asarray(im, dtype=np.float32) / 255.0

        # 按 MNIST 规范居中（放在 20×20 框内再整体居中）
        out = np.zeros((28, 28), dtype=np.float32)
        top = (28 - new_h) // 2
        left = (28 - new_w) // 2
        out[top:top + new_h, left:left + new_w] = small
        return out

    # ---------------------------------------------------------------- 预测
    def _schedule_predict(self, delay_ms):
        self._cancel_pending()
        self.delay_id = self.root.after(delay_ms, self.predict)

    def _cancel_pending(self):
        if self.delay_id is not None:
            self.root.after_cancel(self.delay_id)
            self.delay_id = None

    def predict(self):
        self.delay_id = None

        img28 = self.preprocess()
        if img28 is None:
            return

        self._update_preview(img28)

        # 与训练一致的标准化，再送入模型
        normalized = (img28 - MNIST_MEAN) / MNIST_STD
        tensor = torch.from_numpy(normalized).view(1, 1, 28, 28).float()

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0].numpy()

        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        self.result_digit.config(text=str(pred))
        self.result_conf.config(text=f"置信度 {conf * 100:.1f}%")
        self._draw_probabilities(probs)

    # ---------------------------------------------------------------- 可视化
    def _reset_result(self):
        self.result_digit.config(text="—")
        self.result_conf.config(text="置信度 —")

    def _update_preview(self, img28):
        resample = getattr(Image, "Resampling", Image).NEAREST
        im = Image.fromarray((img28 * 255).astype(np.uint8), mode="L")
        im = im.resize((112, 112), resample)
        self._preview_photo = ImageTk.PhotoImage(im)
        self.preview.delete("all")
        self.preview.create_image(0, 0, anchor="nw", image=self._preview_photo)

    def _draw_probabilities(self, probs):
        c = self.prob_canvas
        c.delete("all")

        top_pad, row_h, label_w = 10, 18, 26
        bar_x = label_w
        bar_max = 250 - label_w - 52

        for i in range(10):
            y = top_pad + i * row_h
            p = 0.0 if probs is None else float(probs[i])
            is_best = probs is not None and int(np.argmax(probs)) == i

            c.create_text(bar_x - 8, y + row_h / 2 - 2, text=str(i), anchor="e",
                          fill=COLOR_TEXT if is_best else COLOR_MUTED,
                          font=("Microsoft YaHei UI", 9, "bold" if is_best else "normal"))

            c.create_rectangle(bar_x, y, bar_x + bar_max, y + row_h - 5,
                               fill=COLOR_BAR_BG, outline="")

            if p > 0:
                w = max(2, bar_max * p)
                c.create_rectangle(bar_x, y, bar_x + w, y + row_h - 5,
                                   fill=COLOR_ACCENT if is_best else "#9db6ee", outline="")

            c.create_text(bar_x + bar_max + 8, y + row_h / 2 - 2,
                          text=f"{p * 100:4.1f}%", anchor="w",
                          fill=COLOR_TEXT if is_best else COLOR_MUTED,
                          font=("Microsoft YaHei UI", 9))

    # ---------------------------------------------------------------- 清空
    def clear(self):
        self._cancel_pending()
        self.canvas.delete("all")
        self.image.fill(0.0)
        self.last_point = None
        self._reset_result()
        self._draw_probabilities(None)
        self.preview.delete("all")
        self._preview_photo = None

    # ---------------------------------------------------------------- 运行
    def run(self):
        self.root.mainloop()


def main():
    model = load_trained_model()
    app = MNISTPanelApp(model)
    app.run()


if __name__ == "__main__":
    main()
