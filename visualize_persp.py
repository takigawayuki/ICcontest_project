"""
visualize_persp.py  —  yolo_persp_crop.py 管线可视化

每张车牌 5 列：
  ① 原图局部（bbox+60px）+ 绿色检测框
  ② ROI（含 PAD 扩边）
  ③ 全图 adaptiveThreshold 裁出的 ROI 二值图 + 最大轮廓 + minAreaRect 4 角点
  ④ 透视变换中间图（376×96）
  ⑤ 最终输出 94×24（×6 放大）
"""

import os, sys, random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_persp_crop import (
    yolo_detect, find_quad, order_points, perspective_warp,
    get_plate_text, read_image,
    YOLO_MODEL, CCPD2019_DIR, CCPD2020_DIR,
)
from ultralytics import YOLO

N_SAMPLES   = 5
FINAL_SCALE = 6
DISP_H      = 150
MID_W, MID_H = 376, 96

STEP_NAMES = ["① 原图局部\nYOLO bbox",
              "② ROI\n(+PAD)",
              "③ 全图二值化ROI\n轮廓+minAreaRect",
              "④ 透视变换\n376×96",
              "⑤ 最终输出\n94×24"]

# rb lb lt rt 顺序
PT_COLORS_BGR = [(0, 0, 255), (0, 140, 255), (0, 210, 0), (255, 60, 0)]
PT_LABELS     = ['RB', 'LB', 'LT', 'RT']


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 \
           else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def fit_h(img, h):
    oh, ow = img.shape[:2]
    if oh == 0: return img
    return cv2.resize(img, (max(1, int(ow * h / oh)), h),
                      interpolation=cv2.INTER_NEAREST)

def draw_quad_on_img(img_bgr, quad_rb_lb_lt_rt, r=5):
    vis = img_bgr.copy()
    pts = quad_rb_lb_lt_rt.astype(np.int32)
    cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (x, y), r, PT_COLORS_BGR[i], -1)
        cv2.putText(vis, PT_LABELS[i], (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, PT_COLORS_BGR[i], 1, cv2.LINE_AA)
    return vis


def process_steps(model, src_path, filename):
    plate_text = get_plate_text(filename)
    if plate_text is None:
        return None
    img = read_image(src_path)
    if img is None:
        return None
    H, W = img.shape[:2]

    bbox = yolo_detect(model, img)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox

    roi = img[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return None

    # 颜色类型（仅用于标签）
    hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total = roi.shape[0] * roi.shape[1]
    br    = np.sum(cv2.inRange(hsv, (100,60,60),(130,255,255)) > 0) / total
    gr    = np.sum(cv2.inRange(hsv, (35, 60,60),(85, 255,255)) > 0) / total
    ptype = 'blue' if br > 0.05 and br >= gr else ('green' if gr > 0.05 else 'unknown')

    # ① 原图局部 + 检测框
    p0 = 60
    r0 = img[max(0,y1-p0):min(H,y2+p0), max(0,x1-p0):min(W,x2+p0)].copy()
    ox, oy = max(0, x1-p0), max(0, y1-p0)
    cv2.rectangle(r0, (x1-ox, y1-oy), (x2-ox, y2-oy), (0,255,0), 2)
    s0 = r0
    l0 = "bbox {}×{}  [{}]".format(x2-x1, y2-y1, ptype)

    # ② ROI
    s1 = roi
    l1 = "ROI {}×{}".format(x2-x1, y2-y1)

    # ③ 全图二值化 ROI + 最大轮廓 + minAreaRect
    binary_roi, box = find_quad(img, x1, y1, x2, y2)
    vis3 = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR)

    if box is not None:
        # 画最大轮廓（绿色）
        gray_full   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bin_full    = cv2.adaptiveThreshold(gray_full, 255,
                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv2.THRESH_BINARY_INV, 11, 2)
        bin_crop    = bin_full[y1:y2, x1:x2]
        contours, _ = cv2.findContours(bin_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(vis3, [largest], -1, (0, 200, 0), 1)

        quad = order_points(box)
        vis3 = draw_quad_on_img(vis3, quad)
        l2   = "轮廓OK · minAreaRect"
        fallback = False
    else:
        cv2.putText(vis3, "NO QUAD", (4, vis3.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        l2 = "FALLBACK (bbox)"
        h_r, w_r = roi.shape[:2]
        box = np.array([[w_r,h_r],[0,h_r],[0,0],[w_r,0]], dtype=np.float32)
        quad = order_points(box)
        fallback = True
    s2 = vis3

    # ④ 透视变换 → 376×96
    dst    = np.array([[MID_W,MID_H],[0,MID_H],[0,0],[MID_W,0]], dtype=np.float32)
    M      = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(roi, M, (MID_W,MID_H), flags=cv2.INTER_CUBIC)
    s3 = warped
    l3 = "透视变换 376×96" + (" [fallback]" if fallback else "")

    # ⑤ 94×24
    s4 = cv2.resize(warped, (94, 24), interpolation=cv2.INTER_AREA)
    l4 = "94×24"

    return [s0,s1,s2,s3,s4], [l0,l1,l2,l3,l4], plate_text, ptype


def draw_dataset(results, title):
    n, n_col = len(results), len(STEP_NAMES)
    fig = plt.figure(figsize=(n_col * 3.2, n * 3.0))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    gs  = gridspec.GridSpec(n, n_col, figure=fig, hspace=0.18, wspace=0.06)

    for row, (steps, labels, plate_text, ptype) in enumerate(results):
        for col in range(n_col):
            ax = fig.add_subplot(gs[row, col])
            if col == n_col - 1:
                disp = bgr2rgb(cv2.resize(steps[col],
                               (94*FINAL_SCALE, 24*FINAL_SCALE),
                               interpolation=cv2.INTER_NEAREST))
            else:
                disp = bgr2rgb(fit_h(steps[col], DISP_H))
            ax.imshow(disp, interpolation='nearest', aspect='auto')
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(STEP_NAMES[col], fontsize=8, fontweight='bold', pad=3)
            ax.set_xlabel(labels[col], fontsize=6.5, color='#444', labelpad=2)
            if col == 0:
                color = '#003399' if ptype == 'blue' else '#006600'
                ax.set_ylabel("{}\n[{}]".format(plate_text, ptype),
                              fontsize=9, rotation=0, labelpad=75,
                              ha='right', va='center', color=color, fontweight='bold')
    plt.tight_layout(rect=[0.09, 0, 1, 1])
    plt.show()


def sample_ccpd2019(n):
    txt = os.path.join(CCPD2019_DIR, 'splits', 'train.txt')
    with open(txt, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    chosen = random.sample(lines, min(n * 6, len(lines)))
    return [(os.path.join(CCPD2019_DIR, p.replace('/', os.sep)), os.path.basename(p))
            for p in chosen]

def sample_ccpd2020(n):
    src   = os.path.join(CCPD2020_DIR, 'train')
    files = [f for f in os.listdir(src) if f.endswith('.jpg')]
    chosen = random.sample(files, min(n * 6, len(files)))
    return [(os.path.join(src, f), f) for f in chosen]

def collect(model, file_list, n):
    results = []
    for src_path, fname in file_list:
        if len(results) >= n: break
        if not os.path.isfile(src_path): continue
        ret = process_steps(model, src_path, fname)
        if ret:
            results.append(ret)
            print("  [{}/{}] {} [{}]".format(len(results), n, ret[2], ret[3]))
    return results


def main():
    print("加载 YOLO 模型...")
    model = YOLO(YOLO_MODEL)

    print("\n--- CCPD2019（蓝牌）---")
    draw_dataset(collect(model, sample_ccpd2019(N_SAMPLES), N_SAMPLES),
                 "CCPD2019（蓝牌）· 全图adaptiveThreshold→minAreaRect→透视变换")

    print("\n--- CCPD2020（绿牌）---")
    draw_dataset(collect(model, sample_ccpd2020(N_SAMPLES), N_SAMPLES),
                 "CCPD2020（绿牌）· 全图adaptiveThreshold→minAreaRect→透视变换")
    print("完成。")


if __name__ == "__main__":
    main()
