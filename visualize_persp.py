"""
visualize_persp.py  —  yolo_persp_crop.py 管线可视化

每张车牌 5 列：
  ① 原图局部（bbox+60px）+ 绿色检测框
  ② ROI（bbox+5%）
  ③ 颜色掩膜闭运算叠加图 + minAreaRect 四角点
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
    yolo_detect, find_quad, perspective_warp,
    get_plate_text, read_image,
    YOLO_MODEL, CCPD2019_DIR, CCPD2020_DIR,
)
from ultralytics import YOLO

N_SAMPLES   = 5
RANDOM_SEED = 42
FINAL_SCALE = 6
DISP_H      = 150
MID_W, MID_H = 376, 96

STEP_NAMES = ["① 原图局部\nYOLO bbox",
              "② ROI\n(+5%)",
              "③ Canny边缘\n四角点",
              "④ 透视变换\n376×96",
              "⑤ 最终输出\n94×24"]

PT_COLORS = [(0,0,255),(0,140,255),(0,210,0),(255,60,0)]
PT_LABELS = ['RB','LB','LT','RT']


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape)==3 \
           else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def fit_h(img, h):
    oh, ow = img.shape[:2]
    if oh == 0: return img
    return cv2.resize(img, (max(1, int(ow*h/oh)), h), interpolation=cv2.INTER_NEAREST)

def draw_quad(img, quad, r=5):
    vis = img.copy()
    pts = quad.astype(np.int32)
    cv2.polylines(vis, [pts], True, (0,255,255), 2)
    for i,(x,y) in enumerate(pts):
        cv2.circle(vis, (x,y), r, PT_COLORS[i], -1)
        cv2.putText(vis, PT_LABELS[i], (x+3,y-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, PT_COLORS[i], 1, cv2.LINE_AA)
    return vis


def process_steps(model, src_path, filename):
    plate_text = get_plate_text(filename)
    if plate_text is None: return None
    img = read_image(src_path)
    if img is None: return None
    H, W = img.shape[:2]

    bbox = yolo_detect(model, img)
    if bbox is None: return None
    x1, y1, x2, y2 = bbox

    roi = img[y1:y2, x1:x2].copy()
    if roi.size == 0: return None

    # 判断颜色仅用于显示标签
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total = roi.shape[0] * roi.shape[1]
    br = np.sum(cv2.inRange(hsv,(100,60,60),(130,255,255))>0)/total
    gr = np.sum(cv2.inRange(hsv,(35,60,60),(85,255,255))>0)/total
    ptype = 'blue' if br>0.05 and br>=gr else ('green' if gr>0.05 else 'unknown')

    # ① 原图局部
    p0 = 60
    r = img[max(0,y1-p0):min(H,y2+p0), max(0,x1-p0):min(W,x2+p0)].copy()
    cv2.rectangle(r, (x1-max(0,x1-p0), y1-max(0,y1-p0)),
                     (x2-max(0,x1-p0), y2-max(0,y1-p0)), (0,255,0), 2)
    s0, l0 = r, "bbox {}×{}  [{}]".format(x2-x1, y2-y1, ptype)

    # ② ROI
    s1, l1 = roi, "ROI {}×{}".format(x2-x1, y2-y1)

    # ③ Canny 边缘 + minAreaRect 四角点
    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    k     = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges_d = cv2.dilate(edges, k, iterations=2)
    # 边缘图转 BGR 用于显示
    vis3  = cv2.cvtColor(edges_d, cv2.COLOR_GRAY2BGR)
    quad  = find_quad(roi)
    if quad is not None:
        vis3 = draw_quad(vis3, quad)
        l2   = "Canny+minAreaRect OK"
        ok   = True
    else:
        cv2.putText(vis3, "NO QUAD", (4, vis3.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        l2 = "FALLBACK"
        ok = False
    s2 = vis3

    # ④ 透视变换 → 376×96
    if ok:
        dst    = np.array([[MID_W,MID_H],[0,MID_H],[0,0],[MID_W,0]], dtype=np.float32)
        M      = cv2.getPerspectiveTransform(quad, dst)
        warped = cv2.warpPerspective(roi, M, (MID_W,MID_H), flags=cv2.INTER_CUBIC)
        s3, l3 = warped, "透视变换 376×96"
    else:
        s3 = cv2.resize(roi, (MID_W, MID_H), interpolation=cv2.INTER_CUBIC)
        l3 = "resize fallback"

    # ⑤ 94×24
    s4 = cv2.resize(s3, (94, 24), interpolation=cv2.INTER_AREA)
    l4 = "94×24"

    return [s0,s1,s2,s3,s4], [l0,l1,l2,l3,l4], plate_text, ptype


def draw_dataset(results, title):
    n, n_col = len(results), len(STEP_NAMES)
    fig = plt.figure(figsize=(n_col*3.0, n*2.8))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    gs  = gridspec.GridSpec(n, n_col, figure=fig, hspace=0.15, wspace=0.06)

    for row, (steps, labels, plate_text, ptype) in enumerate(results):
        for col in range(n_col):
            ax = fig.add_subplot(gs[row, col])
            if col == n_col-1:
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
                color = '#003399' if ptype=='blue' else '#006600'
                ax.set_ylabel("{}\n[{}]".format(plate_text, ptype),
                              fontsize=9, rotation=0, labelpad=72,
                              ha='right', va='center', color=color, fontweight='bold')
    plt.tight_layout(rect=[0.09, 0, 1, 1])
    plt.show()


def sample_ccpd2019(n):
    txt = os.path.join(CCPD2019_DIR, 'splits', 'train.txt')
    with open(txt, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    chosen = random.sample(lines, min(n*6, len(lines)))
    return [(os.path.join(CCPD2019_DIR, p.replace('/', os.sep)), os.path.basename(p))
            for p in chosen]

def sample_ccpd2020(n):
    src   = os.path.join(CCPD2020_DIR, 'train')
    files = [f for f in os.listdir(src) if f.endswith('.jpg')]
    chosen = random.sample(files, min(n*6, len(files)))
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
    random.seed(RANDOM_SEED)
    print("加载 YOLO 模型...")
    model = YOLO(YOLO_MODEL)

    print("\n--- CCPD2019（蓝牌）---")
    draw_dataset(collect(model, sample_ccpd2019(N_SAMPLES), N_SAMPLES),
                 "CCPD2019（蓝牌）· YOLO→颜色掩膜→minAreaRect→透视变换")

    print("\n--- CCPD2020（绿牌）---")
    draw_dataset(collect(model, sample_ccpd2020(N_SAMPLES), N_SAMPLES),
                 "CCPD2020（绿牌）· YOLO→颜色掩膜→minAreaRect→透视变换")
    print("完成。")


if __name__ == "__main__":
    main()
