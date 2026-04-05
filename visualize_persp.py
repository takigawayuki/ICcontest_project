"""
visualize_persp.py
==================
yolo_persp_crop.py 可视化（新版：直接用 CCPD GT 四角点）

每张车牌展示 4 列：
  ① 原图 + GT 四角点（rb→lb→lt→rt，彩色圆点+连线）
  ② 以 bbox 为中心的裁剪区域（+30px margin）+ 四角点
  ③ 透视变换后（376×96 中间图）
  ④ 最终输出 94×24（× FINAL_SCALE 放大）

运行：
    python visualize_persp.py
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
    parse_ccpd_filename, read_image,
    CCPD2019_DIR, CCPD2020_DIR,
)

# ===== 配置 =====
N_SAMPLES   = 5
RANDOM_SEED = 42
FINAL_SCALE = 6     # 94×24 → 564×144 放大显示
MID_W, MID_H = 376, 96   # 透视变换中间尺寸
# ================

STEP_NAMES = [
    "① 车牌全景\n(bbox+80px)",
    "② 紧贴裁剪\n(bbox+15px)",
    "③ 透视变换\n(376×96)",
    "④ 最终输出\n94×24",
]
# 四角点颜色：rb=红, lb=橙, lt=绿, rt=蓝
PT_COLORS_BGR = [(0,0,255),(0,100,255),(0,220,0),(255,80,0)]
PT_LABELS     = ['RB','LB','LT','RT']


def bgr2rgb(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_quad(img, four_pts, radius=6, thickness=2):
    """在图像上画四角点 + 连线（顺序 rb→lb→lt→rt）"""
    vis = img.copy()
    pts = four_pts.astype(np.int32)
    # 画四边形
    cv2.polylines(vis, [pts], isClosed=True, color=(0,255,255), thickness=thickness)
    # 画各角点
    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (x, y), radius, PT_COLORS_BGR[i], -1)
        cv2.putText(vis, PT_LABELS[i], (x+4, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, PT_COLORS_BGR[i], 1,
                    cv2.LINE_AA)
    return vis


def detect_plate_type_simple(img, bbox):
    """仅用于显示标签，判断蓝/绿牌"""
    x1, y1, x2, y2 = bbox
    H, W = img.shape[:2]
    roi = img[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
    if roi.size == 0: return 'unknown'
    hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total = roi.shape[0] * roi.shape[1]
    blue_r  = np.sum(cv2.inRange(hsv, (100,60,60), (130,255,255)) > 0) / total
    green_r = np.sum(cv2.inRange(hsv, (35, 60,60), (85, 255,255)) > 0) / total
    if blue_r  > 0.05 and blue_r >= green_r: return 'blue'
    if green_r > 0.05 and green_r >  blue_r: return 'green'
    return 'unknown'


def process_steps(src_path, filename):
    """
    读取 CCPD 图像，用 GT 四角点做透视变换，收集各阶段图像。
    返回 (steps, labels, plate_text, ptype) 或 None。
    """
    plate_text, bbox, four_pts = parse_ccpd_filename(filename)
    if plate_text is None or four_pts is None: return None

    img = read_image(src_path)
    if img is None: return None
    H, W = img.shape[:2]

    # 判断车牌颜色（辅助显示）
    ptype = detect_plate_type_simple(img, bbox) if bbox else 'unknown'

    # ① bbox 大裁剪区域（+80px margin）+ GT 四角点 → 车牌全景
    if bbox:
        x1, y1, x2, y2 = bbox
        pad0 = 80
        cx10 = max(0, x1-pad0); cy10 = max(0, y1-pad0)
        cx20 = min(W, x2+pad0); cy20 = min(H, y2+pad0)
        crop0 = img[cy10:cy20, cx10:cx20].copy()
        pts0  = four_pts - np.array([cx10, cy10], dtype=np.float32)
        s0    = draw_quad(crop0, pts0, radius=6)
        l0    = "bbox+80px  {}×{}  原图{}×{}".format(cx20-cx10, cy20-cy10, W, H)
    else:
        # 无 bbox 时缩略全图
        s0 = draw_quad(img, four_pts)
        l0 = "原图 {}×{}".format(W, H)

    # ② 紧贴 bbox 裁剪（+15px margin）+ GT 四角点 → 车牌放大
    if bbox:
        x1, y1, x2, y2 = bbox
        pad1 = 15
        cx11 = max(0, x1-pad1); cy11 = max(0, y1-pad1)
        cx21 = min(W, x2+pad1); cy21 = min(H, y2+pad1)
        crop1 = img[cy11:cy21, cx11:cx21].copy()
        pts1  = four_pts - np.array([cx11, cy11], dtype=np.float32)
        s1    = draw_quad(crop1, pts1, radius=4)
        l1    = "bbox+15px  {}×{}".format(cx21-cx11, cy21-cy11)
    else:
        s1 = s0.copy()
        l1 = "无 bbox 信息"

    # ③ 透视变换 → 376×96 中间图
    dst = np.array([[MID_W, MID_H], [0, MID_H], [0, 0], [MID_W, 0]], dtype=np.float32)
    M      = cv2.getPerspectiveTransform(four_pts, dst)
    warped = cv2.warpPerspective(img, M, (MID_W, MID_H), flags=cv2.INTER_CUBIC)
    s2 = warped
    l2 = "透视变换 376×96"

    # ④ resize → 94×24
    final = cv2.resize(warped, (94, 24), interpolation=cv2.INTER_AREA)
    s3 = final
    l3 = "输出 94×24"

    steps  = [s0, s1, s2, s3]
    labels = [l0, l1, l2, l3]
    return steps, labels, plate_text, ptype


def fit_h(img, h):
    oh, ow = img.shape[:2]
    if oh == 0: return img
    w = max(1, int(ow * h / oh))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)


def draw_dataset(results, title):
    n     = len(results)
    n_col = len(STEP_NAMES)
    DISP_H = 160   # 统一显示高度（前3列）

    fig_w = n_col * 3.6
    fig_h = n * 2.8
    fig   = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    gs = gridspec.GridSpec(n, n_col, figure=fig, hspace=0.12, wspace=0.06)

    for row, (steps, labels, plate_text, ptype) in enumerate(results):
        for col in range(n_col):
            ax = fig.add_subplot(gs[row, col])

            if col == n_col - 1:
                # 最终 94×24 放大
                disp = bgr2rgb(cv2.resize(steps[col],
                    (94 * FINAL_SCALE, 24 * FINAL_SCALE),
                    interpolation=cv2.INTER_NEAREST))
            else:
                disp = bgr2rgb(fit_h(steps[col], DISP_H))

            ax.imshow(disp, interpolation='nearest', aspect='auto')
            ax.set_xticks([]); ax.set_yticks([])

            if row == 0:
                ax.set_title(STEP_NAMES[col], fontsize=8,
                             fontweight='bold', pad=3)

            ax.set_xlabel(labels[col], fontsize=6.5,
                          color='#444444', labelpad=2)

            if col == 0:
                color = '#003399' if ptype == 'blue' else '#006600'
                ax.set_ylabel(
                    "{}\n[{}]".format(plate_text, ptype),
                    fontsize=9, rotation=0, labelpad=70,
                    ha='right', va='center',
                    color=color, fontweight='bold')

    plt.tight_layout(rect=[0.08, 0, 1, 1])
    plt.show()


def sample_ccpd2019(n):
    txt = os.path.join(CCPD2019_DIR, 'splits', 'train.txt')
    with open(txt, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    chosen = random.sample(lines, min(n * 6, len(lines)))
    return [(os.path.join(CCPD2019_DIR, p.replace('/', os.sep)),
             os.path.basename(p)) for p in chosen]


def sample_ccpd2020(n):
    src = os.path.join(CCPD2020_DIR, 'train')
    files = [f for f in os.listdir(src) if f.endswith('.jpg')]
    chosen = random.sample(files, min(n * 6, len(files)))
    return [(os.path.join(src, f), f) for f in chosen]


def collect(file_list, n):
    results = []
    for src_path, fname in file_list:
        if len(results) >= n: break
        if not os.path.isfile(src_path): continue
        ret = process_steps(src_path, fname)
        if ret:
            results.append(ret)
            print("  [{}/{}] {} [{}]".format(len(results), n, ret[2], ret[3]))
    return results


def main():
    random.seed(RANDOM_SEED)

    print("\n--- CCPD2019（蓝牌）{} 张 ---".format(N_SAMPLES))
    r2019 = collect(sample_ccpd2019(N_SAMPLES), N_SAMPLES)
    draw_dataset(r2019, "CCPD2019（蓝牌）· GT 四角点透视变换 各阶段")

    print("\n--- CCPD2020（绿牌）{} 张 ---".format(N_SAMPLES))
    r2020 = collect(sample_ccpd2020(N_SAMPLES), N_SAMPLES)
    draw_dataset(r2020, "CCPD2020（绿牌）· GT 四角点透视变换 各阶段")

    print("完成。")


if __name__ == "__main__":
    main()
