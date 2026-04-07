"""
visualize_persp.py  —  CCPD GT 四角点透视变换可视化

每张车牌 4 列：
  ① 原图（含 GT 四角点标注）
  ② GT 四角点放大图（bbox 区域 +60px 边距）
  ③ 透视变换中间图（376×96）
  ④ 最终输出 94×24（×6 放大）
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
    get_gt_corners, get_plate_text, read_image, perspective_warp,
    enhance_plate, roi_mean_l,
    CCPD2019_DIR, CCPD2020_DIR,
)

N_SAMPLES    = 5
FINAL_SCALE  = 6
DISP_H       = 180
MID_W, MID_H = 376, 96

STEP_NAMES = ["① 原图\nGT 四角点",
              "② 局部放大\nGT 四角点",
              "③ 透视变换\n376×96",
              "④ 原始输出\n94×24",
              "⑤ 增强后\n94×24"]

# rb lb lt rt 对应颜色
PT_COLORS_BGR = [(0, 0, 255), (0, 140, 255), (0, 210, 0), (255, 60, 0)]
PT_LABELS     = ['RB', 'LB', 'LT', 'RT']


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 \
           else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def fit_h(img, h):
    oh, ow = img.shape[:2]
    if oh == 0: return img
    return cv2.resize(img, (max(1, int(ow * h / oh)), h),
                      interpolation=cv2.INTER_LINEAR)

def draw_corners(img, corners_rb_lb_lt_rt, r=6):
    """画 GT 四角点和连线。"""
    vis = img.copy()
    pts = corners_rb_lb_lt_rt.astype(np.int32)
    cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (x, y), r, PT_COLORS_BGR[i], -1)
        cv2.putText(vis, PT_LABELS[i], (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, PT_COLORS_BGR[i], 1, cv2.LINE_AA)
    return vis


def process_steps(src_path, filename):
    plate_text = get_plate_text(filename)
    if plate_text is None:
        return None
    corners = get_gt_corners(filename)
    if corners is None:
        return None
    img = read_image(src_path)
    if img is None:
        return None
    H, W = img.shape[:2]

    # 颜色类型
    xs = corners[:, 0].astype(int)
    ys = corners[:, 1].astype(int)
    x1c, y1c = max(0, xs.min()), max(0, ys.min())
    x2c, y2c = min(W, xs.max()), min(H, ys.max())
    roi_c = img[y1c:y2c, x1c:x2c]
    if roi_c.size > 0:
        hsv = cv2.cvtColor(roi_c, cv2.COLOR_BGR2HSV)
        tot = roi_c.shape[0] * roi_c.shape[1]
        br  = np.sum(cv2.inRange(hsv, (95,35,35),(135,255,255)) > 0) / tot
        gr  = np.sum(cv2.inRange(hsv, (30,35,35),(95, 255,255)) > 0) / tot
        ptype = 'blue' if br > gr and br > 0.05 else ('green' if gr > 0.05 else 'unk')
    else:
        ptype = 'unk'

    # ① 原图 + GT 四角点（缩小显示）
    scale = min(1.0, 640 / max(H, W))
    img_small = cv2.resize(img, (int(W*scale), int(H*scale)))
    corners_small = corners * scale
    s0 = draw_corners(img_small, corners_small, r=max(3, int(6*scale)))
    l0 = "原图 {}×{}  [{}]".format(W, H, ptype)

    # ② 局部放大（GT bbox 区域 +60px）
    p0 = 60
    rx1, ry1 = max(0, x1c-p0), max(0, y1c-p0)
    rx2, ry2 = min(W, x2c+p0), min(H, y2c+p0)
    crop = img[ry1:ry2, rx1:rx2].copy()
    corners_crop = corners - np.array([rx1, ry1], dtype=np.float32)
    s1 = draw_corners(crop, corners_crop)
    l1 = "GT 角点  [{}→{}→{}→{}]".format(*PT_LABELS)

    # ③ 透视变换 → 376×96（展示用中间图）
    src_pts = corners.astype(np.float32)
    dst_pts = np.array([[MID_W,MID_H],[0,MID_H],[0,0],[MID_W,0]], dtype=np.float32)
    M       = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped  = cv2.warpPerspective(img, M, (MID_W, MID_H), flags=cv2.INTER_CUBIC)
    s2 = warped
    l2 = "透视变换 376×96"

    # ④ 94×24（原始）
    s3 = cv2.resize(warped, (94, 24), interpolation=cv2.INTER_AREA)
    l3 = "94×24（原始）"

    # ⑤ 94×24（增强后）—— 亮度在原图四角点区域内判断
    mean_l = roi_mean_l(img, corners)
    s4 = enhance_plate(s3, mean_l)
    if   mean_l > 211: emode = "Gamma压暗(强光)"
    elif mean_l < 146: emode = "线性拉亮+CLAHE"
    else:              emode = "无需增强"
    l4 = "原图ROI亮度={:.0f} {}\n增强后L={:.0f}".format(
        mean_l, emode,
        float(np.mean(cv2.cvtColor(s4, cv2.COLOR_BGR2LAB)[:, :, 0])))

    return [s0, s1, s2, s3, s4], [l0, l1, l2, l3, l4], plate_text, ptype


def draw_dataset(results, title):
    n, n_col = len(results), len(STEP_NAMES)
    fig = plt.figure(figsize=(n_col * 3.2, n * 3.0))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    gs  = gridspec.GridSpec(n, n_col, figure=fig, hspace=0.20, wspace=0.06)

    for row, (steps, labels, plate_text, ptype) in enumerate(results):
        for col in range(n_col):
            ax = fig.add_subplot(gs[row, col])
            if col >= 3:   # ④ 原始 94×24 和 ⑤ 增强后 94×24 都放大显示
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
                              fontsize=9, rotation=0, labelpad=80,
                              ha='right', va='center', color=color, fontweight='bold')
    plt.tight_layout(rect=[0.10, 0, 1, 1])
    plt.show()


def sample_ccpd2019(n, split='train'):
    txt = os.path.join(CCPD2019_DIR, 'splits', '{}.txt'.format(split))
    with open(txt, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    chosen = random.sample(lines, min(n * 4, len(lines)))
    return [(os.path.join(CCPD2019_DIR, p.replace('/', os.sep)), os.path.basename(p))
            for p in chosen]

def sample_ccpd2020(n):
    src   = os.path.join(CCPD2020_DIR, 'train')
    files = [f for f in os.listdir(src) if f.endswith('.jpg')]
    chosen = random.sample(files, min(n * 4, len(files)))
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
    print("\n--- CCPD2019 train（蓝牌）---")
    draw_dataset(collect(sample_ccpd2019(N_SAMPLES, 'train'), N_SAMPLES),
                 "CCPD2019 train（蓝牌）· GT四角点→透视变换（倾斜完全校正）")

    print("\n--- CCPD2019 test（蓝牌·困难场景）---")
    draw_dataset(collect(sample_ccpd2019(N_SAMPLES, 'test'), N_SAMPLES),
                 "CCPD2019 test（蓝牌·困难）· GT四角点→透视变换")

    print("\n--- CCPD2020（绿牌）---")
    draw_dataset(collect(sample_ccpd2020(N_SAMPLES), N_SAMPLES),
                 "CCPD2020（绿牌）· GT四角点→透视变换（倾斜完全校正）")
    print("完成。")


if __name__ == "__main__":
    main()
