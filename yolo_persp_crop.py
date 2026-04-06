"""
yolo_persp_crop.py
==================
流程（训练 & 实时完全一致）：
  1. YOLO 推理 → bbox（含 PAD 扩边）
  2. CLAHE 增强全图（改善暗图 / 模糊图）
  3. Canny 边缘检测（全图，避免 ROI 裁边伪影）
     → 裁出 ROI 内的边缘图
     → 全部边缘点 → convex hull → minAreaRect → 4 个有向角点
  4. order_points 排序 → 透视变换 → 94×24

为何用"全部边缘点 convex hull + minAreaRect"而非"最大轮廓 + minAreaRect"：
  - adaptiveThreshold BINARY_INV 对蓝牌（白字蓝底）结果不稳定：
    白色字符比蓝色背景亮，BINARY_INV 可能把字符变成黑色、背景变白，
    "最大轮廓"变成背景区域而非字符 / 车牌边界。
  - Canny 只提取亮度梯度边缘，蓝/绿牌通用，字符笔画和车牌边框都会被检测。
  - 将 ROI 内全部边缘点做 convex hull：凸包外轮廓等同于车牌区域的包络，
    minAreaRect 直接给出最小有向外接矩形，即使倾斜也精确。

实时调用：
    from yolo_persp_crop import process_image
    out = process_image(model, img_bgr)   # 返回 94×24 BGR 或 None
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

# ===== 路径配置 =====
BASE_DIR     = r"c:\Users\Y9000P\Downloads\2026ICContest\ICcontest_project"
YOLO_MODEL   = os.path.join(BASE_DIR, r"runs\new_plate_detect_merged\weights\best.pt")
CCPD2019_DIR = r"D:\Tempcode\26IC\车牌数据集\中国车牌\CCPD2019"
CCPD2020_DIR = r"D:\Tempcode\26IC\车牌数据集\中国车牌\CCPD2020\ccpd_green"
OUTPUT_DIR   = os.path.join(BASE_DIR, "LPR_DATA_PERSP")
YOLO_CONF    = 0.3
PAD          = 0.12   # 扩边比例：让车牌边框完整进入 ROI，Canny 可看到边界
# ====================


# -------------------------------------------------------
# Step 1: YOLO 推理 → bbox
# -------------------------------------------------------

def yolo_detect(model, img):
    """返回加 PAD 后的 bbox (x1,y1,x2,y2)，或 None。"""
    best_box, best_conf = None, 0.0
    for r in model(img, verbose=False, conf=YOLO_CONF):
        if r.boxes is None:
            continue
        for box in r.boxes:
            c = float(box.conf[0])
            if c > best_conf:
                best_conf = c
                best_box = box.xyxy[0].cpu().numpy().astype(int)
    if best_box is None:
        return None
    H, W = img.shape[:2]
    x1, y1, x2, y2 = best_box
    pw = max(4, int((x2 - x1) * PAD))
    ph = max(4, int((y2 - y1) * PAD))
    return (max(0, x1 - pw), max(0, y1 - ph),
            min(W, x2 + pw), min(H, y2 + ph))


# -------------------------------------------------------
# Step 2+3: CLAHE + Canny → convex hull → minAreaRect
# -------------------------------------------------------

def _make_edge_img(img):
    """
    对原图做 CLAHE 增强后 Canny 边缘检测。
    CLAHE 对暗图 / 低对比度图效果显著（CCPD2019 test 困难场景）。
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur    = cv2.GaussianBlur(enhanced, (5, 5), 1.2)
    return cv2.Canny(blur, 20, 60)


def find_quad(img, x1, y1, x2, y2):
    """
    在完整原图上做 CLAHE+Canny，裁出 ROI 边缘图，
    取全部边缘点做 convex hull → minAreaRect，得到有向 4 角点（ROI 坐标系）。

    返回 (edge_roi_uint8, box_4pts_float32) 或 (edge_roi, None)。
    edge_roi 用于可视化。
    """
    edges_full = _make_edge_img(img)
    edge_roi   = edges_full[y1:y2, x1:x2].copy()
    h_roi, w_roi = edge_roi.shape[:2]

    ys, xs = np.where(edge_roi > 0)
    if len(xs) < 30:
        return edge_roi, None

    pts  = np.stack([xs, ys], axis=1).astype(np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)

    # 凸包面积过小（边缘太稀疏 / 全是噪声）
    if cv2.contourArea(hull) < h_roi * w_roi * 0.04:
        return edge_roi, None

    rect = cv2.minAreaRect(hull)
    (cx, cy), (rw, rh), angle = rect

    # 保证宽 > 高（车牌横向更长）
    if rw < rh:
        rw, rh = rh, rw
        angle  = angle + 90

    box = cv2.boxPoints(((cx, cy), (rw, rh), angle)).astype(np.float32)
    return edge_roi, box


# -------------------------------------------------------
# Step 4: 质心角度排序 → rb→lb→lt→rt
# -------------------------------------------------------

def order_points(pts):
    """
    按质心角度升序排列（arctan2），起点调整为左上角（x+y 最小）。
    输出顺序 rb→lb→lt→rt（与 perspective_warp 的 dst 对应）。
    """
    centroid = np.mean(pts, axis=0)
    angles   = np.arctan2(pts[:, 1] - centroid[1],
                          pts[:, 0] - centroid[0])
    sorted_idx = np.argsort(angles)
    sorted_pts = pts[sorted_idx]

    top_left_idx = int(np.argmin(sorted_pts.sum(axis=1)))
    ordered = np.roll(sorted_pts, -top_left_idx, axis=0)   # lt rt rb lb

    lt, rt, rb, lb = ordered[0], ordered[1], ordered[2], ordered[3]
    return np.array([rb, lb, lt, rt], dtype=np.float32)


# -------------------------------------------------------
# Step 5: 透视变换 → 94×24
# -------------------------------------------------------

def perspective_warp(img, quad, out_w=94, out_h=24):
    """
    quad: (4,2) float32，顺序 rb→lb→lt→rt。
    先 warp 到 4× 中间尺寸（376×96）保证质量，再 INTER_AREA resize。
    """
    mid_w, mid_h = out_w * 4, out_h * 4
    dst = np.array([[mid_w, mid_h],
                    [0,     mid_h],
                    [0,     0    ],
                    [mid_w, 0    ]], dtype=np.float32)
    M      = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img, M, (mid_w, mid_h), flags=cv2.INTER_CUBIC)
    return cv2.resize(warped, (out_w, out_h), interpolation=cv2.INTER_AREA)


# -------------------------------------------------------
# 完整流程（训练 & 实时共用）
# -------------------------------------------------------

def process_image(model, img, out_w=94, out_h=24):
    """
    输入 BGR 原图，返回 94×24 BGR 或 None。
    fallback：边缘点不足时用 bbox 四角（无倾斜矫正，保证至少有输出）。
    """
    bbox = yolo_detect(model, img)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    _, box = find_quad(img, x1, y1, x2, y2)
    if box is None:
        h_roi, w_roi = roi.shape[:2]
        box = np.array([[w_roi, h_roi], [0, h_roi],
                        [0, 0],         [w_roi, 0]], dtype=np.float32)

    quad = order_points(box)
    return perspective_warp(roi, quad, out_w, out_h)


# -------------------------------------------------------
# CCPD 文件名解析
# -------------------------------------------------------

PROVINCES = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑",
             "苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤",
             "桂","琼","川","贵","云","藏","陕","甘","青","宁",
             "新","警","学","O"]
ALPHABETS  = ['A','B','C','D','E','F','G','H','J','K',
              'L','M','N','P','Q','R','S','T','U','V',
              'W','X','Y','Z','O']
ADS        = ['A','B','C','D','E','F','G','H','J','K',
              'L','M','N','P','Q','R','S','T','U','V',
              'W','X','Y','Z',
              '0','1','2','3','4','5','6','7','8','9','O']

def decode_plate(s):
    parts = s.split('_')
    chars = []
    for i, p in enumerate(parts):
        idx = int(p)
        if   i == 0: chars.append(PROVINCES[idx])
        elif i == 1: chars.append(ALPHABETS[idx])
        else:        chars.append(ADS[idx])
    return ''.join(chars)

def get_plate_text(filename):
    stem = os.path.splitext(filename)[0].split('-')
    if len(stem) < 5:
        return None
    try:
        return decode_plate(stem[4])
    except Exception:
        return None

def read_image(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# -------------------------------------------------------
# 批量处理
# -------------------------------------------------------

def batch_convert(model, file_list, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    ok = skip = 0
    total = len(file_list)
    for i, (src_path, fname) in enumerate(file_list):
        if not os.path.isfile(src_path):
            skip += 1; continue
        plate_text = get_plate_text(fname)
        if plate_text is None:
            skip += 1; continue
        img = read_image(src_path)
        if img is None:
            skip += 1; continue
        out = process_image(model, img)
        if out is None:
            skip += 1; continue
        dst = os.path.join(out_dir, '{}.jpg'.format(plate_text))
        cv2.imencode('.jpg', out)[1].tofile(dst)
        ok += 1
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print('  [{}] {}/{} | ok={} skip={}'.format(
                tag, i + 1, total, ok, skip))
    return ok


def collect_ccpd2019(split):
    txt = os.path.join(CCPD2019_DIR, 'splits', '{}.txt'.format(split))
    with open(txt, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    return [(os.path.join(CCPD2019_DIR, p.replace('/', os.sep)),
             os.path.basename(p)) for p in lines]

def collect_ccpd2020(split):
    src = os.path.join(CCPD2020_DIR, split)
    return [(os.path.join(src, f), f)
            for f in os.listdir(src) if f.endswith('.jpg')]


def main():
    print('加载 YOLO 模型...')
    model = YOLO(YOLO_MODEL)
    for split in ['train', 'val', 'test']:
        out_dir = os.path.join(OUTPUT_DIR, split)
        print('\n========== {} =========='.format(split))
        n1 = batch_convert(model, collect_ccpd2019(split), out_dir, 'CCPD2019')
        n2 = batch_convert(model, collect_ccpd2020(split), out_dir, 'CCPD2020')
        print('  合计: {}'.format(n1 + n2))
    print('\n完成！输出目录: {}'.format(OUTPUT_DIR))


if __name__ == '__main__':
    main()
