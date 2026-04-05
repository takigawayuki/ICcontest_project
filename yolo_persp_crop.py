"""
yolo_persp_crop.py
==================
流程（训练 & 实时完全一致）：
  1. YOLO 推理 → bbox，加 5% padding 裁剪 ROI
  2. ROI 上颜色掩膜 → 闭运算 → 最大轮廓 → minAreaRect（旋转矩形）
  3. 透视变换 → 94×24（倾斜校正）
  fallback：ROI 直接 resize

实时调用：
    from yolo_persp_crop import process_image
    out = process_image(model, img_bgr)
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
PAD          = 0.05   # bbox 各边扩边 5%，防止倾斜角点被截断
# ====================


# -------------------------------------------------------
# Step 1: YOLO → ROI
# -------------------------------------------------------

def yolo_detect(model, img):
    """返回加 PAD 后的 bbox (x1,y1,x2,y2)，或 None。"""
    best_box, best_conf = None, 0.0
    for r in model(img, verbose=False, conf=YOLO_CONF):
        if r.boxes is None: continue
        for box in r.boxes:
            c = float(box.conf[0])
            if c > best_conf:
                best_conf = c
                best_box = box.xyxy[0].cpu().numpy().astype(int)
    if best_box is None: return None

    H, W = img.shape[:2]
    x1, y1, x2, y2 = best_box
    pw = max(3, int((x2 - x1) * PAD))
    ph = max(3, int((y2 - y1) * PAD))
    return (max(0, x1-pw), max(0, y1-ph),
            min(W, x2+pw), min(H, y2+ph))


# -------------------------------------------------------
# Step 2: 颜色掩膜 → minAreaRect
# -------------------------------------------------------

def detect_plate_type(roi):
    hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total = roi.shape[0] * roi.shape[1]
    blue_r  = np.sum(cv2.inRange(hsv, (100, 60, 60), (130, 255, 255)) > 0) / total
    green_r = np.sum(cv2.inRange(hsv, (35,  60, 60), (85,  255, 255)) > 0) / total
    if blue_r  > 0.05 and blue_r >= green_r: return 'blue'
    if green_r > 0.05 and green_r >  blue_r: return 'green'
    return 'unknown'


def _sort_quad(pts):
    """任意顺序 4 点 → rb→lb→lt→rt。"""
    pts  = pts.reshape(4, 2).astype(np.float32)
    s    = pts.sum(axis=1)
    diff = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmax(s)],    # rb
                     pts[np.argmax(diff)],  # lb
                     pts[np.argmin(s)],     # lt
                     pts[np.argmin(diff)]], dtype=np.float32)  # rt


def find_quad(roi):
    """
    边缘检测 → 最大轮廓 → minAreaRect → 4 角点。
    不依赖颜色分割（颜色掩膜会填满整个 ROI，minAreaRect 退化为 ROI 边界）。
    改用 Canny 边缘检测车牌边框，能正确捕捉倾斜/旋转角度。
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 膨胀连接断裂边缘
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < roi.shape[0] * roi.shape[1] * 0.10:
        return None

    box = cv2.boxPoints(cv2.minAreaRect(cnt))
    return _sort_quad(box)


# -------------------------------------------------------
# Step 3: 透视变换 → 94×24
# -------------------------------------------------------

def perspective_warp(roi, quad, out_w=94, out_h=24):
    """quad: (4,2) rb→lb→lt→rt，ROI 坐标系。先 warp 到 4× 再 resize。"""
    mid_w, mid_h = out_w * 4, out_h * 4
    dst = np.array([[mid_w, mid_h],
                    [0,     mid_h],
                    [0,     0    ],
                    [mid_w, 0    ]], dtype=np.float32)
    M      = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(roi, M, (mid_w, mid_h), flags=cv2.INTER_CUBIC)
    return cv2.resize(warped, (out_w, out_h), interpolation=cv2.INTER_AREA)


# -------------------------------------------------------
# 完整流程（训练 & 实时共用）
# -------------------------------------------------------

def process_image(model, img, out_w=94, out_h=24):
    """输入 BGR 原图，返回 94×24 BGR 或 None。"""
    bbox = yolo_detect(model, img)
    if bbox is None: return None
    x1, y1, x2, y2 = bbox
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return None

    quad = find_quad(roi)
    if quad is not None:
        return perspective_warp(roi, quad, out_w, out_h)
    else:
        return cv2.resize(roi, (out_w, out_h), interpolation=cv2.INTER_AREA)


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
    if len(stem) < 5: return None
    try:    return decode_plate(stem[4])
    except: return None

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
        if not os.path.isfile(src_path): skip += 1; continue
        plate_text = get_plate_text(fname)
        if plate_text is None: skip += 1; continue
        img = read_image(src_path)
        if img is None: skip += 1; continue
        out = process_image(model, img)
        if out is None: skip += 1; continue
        dst = os.path.join(out_dir, '{}.jpg'.format(plate_text))
        cv2.imencode('.jpg', out)[1].tofile(dst)
        ok += 1
        if (i+1) % 500 == 0 or (i+1) == total:
            print('  [{}] {}/{} | ok={} skip={}'.format(tag, i+1, total, ok, skip))
    return ok


def collect_ccpd2019(split):
    txt = os.path.join(CCPD2019_DIR, 'splits', '{}.txt'.format(split))
    with open(txt, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    return [(os.path.join(CCPD2019_DIR, p.replace('/', os.sep)), os.path.basename(p))
            for p in lines]

def collect_ccpd2020(split):
    src = os.path.join(CCPD2020_DIR, split)
    return [(os.path.join(src, f), f) for f in os.listdir(src) if f.endswith('.jpg')]


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
