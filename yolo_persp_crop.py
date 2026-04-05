"""
yolo_persp_crop.py
==================
训练数据制作流程（CCPD 数据集）：
  1. 解析 CCPD 文件名 → 得到 4 个精确角点（GT landmark）
  2. 透视变换：4 角点直接拉平到 94×24

实时推理流程（无文件名信息）：
  1. YOLO 推理 → bbox
  2. 直接裁剪 + resize 到 94×24（无角点信息时的 fallback）

CCPD 文件名格式：
  xxx-area-bbox-landmark-chars-...
  landmark (parts[3]): "x1&y1_x2&y2_x3&y3_x4&y4"
  角点顺序：右下(rb) → 左下(lb) → 左上(lt) → 右上(rt)
"""

import os, sys, random
import cv2
import numpy as np
from typing import Optional, Tuple
from ultralytics import YOLO

# ===== 路径配置 =====
BASE_DIR     = r"c:\Users\Y9000P\Downloads\2026ICContest\ICcontest_project"
YOLO_MODEL   = os.path.join(BASE_DIR, r"runs\new_plate_detect_merged\weights\best.pt")
CCPD2019_DIR = r"D:\Tempcode\26IC\车牌数据集\中国车牌\CCPD2019"
CCPD2020_DIR = r"D:\Tempcode\26IC\车牌数据集\中国车牌\CCPD2020\ccpd_green"
OUTPUT_DIR   = os.path.join(BASE_DIR, "LPR_DATA_PERSP")
YOLO_CONF    = 0.3
# ====================


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


def parse_ccpd_filename(filename):
    """
    返回 (plate_text, bbox, four_pts) 或 (None, None, None)。
    four_pts: np.float32 (4,2)，顺序 rb→lb→lt→rt（图像坐标系）
    """
    stem  = os.path.splitext(filename)[0]
    parts = stem.split('-')
    if len(parts) < 5:
        return None, None, None
    try:
        plate_text = decode_plate(parts[4])
    except Exception:
        return None, None, None
    try:
        bp = parts[2].split('_')
        x1, y1 = map(int, bp[0].split('&'))
        x2, y2 = map(int, bp[1].split('&'))
        bbox = (x1, y1, x2, y2)
    except Exception:
        bbox = None
    try:
        raw = parts[3].split('_')
        pts = [list(map(int, p.split('&'))) for p in raw]
        four_pts = np.array(pts, dtype=np.float32)   # (4,2): rb,lb,lt,rt
    except Exception:
        four_pts = None
    return plate_text, bbox, four_pts


def read_image(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# -------------------------------------------------------
# 透视变换 → 94×24
# -------------------------------------------------------

def perspective_warp(img, four_pts, out_w=94, out_h=24):
    """
    four_pts: np.float32 (4,2)，顺序 rb→lb→lt→rt（图像坐标系）
    先 warp 到 4× 中间尺寸保证插值质量，再 INTER_AREA resize。
    """
    mid_w, mid_h = out_w * 4, out_h * 4   # 376×96
    # rb→lb→lt→rt 对应目标: 右下→左下→左上→右上
    dst = np.array([[mid_w, mid_h],
                    [0,     mid_h],
                    [0,     0    ],
                    [mid_w, 0    ]], dtype=np.float32)
    M      = cv2.getPerspectiveTransform(four_pts, dst)
    warped = cv2.warpPerspective(img, M, (mid_w, mid_h), flags=cv2.INTER_CUBIC)
    return cv2.resize(warped, (out_w, out_h), interpolation=cv2.INTER_AREA)


# -------------------------------------------------------
# YOLO 推理（仅实时推理时使用）
# -------------------------------------------------------

def yolo_detect(model, img):
    """
    返回 bbox (x1,y1,x2,y2) 或 None。
    仅在没有 CCPD 文件名角点时用作 fallback。
    """
    best_box, best_conf = None, 0.0
    for r in model(img, verbose=False, conf=YOLO_CONF):
        if r.boxes is None: continue
        for box in r.boxes:
            c = float(box.conf[0])
            if c > best_conf:
                best_conf = c
                best_box = box.xyxy[0].cpu().numpy().astype(int)
    return best_box


# -------------------------------------------------------
# 主流程：CCPD 图像处理
# -------------------------------------------------------

def process_ccpd(img, four_pts, bbox=None, out_w=94, out_h=24):
    """
    img      : BGR 原图（完整图像，非裁剪）
    four_pts : CCPD GT 角点 (4,2)，图像坐标系，rb→lb→lt→rt
    bbox     : fallback bbox，当 four_pts 为 None 时使用
    返回 94×24 BGR 图像，或 None。
    """
    if four_pts is not None:
        # 主路：直接透视变换
        return perspective_warp(img, four_pts, out_w, out_h)

    # fallback：用 bbox 裁剪 + resize
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        H, W = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            return cv2.resize(roi, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return None


def process_runtime(model, img, out_w=94, out_h=24):
    """
    实时推理接口（无 CCPD 文件名）。
    YOLO bbox → 裁剪 → resize。
    """
    box = yolo_detect(model, img)
    if box is None: return None
    x1, y1, x2, y2 = box
    H, W = img.shape[:2]
    # 加 5% padding
    pw = max(4, int((x2-x1)*0.05))
    ph = max(4, int((y2-y1)*0.05))
    x1, y1 = max(0, x1-pw), max(0, y1-ph)
    x2, y2 = min(W, x2+pw), min(H, y2+ph)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return None
    return cv2.resize(roi, (out_w, out_h), interpolation=cv2.INTER_AREA)


# -------------------------------------------------------
# 批量处理
# -------------------------------------------------------

def batch_convert(file_list, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    ok = skip = 0
    total = len(file_list)
    for i, (src_path, fname) in enumerate(file_list):
        if not os.path.isfile(src_path): skip += 1; continue
        plate_text, bbox, four_pts = parse_ccpd_filename(fname)
        if plate_text is None: skip += 1; continue

        img = read_image(src_path)
        if img is None: skip += 1; continue

        out = process_ccpd(img, four_pts, bbox)
        if out is None: skip += 1; continue

        dst = os.path.join(out_dir, '{}.jpg'.format(plate_text))
        cv2.imencode('.jpg', out)[1].tofile(dst)
        ok += 1

        if (i+1) % 2000 == 0 or (i+1) == total:
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
    for split in ['train', 'val', 'test']:
        out_dir = os.path.join(OUTPUT_DIR, split)
        print('\n========== {} =========='.format(split))
        n1 = batch_convert(collect_ccpd2019(split), out_dir, 'CCPD2019')
        n2 = batch_convert(collect_ccpd2020(split), out_dir, 'CCPD2020')
        print('  合计: {}'.format(n1 + n2))
    print('\n完成！输出目录: {}'.format(OUTPUT_DIR))


if __name__ == '__main__':
    main()
