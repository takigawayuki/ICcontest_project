"""
yolo_persp_crop.py
==================
两种模式：

【批量训练数据模式】 batch_convert() 调用
  直接从 CCPD 文件名解析 GT 四角点（精确的倾斜车牌坐标）→ 透视变换
  文件名格式：xxx-xx-bbox-vertices-plate-brightness-blur.jpg
  vertices 字段（index 3）：x1&y1_x2&y2_x3&y3_x4&y4，顺序 rb→lb→lt→rt

【实时推理模式】 process_image() 调用
  YOLO 检测 → bbox 四顶点 → 透视变换（等同于直接裁剪）
  实时场景摄像头角度通常接近正面，bbox 裁剪效果可接受

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
PAD          = 0.08
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

def get_plate_text(filename):
    stem = os.path.splitext(filename)[0].split('-')
    if len(stem) < 5:
        return None
    try:
        return decode_plate(stem[4])
    except Exception:
        return None

def get_gt_corners(filename):
    """
    从 CCPD 文件名解析 GT 四角点（图像坐标系）。
    vertices 字段（index 3）格式：x1&y1_x2&y2_x3&y3_x4&y4
    顺序：rb→lb→lt→rt（右下→左下→左上→右上）

    返回 (4,2) float32，顺序 rb→lb→lt→rt，或 None。
    """
    stem = os.path.splitext(filename)[0].split('-')
    if len(stem) < 4:
        return None
    try:
        verts = stem[3].split('_')
        pts = []
        for v in verts:
            xy = v.split('&')
            pts.append([int(xy[0]), int(xy[1])])
        if len(pts) != 4:
            return None
        return np.array(pts, dtype=np.float32)  # rb lb lt rt
    except Exception:
        return None

def read_image(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# -------------------------------------------------------
# 透视变换 → 94×24
# -------------------------------------------------------

def perspective_warp(img, quad_rb_lb_lt_rt, out_w=94, out_h=24):
    """
    quad: (4,2) float32，顺序 rb→lb→lt→rt，图像坐标系。
    先 warp 到 4× 中间尺寸（376×96）保证质量，再 INTER_AREA resize。
    """
    mid_w, mid_h = out_w * 4, out_h * 4
    src = quad_rb_lb_lt_rt.astype(np.float32)
    dst = np.array([[mid_w, mid_h],
                    [0,     mid_h],
                    [0,     0    ],
                    [mid_w, 0    ]], dtype=np.float32)
    M      = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (mid_w, mid_h), flags=cv2.INTER_CUBIC)
    return cv2.resize(warped, (out_w, out_h), interpolation=cv2.INTER_AREA)


# -------------------------------------------------------
# YOLO 推理（实时模式用）
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
# 实时推理主函数
# -------------------------------------------------------

def process_image(model, img, out_w=94, out_h=24):
    """
    实时调用接口：YOLO 检测 → bbox 四角点 → 透视变换 → 94×24。
    实时场景通常正面拍摄，bbox 裁剪已能满足 LPRNet 输入要求。
    返回 94×24 BGR 或 None。
    """
    bbox = yolo_detect(model, img)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return cv2.resize(roi, (out_w, out_h), interpolation=cv2.INTER_AREA)


# -------------------------------------------------------
# 批量处理（使用 GT 四角点，倾斜校正精确）
# -------------------------------------------------------

def batch_convert_with_gt(file_list, out_dir, tag):
    """
    从 CCPD 文件名解析 GT 四角点，直接做透视变换。
    不需要 YOLO，速度更快，倾斜校正完全准确。
    """
    os.makedirs(out_dir, exist_ok=True)
    ok = skip = 0
    total = len(file_list)
    for i, (src_path, fname) in enumerate(file_list):
        if not os.path.isfile(src_path):
            skip += 1; continue
        plate_text = get_plate_text(fname)
        if plate_text is None:
            skip += 1; continue
        corners = get_gt_corners(fname)
        if corners is None:
            skip += 1; continue
        img = read_image(src_path)
        if img is None:
            skip += 1; continue

        # corners 是图像坐标系的 rb→lb→lt→rt，直接透视变换
        out = perspective_warp(img, corners)
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
    print('使用 CCPD GT 四角点批量生成训练数据（无需 YOLO，倾斜校正精确）...')
    for split in ['train', 'val', 'test']:
        out_dir = os.path.join(OUTPUT_DIR, split)
        print('\n========== {} =========='.format(split))
        n1 = batch_convert_with_gt(collect_ccpd2019(split), out_dir, 'CCPD2019')
        n2 = batch_convert_with_gt(collect_ccpd2020(split), out_dir, 'CCPD2020')
        print('  合计: {}'.format(n1 + n2))
    print('\n完成！输出目录: {}'.format(OUTPUT_DIR))


if __name__ == '__main__':
    main()
