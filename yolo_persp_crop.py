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
# 图像增强（透视变换后，94×24 BGR）
# -------------------------------------------------------

def _adaptive_gamma_darken(l_channel, n_bx=3, n_by=3):
    """
    分块局部 Gamma 压暗（针对强光/反光）。
    gamma > 1 才能压暗：output = (input/255)^gamma * 255
      gamma=1 不变；gamma=2 明显压暗；gamma=3 大幅压暗
    每块根据局部均值动态计算 gamma，均值越高 gamma 越大（压暗越狠）。
    gamma 范围 [1.5, 3.0]。
    l_channel: uint8 单通道（LAB 的 L 分量）
    """
    h, w = l_channel.shape
    out = np.empty_like(l_channel)
    lut_cache = {}
    for by in range(n_by):
        y1 = int(by * h / n_by)
        y2 = int((by + 1) * h / n_by) if by < n_by - 1 else h
        for bx in range(n_bx):
            x1 = int(bx * w / n_bx)
            x2 = int((bx + 1) * w / n_bx) if bx < n_bx - 1 else w
            block = l_channel[y1:y2, x1:x2]
            if block.size == 0:
                continue
            local_mean = float(np.mean(block))
            # 均值越高 gamma 越大（压暗越狠）；范围 [1.5, 3.0]
            gamma = float(np.clip(1.0 + (local_mean - 128.0) / 80.0, 1.5, 3.0))
            g_key = round(gamma, 2)
            if g_key not in lut_cache:
                lut = np.array([int(((i / 255.0) ** gamma) * 255 + 0.5)
                                for i in range(256)], dtype=np.uint8)
                lut_cache[g_key] = lut
            out[y1:y2, x1:x2] = lut_cache[g_key][block]
    return out


def roi_mean_l(img, corners):
    """
    在原图四角点 bbox 内计算 LAB-L 均值，用于亮度判断。
    corners: (4,2) float32，rb→lb→lt→rt
    """
    H, W = img.shape[:2]
    xs = corners[:, 0].astype(int)
    ys = corners[:, 1].astype(int)
    x1 = max(0, xs.min()); y1 = max(0, ys.min())
    x2 = min(W, xs.max()); y2 = min(H, ys.max())
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return 128.0
    return float(np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)[:, :, 0]))


ENHANCE_BRIGHT_THR = 211   # 原图 ROI L均值 > 211：强光处理（211~227 范围需压暗）
ENHANCE_DARK_THR   = 146   # 原图 ROI L均值 < 146：暗场处理（121~146 范围需增强）


def enhance_plate(plate_bgr, mean_l):
    """
    对 94×24 BGR 车牌图像做自适应增强。
    mean_l: 由原图四角点 bbox 内计算的 LAB-L 均值（roi_mean_l 返回值）。
      - 强光/反光（mean_l > ENHANCE_BRIGHT_THR）：局部自适应 Gamma 压暗高光
      - 正常光线（范围内）：直接返回原图，不做任何处理
      - 暗/夜间（mean_l < ENHANCE_DARK_THR）：线性拉亮到目标均值 ~110 + CLAHE（clip=4.0）
    在 LAB 色彩空间仅处理亮度通道，不影响色相。
    """
    if ENHANCE_DARK_THR <= mean_l <= ENHANCE_BRIGHT_THR:
        return plate_bgr   # 正常光线，不处理

    lab = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if mean_l > ENHANCE_BRIGHT_THR:
        # 强光：局部自适应 Gamma（3×3 分块）
        l_out = _adaptive_gamma_darken(l, n_bx=3, n_by=3)
    else:
        # 暗/夜间：先线性拉亮到目标均值 ~110，再 CLAHE 增强对比度
        scale = min(2.8, 110.0 / max(float(mean_l), 1.0))
        l_boosted = np.clip(l.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        l_out = clahe.apply(l_boosted)

    return cv2.cvtColor(cv2.merge([l_out, a, b]), cv2.COLOR_LAB2BGR)


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

        # corners 是图像坐标系的 rb→lb→lt→rt，直接透视变换 + 图像增强
        # 亮度在原图四角点区域内计算，避免 warp 后失真
        mean_l = roi_mean_l(img, corners)
        out = enhance_plate(perspective_warp(img, corners), mean_l)
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
