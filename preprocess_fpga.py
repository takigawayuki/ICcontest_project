"""
FPGA 模拟预处理管线: CCPD → LPRNet 训练格式
==============================================
新管线（按车牌颜色分支处理）：
  1. YOLO 推理定位车牌 bbox
  2. 裁剪 ROI；小目标先 2× 超分
  3. 非锐化掩膜（Unsharp Mask）对抗模糊
  4. 车牌颜色判断（蓝/绿）
  5. 颜色通道提取 + 二值化
       蓝牌: BGR→YCbCr, 取 Cb 通道, Otsu 二值化
       绿牌: BGR→HSV,   取 S  通道, Otsu 二值化
  6. 腐蚀（去噪）→ Sobel 垂直边缘 → 膨胀（连接边缘）
  7. minAreaRect → 仿射矫正（±15° 门限）
  8. 光照自适应增强（强光→局部Gamma，夜间→CLAHE，正常→轻度CLAHE）
  9. Resize → 94×24 输出

输出目录: LPR_DATA_FPGA/  train/ val/ test/
"""

import os
import cv2
import numpy as np
from typing import Optional, List, Tuple
from ultralytics import YOLO

# ===== 路径配置 =====
BASE_DIR     = r"c:\Users\Y9000P\Downloads\2026ICContest\ICcontest_project"
YOLO_MODEL   = os.path.join(BASE_DIR, r"runs\new_plate_detect_merged\weights\best.pt")
CCPD2019_DIR = r"D:\Tempcode\26IC\车牌数据集\中国车牌\CCPD2019"
CCPD2020_DIR = r"D:\Tempcode\26IC\车牌数据集\中国车牌\CCPD2020\ccpd_green"
OUTPUT_DIR   = os.path.join(BASE_DIR, "LPR_DATA_FPGA")

USE_YOLO  = True   # False = 用文件名 bbox（快速调试）
YOLO_CONF = 0.3
# ====================

PROVINCES = [
    "皖","沪","津","渝","冀","晋","蒙","辽","吉","黑",
    "苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤",
    "桂","琼","川","贵","云","藏","陕","甘","青","宁",
    "新","警","学","O"
]
ALPHABETS = ['A','B','C','D','E','F','G','H','J','K',
             'L','M','N','P','Q','R','S','T','U','V',
             'W','X','Y','Z','O']
ADS = ['A','B','C','D','E','F','G','H','J','K',
       'L','M','N','P','Q','R','S','T','U','V',
       'W','X','Y','Z',
       '0','1','2','3','4','5','6','7','8','9','O']


# -------------------------------------------------------
# 工具：CCPD 文件名解析
# -------------------------------------------------------

def decode_plate(plate_str: str) -> str:
    parts = plate_str.split('_')
    chars = []
    for i, p in enumerate(parts):
        idx = int(p)
        if   i == 0: chars.append(PROVINCES[idx])
        elif i == 1: chars.append(ALPHABETS[idx])
        else:        chars.append(ADS[idx])
    return ''.join(chars)


def parse_ccpd_filename(filename: str):
    """
    解析 CCPD 文件名。
    返回: (plate_text, bbox, four_pts) 或 (None, None, None)
      bbox      = (x1,y1,x2,y2)，来自 parts[2]，YOLO fallback 用
      four_pts  = np.float32 shape(4,2)，来自 parts[3]
                  顺序：右下→左下→左上→右上
                  透视矫正 dst 对应：(w,h)→(0,h)→(0,0)→(w,0)
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
        four_pts = np.array(pts, dtype=np.float32)  # (4,2): rb,lb,lt,rt
    except Exception:
        four_pts = None
    return plate_text, bbox, four_pts


def read_image(path: str) -> Optional[np.ndarray]:
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# -------------------------------------------------------
# 模块一：YOLO 推理 + 裁剪
# -------------------------------------------------------

def detect_and_crop(model: YOLO, img: np.ndarray,
                    fallback_bbox=None) -> Optional[np.ndarray]:
    best_box, best_conf = None, 0.0
    for result in model(img, verbose=False, conf=YOLO_CONF):
        if result.boxes is None: continue
        for box in result.boxes:
            c = float(box.conf[0])
            if c > best_conf:
                best_conf = c
                best_box = box.xyxy[0].cpu().numpy().astype(int)

    if best_box is not None:
        x1, y1, x2, y2 = best_box
    elif fallback_bbox is not None:
        x1, y1, x2, y2 = fallback_bbox
    else:
        return None

    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1: return None
    return img[y1:y2, x1:x2]


# -------------------------------------------------------
# 模块二：图像质量评估
# -------------------------------------------------------

def assess_quality(roi: np.ndarray) -> dict:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(hsv[:, :, 2]))
    h, w = roi.shape[:2]
    return {
        'brightness': brightness,
        'is_bright':  brightness > 190,
        'is_dark':    brightness < 60,
        'is_small':   (h * w) < 1500,
    }


# -------------------------------------------------------
# 模块三：超分（双三次，仅小目标触发）
# -------------------------------------------------------

def super_resolve(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    return cv2.resize(roi, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)


# -------------------------------------------------------
# 模块四：非锐化掩膜（Unsharp Mask）
# -------------------------------------------------------

def unsharp_mask(img: np.ndarray,
                 sigma: float = 1.2,
                 strength: float = 1.5) -> np.ndarray:
    """
    img_sharp = img + strength * (img - gaussian_blur(img))
    对模糊 ROI 有效；sigma/strength 可调。
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)


# -------------------------------------------------------
# 模块五：车牌颜色判断 + 颜色通道二值化
# -------------------------------------------------------

def detect_plate_type(roi: np.ndarray) -> str:
    """
    判断车牌颜色类型：'blue' / 'green' / 'unknown'
    用 HSV 范围统计背景颜色占比。
    """
    hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total = roi.shape[0] * roi.shape[1]
    blue_ratio  = np.sum(cv2.inRange(hsv, (100, 60, 60), (130, 255, 255)) > 0) / total
    green_ratio = np.sum(cv2.inRange(hsv, (35,  60, 60), (85,  255, 255)) > 0) / total
    if blue_ratio > 0.05 and blue_ratio >= green_ratio:
        return 'blue'
    if green_ratio > 0.05 and green_ratio > blue_ratio:
        return 'green'
    return 'unknown'


def extract_binary_mask(roi: np.ndarray, plate_type: str) -> np.ndarray:
    """
    按颜色类型提取二值掩膜（字符=白，背景=黑）：
    - 蓝牌: BGR→YCrCb, Cb 通道 + BINARY_INV
            蓝背景 Cb 高 → INV 后背景变黑，白/黄字变白
    - 绿牌: BGR→HSV, V 通道 + BINARY_INV
            绿背景较暗 → INV 后背景变黑，亮字变白
    - unknown: 灰度 + BINARY_INV
    返回单通道二值图（字符白=255，背景黑=0）。
    """
    if plate_type == 'blue':
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        ch = ycrcb[:, :, 2]   # Cb
    elif plate_type == 'green':
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ch = hsv[:, :, 2]     # V（亮度）
    else:
        ch = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(ch, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


# -------------------------------------------------------
# 模块六：腐蚀 → Sobel → 膨胀（边缘增强）
# -------------------------------------------------------

def morph_edge_pipeline(binary: np.ndarray) -> np.ndarray:
    """
    输入：二值掩膜（背景=白色）
    输出：细边缘图（用于 HoughLinesP 角度估计）
    步骤：腐蚀(去噪) → Sobel水平+垂直边缘 → 轻度膨胀
    """
    # 腐蚀去噪（小核，保留主要结构）
    k_e = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(binary, k_e, iterations=1)

    # Sobel 双向边缘（水平+垂直）
    sx = cv2.Sobel(eroded, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(eroded, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    if mag.max() == 0:
        return eroded
    mag_norm = (mag / mag.max() * 255).astype(np.uint8)
    # 高阈值：只保留强边缘，避免填满整图
    _, edge = cv2.threshold(mag_norm, 80, 255, cv2.THRESH_BINARY)

    # 轻度膨胀（仅 1 次，小核）
    k_d = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    dilated = cv2.dilate(edge, k_d, iterations=1)
    return dilated


# -------------------------------------------------------
# 模块七：几何矫正
# -------------------------------------------------------

def correct_perspective(img: np.ndarray, four_pts: np.ndarray,
                        dst_w: int = 376, dst_h: int = 96) -> np.ndarray:
    """
    用 CCPD 文件名中的 4 角点做透视变换（训练数据生成专用）。
    four_pts 顺序：右下→左下→左上→右上
    dst 对应：  (w,h)→(0,h)→(0,0)→(w,0)
    dst_w/h 默认 376×96（车牌标准比例约 4:1，留余量）。
    """
    dst = np.array([[dst_w, dst_h], [0, dst_h], [0, 0], [dst_w, 0]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(four_pts, dst)
    return cv2.warpPerspective(img, M, (dst_w, dst_h),
                               flags=cv2.INTER_CUBIC)


def correct_skew_hough(roi: np.ndarray, edge_map: np.ndarray) -> np.ndarray:
    """
    用 HoughLinesP 估计倾斜角做仿射矫正（实时推理场景 fallback）。
    超出 ±15° 时跳过，避免误矫正。
    """
    h, w = roi.shape[:2]
    min_len = max(10, int(w * 0.20))
    lines = cv2.HoughLinesP(edge_map, 1, np.pi / 180,
                            threshold=15,
                            minLineLength=min_len,
                            maxLineGap=8)
    if lines is None:
        return roi

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1: continue
        a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if abs(a) < 30:
            angles.append(a)

    if not angles:
        return roi

    angle = float(np.median(angles))
    if abs(angle) > 15:
        return roi

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(roi, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# -------------------------------------------------------
# 模块八：光照自适应增强
# -------------------------------------------------------

def adaptive_gamma(roi: np.ndarray, block_size: int = 8) -> np.ndarray:
    """局部自适应 Gamma（强光）：均值越高 gamma 越大，压暗高光"""
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L   = lab[:, :, 0].astype(np.float32)
    out = L.copy()
    h, w = L.shape
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            blk  = L[y:y+block_size, x:x+block_size]
            mean = float(np.mean(blk))
            g    = 1.6 if mean > 210 else (1.3 if mean > 170 else 1.0)
            out[y:y+block_size, x:x+block_size] = np.power(blk/255.0, g)*255.0
    lab[:, :, 0] = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def clahe_enhance(roi: np.ndarray, clip: float = 3.0) -> np.ndarray:
    """CLAHE（夜间低光）"""
    lab   = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def normal_enhance(roi: np.ndarray) -> np.ndarray:
    """轻度 CLAHE（正常光照）"""
    return clahe_enhance(roi, clip=1.5)


# -------------------------------------------------------
# 完整预处理管线
# -------------------------------------------------------

def preprocess(roi: np.ndarray,
               four_pts: Optional[np.ndarray] = None) -> np.ndarray:
    """
    输入：BGR 车牌 ROI（任意尺寸）
          four_pts: CCPD 4角点（在原图坐标系），有则做透视矫正
    输出：94×24 BGR
    """
    q = assess_quality(roi)

    # ① 小目标超分
    if q['is_small']:
        roi = super_resolve(roi)

    # ② 锐化（对抗模糊）
    roi = unsharp_mask(roi)

    # ③ 判断颜色类型
    ptype = detect_plate_type(roi)

    # ④ 颜色通道二值化
    binary = extract_binary_mask(roi, ptype)

    # ⑤ 腐蚀→Sobel→膨胀（供 Hough fallback 用）
    edge_map = morph_edge_pipeline(binary)

    # ⑥ 几何矫正：优先透视变换，否则 Hough 仿射
    if four_pts is not None:
        roi = correct_perspective(roi, four_pts)
    else:
        roi = correct_skew_hough(roi, edge_map)

    # ⑦ 光照增强
    if q['is_bright']:
        roi = adaptive_gamma(roi)
    elif q['is_dark']:
        roi = clahe_enhance(roi)
    else:
        roi = normal_enhance(roi)

    # ⑧ 输出尺寸
    return cv2.resize(roi, (94, 24), interpolation=cv2.INTER_CUBIC)


# -------------------------------------------------------
# 批量处理
# -------------------------------------------------------

def process_one(model, src_path: str, filename: str, out_dir: str) -> bool:
    plate_text, fallback_bbox, four_pts = parse_ccpd_filename(filename)
    if plate_text is None: return False

    img = read_image(src_path)
    if img is None: return False

    if USE_YOLO and model is not None:
        roi       = detect_and_crop(model, img, fallback_bbox)
        four_pts  = None   # 实时推理无文件名角点，用 Hough fallback
    else:
        if fallback_bbox is None: return False
        x1, y1, x2, y2 = fallback_bbox
        roi = img[y1:y2, x1:x2]
        # four_pts 在原图坐标系，需要转为 ROI 坐标系
        if four_pts is not None:
            four_pts = four_pts - np.array([x1, y1], dtype=np.float32)

    if roi is None or roi.size == 0: return False

    processed = preprocess(roi, four_pts)
    out_path  = os.path.join(out_dir, "{}.jpg".format(plate_text))
    cv2.imencode('.jpg', processed)[1].tofile(out_path)
    return True


def convert(model, file_list: List[Tuple], out_dir: str, tag: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    ok = skip = 0
    total = len(file_list)
    for i, (src_path, filename) in enumerate(file_list):
        if process_one(model, src_path, filename, out_dir): ok   += 1
        else:                                                skip += 1
        if (i+1) % 2000 == 0 or (i+1) == total:
            print("  [{}] {}/{} | 成功 {} 跳过 {}".format(tag, i+1, total, ok, skip))
    return ok


def collect_ccpd2019(split: str) -> List[Tuple]:
    txt = os.path.join(CCPD2019_DIR, "splits", "{}.txt".format(split))
    with open(txt, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    return [(os.path.join(CCPD2019_DIR, p.replace('/', os.sep)), os.path.basename(p))
            for p in lines]


def collect_ccpd2020(split: str) -> List[Tuple]:
    src_dir = os.path.join(CCPD2020_DIR, split)
    return [(os.path.join(src_dir, f), f)
            for f in os.listdir(src_dir) if f.endswith('.jpg')]


def main():
    model = None
    if USE_YOLO:
        print("加载 YOLO 模型...")
        model = YOLO(YOLO_MODEL)
    else:
        print("USE_YOLO=False，使用 CCPD 文件名 bbox")

    for split in ['train', 'val', 'test']:
        out_dir = os.path.join(OUTPUT_DIR, split)
        print("\n========== {} ==========".format(split))
        n2019 = convert(model, collect_ccpd2019(split), out_dir, "CCPD2019/{}".format(split))
        n2020 = convert(model, collect_ccpd2020(split), out_dir, "CCPD2020/{}".format(split))
        print("  {} 合计: {} 张".format(split, n2019 + n2020))

    print("\n全部完成！输出目录: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
