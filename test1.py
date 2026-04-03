"""
CCPD → LPRNet 训练格式转换脚本（YOLO检测 + 透视校正版）

流程：
  原图 → YOLOv8 检测车牌 bbox → 裁出 ROI
       → 透视校正（基于字符轮廓拟合上下边界）
       → resize 到 94×24
       → 以车牌号命名保存

输出目录：
  LPR_DATA_V2/
  ├── train/
  ├── val/
  └── test/
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

# ==================== 路径配置 ====================
CCPD2019_DIR = r"D:\Tempcode\26IC\车牌数据集\中国车牌\CCPD2019"
CCPD2020_DIR = r"D:\Tempcode\26IC\车牌数据集\中国车牌\CCPD2020\ccpd_green"

BASE       = r"c:\Users\Y9000P\Downloads\2026ICContest\ICcontest_project"
OUTPUT_DIR = os.path.join(BASE, "LPR_DATA_V2")
MODEL_PATH = os.path.join(BASE, "runs", "new_plate_detect_merged", "weights", "best.pt")

CONF_THRESH = 0.25
TARGET_W, TARGET_H = 94, 24  # LPRNet 输入尺寸
# =================================================

# ---------- CCPD 字符映射表 ----------
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新", "警", "学", "O"
]
ALPHABETS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'O'
]
ADS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O'
]


def decode_plate(plate_str: str) -> str:
    parts = plate_str.split('_')
    chars = []
    for i, p in enumerate(parts):
        idx = int(p)
        if i == 0:
            chars.append(PROVINCES[idx])
        elif i == 1:
            chars.append(ALPHABETS[idx])
        else:
            chars.append(ADS[idx])
    return ''.join(chars)


# ---------- 透视校正 ----------
def order_points(pts):
    """对四个角点排序：左上、右上、右下、左下"""
    # 先按y坐标排序，分出上面两个点和下面两个点
    pts = pts[np.argsort(pts[:, 1])]
    top_pts = pts[:2]
    bottom_pts = pts[2:]

    # 上面两个点按x排序：左上、右上
    top_pts = top_pts[np.argsort(top_pts[:, 0])]
    # 下面两个点按x排序：左下、右下
    bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]

    # 返回：左上、右上、右下、左下
    return np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.float32)


def perspective_correction(roi):
    """基于最小外接矩形的透视校正，返回 TARGET_W×TARGET_H 图像。
    校正失败时直接 resize 返回。"""

    # 灰度化
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 使用Otsu自动二值化（比自适应阈值更稳定）
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学闭运算：将字符连接成一个整体
    # 横向kernel较大，纵向较小，符合车牌字符横向排列的特点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(roi, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)

    # 获取最大轮廓（通常是整个车牌区域）
    cnt = max(contours, key=cv2.contourArea)

    # 获取最小外接矩形（可以是旋转的）
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)

    # 检查box的有效性
    if len(box) != 4:
        return cv2.resize(roi, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)

    # 排序角点：左上、右上、右下、左下
    box = order_points(box.astype(np.float32))

    # 目标矩形的四个角点（94x24）
    dst = np.array([
        [0, 0],              # 左上
        [TARGET_W - 1, 0],   # 右上
        [TARGET_W - 1, TARGET_H - 1],  # 右下
        [0, TARGET_H - 1]    # 左下
    ], dtype=np.float32)

    # 计算透视变换矩阵并应用
    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(roi, M, (TARGET_W, TARGET_H))

    return warped


# ---------- 核心处理函数 ----------
def process_image(model, img_path: str, filename: str, dst_dir: str) -> bool:
    """YOLO检测 → 透视校正 → 保存，返回是否成功"""
    try:
        parts = os.path.splitext(filename)[0].split('-')
        if len(parts) < 6:
            return False  # 无车牌负样本，跳过

        plate_text = decode_plate(parts[4])

        # 读图（支持中文路径）
        buf = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return False

        # YOLO 推理
        results = model(img, conf=CONF_THRESH, verbose=False)
        boxes   = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return False

        # 取置信度最高的框
        best_idx = int(boxes.conf.argmax())
        x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].tolist())

        # 边界保护
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        # 透视校正 + resize
        plate_img = perspective_correction(roi)

        # 保存
        dst_path = os.path.join(dst_dir, plate_text + ".jpg")
        cv2.imencode('.jpg', plate_img)[1].tofile(dst_path)
        return True

    except Exception:
        return False


# ---------- 数据集转换 ----------
def convert_ccpd2019(model, split: str, out_dir: str) -> int:
    txt_path = os.path.join(CCPD2019_DIR, "splits", f"{split}.txt")
    with open(txt_path, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    ok = skip = 0
    for i, rel_path in enumerate(lines):
        fname    = os.path.basename(rel_path)
        src_path = os.path.join(CCPD2019_DIR, rel_path.replace('/', os.sep))
        if process_image(model, src_path, fname, out_dir):
            ok += 1
        else:
            skip += 1
        if (i + 1) % 10000 == 0:
            print(f"  CCPD2019/{split}: {i+1}/{len(lines)}")

    print(f"  CCPD2019/{split} 完成: 成功 {ok}，跳过 {skip}")
    return ok


def convert_ccpd2020(model, split: str, out_dir: str) -> int:
    src_dir = os.path.join(CCPD2020_DIR, split)
    files   = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]

    ok = skip = 0
    for i, fname in enumerate(files):
        src_path = os.path.join(src_dir, fname)
        if process_image(model, src_path, fname, out_dir):
            ok += 1
        else:
            skip += 1
        if (i + 1) % 2000 == 0:
            print(f"  CCPD2020/{split}: {i+1}/{len(files)}")

    print(f"  CCPD2020/{split} 完成: 成功 {ok}，跳过 {skip}")
    return ok


def main():
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    print(f"加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    for split in ['train', 'val', 'test']:
        out_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n========== {split} ==========")
        n2019 = convert_ccpd2019(model, split, out_dir)
        n2020 = convert_ccpd2020(model, split, out_dir)
        print(f"  {split} 合计: {n2019 + n2020} 张")

    print(f"\n全部完成，输出目录: {OUTPUT_DIR}")


def preview(n=10):
    """从 CCPD2019 和 CCPD2020 各随机抽 n 张，处理后拼图显示"""
    import random
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    model = YOLO(MODEL_PATH)

    samples = []

    # CCPD2019：从 splits/train.txt 随机抽
    txt_path = os.path.join(CCPD2019_DIR, "splits", "train.txt")
    with open(txt_path, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    for rel_path in random.sample(lines, n * 3):  # 多抽一些防止检测失败
        fname    = os.path.basename(rel_path)
        src_path = os.path.join(CCPD2019_DIR, rel_path.replace('/', os.sep))
        parts    = os.path.splitext(fname)[0].split('-')
        if len(parts) < 6:
            continue
        buf = np.fromfile(src_path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        results = model(img, conf=CONF_THRESH, verbose=False)
        boxes   = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        best_idx = int(boxes.conf.argmax())
        x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].tolist())
        h, w = img.shape[:2]
        roi = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if roi.size == 0:
            continue
        plate = perspective_correction(roi)
        plate_text = decode_plate(parts[4])
        samples.append(('2019', plate_text, roi.copy(), plate))
        if len(samples) >= n:
            break

    # CCPD2020：从 train 目录随机抽
    src_dir = os.path.join(CCPD2020_DIR, "train")
    files   = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
    count   = 0
    for fname in random.sample(files, min(n * 3, len(files))):
        src_path = os.path.join(src_dir, fname)
        parts    = os.path.splitext(fname)[0].split('-')
        if len(parts) < 6:
            continue
        buf = np.fromfile(src_path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        results = model(img, conf=CONF_THRESH, verbose=False)
        boxes   = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        best_idx = int(boxes.conf.argmax())
        x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].tolist())
        h, w = img.shape[:2]
        roi = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        if roi.size == 0:
            continue
        plate = perspective_correction(roi)
        plate_text = decode_plate(parts[4])
        samples.append(('2020', plate_text, roi.copy(), plate))
        count += 1
        if count >= n:
            break

    # 拼图：每行左边是裁剪前ROI，右边是校正后，最右打标签
    cell_w, cell_h = TARGET_W * 3, TARGET_H * 3  # 放大3倍方便看
    gap = 10
    canvas_h = (cell_h + gap) * len(samples) + gap
    canvas_w = cell_w * 2 + gap * 3
    canvas   = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240

    for i, (dataset, label, roi_img, plate_img) in enumerate(samples):
        y_off = i * (cell_h + gap) + gap

        # 左：裁剪前 ROI（等比缩放到 cell_h 高度）
        roi_h, roi_w = roi_img.shape[:2]
        scale   = cell_h / roi_h
        roi_show = cv2.resize(roi_img, (int(roi_w * scale), cell_h), interpolation=cv2.INTER_LINEAR)
        roi_show = roi_show[:, :cell_w]  # 超宽截断
        canvas[y_off:y_off+cell_h, gap:gap+roi_show.shape[1]] = roi_show

        # 右：校正后
        corrected_show = cv2.resize(plate_img, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
        x_off = cell_w + gap * 2
        canvas[y_off:y_off+cell_h, x_off:x_off+cell_w] = corrected_show

    # 保存并显示
    os.makedirs("preview_output", exist_ok=True)
    out_path = "preview_output/plate_preview.jpg"
    cv2.imwrite(out_path, canvas)
    print(f"预览图已保存: {out_path}")
    cv2.imshow("Plate Preview", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "preview":
        preview()
    else:
        main()
