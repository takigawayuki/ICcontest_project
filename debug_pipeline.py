"""
调试脚本：可视化 YOLO检测 → 裁剪 → 二值化 → 轮廓 → 透视校正 每个阶段
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import random
from ultralytics import YOLO

CCPD2019_DIR = r"D:\Tempcode\26IC\车牌数据集\中国车牌\CCPD2019"
MODEL_PATH   = r"runs/new_plate_detect_merged/weights/best.pt"
TARGET_W, TARGET_H = 94, 24



def to_bgr(img):
    """灰度图转BGR，方便拼接"""
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def resize_to_h(img, h):
    """等比缩放到指定高度"""
    oh, ow = img.shape[:2]
    w = max(1, int(ow * h / oh))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def add_label(img, text):
    """在图像顶部加标题条"""
    label_h = 20
    canvas  = np.zeros((img.shape[0] + label_h, img.shape[1], 3), dtype=np.uint8)
    canvas[label_h:] = img
    cv2.putText(canvas, text, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return canvas


def debug_one(img_path):
    model  = YOLO(MODEL_PATH)
    stages = []  # 每个元素：(标题, 图像)

    # ── 阶段1：原图 ──
    buf = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    stages.append(("1.Original", img.copy()))

    # ── 阶段2：YOLO检测框 ──
    results  = model(img, conf=0.25, verbose=False)
    boxes    = results[0].boxes
    if boxes is None or len(boxes) == 0:
        print("未检测到车牌")
        return

    best_idx = int(boxes.conf.argmax())
    x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].tolist())
    conf     = float(boxes.conf[best_idx])
    det_img  = img.copy()
    cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(det_img, f"{conf:.2f}", (x1, max(y1-5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    stages.append(("2.YOLO Detection", det_img))

    # ── 阶段3：裁剪ROI ──
    h, w = img.shape[:2]
    roi  = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    stages.append(("3.Cropped ROI", roi.copy()))

    # ── 阶段4：灰度化 ──
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    stages.append(("4.Grayscale", gray.copy()))

    # ── 阶段5：Otsu二值化 ──
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    stages.append(("5.Binary(Otsu)", binary.copy()))

    # ── 阶段6：形态学闭运算 ──
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    stages.append(("6.Morphology Close", closed.copy()))

    # ── 阶段7：找最大轮廓和最小外接矩形 ──
    contour_img = roi.copy()
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 获取最大轮廓
        cnt = max(contours, key=cv2.contourArea)

        # 绘制轮廓
        cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 2)

        # 获取最小外接矩形
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        # 绘制旋转矩形
        cv2.drawContours(contour_img, [box], 0, (0, 0, 255), 2)

        # 标注四个角点
        for i, pt in enumerate(box):
            cv2.circle(contour_img, tuple(pt), 3, (255, 0, 0), -1)
            cv2.putText(contour_img, str(i), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    stages.append(("7.MinAreaRect", contour_img))
    print(f"  轮廓数量: {len(contours)}")

    # ── 阶段8：透视校正 ──
    if contours and len(box) == 4:
        # 导入order_points函数（从test1.py复制）
        def order_points(pts):
            pts = pts[np.argsort(pts[:, 1])]
            top_pts = pts[:2]
            bottom_pts = pts[2:]
            top_pts = top_pts[np.argsort(top_pts[:, 0])]
            bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]
            return np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.float32)

        box_sorted = order_points(box.astype(np.float32))
        dst = np.array([
            [0, 0],
            [TARGET_W - 1, 0],
            [TARGET_W - 1, TARGET_H - 1],
            [0, TARGET_H - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(box_sorted, dst)
        corrected = cv2.warpPerspective(roi, M, (TARGET_W, TARGET_H))
        stages.append(("8.Perspective Corrected", corrected))
    else:
        print("  轮廓不足，降级resize")
        stages.append(("8.Fallback Resize", cv2.resize(roi, (TARGET_W, TARGET_H))))

    # ── 拼成两行大图 ──
    # 第一行：原图、检测框（大图）
    # 第二行：ROI、灰度、二值化、轮廓、边界线、校正结果（小图）
    row1 = stages[:2]   # 原图、检测框
    row2 = stages[2:]   # ROI 之后

    GAP  = 6
    R1_H = 480  # 第一行统一高度
    R2_H = 120  # 第二行统一高度

    def make_row(items, row_h, max_cell_w=None):
        cells = []
        for title, s_img in items:
            cell = resize_to_h(to_bgr(s_img), row_h)
            if max_cell_w and cell.shape[1] > max_cell_w:
                cell = cv2.resize(cell, (max_cell_w, row_h))
            cell = add_label(cell, title)
            cells.append(cell)
        row_w = sum(c.shape[1] for c in cells) + GAP * (len(cells) - 1)
        row   = np.full((cells[0].shape[0], row_w, 3), 50, dtype=np.uint8)
        x = 0
        for c in cells:
            row[:c.shape[0], x:x+c.shape[1]] = c
            x += c.shape[1] + GAP
        return row

    r1 = make_row(row1, R1_H, max_cell_w=600)
    r2 = make_row(row2, R2_H, max_cell_w=280)

    canvas_w = max(r1.shape[1], r2.shape[1])
    canvas_h = r1.shape[0] + GAP + r2.shape[0]
    canvas   = np.full((canvas_h, canvas_w, 3), 30, dtype=np.uint8)
    canvas[:r1.shape[0], :r1.shape[1]] = r1
    canvas[r1.shape[0]+GAP:r1.shape[0]+GAP+r2.shape[0], :r2.shape[1]] = r2

    os.makedirs("preview_output", exist_ok=True)
    out_path = "preview_output/debug_pipeline.jpg"
    cv2.imwrite(out_path, canvas)
    print(f"已保存: {out_path}, 尺寸: {canvas.shape[1]}x{canvas.shape[0]}")
    print("stages:", [t for t, _ in stages])

    MAX_W = 1920
    if canvas.shape[1] > MAX_W:
        scale  = MAX_W / canvas.shape[1]
        canvas = cv2.resize(canvas, (MAX_W, int(canvas.shape[0] * scale)))
    cv2.imshow("Pipeline Debug", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # 从 CCPD2019 train.txt 随机取一张
    txt_path = os.path.join(CCPD2019_DIR, "splits", "train.txt")
    with open(txt_path, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    for rel_path in random.sample(lines, 20):
        fname    = os.path.basename(rel_path)
        src_path = os.path.join(CCPD2019_DIR, rel_path.replace('/', os.sep))
        parts    = os.path.splitext(fname)[0].split('-')
        if len(parts) < 6:
            continue
        print(f"处理: {fname}")
        debug_one(src_path)
        break


if __name__ == "__main__":
    main()
