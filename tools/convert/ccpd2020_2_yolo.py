"""
CCPD2020 --> YOLOv8 格式转换脚本
原数据集路径：车牌数据集/中国车牌/CCPD2020/ccpd_green/
输出路径：    CCPD_YOLO/

CCPD 文件名格式（以 - 分隔）：
  [0] ID
  [1] 面积比_倾斜角
  [2] BBox: x1&y1_x2&y2   <-- 我们需要这个
  [3] 四角点坐标
  [4] 车牌字符编码
  [5] 亮度
  [6] 模糊度

YOLOv8 标注格式（归一化）：
  class_id  x_center  y_center  width  height
"""

import os
import shutil
from PIL import Image

# ==================== 路径配置 ====================
SOURCE_DIR = r"D:\Tempcode\26IC\车牌数据集\中国车牌\CCPD2020\ccpd_green"
OUTPUT_DIR = r"c:\Users\Y9000P\Downloads\2026ICContest\ICcontest_project\CCPD2020_YOLO"
# =================================================

SPLITS = ["train", "val", "test"]


def parse_bbox(filename, img_w, img_h):
    """
    从 CCPD 文件名解析 BBox，返回 YOLOv8 归一化格式
    (x_center, y_center, width, height)，均在 0~1 之间
    """
    parts = filename.split("-")
    # parts[2] 形如 "311&485_406&524"
    bbox_str = parts[2].split("_")
    x1, y1 = map(int, bbox_str[0].split("&"))
    x2, y2 = map(int, bbox_str[1].split("&"))

    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    w        = (x2 - x1) / img_w
    h        = (y2 - y1) / img_h

    return x_center, y_center, w, h


def create_dirs():
    for split in SPLITS:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)


def convert_split(split):
    src_img_dir = os.path.join(SOURCE_DIR, split)
    dst_img_dir = os.path.join(OUTPUT_DIR, "images", split)
    dst_lbl_dir = os.path.join(OUTPUT_DIR, "labels", split)

    files = [f for f in os.listdir(src_img_dir) if f.endswith(".jpg")]
    total = len(files)
    success = 0
    skip = 0

    print(f"\n[{split}] 共 {total} 张，开始转换...")

    for i, fname in enumerate(files, 1):
        try:
            img_path = os.path.join(src_img_dir, fname)

            # 获取图片尺寸
            with Image.open(img_path) as img:
                img_w, img_h = img.size

            # 解析 BBox
            x_c, y_c, w, h = parse_bbox(fname, img_w, img_h)

            # 合法性检查
            if not (0 < x_c < 1 and 0 < y_c < 1 and 0 < w <= 1 and 0 < h <= 1):
                skip += 1
                continue

            # 写 label txt（class_id = 0）
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(dst_lbl_dir, stem + ".txt")
            with open(lbl_path, "w") as f:
                f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

            # 复制图片
            shutil.copy2(img_path, os.path.join(dst_img_dir, fname))

            success += 1

        except Exception as e:
            print(f"  跳过 {fname}：{e}")
            skip += 1

        if i % 1000 == 0:
            print(f"  进度：{i}/{total}")

    print(f"[{split}] 完成：成功 {success} 张，跳过 {skip} 张")
    return success


def write_yaml(train_n, val_n, test_n):
    yaml_path = os.path.join(OUTPUT_DIR, "plate2020.yaml")
    content = f"""# 车牌检测数据集配置
# 转换自 CCPD2020 ccpd_green
# train: {train_n} 张  val: {val_n} 张  test: {test_n} 张

path: {OUTPUT_DIR}
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  0: plate
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n已生成配置文件：{yaml_path}")


def main():
    print("=" * 50)
    print("CCPD2020 --> YOLOv8 格式转换")
    print("=" * 50)
    print(f"数据源：{SOURCE_DIR}")
    print(f"输出到：{OUTPUT_DIR}")

    create_dirs()

    counts = {}
    for split in SPLITS:
        counts[split] = convert_split(split)

    write_yaml(counts["train"], counts["val"], counts["test"])

    total = sum(counts.values())
    print("\n" + "=" * 50)
    print(f"全部完成！共转换 {total} 张")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 50)
    print("\n训练命令：")
    print(f"  yolo train model=yolov8n.pt data={OUTPUT_DIR}/plate.yaml epochs=100 imgsz=640 batch=16")


if __name__ == "__main__":
    main()
