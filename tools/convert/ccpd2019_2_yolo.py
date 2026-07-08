"""
CCPD2019 --> YOLOv8 格式转换脚本（修正版）

正确划分方式：完全按官方 splits/ 三个文件划分
  splits/train.txt → train（ccpd_base 中的训练图片）
  splits/val.txt   → val  （ccpd_base 中的验证图片）
  splits/test.txt  → test （ccpd_blur/challenge/db/fn/rotate/tilt/weather 中的测试图片）

ccpd_np（无车牌负样本）→ train
"""

import os
import shutil
from PIL import Image

# ==================== 路径配置 ====================
SOURCE_DIR = r"D:\Tempcode\26IC\车牌数据集\中国车牌\CCPD2019"
OUTPUT_DIR = r"c:\Users\Y9000P\Downloads\2026ICContest\ICcontest_project\CCPD2019_YOLO"
# =================================================


def parse_bbox(filename, img_w, img_h):
    """从 CCPD 文件名解析 BBox，返回 YOLOv8 归一化格式"""
    parts = filename.split("-")
    bbox_str = parts[2].split("_")
    x1, y1 = map(int, bbox_str[0].split("&"))
    x2, y2 = map(int, bbox_str[1].split("&"))

    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    w        = (x2 - x1) / img_w
    h        = (y2 - y1) / img_h

    return x_center, y_center, w, h


def create_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)


def convert_file(img_path, dst_split, has_plate=True, prefix=""):
    """转换单张图片，返回是否成功"""
    try:
        fname = os.path.basename(img_path)
        stem  = os.path.splitext(fname)[0]
        if prefix:
            fname = prefix + "_" + fname
            stem  = prefix + "_" + stem

        with Image.open(img_path) as img:
            img_w, img_h = img.size

        dst_img = os.path.join(OUTPUT_DIR, "images", dst_split, fname)
        dst_lbl = os.path.join(OUTPUT_DIR, "labels", dst_split, stem + ".txt")

        if has_plate:
            x_c, y_c, w, h = parse_bbox(fname, img_w, img_h)
            if not (0 < x_c < 1 and 0 < y_c < 1 and 0 < w <= 1 and 0 < h <= 1):
                return False
            with open(dst_lbl, "w") as f:
                f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
        else:
            open(dst_lbl, "w").close()

        shutil.copy2(img_path, dst_img)
        return True

    except Exception:
        return False


def convert_from_txt(txt_filename, dst_split):
    """按 splits/*.txt 文件转换到对应 split"""
    txt_path = os.path.join(SOURCE_DIR, "splits", txt_filename)
    print(f"\n[{txt_filename}] → {dst_split}")

    with open(txt_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    total   = len(lines)
    success = 0
    skip    = 0

    print(f"  共 {total} 张")

    for i, rel_path in enumerate(lines, 1):
        # rel_path 格式如：ccpd_base/filename.jpg 或 ccpd_blur/filename.jpg
        subset  = rel_path.split("/")[0]
        fname   = os.path.basename(rel_path)
        img_path = os.path.join(SOURCE_DIR, subset, fname)

        if not os.path.exists(img_path):
            skip += 1
            continue

        if convert_file(img_path, dst_split, prefix=subset):
            success += 1
        else:
            skip += 1

        if i % 10000 == 0:
            print(f"  进度：{i}/{total}")

    print(f"  完成：{success} 张，跳过 {skip} 张")
    return success


def convert_np_to_train():
    """ccpd_np：无车牌负样本，空标注 → train"""
    np_dir  = os.path.join(SOURCE_DIR, "ccpd_np")
    files   = [f for f in os.listdir(np_dir) if f.endswith(".jpg")]
    success = 0

    print(f"\n[ccpd_np] 共 {len(files)} 张（负样本）→ train")

    for fname in files:
        img_path = os.path.join(np_dir, fname)
        if convert_file(img_path, "train", has_plate=False):
            success += 1

    print(f"  完成：{success} 张")
    return success


def write_yaml(train_n, val_n, test_n):
    yaml_path = os.path.join(OUTPUT_DIR, "plate2019.yaml")
    content = f"""# CCPD2019 蓝牌数据集配置（修正版，按官方 splits 划分）
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
    print("=" * 55)
    print("CCPD2019 --> YOLOv8 格式转换（修正版）")
    print("=" * 55)
    print(f"数据源：{SOURCE_DIR}")
    print(f"输出到：{OUTPUT_DIR}")

    create_dirs()

    # 按官方三个 txt 文件严格划分
    train_n = convert_from_txt("train.txt", "train")
    val_n   = convert_from_txt("val.txt",   "val")
    test_n  = convert_from_txt("test.txt",  "test")

    # 负样本加入 train
    train_n += convert_np_to_train()

    write_yaml(train_n, val_n, test_n)

    print("\n" + "=" * 55)
    print(f"全部完成！train {train_n} 张，val {val_n} 张，test {test_n} 张")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
