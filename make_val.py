"""
构建混合验证集
从 CCPD2019_YOLO/val（蓝牌）中随机抽取 1000 张
与 CCPD_YOLO/val（绿牌 1001 张）合并
输出到 VAL_MIXED/（共约 2001 张）
"""

import os
import shutil
import random

# ==================== 路径配置 ====================
BASE        = r"c:\Users\Y9000P\Downloads\2026ICContest\ICcontest_project"
GREEN_VAL   = os.path.join(BASE, "CCPD_YOLO", "images", "val")
BLUE_VAL    = os.path.join(BASE, "CCPD2019_YOLO", "images", "val")
OUTPUT_DIR  = os.path.join(BASE, "VAL_MIXED")
SAMPLE_N    = 1000   # 从蓝牌 val 里抽取的数量
# =================================================

random.seed(42)


def create_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", "val"), exist_ok=True)


def copy_files(src_img_dir, files, prefix=""):
    """复制图片和对应标注到 VAL_MIXED"""
    src_lbl_dir = src_img_dir.replace("images", "labels")
    success = 0
    for fname in files:
        stem    = os.path.splitext(fname)[0]
        dst_img = os.path.join(OUTPUT_DIR, "images", "val", prefix + fname)
        dst_lbl = os.path.join(OUTPUT_DIR, "labels", "val", prefix + stem + ".txt")

        # 文件名冲突时加前缀
        if os.path.exists(dst_img):
            prefix_fix = "dup_" + prefix
            dst_img = os.path.join(OUTPUT_DIR, "images", "val", prefix_fix + fname)
            dst_lbl = os.path.join(OUTPUT_DIR, "labels", "val", prefix_fix + stem + ".txt")

        shutil.copy2(os.path.join(src_img_dir, fname), dst_img)
        src_lbl = os.path.join(src_lbl_dir, stem + ".txt")
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            open(dst_lbl, "w").close()
        success += 1
    return success


def main():
    print("=" * 50)
    print("构建混合验证集")
    print("=" * 50)

    create_dirs()

    # 全部绿牌 val
    green_files = [f for f in os.listdir(GREEN_VAL) if f.endswith(".jpg")]
    print(f"\n绿牌 val：{len(green_files)} 张（全部加入）")
    n1 = copy_files(GREEN_VAL, green_files, prefix="green_")

    # 随机抽取蓝牌 val
    blue_files = [f for f in os.listdir(BLUE_VAL) if f.endswith(".jpg")]
    sampled    = random.sample(blue_files, min(SAMPLE_N, len(blue_files)))
    print(f"蓝牌 val：从 {len(blue_files)} 张中随机抽取 {len(sampled)} 张")
    n2 = copy_files(BLUE_VAL, sampled, prefix="blue_")

    total = n1 + n2
    print(f"\n混合 val 共：{total} 张（绿牌 {n1} + 蓝牌 {n2}）")
    print(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
