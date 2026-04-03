"""
CCPD → LPRNet 训练格式转换脚本
将 CCPD2019（蓝牌）和 CCPD2020 ccpd_green（绿牌）裁剪为 94×24 车牌小图，
文件名即车牌号（如 皖A12345.jpg），供 LPRNet 训练使用。

输出目录：
  LPR_DATA/
  ├── train/   ← CCPD2019 train(100,000) + CCPD2020 train(5,769)  ≈ 105,769 张
  ├── val/     ← CCPD2019 val(99,996)   + CCPD2020 val(1,001)    ≈ 100,997 张
  └── test/    ← CCPD2019 test(141,982) + CCPD2020 test(5,006)   ≈ 146,988 张

  注：CCPD2019 中的 ccpd_np（无车牌负样本）会被自动跳过，实际数量略少于上述估算。
"""

import os
import cv2
import numpy as np

# ==================== 路径配置 ====================
# 数据集路径
CCPD2019_DIR = r"D:\Tempcode\26IC\车牌数据集\中国车牌\CCPD2019"
CCPD2020_DIR = r"D:\Tempcode\26IC\车牌数据集\中国车牌\CCPD2020\ccpd_green"

# 输出目录
BASE = r"c:\Users\Y9000P\Downloads\2026ICContest\ICcontest_project"
OUTPUT_DIR = os.path.join(BASE, "LPR_DATA")
# ==================================================

# ---------- CCPD 字符映射表 ----------
# 省份（index 0-33）
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新", "警", "学", "O"
]
# 第二位：字母（不含 I）
ALPHABETS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'O'
]
# 第三位起：字母+数字
ADS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O'
]


def decode_plate(plate_str: str) -> str:
    """
    将 CCPD 文件名中的车牌索引字段解码为车牌字符串。
    plate_str 示例: '0_0_3_24_28_24_31_33'（7位或8位）
    """
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


def crop_and_save(img_path: str, filename: str, dst_path: str) -> bool:
    """
    从 CCPD 图片中裁剪车牌区域，resize 到 94×24，保存到 dst_path。
    返回 True 表示成功，False 表示跳过（无牌/解析失败）。
    """
    try:
        parts = os.path.splitext(filename)[0].split('-')
        if len(parts) < 6:
            return False  # ccpd_np 等无效格式

        # 解码车牌号
        plate_text = decode_plate(parts[4])

        # 解析裁剪坐标（第三字段）
        box_parts = parts[2].split('_')
        x1, y1 = map(int, box_parts[0].split('&'))
        x2, y2 = map(int, box_parts[1].split('&'))

        # 读取原图（支持中文路径）
        buf = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return False

        # 裁剪 + resize
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        crop = cv2.resize(crop, (94, 24), interpolation=cv2.INTER_CUBIC)

        # 保存（用 imencode+tofile 支持中文文件名）
        cv2.imencode('.jpg', crop)[1].tofile(dst_path.format(plate=plate_text))
        return True

    except Exception:
        return False


def convert_ccpd2019(split: str, out_dir: str):
    """
    用 splits/{split}.txt 中的路径列表转换 CCPD2019。
    split: 'train' / 'val' / 'test'
    """
    txt_path = os.path.join(CCPD2019_DIR, "splits", f"{split}.txt")
    with open(txt_path, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    ok = skip = 0
    for i, rel_path in enumerate(lines):
        fname    = os.path.basename(rel_path)
        src_path = os.path.join(CCPD2019_DIR, rel_path.replace('/', os.sep))
        dst_path = os.path.join(out_dir, "{plate}.jpg")

        if crop_and_save(src_path, fname, dst_path):
            ok += 1
        else:
            skip += 1

        if (i + 1) % 10000 == 0:
            print(f"  CCPD2019/{split}: {i+1}/{len(lines)} 处理中...")

    print(f"  CCPD2019/{split} 完成: 成功 {ok}，跳过 {skip}")
    return ok


def convert_ccpd2020(split: str, out_dir: str):
    """
    转换 CCPD2020/ccpd_green/{split}/ 目录下的绿牌。
    """
    src_dir = os.path.join(CCPD2020_DIR, split)
    files   = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]

    ok = skip = 0
    for i, fname in enumerate(files):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(out_dir, "{plate}.jpg")

        if crop_and_save(src_path, fname, dst_path):
            ok += 1
        else:
            skip += 1

        if (i + 1) % 2000 == 0:
            print(f"  CCPD2020/{split}: {i+1}/{len(files)} 处理中...")

    print(f"  CCPD2020/{split} 完成: 成功 {ok}，跳过 {skip}")
    return ok


def main():
    for split in ['train', 'val', 'test']:
        out_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n========== {split} ==========")
        n2019 = convert_ccpd2019(split, out_dir)
        n2020 = convert_ccpd2020(split, out_dir)
        print(f"  {split} 合计: {n2019 + n2020} 张")

    print("\n全部完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("LPRNet 训练时将路径指向:")
    print(f"  train: {os.path.join(OUTPUT_DIR, 'train')}")
    print(f"  val:   {os.path.join(OUTPUT_DIR, 'val')}")
    print(f"  test:  {os.path.join(OUTPUT_DIR, 'test')}")


if __name__ == "__main__":
    main()
