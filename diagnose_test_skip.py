"""
诊断 CCPD2019 test.txt 转换丢失原因
运行后输出每个子集的：成功/文件不存在/parse异常/bbox越界 的精确数量
"""

import os
from PIL import Image

SOURCE_DIR = r"D:\Tempcode\26IC\车牌数据集\中国车牌\CCPD2019"
TXT_PATH   = os.path.join(SOURCE_DIR, "splits", "test.txt")


def parse_bbox(filename, img_w, img_h):
    parts = filename.split("-")
    bbox_str = parts[2].split("_")
    x1, y1 = map(int, bbox_str[0].split("&"))
    x2, y2 = map(int, bbox_str[1].split("&"))
    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return x_center, y_center, w, h


def main():
    with open(TXT_PATH) as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"test.txt 共 {len(lines)} 条\n")

    # 按子集统计
    stats = {}  # subset -> {total, ok, no_file, parse_err, bbox_invalid}

    for rel_path in lines:
        subset = rel_path.split("/")[0]
        fname  = os.path.basename(rel_path)

        if subset not in stats:
            stats[subset] = dict(total=0, ok=0, no_file=0, parse_err=0, bbox_invalid=0)
        stats[subset]["total"] += 1

        img_path = os.path.join(SOURCE_DIR, subset, fname)

        if not os.path.exists(img_path):
            stats[subset]["no_file"] += 1
            continue

        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
            x_c, y_c, w, h = parse_bbox(fname, img_w, img_h)
        except Exception as e:
            stats[subset]["parse_err"] += 1
            if stats[subset]["parse_err"] <= 3:
                print(f"  [parse_err] {subset}/{fname}: {e}")
            continue

        if not (0 < x_c < 1 and 0 < y_c < 1 and 0 < w <= 1 and 0 < h <= 1):
            stats[subset]["bbox_invalid"] += 1
            continue

        stats[subset]["ok"] += 1

    # 汇总输出
    print(f"{'子集':<20} {'total':>7} {'ok':>7} {'no_file':>8} {'parse_err':>10} {'bbox_invalid':>13}")
    print("-" * 70)
    grand = dict(total=0, ok=0, no_file=0, parse_err=0, bbox_invalid=0)
    for subset, s in sorted(stats.items()):
        print(f"{subset:<20} {s['total']:>7} {s['ok']:>7} {s['no_file']:>8} {s['parse_err']:>10} {s['bbox_invalid']:>13}")
        for k in grand:
            grand[k] += s[k]
    print("-" * 70)
    print(f"{'合计':<20} {grand['total']:>7} {grand['ok']:>7} {grand['no_file']:>8} {grand['parse_err']:>10} {grand['bbox_invalid']:>13}")


if __name__ == "__main__":
    main()
