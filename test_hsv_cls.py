"""
快速测试 HSV 分类效果，各取若干张蓝牌（CCPD2019 train）和绿牌（CCPD2020），
展示分类结果统计 + 可视化 unk 样本。
"""
import os, random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties

from yolo_persp_crop import (
    get_gt_corners, read_image, perspective_warp, hsv_classify,
    roi_mean_l, enhance_plate,
    collect_ccpd2019, collect_ccpd2020,
)
from cv2_chinese import find_font

# matplotlib 中文字体
_mpl_font = None
_font_path = find_font()
if _font_path:
    _mpl_font = FontProperties(fname=_font_path, size=8)

N_EACH  = 10000  # 每类取样数量
N_SHOW  = 20    # 可视化时最多展示的 unk 样本数
SEED    = 42
random.seed(SEED)


def test_batch(file_list, expected_color, label):
    cnt = {'blue': 0, 'green': 0, 'unk': 0}
    unk_samples = []   # 存 (warped_bgr, plate_text) 用于可视化

    for src_path, fname in file_list:
        corners = get_gt_corners(fname)
        if corners is None:
            continue
        img = read_image(src_path)
        if img is None:
            continue
        warped  = perspective_warp(img, corners)
        mean_l  = roi_mean_l(img, corners)
        enhanced = enhance_plate(warped, mean_l)
        color   = hsv_classify(enhanced)
        cnt[color] += 1
        if color == 'unk' and len(unk_samples) < N_SHOW:
            unk_samples.append(enhanced)

    total = sum(cnt.values())
    correct = cnt[expected_color]
    print(f'\n[{label}]  期望={expected_color}  共处理={total}')
    print(f'  blue={cnt["blue"]}  green={cnt["green"]}  unk={cnt["unk"]}')
    print(f'  正确率={correct/total*100:.1f}%  unk率={cnt["unk"]/total*100:.1f}%')
    return cnt, unk_samples


def show_unk(unk_blue, unk_green):
    samples = [('unk(应为blue)', img) for img in unk_blue] + \
              [('unk(应为green)', img) for img in unk_green]
    if not samples:
        print('\n没有 unk 样本，分类全部成功！')
        return

    n = len(samples)
    cols = min(n, 10)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 1.5, rows * 1.2))
    fp12 = FontProperties(fname=_font_path, size=12) if _font_path else None
    fig.suptitle('HSV 分类失败（unk）样本', fontsize=12, fontweight='bold',
                 fontproperties=fp12)
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.1)
    for idx, (title, bgr) in enumerate(samples):
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        ax.imshow(cv2.cvtColor(
            cv2.resize(bgr, (94*4, 24*4), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=6,
                     fontproperties=_mpl_font if _mpl_font else None)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('test_hsv_unk.png', dpi=120, bbox_inches='tight')
    print(f'\nunk 样本已保存到 test_hsv_unk.png（共 {n} 张）')
    plt.show()


def main():
    # 取样
    blue_all  = collect_ccpd2019('train')
    green_all = collect_ccpd2020('train')
    blue_sample  = random.sample(blue_all,  min(N_EACH, len(blue_all)))
    green_sample = random.sample(green_all, min(N_EACH, len(green_all)))

    cnt_b, unk_b = test_batch(blue_sample,  'blue',  'CCPD2019 蓝牌')
    cnt_g, unk_g = test_batch(green_sample, 'green', 'CCPD2020 绿牌')

    print('\n======= 汇总 =======')
    total = sum(cnt_b.values()) + sum(cnt_g.values())
    correct = cnt_b['blue'] + cnt_g['green']
    print(f'综合正确率: {correct}/{total} = {correct/total*100:.1f}%')

    show_unk(unk_b, unk_g)


if __name__ == '__main__':
    main()
