"""
预处理管线可视化脚本 v3
========================
每张车牌显示 3 行：
  行1 (结果图)  : 原图bbox | ROI | 锐化 | 二值掩膜 | Sobel膨胀边缘 | 矫正后 | 光照增强 | 94×24
  行2 (差分图)  : -        | Δ   | Δ   | -        | -             | Δ      | Δ        | Δ
  行3 (通道图)  : 原始ROI放大显示（对比锐化前后）

实际只画 行1 + 行2，共 2 行/样本，让界面不过于拥挤。

运行：
    conda activate LPR
    python visualize_preprocess.py
"""

import os, sys, random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess_fpga import (
    parse_ccpd_filename, read_image, assess_quality,
    super_resolve, unsharp_mask,
    detect_plate_type, extract_binary_mask, morph_edge_pipeline,
    correct_perspective, correct_skew_hough,
    adaptive_gamma, clahe_enhance, normal_enhance,
    YOLO_MODEL, CCPD2019_DIR, CCPD2020_DIR,
)
from ultralytics import YOLO

# ===== 配置 =====
N_SAMPLES   = 5
YOLO_CONF   = 0.3
RANDOM_SEED = 42
FINAL_SCALE = 6     # 94×24 放大倍数（显示用）
DIFF_AMP    = 4     # 差分放大倍数
IMG_H       = 80    # 统一显示高度（像素）
# ================

STEP_NAMES = [
    "① 原图\nbbox框选",
    "② 裁剪ROI\n(+超分)",
    "③ Unsharp\n锐化",
    "④ 颜色通道\n二值化",
    "⑤ 腐蚀→Sobel\n→膨胀边缘",
    "⑥ 几何矫正\n(透视/仿射)",
    "⑦ 光照增强",
    "⑧ 最终\n94×24",
]


def bgr2rgb(img):
    if len(img.shape) == 2:          # 灰度图
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fit_h(img, h):
    oh, ow = img.shape[:2]
    if oh == 0: return img
    w = max(1, int(ow * h / oh))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)


def make_diff(a, b):
    """差分图：灰度差×DIFF_AMP，jet 色彩，无变化=黑色"""
    # 统一到同一尺寸再比较
    h = min(a.shape[0], b.shape[0], IMG_H)
    def _resize(x):
        if len(x.shape) == 2:
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        return cv2.resize(x, (int(x.shape[1]*h/x.shape[0]), h),
                          interpolation=cv2.INTER_NEAREST)
    ar, br = _resize(a), _resize(b)
    mw = min(ar.shape[1], br.shape[1])
    ar, br = ar[:, :mw], br[:, :mw]
    diff = np.mean(np.abs(ar.astype(np.int16) - br.astype(np.int16)), axis=2)
    diff_amp = np.clip(diff * DIFF_AMP, 0, 255).astype(np.uint8)
    jet = cv2.applyColorMap(diff_amp, cv2.COLORMAP_JET)
    jet[diff_amp == 0] = 0
    mean_d = round(float(np.mean(diff)), 1)
    return jet, mean_d


def imshow(ax, img, title=None, label=None, h=IMG_H):
    disp = bgr2rgb(fit_h(img, h))
    ax.imshow(disp, interpolation='nearest', aspect='auto')
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=7.5, fontweight='bold', pad=2)
    if label: ax.set_xlabel(label, fontsize=6.5, color='#444', labelpad=1)


def crop_bbox_region(img, bbox, pad=25):
    x1, y1, x2, y2 = bbox
    H, W = img.shape[:2]
    x1p, y1p = max(0, x1-pad), max(0, y1-pad)
    x2p, y2p = min(W, x2+pad), min(H, y2+pad)
    region = img[y1p:y2p, x1p:x2p].copy()
    rx1, ry1 = x1-x1p, y1-y1p
    rx2, ry2 = x2-x1p, y2-y1p
    cv2.rectangle(region, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
    return region


def process_one(model, src_path, filename):
    """
    完整走一遍管线，收集所有中间步骤。
    返回 (steps, labels, plate_text, ptype) 或 None
    steps 共 8 项（BGR 或灰度 ndarray）
    """
    plate_text, fallback_bbox, four_pts = parse_ccpd_filename(filename)
    if plate_text is None: return None

    img = read_image(src_path)
    if img is None: return None

    # YOLO 推理
    best_box, best_conf = None, 0.0
    for r in model(img, verbose=False, conf=YOLO_CONF):
        if r.boxes is None: continue
        for box in r.boxes:
            c = float(box.conf[0])
            if c > best_conf:
                best_conf = c
                best_box = box.xyxy[0].cpu().numpy().astype(int)

    if best_box is not None:
        x1, y1, x2, y2 = best_box
        det_label = "YOLO conf={:.2f}".format(best_conf)
    elif fallback_bbox is not None:
        x1, y1, x2, y2 = fallback_bbox
        best_box = fallback_bbox
        det_label = "fallback bbox"
    else:
        return None

    # ① 原图 + bbox（裁出局部区域展示）
    s0 = crop_bbox_region(img, (x1, y1, x2, y2), pad=30)

    # 裁剪 ROI
    H, W = img.shape[:2]
    roi = img[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
    if roi.size == 0: return None

    q = assess_quality(roi)
    work = roi.copy()

    # ② 裁剪/超分
    if q['is_small']:
        work = super_resolve(work)
        roi_label = "小目标→2×超分 亮度={:.0f}".format(q['brightness'])
    else:
        roi_label = "亮度={:.0f} {}".format(
            q['brightness'],
            "[强光]" if q['is_bright'] else ("[夜间]" if q['is_dark'] else "[正常]"))
    s1 = work.copy()

    # ③ 锐化
    work = unsharp_mask(work)
    s2 = work.copy()

    # ④ 颜色判断 + 二值化
    ptype  = detect_plate_type(work)
    binary = extract_binary_mask(work, ptype)
    s3 = binary   # 灰度

    # ⑤ 腐蚀→Sobel→膨胀
    edge_map = morph_edge_pipeline(binary)
    s4 = edge_map   # 灰度

    # ⑥ 几何矫正：有 4 角点用透视，否则 Hough 仿射
    if four_pts is not None:
        # 转为 ROI 坐标系
        pts_roi = four_pts - np.array([x1, y1], dtype=np.float32)
        work = correct_perspective(work, pts_roi)
        corr_label = "4角点透视变换"
    else:
        work = correct_skew_hough(work, edge_map)
        corr_label = "Hough仿射矫正"
    s5 = work.copy()

    # ⑦ 光照增强
    if q['is_bright']:
        work = adaptive_gamma(work)
        light_label = "局部Gamma"
    elif q['is_dark']:
        work = clahe_enhance(work)
        light_label = "CLAHE 3.0"
    else:
        work = normal_enhance(work)
        light_label = "CLAHE 1.5"
    s6 = work.copy()

    # ⑧ 最终
    s7 = cv2.resize(work, (94, 24), interpolation=cv2.INTER_CUBIC)

    steps  = [s0, s1, s2, s3, s4, s5, s6, s7]
    labels = [det_label, roi_label, "sigma=1.2 strength=1.5",
              "{} 型→{}通道".format(ptype,
                  "Cb+INV(YCbCr)" if ptype=="blue" else "V+INV(HSV)"),
              "腐蚀→Sobel→膨胀", corr_label,
              light_label, "→LPRNet"]
    return steps, labels, plate_text, ptype


def draw_dataset(results, title):
    n     = len(results)
    n_col = len(STEP_NAMES)
    # 每个样本占 2 行：结果图 + 差分图
    fig_w = n_col * 2.5
    fig_h = n * 2.6
    fig   = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.005)

    gs = gridspec.GridSpec(n * 2, n_col, figure=fig, hspace=0.08, wspace=0.07)

    for i, (steps, labels, plate_text, ptype) in enumerate(results):
        row_img  = i * 2
        row_diff = i * 2 + 1

        for col in range(n_col):
            # --- 结果图 ---
            ax = fig.add_subplot(gs[row_img, col])
            if col == n_col - 1:   # 最终 94×24
                disp_img = cv2.resize(steps[col],
                    (94 * FINAL_SCALE, 24 * FINAL_SCALE),
                    interpolation=cv2.INTER_NEAREST)
                ax.imshow(bgr2rgb(disp_img), interpolation='nearest', aspect='auto')
            else:
                disp_img = fit_h(steps[col], IMG_H)
                ax.imshow(bgr2rgb(disp_img), interpolation='nearest', aspect='auto')
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(STEP_NAMES[col], fontsize=7.5, fontweight='bold', pad=2)
            if col == 0:
                color = '#aa0000' if ptype == 'blue' else '#006600'
                ax.set_ylabel(
                    "{}\n[{}]".format(plate_text, ptype),
                    fontsize=8.5, rotation=0, labelpad=60,
                    ha='right', va='center',
                    color=color, fontweight='bold')

            # --- 差分图 ---
            ax2 = fig.add_subplot(gs[row_diff, col])
            if col in (0, 3, 4):   # 这几列没有可对比的前步（或是掩膜图）
                ax2.set_facecolor('#0a0a0a')
                ax2.text(0.5, 0.5, labels[col],
                         ha='center', va='center',
                         fontsize=5.5, color='#aaaaaa',
                         transform=ax2.transAxes, wrap=True)
            else:
                diff, mean_d = make_diff(steps[col-1], steps[col])
                ax2.imshow(bgr2rgb(diff), interpolation='nearest', aspect='auto')
                ax2.set_xlabel(
                    "Δ={} {}".format(mean_d, labels[col]),
                    fontsize=6, color='#555', labelpad=1)
            ax2.set_xticks([]); ax2.set_yticks([])

    plt.tight_layout(rect=[0.07, 0, 1, 1])
    plt.show()


def sample_ccpd2019(n):
    txt = os.path.join(CCPD2019_DIR, "splits", "train.txt")
    with open(txt, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    chosen = random.sample(lines, min(n * 5, len(lines)))
    return [(os.path.join(CCPD2019_DIR, p.replace('/', os.sep)), os.path.basename(p))
            for p in chosen]


def sample_ccpd2020(n):
    src_dir = os.path.join(CCPD2020_DIR, "train")
    files   = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
    chosen  = random.sample(files, min(n * 5, len(files)))
    return [(os.path.join(src_dir, f), f) for f in chosen]


def collect(model, file_list, n):
    results = []
    for src_path, filename in file_list:
        if len(results) >= n: break
        ret = process_one(model, src_path, filename)
        if ret:
            results.append(ret)
            print("  [{}/{}] {} [{}]".format(len(results), n, ret[2], ret[3]))
    return results


def main():
    random.seed(RANDOM_SEED)
    print("加载 YOLO 模型...")
    model = YOLO(YOLO_MODEL)

    print("\n--- CCPD2019（蓝牌）{} 张 ---".format(N_SAMPLES))
    r2019 = collect(model, sample_ccpd2019(N_SAMPLES), N_SAMPLES)
    draw_dataset(r2019, "CCPD2019（蓝牌）预处理管线  |  上行=结果  下行=差分×{}".format(DIFF_AMP))

    print("\n--- CCPD2020（绿牌）{} 张 ---".format(N_SAMPLES))
    r2020 = collect(model, sample_ccpd2020(N_SAMPLES), N_SAMPLES)
    draw_dataset(r2020, "CCPD2020（绿牌）预处理管线  |  上行=结果  下行=差分×{}".format(DIFF_AMP))

    print("完成。")


if __name__ == "__main__":
    main()
