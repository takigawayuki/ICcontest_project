"""
分别评估模型在绿牌和蓝牌上的效果
"""
from ultralytics import YOLO
import os
import shutil
import random


BASE    = r"c:\Users\Y9000P\Downloads\2026ICContest\ICcontest_project"
WEIGHT  = os.path.join(BASE, "runs", "plate_detect_merged", "weights", "best.pt")


def main():
    model = YOLO(WEIGHT)

    # 评估绿牌
    print("=" * 40)
    print("评估绿牌（CCPD2020）")
    print("=" * 40)
    m1 = model.val(data=os.path.join(BASE, "CCPD_YOLO", "plate.yaml"), split="val")
    print(f"mAP50: {m1.box.map50:.4f}  mAP50-95: {m1.box.map:.4f}")

    # 评估蓝牌（从 CCPD2019_YOLO val 中抽样 1000 张）
    print("\n" + "=" * 40)
    print("评估蓝牌（CCPD2019）")
    print("=" * 40)

    blue_val = os.path.join(BASE, "CCPD2019_YOLO", "images", "val")
    files    = random.sample([f for f in os.listdir(blue_val) if f.endswith(".jpg")], 1000)

    tmp_img = os.path.join(BASE, "tmp_blue_val", "images", "val")
    tmp_lbl = os.path.join(BASE, "tmp_blue_val", "labels", "val")
    os.makedirs(tmp_img, exist_ok=True)
    os.makedirs(tmp_lbl, exist_ok=True)

    for f in files:
        shutil.copy2(os.path.join(blue_val, f), os.path.join(tmp_img, f))
        lbl = os.path.join(BASE, "CCPD2019_YOLO", "labels", "val", f.replace(".jpg", ".txt"))
        if os.path.exists(lbl):
            shutil.copy2(lbl, os.path.join(tmp_lbl, f.replace(".jpg", ".txt")))

    tmp_yaml = os.path.join(BASE, "tmp_blue_val", "blue_val.yaml")
    with open(tmp_yaml, "w") as f:
        f.write(f"path: {os.path.join(BASE, 'tmp_blue_val')}\n")
        f.write("train: images/val\n")  # ultralytics 要求 train 字段存在
        f.write("val: images/val\n")
        f.write("nc: 1\nnames:\n  0: plate\n")

    m2 = model.val(data=tmp_yaml, split="val")
    print(f"mAP50: {m2.box.map50:.4f}  mAP50-95: {m2.box.map:.4f}")

    shutil.rmtree(os.path.join(BASE, "tmp_blue_val"))

    print("\n" + "=" * 40)
    print("对比总结")
    print("=" * 40)
    print(f"绿牌 mAP50: {m1.box.map50:.4f}")
    print(f"蓝牌 mAP50: {m2.box.map50:.4f}")


if __name__ == "__main__":
    main()
