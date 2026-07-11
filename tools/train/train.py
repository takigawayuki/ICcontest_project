"""
YOLOv8n 车牌检测训练脚本

"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# ==================== 路径配置 ====================
DATA_YAML = r"c:\Users\Y9000P\Downloads\2026ICContest\ICcontest_project\new_plate_merged.yaml"
PROJECT_DIR = r"c:\Users\Y9000P\Downloads\2026ICContest\ICcontest_project\runs"
# =================================================

def train():
    # 加载 YOLOv8n 预训练模型（首次运行会自动下载）
    model = YOLO("yolov8n.pt")

    model.train(
        data=DATA_YAML,
        epochs=100,         # 训练轮数
        imgsz=640,          # 输入图片尺寸
        batch=16,           # 每批次图片数（显存不足时改为 8）
        workers=2,          # 数据加载线程数（降低内存压力）
        device=0,           # GPU 编号，没有 GPU 则改为 'cpu'
        project=PROJECT_DIR,
        name="new_plate_detect_merged",
        exist_ok=True,      # 允许覆盖已有训练结果
        # 以下为优化参数，保持默认即可
        patience=20,        # 验证集无提升超过20轮则提前停止
        save_period=10,     # 每10轮保存一次权重
        val=True,           # 训练过程中进行验证
    )

    print("\n训练完成！")
    print(f"最优权重：{PROJECT_DIR}/new_plate_detect_merged/weights/best.pt")


def resume():
    """从上次中断处继续训练"""
    last_pt = os.path.join(PROJECT_DIR, "new_plate_detect_merged", "weights", "last.pt")
    if not os.path.exists(last_pt):
        print("未找到 last.pt，请先运行训练")
        return
    model = YOLO(last_pt)
    model.train(resume=True)


def validate():
    """训练完成后验证模型效果"""
    weight_path = os.path.join(PROJECT_DIR, "new_plate_detect_merged", "weights", "best.pt")
    if not os.path.exists(weight_path):
        print("未找到权重文件，请先运行训练")
        return

    model = YOLO(weight_path)
    metrics = model.val(data=DATA_YAML)

    print(f"\n验证结果：")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")


def export_onnx():
    """将训练好的模型导出为 ONNX 格式（用于后续转 RKNN）"""
    weight_path = os.path.join(PROJECT_DIR, "new_plate_detect_merged", "weights", "best.pt")
    if not os.path.exists(weight_path):
        print("未找到权重文件，请先运行训练")
        return

    model = YOLO(weight_path)
    model.export(
        format="onnx",
        opset=12,       # RKNN Toolkit2 推荐使用 opset 12
        simplify=True,  # 简化模型结构
        dynamic=False,  # 固定输入尺寸，部署必须关闭
        imgsz=640,
    )
    print(f"\nONNX 已导出：{weight_path.replace('.pt', '.onnx')}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # 默认执行训练
        train()
    elif sys.argv[1] == "resume":
        resume()
    elif sys.argv[1] == "val":
        validate()
    elif sys.argv[1] == "export":
        export_onnx()
    else:
        print("用法：")
        print("  python train.py          # 训练")
        print("  python train.py val      # 验证")
        print("  python train.py export   # 导出 ONNX")
