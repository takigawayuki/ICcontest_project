# Ultralytics 项目文件树说明

> 项目来源：https://github.com/ultralytics/ultralytics
> 说明时间：2026-03-08

---

## 顶层目录结构

```
ultralytics-main/
├── ultralytics/            ← 核心源码（主要关注此处）
├── docs/                   ← 官方文档（可忽略）
├── examples/               ← 使用示例脚本
├── tests/                  ← 单元测试（可忽略）
├── docker/                 ← Docker 部署配置（可忽略）
├── pyproject.toml          ← Python 包配置文件
├── README.md               ← 英文说明
└── README.zh-CN.md         ← 中文说明
```

---

## 核心源码 `ultralytics/` 详细结构

```
ultralytics/
├── cfg/                        ← 配置文件目录
│   ├── default.yaml                ← 全局默认训练参数（学习率、batch、epoch等）
│   ├── datasets/                   ← 各数据集 yaml 配置模板（coco.yaml 等）
│   ├── models/                     ← 模型结构定义文件（yolov8n.yaml 等）
│   └── trackers/                   ← 目标跟踪器配置
│
├── data/                       ← 数据处理模块
│   ├── dataset.py                  ← 数据集类，负责读取图片和标注文件
│   ├── augment.py                  ← 数据增强（Mosaic、随机翻转、HSV变换等）
│   ├── converter.py            ⭐  ← 数据格式转换工具（可参考写 CCPD 转换脚本）
│   ├── loaders.py                  ← 图片 / 视频 / 摄像头数据加载器
│   ├── build.py                    ← 构建 DataLoader
│   ├── base.py                     ← 数据集基类
│   ├── split.py                    ← 数据集划分工具（train/val/test）
│   ├── utils.py                    ← 数据处理工具函数
│   └── scripts/                    ← 数据下载脚本
│
├── models/                     ← 各模型实现
│   ├── yolo/                   ⭐  ← YOLO 系列（车牌检测主要用这里）
│   │   ├── detect/             ⭐  ← 目标检测任务（本项目使用）
│   │   │   ├── train.py            ← 检测模型训练入口
│   │   │   ├── predict.py          ← 检测模型推理入口
│   │   │   └── val.py              ← 检测模型验证入口
│   │   ├── pose/               ⭐  ← 关键点检测（可用于车牌四角点定位 + 透视矫正）
│   │   │   ├── train.py
│   │   │   ├── predict.py
│   │   │   └── val.py
│   │   ├── classify/               ← 图像分类任务
│   │   ├── segment/                ← 实例分割任务
│   │   ├── obb/                    ← 旋转框检测任务
│   │   ├── world/                  ← YOLO-World 开放词汇检测
│   │   ├── yoloe/                  ← YOLO-E 系列
│   │   └── model.py                ← YOLO 模型统一入口类
│   │
│   ├── fastsam/                ← FastSAM 快速分割模型
│   ├── nas/                    ← NAS 神经架构搜索模型
│   ├── rtdetr/                 ← RT-DETR 实时检测变换器
│   └── sam/                    ← SAM 分割一切模型
│
├── engine/                     ← 训练 / 推理引擎（框架核心）
│   ├── trainer.py                  ← 训练循环逻辑（损失计算、反向传播、保存权重）
│   ├── predictor.py                ← 推理流程逻辑（前处理、后处理）
│   ├── validator.py                ← 验证流程逻辑（计算 mAP 等指标）
│   ├── exporter.py             ⭐  ← 模型导出（ONNX / TensorRT / CoreML 等）
│   ├── model.py                    ← 模型基类（train/predict/export 的统一接口）
│   ├── results.py                  ← 推理结果封装类（BBox、掩码、关键点）
│   └── tuner.py                    ← 超参数自动调优
│
├── nn/                         ← 神经网络组件
│   ├── tasks.py                    ← 模型构建（从 yaml 文件搭建网络结构）
│   ├── autobackend.py              ← 多后端推理支持（PyTorch / ONNX / TFLite 等）
│   ├── text_model.py               ← 文本编码器（CLIP 等）
│   └── modules/                    ← 基础模块实现
│       ├── conv.py                     ← 卷积模块（Conv、DWConv 等）
│       ├── block.py                    ← 核心模块（C2f、SPPF、Bottleneck 等）
│       ├── head.py                     ← 检测头（Detect、Segment、Pose 等）
│       └── transformer.py              ← Transformer 模块
│
├── utils/                      ← 工具函数库
│   ├── loss.py                     ← 损失函数（BBox Loss、DFL Loss 等）
│   ├── metrics.py                  ← 评估指标（mAP、混淆矩阵等）
│   ├── plotting.py                 ← 结果可视化（画框、标签、训练曲线）
│   ├── ops.py                      ← 核心操作（NMS、坐标变换等）
│   ├── torch_utils.py              ← PyTorch 工具（模型参数量、FLOPs 计算等）
│   ├── checks.py                   ← 环境检查（依赖版本、CUDA 等）
│   ├── downloads.py                ← 模型权重自动下载
│   ├── augment.py                  ← 推理时增强（TTA）
│   └── callbacks/                  ← 训练回调（TensorBoard、WandB 等）
│
├── solutions/                  ← 高级应用方案（可忽略）
│   ├── object_counter.py           ← 目标计数
│   ├── speed_estimation.py         ← 速度估计
│   ├── parking_management.py       ← 停车管理
│   └── ...                         ← 其他应用场景
│
├── trackers/                   ← 目标跟踪（可忽略）
│   ├── byte_tracker.py             ← ByteTrack 算法
│   └── bot_sort.py                 ← BoT-SORT 算法
│
├── hub/                        ← Ultralytics Hub 云平台接口（可忽略）
│
└── assets/                     ← 内置测试图片
    ├── bus.jpg
    └── zidane.jpg
```

---

## 本项目（车牌检测）实际需要关注的文件

| 任务 | 文件路径 | 说明 |
|------|----------|------|
| 数据集配置 | `ultralytics/cfg/datasets/` | 在此新建 `plate.yaml` |
| 训练参数参考 | `ultralytics/cfg/default.yaml` | 查看可调整的参数名称 |
| 模型结构参考 | `ultralytics/cfg/models/v8/yolov8n.yaml` | YOLOv8n 网络结构 |
| 导出 ONNX | `ultralytics/engine/exporter.py` | 导出时若报错可查此文件 |
| 数据格式转换参考 | `ultralytics/data/converter.py` | 编写 CCPD 转换脚本时参考 |

---

## 实际使用方式（命令行，无需改源码）

```bash
# 安装
pip install ultralytics

# 训练
yolo train model=yolov8n.pt data=plate.yaml epochs=100 imgsz=640 batch=16

# 验证
yolo val model=runs/detect/train/weights/best.pt data=plate.yaml

# 推理（单张图片）
yolo predict model=best.pt source=test.jpg

# 导出 ONNX（用于后续转 RKNN）
yolo export model=best.pt format=onnx opset=12
```

---

## 训练输出目录结构

```
runs/
└── detect/
    └── train/
        ├── weights/
        │   ├── best.pt         ← 验证集最优权重（导出用这个）
        │   └── last.pt         ← 最后一轮权重
        ├── results.csv         ← 每轮训练指标记录
        ├── confusion_matrix.png← 混淆矩阵
        ├── F1_curve.png        ← F1 曲线
        ├── PR_curve.png        ← PR 曲线
        └── val_batch0_pred.jpg ← 验证集预测可视化
```
