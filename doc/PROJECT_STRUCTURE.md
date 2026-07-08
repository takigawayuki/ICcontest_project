# 项目文件结构

这个工程是一个车牌识别项目，主流程分成两段：

1. YOLO 做车牌检测。
2. LPRNet 做车牌字符识别。

## 主要入口脚本

| 文件 | 用途 |
| --- | --- |
| `train.py` | 训练、验证、导出 YOLO 检测模型。 |
| `yolo_2_onnx.py` | 将训练好的 YOLO 模型导出为 ONNX，可配置 NMS。 |
| `detect_onnx_pipeline.py` | 完整 ONNX 推理流程：YOLO 检测 + LPRNet 识别。 |
| `preprocess_fpga.py` | 面向 FPGA 部署的图像预处理流程。 |
| `yolo_persp_crop.py` | YOLO 检测车牌后做透视裁剪。 |
| `re_yolo_persp_crop.py` | 修订版透视裁剪流程。 |

## 数据集转换与准备

| 文件 | 用途 |
| --- | --- |
| `ccpd2019_2_yolo.py` | 将 CCPD2019 转成 YOLO 格式。 |
| `ccpd2020_2_yolo.py` | 将 CCPD2020 转成 YOLO 格式。 |
| `only_ccpd2lpr.py` | 将 CCPD 风格数据转成 LPRNet 识别训练数据。 |
| `cblprd_green_import.py` | 导入或整理绿牌数据。 |
| `make_val.py` | 生成验证集划分。 |
| `check_duplicates.py` | 检查重复样本。 |
| `diagnose_test_skip.py` | 诊断测试集样本被跳过的原因。 |

## 调试与可视化

| 文件 | 用途 |
| --- | --- |
| `visualize_persp.py` | 可视化透视变换效果。 |
| `visualize_preprocess.py` | 可视化预处理结果。 |
| `test_hsv_cls.py` | 测试基于 HSV 的车牌颜色分类。 |
| `test_yolo_onnx.py` | 测试导出的 YOLO ONNX 模型。 |
| `cv2_chinese.py` | OpenCV 绘制中文文字的辅助函数。 |

## 配置文件

| 文件 | 用途 |
| --- | --- |
| `new_plate_merged.yaml` | 当前 `train.py` 使用的合并版 YOLO 数据集配置。 |
| `plate_merged.yaml` | 早期合并版 YOLO 数据集配置。 |
| `environment_LPR.yml` | Conda 环境配置。 |
| `requirements_LPR.txt` | Python 依赖列表。 |

## 重要目录

| 目录 | 用途 |
| --- | --- |
| `CCPD2019_YOLO/` | YOLO 格式的 CCPD2019 数据集。 |
| `CCPD2020_YOLO/` | YOLO 格式的 CCPD2020 数据集。 |
| `LPRNet_Pytorch/` | LPRNet 模型、训练脚本、数据加载器和识别权重。 |
| `LPR_DATA_PERSP2/` | 透视裁剪后的识别数据。 |
| `re_LPR_DATA_PERSP/` | 修订后的识别数据。 |
| `VAL_MIXED/` | 混合验证数据。 |
| `runs/` | YOLO 训练结果和导出的检测模型权重。 |
| `ultralytics-main/` | 本地 Ultralytics 源码。 |
| `doc/` | 项目笔记、参考资料、报告和流程说明。 |
| `debug/` | 临时调试输出和误生成的 shell 文件。 |

## 常用命令

```powershell
# 训练 YOLO 检测模型
python train.py

# 验证 YOLO 检测模型
python train.py val

# 导出 YOLO 检测模型为 ONNX
python train.py export

# 单独运行 YOLO ONNX 导出脚本
python yolo_2_onnx.py

# 测试导出的 YOLO ONNX 模型
python test_yolo_onnx.py

# 运行完整 ONNX 识别流程
python detect_onnx_pipeline.py
```

## 整理原则

- 根目录 Python 脚本暂时保留原位，因为多个脚本写了绝对路径或根目录相对路径。
- 大型生成结果应放在 `runs/`、`debug/`、数据集目录或权重目录，不建议堆在根目录。
- 新的说明、报告、参考资料放进 `doc/`。
- 新的临时图片、实验输出、排查文件放进 `debug/`。
