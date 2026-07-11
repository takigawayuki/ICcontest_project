# build_yolo_dataset.py 脚本说明

本文档记录 `tools/convert/build_yolo_dataset.py` 的用途、输入输出、转换规则和推荐使用方式，方便后续继续做分区赛数据集转换时快速回忆。

## 脚本定位

`build_yolo_dataset.py` 用来构建分区赛最终的四类 YOLO 检测数据集：

```text
0 plate
1 person
2 car
3 traffic_light
```

输出目录默认为：

```text
datasets/yolo/
```

它只负责目标检测数据集，不负责 LPR 车牌字符识别数据、不负责道路分割数据，也不把 WTS 视频直接混进 YOLO 训练集。

## 当前输入路径

车牌、BDD100K 等图片和标注数据根目录：

```text
D:\Tempcode\26IC\车牌数据集\中国车牌
```

WTS 视频数据路径：

```text
D:\Tempcode\26IC\车牌数据集\中国车牌\WTS_DATASET_TEST
```

注意：WTS 只用于生成视频清单，方便后续违法行为验证/演示，不作为 YOLO 训练数据。

## 数据来源和用途

| 数据集 | 进入 YOLO | 类别 | 用途 |
| --- | --- | --- | --- |
| `CCPD2019` | 是 | `0 plate` | 蓝牌/普通车牌检测。 |
| `CCPD2020` | 是 | `0 plate` | 新能源绿牌检测。 |
| `CRPD_multi` | 是 | `0 plate` | 多车牌场景检测。 |
| `CRPD_double` | 是 | `0 plate` | 双车牌/多车牌场景检测。 |
| `CLPD` | 是 | `0 plate` | 少量真实车牌补充。 |
| `BDD100K` | 是 | `1 person`, `2 car`, `3 traffic_light` | 行人、车辆、交通灯检测。 |
| `BDD100K drivable_maps` | 否 | - | 道路区域分割，单独做 `datasets/drivable`。 |
| `CBLPRD-330k_v1` | 否 | - | 特殊车牌/LPR 识别，不直接进 YOLO。 |
| `WTS_DATASET_TEST` | 否 | - | 视频违法行为验证和演示。 |

## 输出结构

正式转换后输出类似：

```text
datasets/yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── meta/
│   ├── source_manifest.csv
│   ├── bdd_traffic_light_color.csv
│   └── wts_video_manifest.csv
└── yolo.yaml
```

其中：

```text
source_manifest.csv
```

记录每张输出图片来自哪个原始数据集和原始路径。

```text
bdd_traffic_light_color.csv
```

额外保存 BDD100K 中交通灯的颜色信息。颜色不写进 YOLO 类别，YOLO 只检测 `traffic_light` 位置。

```text
wts_video_manifest.csv
```

可选生成，记录 WTS 视频路径和类型，供后续行为模块使用。

## 转换规则

### CCPD2019 / CCPD2020

从文件名解析 bbox：

```text
...-x1&y1_x2&y2-...
```

转换成 YOLO：

```text
0 x_center y_center width height
```

### CRPD_multi / CRPD_double

每行标签包含四角点：

```text
x1 y1 x2 y2 x3 y3 x4 y4 type plate_text
```

脚本取四角点外接矩形，统一转成：

```text
0 x_center y_center width height
```

同一张图有多个车牌时，label 文件保留多行。

### CLPD

从 `CLPD.csv` 读取：

```text
path,x1,y1,x2,y2,x3,y3,x4,y4,label
```

同样使用四角点外接矩形，类别固定为 `0 plate`。

### BDD100K

从 JSON 中提取：

```text
person        -> 1
car           -> 2
traffic light -> 3
```

交通灯颜色来自：

```text
attributes.trafficLightColor
```

但颜色只保存到 `meta/bdd_traffic_light_color.csv`，不作为 YOLO 类别。

## 划分方式

脚本按来源分别做图像级划分，再合并各来源的 train/val/test。

默认优先使用文档中规划的数量：

```text
CCPD2019    273582 / 34198 / 34198
CCPD2020      9420 /  1178 /  1178
CRPD_multi    1268 /   158 /   159
CRPD_double   4882 /   610 /   610
CLPD           960 /   120 /   120
BDD100K      80000 / 10000 / 10000
```

如果本地某个来源实际数量和规划不一致，脚本会退回到 8:1:1 图像级划分，并打印 warning。

## 常用命令

先做 dry-run，只统计不复制：

```powershell
python tools\convert\build_yolo_dataset.py --dry-run
```

只检查某几个来源：

```powershell
python tools\convert\build_yolo_dataset.py --sources clpd --dry-run
python tools\convert\build_yolo_dataset.py --sources bdd100k --dry-run --write-wts-manifest
python tools\convert\build_yolo_dataset.py --sources ccpd2020,crpd_multi,crpd_double --dry-run
python tools\convert\build_yolo_dataset.py --sources ccpd2019 --dry-run
```

正式转换：

```powershell
python tools\convert\build_yolo_dataset.py --write-wts-manifest
```

如果想用硬链接减少磁盘占用，可以用：

```powershell
python tools\convert\build_yolo_dataset.py --copy-mode hardlink --write-wts-manifest
```

注意：硬链接要求源数据和输出目录在同一磁盘分区时效果最好；失败时脚本会自动退回复制。

如果输出目录非空，脚本默认会拒绝写入。确认要继续时才使用：

```powershell
python tools\convert\build_yolo_dataset.py --allow-nonempty --write-wts-manifest
```

如果 YOLO 图片已经转换完成，只是因为 WTS 路径变化导致 `wts_video_manifest.csv` 没有生成，不需要重新转换 46 万张图片，可以只补写 WTS 视频清单：

```powershell
python tools\convert\build_yolo_dataset.py --only-wts-manifest
```

当前默认 WTS 路径是：

```text
D:\Tempcode\26IC\车牌数据集\中国车牌\WTS_DATASET_TEST
```

如果 WTS 又移动了位置，可以临时指定：

```powershell
python tools\convert\build_yolo_dataset.py --only-wts-manifest --wts-root "新的WTS路径"
```

## 已做过的 dry-run 结果

当前脚本已经验证过以下来源：

```text
CLPD:
  1200 图，1200 plate，正常。

BDD100K:
  100000 图。
  person/car/traffic_light 正常。
  WTS 视频清单可识别 664 个视频。

CCPD2020 + CRPD_multi + CRPD_double:
  基本正常。
  有 4 个样本被保护性跳过，通常是空标注、异常 bbox 或文件问题。

CCPD2019:
  341978 图，341978 plate，正常。
```

## WTS 为什么不进 YOLO 训练

WTS 的定位是视频行为验证/演示，不是检测模型主训练集。

本项目中：

```text
图片数据 -> 训练 YOLO / LPR / drivable 分割
视频数据 -> 验证跟踪、违法规则和端到端演示
```

WTS 推荐用于：

```text
YOLO 四类检测
-> drivable 分割
-> person/car 跟踪
-> 行人横穿、闯红灯、车辆逆行等疑似违法规则
```

不要把 WTS public test 视频或抽帧直接混入 YOLO 训练集，除非重新标注并确认不会造成评测数据泄漏。

## 后续步骤

1. 先正式运行 `build_yolo_dataset.py` 生成 `datasets/yolo`。
2. 抽样可视化检查 `plate/person/car/traffic_light` 四类框是否正确。
3. 用 `datasets/yolo/yolo.yaml` 训练四类 YOLO。
4. 单独整理/训练 `datasets/drivable`。
5. LPR 继续走车牌裁剪图和 CBLPRD 特殊牌路线。
6. 最后用 WTS overhead_view 视频验证违法行为模块。

## 转换后体积和训练规模建议

按当前规划，完整 YOLO 四类数据集大约包含：

```text
train: 370112 张
val:    46264 张
test:   46265 张
total: 462641 张
```

如果使用默认 `copy` 模式复制图片，`datasets/yolo` 预计会占用约：

```text
32-36 GiB
```

实际大小主要由图片决定，YOLO label 和 meta CSV 很小。当前原始图片大致体积如下：

| 来源 | 原始图片数 | 原始体积 |
| --- | ---: | ---: |
| CCPD2019 | 约 35.5 万 | 约 23.5 GiB |
| CCPD2020 | 约 1.18 万 | 约 0.86 GiB |
| CRPD_multi | 约 1585 张实际训练图 | 约 1 GiB 级 |
| CRPD_double | 约 6102 张实际训练图 | 约 4 GiB 级 |
| CLPD | 1200 | 约 0.08 GiB |
| BDD100K | 10 万 | 约 5.39 GiB |

如果磁盘空间紧张，优先使用：

```powershell
python tools\convert\build_yolo_dataset.py --copy-mode hardlink --write-wts-manifest
```

`hardlink` 模式通常不会重复占用三十多 GiB 图片空间，只会新增 label、meta 等小文件。硬链接要求源数据和输出目录最好在同一磁盘分区；失败时脚本会自动退回普通复制。

### 46 万张是否适合直接训练

从数据覆盖角度看，46 万张是合理的，因为它覆盖：

```text
plate: CCPD2019 + CCPD2020 + CRPD_multi + CRPD_double + CLPD
person/car/traffic_light: BDD100K
```

对应比赛要求：

```text
车牌标记
多车牌标记
行人识别
车辆检测
交通灯检测
```

但从第一版训练效率看，直接训练 46 万张不一定是最优。数据量偏大，训练耗时长，而且 BDD100K 的 `car` 框数量很多，可能会让训练重点偏向车辆检测；`traffic_light` 又是小目标，需要额外关注 mAP。

推荐分两步：

### 第一版快速训练集

先做一个约 20 万张左右的采样版，用来验证流程是否跑通：

```text
CCPD2019: 120k-160k
CCPD2020: 全部
CRPD_multi: 全部
CRPD_double: 全部
CLPD: 全部
BDD100K: 40k-60k
```

目标是快速确认：

```text
label 没错
类别没错
yolo.yaml 能被训练脚本读取
plate/person/car/traffic_light 都能正常收敛
```

### 第二版正式训练集

确认流程无误后，再训练完整 46 万张全量数据集。全量版本更适合最终模型，但不建议一开始就用它排错。

### 训练时重点观察

不要只看总体 mAP，至少要单独观察：

```text
plate mAP 和召回率
多车牌场景召回率
traffic_light mAP
person 检测效果
car 是否过度主导训练
```

如果发现 `car` 数量过大影响 `plate`，后续可以对 BDD100K 做采样，或者降低 BDD100K 在训练集中的比例。
