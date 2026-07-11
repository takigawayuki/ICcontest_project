# YOLO 四类数据集转换大纲

## 目标

本阶段规划并转换最终 YOLO 检测数据集，输出目录使用：

```text
datasets/yolo/
```

为了向分区赛评分项靠拢，YOLO 类别从原来的三类扩展为四类：

```text
0 plate
1 person
2 car
3 traffic_light
```

说明：

- `plate` 只负责检测车牌位置，不区分蓝牌、绿牌、黄牌和特殊牌。
- 车牌颜色、位数、特殊类型和车牌号放到 LPR 阶段处理。
- `traffic_light` 只作为一个 YOLO 类别，不拆成红灯、黄灯、绿灯，避免小目标类别过细影响检测精度。
- 红绿灯颜色信息从 BDD100K 的 `trafficLightColor` 中额外保存到元数据，供后处理和违法规则使用。

## 推荐目录结构

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
│   └── bdd_traffic_light_color.csv
└── yolo.yaml
```

`yolo.yaml` 建议内容：

```yaml
path: C:/Users/Y9000P/Downloads/2026ICContest/ICcontest_project/datasets/yolo
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: plate
  1: person
  2: car
  3: traffic_light
```

## 数据来源总览

当前原始数据根目录：

```text
D:\Tempcode\26IC\车牌数据集\中国车牌
```

| 数据集 | 用途 | YOLO 类别 | 处理结论 |
| --- | --- | --- | --- |
| `CCPD2019` | 蓝牌/普通车牌检测 | `0 plate` | 从文件名 bbox 转 YOLO。 |
| `CCPD2020` | 新能源绿牌检测 | `0 plate` | 从文件名 bbox 转 YOLO。 |
| `CRPD_multi` | 多车牌检测 | `0 plate` | 四角点转外接矩形。 |
| `CRPD_double` | 双车牌/多车牌检测 | `0 plate` | 四角点转外接矩形。 |
| `CLPD` | 普通/少量新能源车牌检测 | `0 plate` | `CLPD.csv` 四角点转外接矩形。 |
| `BDD100K` | 行人、车辆、交通灯检测 | `1 person`、`2 car`、`3 traffic_light` | 完整 100k 图片和 100k JSON，可直接匹配转换。 |
| `BDD100K drivable_maps` | 道路/可行驶区域分割 | 不进 YOLO | 单独转换为 `datasets/drivable`，用于违法行为规则。 |
| `CBLPRD-330k_v1` | LPR 识别数据 | 不直接进 YOLO | 缺少真实场景 bbox，放到 LPR 阶段。 |

## 原始规模盘点

下面是原始数据当前已有 split 的盘点，用来确认来源规模；正式 `datasets/yolo` 会重新按来源分层 8:1:1 划分。

### plate 来源

| 来源 | 原始 split | 图片数 | plate 实例数 | 备注 |
| --- | --- | ---: | ---: | --- |
| `CCPD2019` | train | 100000 | 100000 | 来自 `splits/train.txt`。 |
| `CCPD2019` | val | 99996 | 99996 | 来自 `splits/val.txt`。 |
| `CCPD2019` | test | 141982 | 141982 | 来自 `splits/test.txt`。 |
| `CCPD2020` | train | 5769 | 5769 | `ccpd_green/train`。 |
| `CCPD2020` | val | 1001 | 1001 | `ccpd_green/val`。 |
| `CCPD2020` | test | 5006 | 5006 | `ccpd_green/test`。 |
| `CRPD_multi` | train | 1000 | 3196 | 多车牌。 |
| `CRPD_multi` | val | 250 | 804 | 多车牌。 |
| `CRPD_multi` | test | 335 | 1086 | 多车牌。 |
| `CRPD_double` | train | 4000 | 7966 | 双车牌/多车牌。 |
| `CRPD_double` | val | 1000 | 1998 | 双车牌/多车牌。 |
| `CRPD_double` | test | 1102 | 2201 | 双车牌/多车牌。 |
| `CLPD` | all | 1200 | 1200 | 后续按图片级 8:1:1 划分。 |

plate 总计：

| 图片数 | plate 实例数 |
| ---: | ---: |
| 362641 | 372805 |

### BDD100K 来源

BDD100K 已经更新为完整 100k：

```text
BDD100K/
├── bdd100k_images_100k/100k/train,val,test
└── bdd100k_labels/100k/train,val,test
```

原始 split 统计：

| split | 图片数 | JSON 数 | person 框 | car 框 | traffic_light 框 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 70000 | 70000 | 91405 | 713917 | 186301 |
| val | 10000 | 10000 | 13262 | 102506 | 26891 |
| test | 20000 | 20000 | 24641 | 205094 | 52840 |
| total | 100000 | 100000 | 129308 | 1021517 | 266032 |

BDD100K 交通灯颜色统计：

| split | green | red | yellow | none |
| --- | ---: | ---: | ---: | ---: |
| train | 79475 | 46178 | 3423 | 57225 |
| val | 11426 | 6606 | 510 | 8349 |
| test | 22276 | 13111 | 1010 | 16443 |
| total | 113177 | 65895 | 4943 | 82017 |

`none` 表示 BDD 标注中没有明确颜色，后续不能直接当作红灯或绿灯。

## 最终划分方法

推荐使用“按来源分层的 8:1:1 图片级划分”。

```text
不是把所有图片完全混在一起随机切。
而是每个数据集内部先按 8:1:1 划分，再把各来源的 train/val/test 分别合并。
```

这样做的原因：

1. 训练集占比更高，避免旧 CCPD2019 的 test/val 过大导致训练数据浪费。
2. 每个来源都保留在 train、val、test 中，验证集和测试集不会只偏向某个数据集。
3. 多车牌数据按图片级划分，同一张图里的多个车牌不会被拆到不同 split。
4. 使用固定随机种子，后续可以复现同一套划分。
5. 保留 `source_manifest.csv`，方便追溯每张图片来自哪个数据集。

推荐图片数量：

| 数据集 | train 图片 | val 图片 | test 图片 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| `CCPD2019` | 273582 | 34198 | 34198 | 341978 |
| `CCPD2020` | 9420 | 1178 | 1178 | 11776 |
| `CRPD_multi` | 1268 | 158 | 159 | 1585 |
| `CRPD_double` | 4882 | 610 | 610 | 6102 |
| `CLPD` | 960 | 120 | 120 | 1200 |
| `BDD100K` | 80000 | 10000 | 10000 | 100000 |
| 合计 | 370112 | 46264 | 46265 | 462641 |

按来源汇总：

| split | plate 来源图片 | BDD 图片 | 最终图片数 |
| --- | ---: | ---: | ---: |
| train | 290112 | 80000 | 370112 |
| val | 36264 | 10000 | 46264 |
| test | 36265 | 10000 | 46265 |
| total | 362641 | 100000 | 462641 |

最终标注框预计规模：

说明：重新按图片划分后，`CRPD_multi`、`CRPD_double`、`BDD100K` 每张图目标数量不同，下面是按总框数 8:1:1 估算的规模。转换完成后需要重新统计真实 label 数量。

| split | plate 框 | person 框 | car 框 | traffic_light 框 | 总框数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 297763 | 103446 | 817214 | 212826 | 1431249 |
| val | 37220 | 12931 | 102152 | 26603 | 178906 |
| test | 37222 | 12931 | 102151 | 26603 | 178907 |
| total | 372805 | 129308 | 1021517 | 266032 | 1789662 |

按类别汇总：

| class_id | 类别 | 框数量 |
| ---: | --- | ---: |
| 0 | plate | 372805 |
| 1 | person | 129308 |
| 2 | car | 1021517 |
| 3 | traffic_light | 266032 |
| 合计 | - | 1789662 |

## 各数据集转换规则

### CCPD2019 / CCPD2020

文件名中有 bbox：

```text
...-x1&y1_x2&y2-四角点-车牌编码-...
```

转成 YOLO：

```text
0 x_center y_center width height
```

注意：

- 类别固定为 `0 plate`。
- 不再沿用原始 split，按最终 8:1:1 划分清单写入 `datasets/yolo`。
- 可同时保留四角点信息供后续 LPR 透视矫正使用。

### CRPD_multi / CRPD_double

标签格式示例：

```text
x1 y1 x2 y2 x3 y3 x4 y4 type plate_text
```

一行代表一个车牌。转 YOLO 时使用四角点外接矩形：

```text
xmin = min(x1, x2, x3, x4)
ymin = min(y1, y2, y3, y4)
xmax = max(x1, x2, x3, x4)
ymax = max(y1, y2, y3, y4)
```

转成：

```text
0 x_center y_center width height
```

注意：

- 一张图片可能有多行标签。
- YOLO label 文件中保留多行。
- 标签中的 `type` 和 `plate_text` 不进入 YOLO，但后续生成 LPR 数据时要保留。

### CLPD

CSV 格式：

```text
path,x1,y1,x2,y2,x3,y3,x4,y4,label
```

转换方法和 CRPD 一样，用四角点外接矩形生成：

```text
0 x_center y_center width height
```

建议：

- 按图片级固定随机种子划分 `train/val/test = 8/1/1`。
- 后续 LPR 裁剪时保留 `label`。

### BDD100K

JSON 中提取：

```text
category == "person"        -> 1
category == "car"           -> 2
category == "traffic light" -> 3
```

使用 `box2d`：

```json
"box2d": {
  "x1": ...,
  "y1": ...,
  "x2": ...,
  "y2": ...
}
```

转成 YOLO：

```text
1 x_center y_center width height
2 x_center y_center width height
3 x_center y_center width height
```

交通灯颜色不要写进 YOLO label，而是额外保存：

```text
datasets/yolo/meta/bdd_traffic_light_color.csv
```

建议字段：

```text
split,image_name,object_id,x1,y1,x2,y2,color
```

其中 `color` 来自：

```text
attributes.trafficLightColor
```

## Drivable 分割数据集

为了避免后期每张图手动画 ROI，违法行为判断不只依赖 YOLO，还需要单独的道路区域分割数据集。

来源：

```text
D:\Tempcode\26IC\车牌数据集\中国车牌\BDD100K\bdd100k_drivable_maps
```

推荐输出目录：

```text
datasets/drivable/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── masks/
│   ├── train/
│   ├── val/
│   └── test/
└── drivable.yaml
```

BDD100K 的 drivable map 通常包含：

```text
0 background
1 direct drivable
2 alternative drivable
```

工程上第一版建议合并成二分类：

```text
0 non_drivable
1 drivable
```

原因：

1. 违法规则只需要知道人/车是否进入机动车可行驶区域。
2. 二分类分割模型更轻，部署延迟更低。
3. 后续如果需要区分主车道和可替代车道，再恢复三分类。

推荐划分方式：

```text
和 BDD100K 检测数据保持一致，按图片级 8:1:1 划分。
```

推荐数量：

| 数据 | train | val | test | 合计 |
| --- | ---: | ---: | ---: | ---: |
| `BDD100K drivable` | 80000 | 10000 | 10000 | 100000 |

注意：

- 如果 `bdd100k_drivable_maps` 中有 160000 个 PNG，转换前需要按图片名和 BDD100K 100k 图片匹配。
- 只使用能找到对应原图的 mask。
- `datasets/drivable` 不参与 YOLO 训练，它是单独的分割模型数据。

## 特殊车牌策略

特殊车牌得分主要靠 LPR，不靠 YOLO 细分类。

YOLO 阶段：

```text
所有车牌统一检测为 0 plate
```

LPR 阶段：

```text
根据颜色、位数、字符规则、模型输出判断特殊车牌类型
```

`CBLPRD-330k_v1` 有大量特殊牌，但缺少真实场景 bbox，所以不直接进入 YOLO。它应该重点进入 LPR，用来覆盖：

```text
黄牌、绿牌、警牌、学牌、挂车、港澳、使领馆、临牌等
```

如果后续测试发现特殊车牌“框不出来”，再补真实场景特殊车牌 bbox 给 YOLO。

如果是“框出来但识别错”，优先补 LPR 数据和规则。

## 违法行为支持方式

YOLO 只负责检测基础目标，不直接判断违法。违法行为由分割、跟踪和规则模块完成。

推荐工程链路：

```text
YOLO 检测 plate/person/car/traffic_light
Drivable 分割得到道路/可行驶区域
交通灯颜色判断得到 red/yellow/green/unknown
视频跟踪得到 person/car 轨迹
规则模块输出疑似违法行为
```

### 行人识别

需要：

```text
person 检测框
```

检测到 `person` 后即可标记行人。

### 行人横穿马路

需要：

```text
person 检测框 + drivable area mask + 连续帧轨迹
```

规则：

```text
person 框底部中心点落入 drivable 区域
并且连续帧轨迹横向穿过道路区域
-> 标记为疑似行人横穿马路 / 行人进入机动车道
```

### 行人闯红灯

需要：

```text
person 检测框 + traffic_light 检测框 + 红灯状态 + drivable area mask + 连续帧轨迹
```

规则：

```text
红灯状态下
person 框底部中心点进入 drivable 区域
并且产生横穿轨迹
-> 标记为疑似行人闯红灯
```

其中红灯状态可以来自：

1. BDD 的 `trafficLightColor` 训练/验证信息。
2. 部署时对 `traffic_light` 框做 HSV 颜色判断。
3. 后续单独训练一个轻量红绿灯颜色分类器。

注意：严格闯红灯还需要知道“哪个交通灯控制哪个方向”，这个关系很复杂。工程版先做“疑似行人闯红灯”更稳。

### 车辆逆行

需要：

```text
car 检测框 + drivable area mask + 视频连续帧跟踪 + 主车流方向估计
```

规则：

```text
只统计落在 drivable 区域内的车辆轨迹
如果某辆车轨迹方向与主车流方向明显相反
-> 标记为疑似车辆逆行
```

注意：

- 车辆逆行不能靠单张图片稳定判断。
- 车辆逆行强依赖视频连续帧、稳定跟踪和主车流方向估计。
- 如果比赛输入主要是图片，优先实现行人横穿马路或行人闯红灯。

## 旧 YOLO 数据集的定位

项目里已有：

```text
CCPD2019_YOLO/
CCPD2020_YOLO/
```

它们不是最终四类 YOLO 数据集，但可以保留作为历史备份和 bbox 解析对照。

后续训练入口推荐只使用：

```text
datasets/yolo/yolo.yaml
```

## 命名建议

为了避免不同数据集文件名冲突，转换时建议给图片加来源前缀：

```text
ccpd2019_xxx.jpg
ccpd2020_xxx.jpg
crpd_multi_xxx.jpg
crpd_double_xxx.jpg
clpd_xxx.jpg
bdd100k_xxx.jpg
```

对应 label 同名：

```text
ccpd2019_xxx.txt
```

## 多车牌处理原则

多车牌图片不要拆成多张 YOLO 图片。

正确方式：

```text
一张原图 -> 一个 label txt -> 多行 plate 框
```

示例：

```text
0 0.5123 0.4211 0.0820 0.0310
0 0.7132 0.3920 0.0750 0.0290
```

同一张原图中的所有目标必须属于同一个 split。

## YOLO 与比赛要求的关系

YOLO 数据集负责解决“检测和标记”部分，不直接完成全部比赛任务。

| 比赛要求 | YOLO 是否直接负责 | YOLO 相关类别 | 后续还需要 |
| --- | --- | --- | --- |
| 车牌标记 | 是 | `0 plate` | 多车牌场景要保证召回率。 |
| 车牌识别 | 否 | `0 plate` 提供裁剪区域 | LPR 识别车牌号。 |
| 一个或多个车牌标记 | 是 | `0 plate` | `CRPD_multi`、`CRPD_double` 提供多车牌训练样本。 |
| 不同角度和光照识别 | 部分负责 | `0 plate` | 透视矫正、光照增强、LPR 鲁棒训练。 |
| 低延迟 | 部分负责 | 4 类轻量检测 | ONNX/量化/FPGA/推理流水线优化。 |
| 特殊车牌识别 | 不直接负责 | `0 plate` 只负责框出特殊车牌 | `CBLPRD-330k_v1` 进入 LPR，做特殊牌识别。 |
| 行人识别 | 是 | `1 person` | 后处理显示行人标记。 |
| 行人横穿马路 | 部分负责 | `1 person` | Drivable 分割 + 跟踪 + 规则。 |
| 行人闯红灯 | 部分负责 | `1 person`、`3 traffic_light` | 交通灯颜色判断 + Drivable 分割 + 跟踪 + 规则。 |
| 车辆逆行 | 部分负责 | `2 car` | 视频跟踪 + Drivable 分割 + 主车流方向估计。 |

因此最终工程至少包含三套数据：

```text
datasets/yolo      -> 目标检测：plate/person/car/traffic_light
datasets/drivable  -> 道路/可行驶区域分割
datasets/lpr       -> 车牌字符和特殊牌识别
```

当前 YOLO 四类设计已经能支撑比赛要求中的检测基础，但不能替代 LPR、分割和违法规则模块。

## 视频数据的定位

YOLO 四类检测数据集仍然以图片训练为主，不需要为了 YOLO 专门找视频重新训练。

视频数据主要用于违法行为模块：

| 用途 | 是否需要视频 | 说明 |
| --- | --- | --- |
| 训练 YOLO `plate/person/car/traffic_light` | 不强制 | 当前图片数据已经覆盖检测训练。 |
| 训练 LPR | 不强制 | LPR 用车牌图和裁剪图训练。 |
| 训练 Drivable 分割 | 不强制 | BDD100K 图片和 mask 可训练。 |
| 测试行人横穿马路 | 建议需要 | 需要连续帧观察行人轨迹。 |
| 测试行人闯红灯 | 建议需要 | 需要红灯状态和行人进入道路区域的时间关系。 |
| 测试车辆逆行 | 强烈需要 | 需要车辆轨迹方向，单张图片无法稳定判断。 |

推荐收集的视频：

```text
路口固定摄像头视频
有红绿灯、行人、车辆的视频
有斑马线或明显道路区域的视频
车辆主行驶方向明显的视频
白天/夜晚/强光/阴影等不同光照视频
```

视频格式建议：

```text
mp4 / avi
720p 或 1080p
15-30 fps
10 秒到 2 分钟
固定视角优先
```

视频不进入 `datasets/yolo`，建议单独保存，例如：

```text
datasets/videos/
├── normal_traffic/
├── pedestrian_crossing/
├── red_light/
└── wrong_way/
```

后续可以从视频中抽帧做测试集，但抽帧图片不要直接混入训练集，除非重新标注并确认不会造成评测数据泄漏。

### 当前 WTS_DATASET_TEST 的定位

本地路径：

```text
D:\Tempcode\26IC\车牌数据集\中国车牌\WTS_DATASET_TEST
```

这批数据不作为 `datasets/yolo` 的主训练来源。它更适合放在视频测试/演示数据中，用来验证违法行为模块。

当前检查到的结构和结论：

| 类型 | 数量 | 典型属性 | 推荐用途 |
| --- | ---: | --- | --- |
| WTS 自有 `overhead_view` 事件视频 | 176 | 约 1080p、30fps、42-89 秒 | 首选，用于行人横穿、闯红灯、车辆逆行规则验证。 |
| WTS 自有 `normal_trimmed/overhead_view` 正常视频 | 35 | 约 1080p、30fps、50-89 秒 | 正常交通对照样例，检查误报。 |
| WTS 自有 `vehicle_view` 视频 | 78 | 1080p，部分较短 | 辅助展示，不作为行为规则主视角。 |
| `external/BDD_PC_5K` 视频 | 375 | 约 720p、30fps、20-53 秒 | 可测试通用检测和跟踪，但不如 WTS 俯视路口贴合违法规则。 |

推荐优先使用：

```text
WTS_DATASET_PUBLIC_TEST/videos/test/public/*/overhead_view/
WTS_DATASET_PUBLIC_TEST/videos/test/public/normal_trimmed/*/overhead_view/
```

这些视频是固定路口/俯视视角，时长和分辨率都满足前面的格式建议，适合做：

```text
YOLO 四类检测 -> drivable 分割 -> person/car 跟踪 -> 违法规则判断
```

注意：

1. 不要把 WTS public test 视频直接混入 YOLO 训练集。
2. 不要把从 WTS public test 抽出的帧直接混入训练集，除非重新标注并确认不会造成评测数据泄漏。
3. WTS 的 caption `event_phase` 和 BBOX 标注可用于调试/验证行为阶段、行人框和车辆框，但不能替代 BDD100K drivable mask。
4. 道路区域分割仍优先使用 `BDD100K drivable_maps` 训练。
5. 第一版违法行为建议先做“疑似行人横穿马路/进入机动车道”，再做“疑似行人闯红灯”，最后有余力再做“疑似车辆逆行”。

## 风险和待确认项

1. `CBLPRD-330k_v1` 不直接进 YOLO，因为没有真实场景 bbox。
2. `BDD100K` 的 `car` 和 `traffic_light` 框数量很多，如果影响 `plate`，后续可以对 BDD 做采样。
3. `traffic_light` 是小目标，训练时需要关注它的 mAP，但不能让它牺牲车牌检测主任务。
4. 所有特殊车牌在 YOLO 中统一标为 `0 plate`，不要在 YOLO 阶段细分。
5. 转换完成后必须抽样可视化检查：单车牌、多车牌、BDD 行人/车辆/交通灯都要看。
6. Drivable 分割不进 YOLO 类别，应该单独训练轻量分割模型。
7. 违法行为建议输出“疑似”标记，避免规则近似导致过度承诺。
