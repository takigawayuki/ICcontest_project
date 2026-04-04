# FPGA 模拟预处理管线说明

> 对应代码：`preprocess_fpga.py`

---

## 整体流程

```
CCPD 原图
    │
    ├─── CCPD 文件名解析 ──────────────────────► ground truth（车牌号）
    │
    ▼
YOLO 推理（new_plate_detect_merged/best.pt）
    │  置信度 < 阈值时，回退到文件名 bbox
    ▼
裁剪车牌 ROI
    │
    ▼
质量评估（亮度 / 面积）
    │
    ├── 面积 < 1500px ──► 双三次插值 2× 超分
    │
    ▼
颜色特征增强（HSV 蓝/绿掩膜）
    │
    ▼
几何矫正（Sobel → 闭运算 → 仿射变换）
    │
    ├── 亮度 > 190 ──► 局部自适应 Gamma 校正
    ├── 亮度 < 60  ──► CLAHE（clipLimit=3.0）
    └── 正常        ──► 轻度 CLAHE（clipLimit=1.5）
    │
    ▼
Resize → 94×24
    │
    ▼
保存（文件名 = 车牌号.jpg）→ LPR_DATA_FPGA/
```

---

## 各模块详解

### 模块一：YOLO 推理 + 裁剪

| 项目 | 说明 |
|------|------|
| 模型 | `runs/new_plate_detect_merged/weights/best.pt` |
| 置信度阈值 | `YOLO_CONF = 0.3` |
| 检测策略 | 取置信度最高的框 |
| Fallback | YOLO 未检出时，使用 CCPD 文件名中的 bbox 坐标 |
| 开关 | `USE_YOLO = False` 可跳过推理，仅跑预处理（调试用）|

---

### 模块二：图像质量评估

通过 HSV 的 V 通道均值判断亮度，通过 ROI 像素面积判断是否为远距离小目标。

| 指标 | 阈值 | 后续处理 |
|------|------|----------|
| 亮度 > 190 | 强光 | 局部自适应 Gamma |
| 亮度 < 60 | 夜间 | CLAHE（强力） |
| 其他 | 正常 | 轻度 CLAHE |
| 面积 < 1500px | 远距离 | 双三次插值 2× |

---

### 模块三：颜色特征增强

1. BGR → HSV
2. 提取蓝牌掩膜（H∈[100,130]，S>80，V>50）
3. 提取绿牌掩膜（H∈[40,80]，S>80，V>50）
4. 掩膜覆盖率 > 5% 时，对 V 通道做 [P5, P95] 百分位对比度拉伸
5. HSV → BGR

> 作用：增强车牌背景与字符之间的对比度。

---

### 模块四：几何矫正

```
灰度图
  │
  ▼ Sobel（ksize=3，仅垂直方向）
边缘图
  │
  ▼ 二值化（阈值50）
  │
  ▼ 形态学闭运算（核 5×3，连接边缘碎片）
  │
  ▼ 找最大轮廓 → minAreaRect → 估计倾斜角
  │
  ├── |angle| > 15° ──► 跳过（防止误矫正）
  └── |angle| ≤ 15° ──► 仿射旋转（INTER_CUBIC，边界复制填充）
```

---

### 模块五：光照自适应增强

#### 强光场景：局部自适应 Gamma

- 在 LAB 色彩空间的 L 通道上操作
- 按 8×8 像素块计算局部均值
- 均值越高，gamma 越大（最高 1.6），压暗过曝区域

| 局部均值 | gamma |
|----------|-------|
| > 210 | 1.6 |
| > 170 | 1.3 |
| 其他 | 1.0（不处理）|

#### 夜间场景：CLAHE

- 色彩空间：LAB（仅处理 L 通道）
- clipLimit = 3.0，tileGridSize = 4×4

#### 正常场景：轻度 CLAHE

- clipLimit = 1.5，tileGridSize = 4×4
- 提升整体对比度，不过度增强

---

### 模块六：远距离超分（双三次插值）

- 触发条件：ROI 面积 < 1500px（约 50×30 以下）
- 操作：双三次插值放大 2×（`INTER_CUBIC`）
- 时机：**在其他预处理之前**执行，放大后再做后续增强

---

### 模块七：输出

- `cv2.resize((94, 24), INTER_CUBIC)`
- 文件名 = 车牌号（如 `皖A12345.jpg`），LPRNet 直接从文件名读标签
- 输出目录结构：

```
LPR_DATA_FPGA/
├── train/   ← CCPD2019 train + CCPD2020 train
├── val/     ← CCPD2019 val   + CCPD2020 val
└── test/    ← CCPD2019 test  + CCPD2020 test
```

---

## 关键参数速查

| 参数 | 默认值 | 位置 | 说明 |
|------|--------|------|------|
| `USE_YOLO` | `True` | 顶部配置 | False = 仅用文件名bbox（快速调试）|
| `YOLO_CONF` | `0.3` | 顶部配置 | YOLO 置信度阈值 |
| 小目标阈值 | `1500 px` | `assess_quality()` | 触发超分的面积门限 |
| 强光阈值 | `190` | `assess_quality()` | V 通道均值 |
| 夜间阈值 | `60` | `assess_quality()` | V 通道均值 |
| 矫正最大角 | `±15°` | `correct_skew()` | 超出则跳过矫正 |
| CLAHE（夜间）| `clipLimit=3.0` | `clahe_enhance()` | |
| CLAHE（正常）| `clipLimit=1.5` | `normal_enhance()` | |
| 输出尺寸 | `94×24` | `preprocess()` | LPRNet 标准输入 |
