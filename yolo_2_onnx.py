from ultralytics import YOLO

model = YOLO(r'C:\\Users\\Y9000P\\Downloads\\2026ICContest\\ICcontest_project\\runs\\new_plate_detect_merged\\weights\\best.pt')

model.export(
    format='onnx',
    opset=12,
    simplify=True,
    dynamic=False,  # ✅ 关闭动态输入（输入尺寸固定为640x640）
    nms=False       # ✅ 关闭 NMS
)

# NMS（Non-Maximum Suppression，非极大值抑制）
# 👉 用来 “去掉重复框，只保留最靠谱的那个框”
# 默认关闭了 NMS，输出的 ONNX 模型中不包含 NMS 的计算图，这样在推理时就需要单独实现 NMS 了

"""

dynamic = False,  # ❗默认是关闭 
nms = False       # ❗默认也是关闭（重点！）

import cv2

# 画框要自己画
for det in preds:
    x1, y1, x2, y2, score, cls = det

    if score > 0.3:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

"""