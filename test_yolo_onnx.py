import cv2
import numpy as np
import onnxruntime as ort

# ====== 1. 加载模型 ======
onnx_path = r"C:\\Users\\Y9000P\\Downloads\\2026ICContest\\ICcontest_project\\runs\\new_plate_detect_merged\\weights\\best.onnx"
session = ort.InferenceSession(onnx_path)

input_name = session.get_inputs()[0].name

# ====== 2. 读取图片 ======
img_path = r"D:\\Tempcode\\26IC\\车牌数据集\\中国车牌\\CCPD2019\\ccpd_tilt\\08-18_45-217&464_565&656-517&558_217&656_265&562_565&464-0_0_13_31_29_31_23-79-32.jpg"  # ❗换成你的测试图片

img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

if img is None:
    print("图片读取失败！检查路径！")
    exit()

orig_img = img.copy()

# ====== 3. 预处理（和YOLO一致）=====
img_resized = cv2.resize(img, (640, 640))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_norm = img_rgb / 255.0
img_trans = np.transpose(img_norm, (2, 0, 1))  # HWC → CHW
input_tensor = np.expand_dims(img_trans, axis=0).astype(np.float32)

# ====== 4. 推理 ======
outputs = session.run(None, {input_name: input_tensor})

preds = outputs[0][0]   # (300, 6)

# ====== 5. 画框 ======
h, w, _ = orig_img.shape

for det in preds:
    x1, y1, x2, y2, score, cls = det

    if score > 0.3:   # 置信度阈值
        # ⚠️ 坐标是640空间，需要映射回原图
        x1 = int(x1 / 640 * w)
        x2 = int(x2 / 640 * w)
        y1 = int(y1 / 640 * h)
        y2 = int(y2 / 640 * h)

        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig_img, f"{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# ====== 6. 显示 ======
cv2.imshow("result", orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows()