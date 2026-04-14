def convert2onnx():
    import torch
    import onnx
    from onnxsim import simplify
    from models.LPRNet import CHARS, LPRNet

    # ===== 1. 构建模型 =====
    lprnet = LPRNet(lpr_max_len=7, phase=False, class_num=len(CHARS), dropout_rate=0) # 蓝牌是7位
    # lprnet = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0) # 绿牌是8位


    # ===== 2. 加载权重 =====
    state_dict = torch.load('weights/lprnet_best.pth', map_location="cpu")
    lprnet.load_state_dict(state_dict)
    lprnet.eval()

    # ===== 3. dummy输入 =====
    dummy_input = torch.randn(1, 3, 24, 94)

    # ===== 4. 导出ONNX =====
    torch.onnx.export(
        lprnet,
        dummy_input,
        "weights/LPRNET.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=12
    )

    print("✅ ONNX导出成功: weights/LPRNET.onnx")

    # ===== 5. 简化模型 =====
    model_onnx = onnx.load("weights/LPRNET.onnx")
    model_simp, check = simplify(model_onnx)

    if check:
        onnx.save(model_simp, "weights/LPRNet_Simplified.onnx")
        print("✅ ONNX简化成功: weights/LPRNet_Simplified.onnx")
    else:
        print("❌ simplify失败")


if __name__ == '__main__':
    convert2onnx()