def convert2onnx():
    import torch
    import onnx
    from onnxsim import simplify
    from model.LPRNet import LPRNet

    onnx_path = "weights/blue_re_run1/blue_LPRNET.onnx"

    # ===== 1. 构建模型 =====
    lprnet = LPRNet(
        lpr_max_len=7,
        phase=False,
        class_num=66, # 我现在的蓝牌模型就是66
        dropout_rate=0
    )

    # lprnet = LPRNet(
    #     lpr_max_len=8,
    #     phase=False,
    #     class_num=66, # 绿牌也是66
    #     dropout_rate=0
    # )

    # ===== 2. 加载权重 =====
    state_dict = torch.load(
        'weights/blue_re_run1/Final_LPRNet_model.pth',
        map_location="cpu",
        weights_only=True
    )
    lprnet.load_state_dict(state_dict)
    lprnet.eval()

    # ===== 3. dummy输入 =====
    dummy_input = torch.randn(1, 3, 24, 94)

    # ===== 4. 导出ONNX =====
    torch.onnx.export(
        lprnet,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"}
        }
    )

    print("✅ ONNX导出成功")

    # ===== 5. 简化模型 =====
    model_onnx = onnx.load(onnx_path)
    model_simp, check = simplify(
        model_onnx,
        input_shapes={"input": [1, 3, 24, 94]}
    )

    if check:
        onnx.save(model_simp, "weights/blue_re_run1/blue_LPRNet_Simplified.onnx")
        print("✅ ONNX简化成功")
    else:
        print("❌ simplify失败")

if __name__ == '__main__':
    convert2onnx()