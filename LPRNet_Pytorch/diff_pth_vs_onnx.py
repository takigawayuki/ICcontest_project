import numpy as np

def main():
    import torch
    from model.LPRNet import LPRNet

    # ===== 1. PyTorch模型（必须和ONNX一致）=====
    # lprnet_mod = LPRNet(
    #     lpr_max_len=7, # 蓝牌
    #     phase=False,
    #     class_num=66,
    #     dropout_rate=0
    # )

    lprnet_mod = LPRNet(
        lpr_max_len=8, # 绿牌
        phase=False,
        class_num=66,
        dropout_rate=0
    )

    lprnet_mod.load_state_dict(
        torch.load('weights/green_re_run1/Final_LPRNet_model.pth', map_location="cpu")
    )
    lprnet_mod.eval()

    # ===== 2. ONNX模型 =====
    import onnxruntime as ort
    lprnet_onnx = ort.InferenceSession(
        "weights/green_re_run1/green_LPRNet_Simplified.onnx"
    )

    input_name = lprnet_onnx.get_inputs()[0].name
    output_name = lprnet_onnx.get_outputs()[0].name

    # ===== 3. 输入 =====
    dummy_torch = torch.randn(1, 3, 24, 94)

    # ===== 4. PyTorch推理 =====
    with torch.no_grad():
        torch_res = lprnet_mod(dummy_torch).numpy()

    # ===== 5. ONNX推理 =====
    dummy_np = dummy_torch.numpy()
    onnx_res = lprnet_onnx.run([output_name], {input_name: dummy_np})[0]

    # ===== 6. 对比 =====
    try:
        np.testing.assert_almost_equal(torch_res, onnx_res, decimal=4)
    except AssertionError:
        print("❌ 不一致")
        print("最大误差:", np.max(np.abs(torch_res - onnx_res)))
    else:
        print("✅ 完全一致（ONNX正确）")

if __name__ == "__main__":
    main()