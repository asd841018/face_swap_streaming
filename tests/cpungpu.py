import onnxruntime as ort, numpy as np

def build_sess(path, ep):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(path, so, providers=[ep, "CPUExecutionProvider"])

def compare(model_path, input_dict, atol=1e-4, rtol=1e-3):
    cpu = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    gpu = build_sess(model_path, ("CUDAExecutionProvider", {
        "cudnn_conv_algo_search": "HEURISTIC",
        "cudnn_conv_use_max_workspace": "0",
        "do_copy_in_default_stream": "1",
        "use_tf32": 0
    }))
    y_cpu = cpu.run(None, input_dict)
    y_gpu = gpu.run(None, input_dict)
    names = [o.name for o in cpu.get_outputs()]
    ok = True
    for n, a, b in zip(names, y_cpu, y_gpu):
        mad = np.max(np.abs(a-b))
        rel = mad / (np.max(np.abs(a))+1e-12)
        print(f"{n:30s} max_abs_diff={mad:.3e} rel={rel:.3e} NaN(cpu/gpu)={(np.isnan(a).any(), np.isnan(b).any())}")
        if not np.allclose(a, b, atol=atol, rtol=rtol):
            ok = False
    print("ALLCLOSE:", ok)
    return ok

if __name__ == "__main__":
    model_fp32 = "models/dynamic_batch_model.onnx"
    model_fp16 = "models/dynamic_batch_model_fp16.onnx"
    input_dict = {
        "target": np.random.randn(1, 3, 128, 128).astype(np.float32),
        "source": np.random.randn(1, 512).astype(np.float32)
    }
    print("Comparing FP32 model:")
    compare(model_fp32, input_dict, atol=1e-5, rtol=1e-4)
    print("\nComparing FP16 model:")
    compare(model_fp16, input_dict, atol=1e-2, rtol=1e-2)