import numpy as np
import onnx
import onnxruntime as ort
import time
import torch

model_path = "models/dynamic_batch_model_fp16.onnx"
# model_path = 'models/inswapper_128_fp16.onnx'
model = onnx.load(model_path)
# with open(model_path, "rb") as f:  # <-- rb = read binary
#     serialized = f.read()
#     print(serialized)
if __name__ == "__main__":
    # providers=["CUDAExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
    # ort_session = ort.InferenceSession(model_path, providers=providers)
    
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 8
    sess_opts.inter_op_num_threads = 8

    providers = [
        ('CUDAExecutionProvider', {}),
        ('TensorrtExecutionProvider', {}),
        "CPUExecutionProvider"
    ]

    ort_session = ort.InferenceSession(model_path, sess_opts, providers=providers)
    batch = 4
    target = np.random.randn(batch, 3, 128, 128).astype(np.float32)
    source = np.random.randn(batch, 512).astype(np.float32)
    for _ in range(50):
        torch.cuda.synchronize()
        s = time.time()
        outputs = ort_session.run(None, {"target": target, "source": source})
        print(f"FPS: {1 / (time.time() - s):}")
    