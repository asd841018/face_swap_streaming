import cv2
from app.model import RealTimeSwapper

if __name__ == "__main__":
    
    # Initialize the model
    print("正在初始化 AI 模型 (RealTimeSwapper)...")
    # providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    # providers = ['CPUExecutionProvider']
    SWAPPER = RealTimeSwapper(providers=providers,
                            face_analysis_name='buffalo_l',
                            inswapper_path='models/dynamic_batch_model.onnx')

    # Load source face for swapping
    print("正在載入來源臉部...")
    source_img = cv2.imread('source/rose.jpeg')
    src_faces = SWAPPER.get_source_face(source_img)
    if not src_faces:
        print("錯誤：在 'source/rose.jpeg' 中找不到來源臉部")
        
    # Target size for processing and output
    target_size = (1280, 720)
    W_OUT, H_OUT = target_size
    
    RTSP_URL = "rtsp://127.0.0.1:8554/cam"
    cap = cv2.VideoCapture(RTSP_URL)
    i = 0
    if not cap.isOpened():
        print("Error: 無法開啟 RTSP 串流")
    else:
        while True:
            ret, frame = cap.read()
            # AI 換臉
            swapped_frame = SWAPPER.swap_into(frame.copy(), src_faces, swap_all=False)
            swapped_resize = cv2.resize(swapped_frame, target_size)
            cv2.imwrite(f'imgs/output_{i:04d}.jpg', swapped_resize)
            print(i)
            i += 1