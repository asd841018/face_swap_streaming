import cv2
import time
from flask import Flask, Response
from app.model import RealTimeSwapper

RTSP_URL = "rtsp://127.0.0.1:8554/cam"

# Initialize the model
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
SWAPPER = RealTimeSwapper(providers=providers,
                          face_analysis_name='buffalo_l',
                          inswapper_path='models/inswapper_128.onnx')

# Generate source face for swapping
source_img = cv2.imread('source/rose.jpeg')
src_faces = SWAPPER.get_source_face(source_img)
# Target size for showing
target_size = (1280, 720)

app = Flask(__name__)

def my_ai_function(frame):
    # 您的 AI 處理
    cv2.rectangle(frame, (50, 50), (150, 150), (0, 255, 0), 2)
    cv2.putText(frame, 'AI PROCESSED', (50, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def gen_frames():  
    print("正在連線到 RTSP...")
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("Error: 無法開啟 RTSP 串流")
        return

    print("連線成功，開始串流")
    while True:
        # 策略性丟棄前 1 幀，以清空緩衝區
        cap.read()
        
        ret, frame = cap.read()
        if not ret:
            print("串流結束")
            break
        
        # Swap
        s = time.time()
        swapped = SWAPPER.swap_into(frame.copy(), src_faces, swap_all=False)
        swapped = cv2.resize(swapped, target_size)
        print(f"Swap FPS: {1 / (time.time() - s)}")
        
        # 2. 將影像編碼為 JPEG
        ret, buffer = cv2.imencode('.jpg', swapped)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # 3. 透過網頁"串流"
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>AI 影像串流</h1><img src='/video_feed'>"

if __name__ == '__main__':
    # 讓外部可以存取，例如在 5000 埠
    app.run(host='0.0.0.0', port=5000)