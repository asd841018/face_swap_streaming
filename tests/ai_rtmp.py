import cv2
import time
import subprocess as sp
import threading
from flask import Flask, Response
from app.model import RealTimeSwapper

# --- 1. 關鍵設定：修改您的串流位址 ---
# (正確) 您必須填入您「本地(家裡)」的「公開 IP」
RTSP_URL = "rtsp://127.0.0.1:8554/cam"

# 這是推送到您「雲端 VM」上運行的 mediamtx 伺服器
RTMP_URL = "rtmp://34.124.152.206:1935/processed_face"
# RTMP_URL = "rtsp://35.240.145.84:8555/test"
# "processed_face" 是您自訂的串流名稱

# ----------------------------------------
# --- A. 全域變數：用於線程間共享影像 ---
global_swapped_frame = None
frame_lock = threading.Lock()
# ----------------------------------------

# Initialize the model
print("正在初始化 AI 模型 (RealTimeSwapper)...")
# providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
providers = ['CPUExecutionProvider']
SWAPPER = RealTimeSwapper(providers=providers,
                          face_analysis_name='buffalo_l',
                          inswapper_path='models/dynamic_batch_model.onnx')

# Load source face for swapping
print("正在載入來源臉部...")
source_img = cv2.imread('source/rose.jpeg')
src_faces = SWAPPER.get_source_face(source_img)
if not src_faces:
    print("錯誤：在 'source/rose.jpeg' 中找不到來源臉部")
    exit()

# Target size for processing and output
target_size = (1280, 720)
W_OUT, H_OUT = target_size

# --- B. AI 處理與 RTMP 推流 (線程 1) ---
def process_and_push_rtmp():
    """
    這個函數會在一個背景線程中運行，
    負責讀取 RTSP、執行 AI、並推流到 RTMP。
    """
    global global_swapped_frame, frame_lock

    # --- 2. 設定 FFmpeg 子程序 ---
    # command = [
    #     'ffmpeg',
    #     '-y', # 覆蓋輸出
    #     '-f', 'rawvideo',
    #     '-vcodec', 'rawvideo',
    #     '-pix_fmt', 'bgr24', # OpenCV 的預設像素格式
    #     '-s', f"{W_OUT}x{H_OUT}", # 影像尺寸 "1280x720"
    #     '-r', '25',       # 假設 25 FPS (請根據您的 AI 速度調整)
    #     '-i', '-',        # 從 stdin 讀取
        
    #     # --- 輸出到 RTMP 的編碼設定 ---
    #     '-c:v', 'libx264',
    #     '-preset', 'veryfast',
    #     '-tune', 'zerolatency',
    #     '-b:v', '3000k',  # 720p 建議的位元率
    #     '-maxrate', '3000k',
    #     '-bufsize', '3000k',
    #     '-g', '50',       # 2 秒一個 I-frame (假設 25fps)
    #     '-an',            # 移除音訊
        
    #     # --- 最終目的地 ---
    #     '-f', 'flv',
    #     RTMP_URL
    # ]
    
    # This is for nvenc encoding
    command = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f"{W_OUT}x{H_OUT}",
    '-r', '25',
    '-i', '-',

    # --- 使用 NVENC ---
    '-c:v', 'h264_nvenc',
    '-preset', 'p4',          # p1(最快) ~ p7(最好畫質)；p3/p4常用
    '-tune', 'll',            # ← 改成 ll 或 ull；或直接移除此行
    '-rc', 'cbr',
    '-b:v', '3000k',
    '-maxrate', '3000k',
    '-bufsize', '3000k',
    '-g', '50',               # GOP = FPS，低延遲
    '-bf', '0',               # 關閉 B-frames 以降低延遲
    '-rc-lookahead', '0',     # 降延遲（關閉 lookahead）
    '-pix_fmt', 'yuv420p',    # 播放相容性
    '-an',

    '-f', 'flv',
    RTMP_URL
    ]
    # command = [
    #     "ffmpeg",
    #     "-y",
    #     "-f", "rawvideo", "-vcodec", "rawvideo",
    #     "-pix_fmt", "bgr24", "-s", f"{W_OUT}x{H_OUT}", "-r", "24",
    #     "-i", "-",
    #     "-c:v", "h264_nvenc", "-preset", "veryfast", "-tune", "zerolatency",
    #     "-b:v", "3000k", "-maxrate", "3000k", "-bufsize", "3000k",
    #     "-g", "24",
    #     "-pix_fmt", "yuv420p",
    #     "-an",
    #     "-rtsp_transport", "tcp",   # 建議走 TCP，較不受 UDP/防火牆影響
    #     "-f", "rtsp",               # ← RTSP 必須用 rtsp muxer
    #     RTMP_URL
    #     ]

    # --- 3. 啟動 FFmpeg 程序 ---
    print(f" [RTMP 線程] 正在啟動 FFmpeg，推流至: {RTMP_URL}")
    try:
        ffmpeg_process = sp.Popen(command, stdin=sp.PIPE)
    except FileNotFoundError:
        print("錯誤：找不到 FFmpeg。請確保 FFmpeg 已安裝並在系統 PATH 中。")
        exit()

    # --- 4. 啟動影像讀取與 AI 處理迴圈 ---
    print(f" [RTMP 線程] 正在連線到 RTSP 串流: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print(" [RTMP 線程] 錯誤: 無法開啟 RTSP 串流")
        if ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        exit()

    print(" [RTMP 線程] 連線成功，開始處理串流...")
    i = 0
    try:
        while True:
            # 策略性跳幀 (每 2 幀讀 1 幀)，幫助 AI 效能跟上
            # cap.read()
            # cap.read() 
            ret, frame = cap.read()
            if not ret:
                print(" [RTMP 線程] RTSP 串流結束")
                break
                
            # --- 5. 執行 AI 處理 (關鍵) ---
            
            # [優化] 先將影像 resize 到目標尺寸，再進行 AI 處理
            # frame_resized = cv2.resize(frame, target_size)
            
            s = time.time()
            # AI 換臉
            swapped_frame = SWAPPER.swap_into(frame.copy(), src_faces, swap_all=False)
            swapped_resize = cv2.resize(swapped_frame, target_size)
            cv2.imwrite(f'imgs/output_{i:04d}.jpg', swapped_resize)
            i += 1
            print(f" [RTMP 線程] Swap FPS: {1 / (time.time() - s):.2f}")
            
            # --- 6. 將處理完的影像幀寫入 FFmpeg 的 stdin ---
            try:
                ffmpeg_process.stdin.write(swapped_resize.tobytes())
            except IOError as e:
                print(f" [RTMP 線程] FFmpeg 寫入錯誤 (可能已中斷): {e}")
                break

            # --- 7. 更新全域變數，供 Flask 使用 ---
            with frame_lock:
                global_swapped_frame = swapped_frame.copy()

    except KeyboardInterrupt:
        print(" [RTMP 線程] 偵測到使用者中斷...")
    finally:
        # --- 8. 清理 ---
        print(" [RTMP 線程] 正在關閉串流...")
        cap.release()
        if ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        print(" [RTMP 線程] 程式已關閉")

# --- C. Flask 網頁伺服器 (主線程) ---
app = Flask(__name__)

def gen_frames():
    """
    這個函數是 Flask 的影像產生器，
    它只會從全域變數中讀取最新的影像幀。
    """
    global global_swapped_frame, frame_lock
    
    print(" [Flask 線程] 網頁串流已連線")
    
    while True:
        local_frame_copy = None
        
        # 從全域變數中安全地讀取最新一幀
        with frame_lock:
            if global_swapped_frame is not None:
                local_frame_copy = global_swapped_frame.copy()
        
        if local_frame_copy is None:
            # 如果 AI 還沒處理好第一幀，稍等一下
            time.sleep(0.1)
            continue

        # --- 將影像編碼為 JPEG ---
        ret, buffer = cv2.imencode('.jpg', local_frame_copy)
        if not ret:
            print(" [Flask 線程] JPEG 編碼失敗")
            continue
            
        frame_bytes = buffer.tobytes()
        
        # --- 透過 HTTP 串流 ---
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # 控制 Flask 的 FPS，避免 CPU 佔用過高
        # (0.04s -> 25 FPS，與 AI 處理速度同步)
        time.sleep(0.04)

@app.route('/video_feed')
def video_feed():
    """網頁影像串流的路由"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """網頁首頁"""
    return "<h1>AI Video Stream (MJPEG)</h1><img src='/video_feed'>"

# --- D. 啟動程式 ---
if __name__ == '__main__':
    # 1. 啟動 AI + RTMP 推流線程 (設為 daemon，主程式結束時自動關閉)
    rtmp_thread = threading.Thread(target=process_and_push_rtmp, daemon=True)
    rtmp_thread.start()
    
    # 2. 在主線程啟動 Flask 伺服器
    # (threaded=True 允許 Flask 處理多個網頁請求)
    print(" [主線程] 啟動 Flask 伺服器於 http://0.0.0.0:5000")
    print(" [主線程] RTMP 串流正在背景推送到 HLS/RTSP...")
    app.run(host='0.0.0.0', port=5000, threaded=True)