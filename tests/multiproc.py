import cv2
import time
import subprocess as sp
import multiprocessing
import os
from app.model import RealTimeSwapper # 假設 model.py 在同一目錄

# --- 1. 關鍵設定：定義多個串流 ---
# 這是您未來要處理的 5 個不同串流
# 為了測試，您可以先用 5 個不同的 RTMP 路徑指向同一個 RTSP 來源
STREAM_CONFIGS = [
    {
        "rtsp_url": "rtsp://127.0.0.1:8554/cam", # 來源 1
        "rtmp_url": "rtmp://35.185.181.19:1935/processed/cam1"
    },
    {
        "rtsp_url": "rtsp://127.0.0.1:8554/cam", # 來源 2
        "rtmp_url": "rtmp://35.185.181.19:1935/processed/cam2" # 輸出 2
    },
    {
        "rtsp_url": "rtsp://127.0.0.1:8554/cam",
        "rtmp_url": "rtmp://35.185.181.19:1935/processed/cam3"
    },
    {
        "rtsp_url": "rtsp://127.0.0.1:8554/cam",
        "rtmp_url": "rtmp://35.185.181.19:1935/processed/cam4"
    },
    {
        "rtsp_url": "rtsp://127.0.0.1:8554/cam",
        "rtmp_url": "rtmp://35.185.181.19:1935/processed/cam5"
    }
]

SOURCE_FACE_PATH = 'source/rose.jpeg'
TARGET_SIZE = (1280, 720) # 處理和輸出的目標尺寸

# --- 2. 工作函數 (這將在獨立的進程中運行) ---
def run_stream_worker(rtsp_url, rtmp_url, source_image_path, percent):
    """
    這個函數會在一個完全獨立的子進程中運行。
    它負責初始化模型、讀取RTSP、AI處理、並推流到RTMP。
    """
    os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(percent)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    print(f"[Worker {os.getpid()}] 函數已啟動，準備初始化...")
    # 關鍵：模型必須在子進程 "內部" 被初始化，而不是在全局
    try:
        print(f"[Worker {os.getpid()}] 正在初始化 AI 模型 (RTS)...")
        # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # 接著才載入你的模型/ONNX
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'gpu_mem_limit': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]
        swapper = RealTimeSwapper(providers=providers,
                                  face_analysis_name='buffalo_l',
                                  inswapper_path='models/inswapper_128.onnx')
        
        print(f"[Worker {os.getpid()}] 正在載入來源臉部: {source_image_path}")
        source_img = cv2.imread(source_image_path)
        if source_img is None:
            print(f"[Worker {os.getpid()}] 錯誤: 無法讀取來源圖片 {source_image_path}")
            return

        src_faces = swapper.get_source_face(source_img)
        if not src_faces:
            print(f"[Worker {os.getpid()}] 錯誤: 在 {source_image_path} 中找不到來源臉部")
            return
            
    except Exception as e:
        print(f"[Worker {os.getpid()}] AI 模型初始化失敗: {e}")
        return

    # --- B. 設定 FFmpeg 子程序 ---
    w_out, h_out = TARGET_SIZE
    # command = [
    #     'ffmpeg',
    #     '-y', # 覆蓋輸出
    #     '-f', 'rawvideo',
    #     '-vcodec', 'rawvideo',
    #     '-pix_fmt', 'bgr24',
    #     '-s', f"{w_out}x{h_out}",
    #     '-r', '25', # 假設 25 FPS
    #     '-i', '-',
        
    #     '-c:v', 'libx264',
    #     '-preset', 'veryfast',
    #     '-tune', 'zerolatency',
    #     '-b:v', '3000k',
    #     '-maxrate', '3000k',
    #     '-bufsize', '3000k',
    #     '-g', '50',
    #     '-an',
        
    #     '-f', 'flv',
    #     rtmp_url # 使用傳入的特定 RTMP URL
    # ]
    
    command = [
    'ffmpeg','-y',
    '-f','rawvideo','-vcodec','rawvideo','-pix_fmt','bgr24',
    '-s', f"{w_out}x{h_out}",
    '-r','7','-i','-',
    # ↓ 用 NVENC，低延遲 + CBR
    '-c:v','h264_nvenc',
    '-preset','p5','-tune','ll','-bf','0',
    '-rc','cbr','-b:v','3000k','-maxrate','3000k','-bufsize','3000k',
    '-g','50',
    '-pix_fmt','yuv420p',
    '-an','-f','flv', rtmp_url
    ]

    # --- C. 啟動 FFmpeg ---
    print(f"[Worker {os.getpid()}] 正在啟動 FFmpeg，推流至: {rtmp_url}")
    try:
        ffmpeg_process = sp.Popen(command, stdin=sp.PIPE)
    except FileNotFoundError:
        print(f"[Worker {os.getpid()}] 錯誤：找不到 FFmpeg。")
        return
    except Exception as e:
        print(f"[Worker {os.getpid()}] 啟動 FFmpeg 失敗: {e}")
        return

    # --- D. 啟動影像讀取與 AI 處理迴圈 ---
    print(f"[Worker {os.getpid()}] 正在連線到 RTSP 串流: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print(f"[Worker {os.getpid()}] 錯誤: 無法開啟 RTSP 串流 {rtsp_url}")
        if ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        return

    print(f"[Worker {os.getpid()}] 連線成功 {rtsp_url}，開始處理串流...")
    try:
        while True:
            # 策略性跳幀 (可選，根據您的 AI 速度調整)
            # cap.read()
            # cap.read() 
            ret, frame = cap.read()
            if not ret:
                print(f"[Worker {os.getpid()}] RTSP 串流 {rtsp_url} 結束")
                break
                
            # --- E. 執行 AI 處理 (關鍵) ---
            s = time.time()
            try:
                # AI 換臉
                swapped_frame = swapper.swap_into(frame.copy(), src_faces, swap_all=False)
                # 統一 resize 到目標尺寸
                swapped_resize = cv2.resize(swapped_frame, TARGET_SIZE)
                print(f"[Worker {os.getpid()}] Swap FPS: {1 / (time.time() - s):.2f}")
            
            except Exception as e:
                print(f"[Worker {os.getpid()}] AI 處理失敗: {e}")
                # 即使處理失敗，也推一個空的（或原始的）幀，避免 ffmpeg 中斷
                swapped_resize = cv2.resize(frame, TARGET_SIZE)


            # --- F. 將處理完的影像幀寫入 FFmpeg 的 stdin ---
            try:
                ffmpeg_process.stdin.write(swapped_resize.tobytes())
            except IOError as e:
                print(f"[Worker {os.getpid()}] FFmpeg 寫入錯誤 (可能已中斷): {e}")
                break
            except Exception as e:
                print(f"[Worker {os.getpid()}] 未知寫入錯誤: {e}")
                break

    except KeyboardInterrupt:
        print(f"[Worker {os.getpid()}] 偵測到使用者中斷...")
    except Exception as e:
        print(f"[Worker {os.getpid()}] 迴圈發生未知錯誤: {e}")
    finally:
        # --- G. 清理 ---
        print(f"[Worker {os.getpid()}] 正在關閉串流 {rtsp_url}...")
        cap.release()
        if ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        print(f"[Worker {os.getpid()}] 程式已關閉")

# --- 3. 主進程 (Manager) ---
if __name__ == '__main__':
    # 在 Windows/macOS 上使用 multiprocessing 必須把啟動代碼放在
    # 'if __name__ == "__main__":' 語句塊中
    
    print(f"[Main {os.getpid()}] 正在啟動 {len(STREAM_CONFIGS)} 個 AI worker 進程...")
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
        
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    processes = []
    
    share_percent = max(1, 100 // len(STREAM_CONFIGS))
    # share_percent = 100
    
    for i, config in enumerate(STREAM_CONFIGS):
        # 為每個串流配置啟動一個新的、獨立的進程
        # p = multiprocessing.Process(
        #     target=run_stream_worker, 
        #     args=(config['rtsp_url'], config['rtmp_url'], SOURCE_FACE_PATH)
        # )
        p = multiprocessing.Process(
            target=run_stream_worker,
            args=(config['rtsp_url'], config['rtmp_url'], SOURCE_FACE_PATH, share_percent)
        )
        p.start()
        processes.append(p)
        print(f"[Main {os.getpid()}] 已啟動 Worker (PID: {p.pid}) 處理 {config['rtsp_url']}")
        # 錯開啟動時間，避免所有模型同時初始化搶佔資源
        time.sleep(5) 

    print(f"[Main {os.getpid()}] 所有 worker 皆已啟動。")
    print(f"[Main {os.getpid()}] 您現在可以開啟 VLC 等播放器，連線到：")
    for config in STREAM_CONFIGS:
        print(f"  -> {config['rtmp_url']}")
    
    try:
        # 等待所有子進程結束
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print(f"[Main {os.getpid()}] 偵測到主程式中斷，正在嘗試終止所有 worker...")
        for p in processes:
            p.terminate() # 強制終止
            p.join()
        print(f"[Main {os.getpid()}] 所有 worker 已終止。")
