import cv2
import numpy as np

def adjust_white_ycrcb(bgr, strength=0.35, brighten=6, mode=None):
    """
    strength: 0~1 冷暖白力度
    brighten: 提亮 Y 的常數（0~15 通常夠）
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    # 冷白：Cb +, Cr -
    # 用 int16 防 overflow
    if mode == "warm":
        Cb2 = np.clip(Cb.astype(np.int16) - int(25*strength), 0, 255).astype(np.uint8)
        Cr2 = np.clip(Cr.astype(np.int16) + int(22*strength), 0, 255).astype(np.uint8)
    if mode == "cold":
        Cb2 = np.clip(Cb.astype(np.int16) + int(25*strength), 0, 255).astype(np.uint8)
        Cr2 = np.clip(Cr.astype(np.int16) - int(22*strength), 0, 255).astype(np.uint8)

    # 微提亮（像棚燈）
    Y2 = cv2.add(Y, brighten)

    out = cv2.merge([Y2, Cr2, Cb2])
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR), out  # 回傳 BGR + ycrcb(後面可重用)

def skin_mask_from_ycrcb(ycrcb):
    # 直接用同一個 ycrcb，不要再做一次 cvtColor（省時間）
    return cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

def fast_pyr_smooth(bgr, mask, strength=0.55, levels=2):
    """
    levels=2 -> 1/4 尺寸做 blur（很快）
    strength: 混合力度
    """
    img = bgr
    m = mask

    # downsample levels 次
    for _ in range(levels):
        img = cv2.pyrDown(img)
        m = cv2.pyrDown(m)

    # 小圖做 Gaussian（超快）
    blur = cv2.GaussianBlur(img, (0,0), 2.2)

    # upsample 回來
    for _ in range(levels):
        blur = cv2.pyrUp(blur)
        m = cv2.pyrUp(m)

    # 對齊大小（避免 odd size 造成差一點點）
    blur = cv2.resize(blur, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    # soft blend（只在 skin mask）
    alpha = (m.astype(np.float32)/255.0)[...,None] * strength
    out = bgr.astype(np.float32)*(1-alpha) + blur.astype(np.float32)*alpha
    return np.clip(out, 0, 255).astype(np.uint8)

def beauty_pipeline(frame, strength=0.35, brighten=6, smooth=0.55, mode="warm"):
    # 1) 調整冷暖白（順便拿 ycrcb 給 mask 用）
    out, ycrcb = adjust_white_ycrcb(frame, strength=strength, brighten=brighten, mode=mode)

    # 2) 膚色 mask（不再 cvtColor）
    mask = skin_mask_from_ycrcb(ycrcb)

    # 3) 超快磨皮（pyramid）
    out = fast_pyr_smooth(out, mask, strength=smooth, levels=2)
    return out