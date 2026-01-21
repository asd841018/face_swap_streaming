import numpy as np
import cv2

def build_fade_curve(contrast=0.12, lift=12, gamma=1.05):
    """
    contrast: 對比 (0~0.25)
    lift: 抬黑 (0~30) 典型復古感
    gamma: 中間調
    """
    x = np.arange(256).astype(np.float32)

    # 先做 gamma
    y = 255.0 * ((x / 255.0) ** (1.0 / gamma))

    # S-curve (用 sigmoid 近似)
    t = (y - 128.0) / 128.0
    y = 128.0 + 128.0 * np.tanh((1.0 + contrast*6.0) * t)

    # 抬黑（讓黑不黑）
    y = y * (255 - lift) / 255.0 + lift

    return np.clip(y, 0, 255).astype(np.uint8)

def apply_curve_bgr(bgr, curve):
    # cv2.LUT 是 1D lookup，非常快
    return cv2.LUT(bgr, curve)

def split_tone(bgr, shadow=(8, 6, -4), highlight=(-6, -2, 10), balance=0.5):
    """
    shadow/highlight: (B,G,R) 的加值，範圍大概 -20~+20
    balance: 0~1，越大越偏高光
    """
    img = bgr.astype(np.float32)
    y = 0.114*img[...,0] + 0.587*img[...,1] + 0.299*img[...,2]
    y = y / 255.0

    # 低亮度權重 / 高亮度權重
    ws = np.clip((0.55 - y) / 0.55, 0, 1)  # shadows
    wh = np.clip((y - 0.45) / 0.55, 0, 1)  # highlights

    ws = (ws**1.6)[...,None]
    wh = (wh**1.6)[...,None]

    s = np.array(shadow, dtype=np.float32)[None,None,:]
    h = np.array(highlight, dtype=np.float32)[None,None,:]

    out = img + ws * s * (1-balance) + wh * h * balance
    return np.clip(out, 0, 255).astype(np.uint8)

def film_grain(bgr, amount=0.12, colored=False):
    """
    amount: 0~0.25
    colored: True 會有彩色顆粒（更復古），False 為灰階顆粒（更乾淨）
    """
    img = bgr.astype(np.float32)
    h, w = img.shape[:2]

    if colored:
        noise = np.random.normal(0, 255*amount, (h, w, 3)).astype(np.float32)
    else:
        n = np.random.normal(0, 255*amount, (h, w, 1)).astype(np.float32)
        noise = np.repeat(n, 3, axis=2)

    # 亮度越低顆粒越明顯
    y = (0.114*img[...,0] + 0.587*img[...,1] + 0.299*img[...,2]) / 255.0
    wgt = np.clip(1.2 - y, 0.2, 1.0)[...,None]

    out = img + noise * wgt
    return np.clip(out, 0, 255).astype(np.uint8)

def vignette(bgr, strength=0.35):
    """
    strength: 0~0.7
    """
    h, w = bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w*0.5, h*0.5
    r = np.sqrt(((xx-cx)/cx)**2 + ((yy-cy)/cy)**2)
    mask = 1.0 - strength * (r**1.6)
    mask = np.clip(mask, 0.0, 1.0)[...,None]

    out = bgr.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)

def vintage_filter(bgr,
                   contrast=0.12, lift=14, gamma=1.05,
                   balance=0.55,
                   grain=0.10, colored_grain=False,
                   vig=0.35):
    curve = build_fade_curve(contrast=contrast, lift=lift, gamma=gamma)
    out = apply_curve_bgr(bgr, curve)

    # 分離色調：暗部微青，高光微暖
    # out = split_tone(out,
    #                  shadow=(10, 6, -6),
    #                  highlight=(-6, -2, 12),
    #                  balance=balance)

    # out = film_grain(out, amount=grain, colored=colored_grain)
    out = vignette(out, strength=vig)
    return out