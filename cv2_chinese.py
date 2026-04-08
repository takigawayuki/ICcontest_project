"""
cv2_chinese.py — OpenCV 中文文字渲染工具
用 PIL 绘制中文后转回 OpenCV BGR，避免 cv2.putText 无法显示中文的问题。

用法：
    from cv2_chinese import put_text, find_font

    img = put_text(img, "皖A12345", (10, 30), font_size=20, color=(255, 255, 255))
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -------------------------------------------------------
# 字体搜索（Windows / Linux / macOS 均可）
# -------------------------------------------------------
_FONT_CANDIDATES = [
    r"C:/Windows/Fonts/simhei.ttf",      # 黑体（Windows）
    r"C:/Windows/Fonts/msyh.ttc",        # 微软雅黑（Windows）
    r"C:/Windows/Fonts/simsun.ttc",      # 宋体（Windows）
    r"C:/Windows/Fonts/STZHONGS.TTF",    # 华文中宋
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",   # Linux
    "/System/Library/Fonts/PingFang.ttc",               # macOS
]

_font_cache: dict = {}   # size → ImageFont


def find_font() -> str:
    """返回第一个存在的中文字体路径，找不到则返回 None。"""
    for p in _FONT_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def _get_font(font_size: int, font_path: str = None) -> ImageFont.FreeTypeFont:
    key = (font_size, font_path)
    if key not in _font_cache:
        path = font_path or find_font()
        if path:
            _font_cache[key] = ImageFont.truetype(path, font_size)
        else:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


# -------------------------------------------------------
# 核心接口
# -------------------------------------------------------

def put_text(img_bgr: np.ndarray,
             text: str,
             pos: tuple,
             font_size: int = 18,
             color: tuple = (255, 255, 255),
             font_path: str = None) -> np.ndarray:
    """
    在 OpenCV BGR 图像上绘制中文（或任意 Unicode）文字。

    参数
    ----
    img_bgr   : OpenCV BGR ndarray
    text      : 要绘制的文字（支持中文）
    pos       : (x, y) 左上角坐标
    font_size : 字号（像素）
    color     : BGR 颜色元组，默认白色
    font_path : 手动指定字体路径，None 则自动搜索

    返回
    ----
    绘制后的 BGR ndarray（与输入同尺寸）
    """
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)
    font    = _get_font(font_size, font_path)
    # PIL color 是 RGB
    rgb_color = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def put_text_with_bg(img_bgr: np.ndarray,
                     text: str,
                     pos: tuple,
                     font_size: int = 18,
                     color: tuple = (255, 255, 255),
                     bg_color: tuple = (0, 0, 0),
                     padding: int = 2,
                     font_path: str = None) -> np.ndarray:
    """
    带背景矩形的中文文字（提高可读性）。
    bg_color 为 BGR 颜色。
    """
    font = _get_font(font_size, font_path)
    # 用临时 PIL 图计算文字尺寸
    tmp  = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = tmp.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = pos
    # 先画背景
    img_bgr = img_bgr.copy()
    cv2.rectangle(img_bgr,
                  (x - padding, y - padding),
                  (x + tw + padding, y + th + padding),
                  bg_color, -1)
    return put_text(img_bgr, text, pos, font_size, color, font_path)
