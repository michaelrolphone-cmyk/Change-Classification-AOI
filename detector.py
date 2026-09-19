"""SMT pixel difference — NumPy only (Pi 3 friendly)."""
from __future__ import annotations

import numpy as np
from PIL import Image


def rgb_to_ycc(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + 0.5
    cr = (r - y) * 0.713 + 0.5
    return y, cb, cr


def sobel_mag(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32)
    pad = np.pad(y, 1, mode="edge")
    tl, t, tr = pad[0:-2, 0:-2], pad[0:-2, 1:-1], pad[0:-2, 2:]
    l, r = pad[1:-1, 0:-2], pad[1:-1, 2:]
    bl, b, br = pad[2:, 0:-2], pad[2:, 1:-1], pad[2:, 2:]
    gx = -tl - 2 * l - bl + tr + 2 * r + br
    gy = -tl - 2 * t - tr + bl + 2 * b + br
    return np.sqrt(gx * gx + gy * gy)


def local_mean(y: np.ndarray) -> np.ndarray:
    pad = np.pad(y.astype(np.float32), 1, mode="edge")
    acc = np.zeros_like(y, dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc += pad[1 + dy : 1 + dy + y.shape[0], 1 + dx : 1 + dx + y.shape[1]]
    return acc / 9.0


def phase_shift_luma(ref_y: np.ndarray, live_y: np.ndarray, n: int = 64) -> tuple[int, int, float]:
    """Translation-only phase correlation on downscaled luma. Returns (dx, dy, peak) in full-res pixels."""
    def down(img):
        im = Image.fromarray(np.clip(img, 0, 1).astype(np.float32), mode="F").resize((n, n), Image.BILINEAR)
        return np.array(im, dtype=np.float32)

    a = down(ref_y)
    b = down(live_y)
    wy = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / max(n - 1, 1))
    w = np.outer(wy, wy)
    fa = np.fft.fft2(a * w)
    fb = np.fft.fft2(b * w)
    cross = fa.conj() * fb
    mag = np.abs(cross) + 1e-8
    corr = np.fft.ifft2(cross / mag).real
    peak_idx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy, dx = int(peak_idx[0]), int(peak_idx[1])
    if dx > n // 2:
        dx -= n
    if dy > n // 2:
        dy -= n
    peak = float(corr[peak_idx])
    scale_x = ref_y.shape[1] / n
    scale_y = ref_y.shape[0] / n
    return int(round(dx * scale_x)), int(round(dy * scale_y)), peak


def _majority3(mask: np.ndarray) -> np.ndarray:
    pad = np.pad(mask.astype(np.uint8), 1, mode="edge")
    acc = np.zeros(mask.shape, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc += pad[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
    return acc >= 5


def find_green_board_box(rgb01: np.ndarray, inset_frac: float = 0.012) -> tuple[int, int, int, int] | None:
    """
    Find the green PCB as the largest box of green pixels that sits inside
    the photo (not the whole frame). rgb01 is HxWx3 in 0..1.
    Returns (x0, y0, x1, y1) exclusive-end, or None if no board found.
    """
    h, w = rgb01.shape[:2]
    # Work on a small copy so a Pi 3 is not chewing 8 MP for this step.
    tw, th = 320, int(round(320 * h / max(w, 1)))
    small = np.array(
        Image.fromarray((np.clip(rgb01, 0, 1) * 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    # White fixture: bright and almost no color. Never treat that as board.
    white = (mn > 0.72) & ((mx - mn) < 0.12)
    # Green soldermask on FR4: green wins, not white, not a black hole.
    green = (
        ~white
        & (g > r + 0.04)
        & (g > b + 0.04)
        & (g > 0.16)
        & (g < 0.90)
        & ((g - np.maximum(r, b)) > 0.03)
    )
    green = _majority3(green)
    # If soldermask is patchy (lots of copper/silkscreen), fall back to "not white".
    if int(green.sum()) < (tw * th * 0.02):
        board = _majority3(~white & (mx < 0.93))
    else:
        board = green
    if int(board.sum()) < (tw * th * 0.02):
        return None
    ys, xs = np.where(board)
    x0s, x1s = int(xs.min()), int(xs.max()) + 1
    y0s, y1s = int(ys.min()), int(ys.max()) + 1
    sx = w / tw
    sy = h / th
    x0 = int(round(x0s * sx))
    y0 = int(round(y0s * sy))
    x1 = int(round(x1s * sx))
    y1 = int(round(y1s * sy))
    # Keep the box strictly inside the frame.
    pad = max(2, int(round(min(w, h) * inset_frac)))
    x0 = max(pad, x0 + pad)
    y0 = max(pad, y0 + pad)
    x1 = min(w - pad, x1 - pad)
    y1 = min(h - pad, y1 - pad)
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    return x0, y0, x1, y1


def draw_box(rgb01: np.ndarray, box: tuple[int, int, int, int], color=(0.0, 0.85, 1.0), t: int = 4) -> np.ndarray:
    out = rgb01.copy()
    x0, y0, x1, y1 = box
    h, w = out.shape[:2]
    t = max(1, t)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    out[y0 : min(h, y0 + t), x0:x1] = color
    out[max(0, y1 - t) : y1, x0:x1] = color
    out[y0:y1, x0 : min(w, x0 + t)] = color
    out[y0:y1, max(0, x1 - t) : x1] = color
    return out


def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    ww = min(w - src_x0, w - dst_x0)
    hh = min(h - src_y0, h - dst_y0)
    if ww <= 0 or hh <= 0:
        return out
    out[dst_y0 : dst_y0 + hh, dst_x0 : dst_x0 + ww] = img[src_y0 : src_y0 + hh, src_x0 : src_x0 + ww]
    return out


def inspect(ref_rgb: np.ndarray, live_rgb: np.ndarray, settings: dict) -> dict:
    """
    ref_rgb / live_rgb: HxWx3 float 0-1 or uint8.
    settings keys: method, y_thresh, c_thresh, e_thresh, gain, floor, highpass, align
    """
    ref = ref_rgb.astype(np.float32)
    live = live_rgb.astype(np.float32)
    if ref.max() > 1.5:
        ref /= 255.0
        live /= 255.0

    if ref.shape != live.shape:
        live_img = Image.fromarray((np.clip(live, 0, 1) * 255).astype(np.uint8))
        live_img = live_img.resize((ref.shape[1], ref.shape[0]), Image.BILINEAR)
        live = np.array(live_img).astype(np.float32) / 255.0

    y_r, cb_r, cr_r = rgb_to_ycc(ref)
    y_l, cb_l, cr_l = rgb_to_ycc(live)

    dx = dy = 0
    peak = 0.0
    if settings.get("align", True):
        dx, dy, peak = phase_shift_luma(y_r, y_l)
        if dx or dy:
            live = shift_image(live, -dx, -dy)
            y_l, cb_l, cr_l = rgb_to_ycc(live)

    dY = y_l - y_r
    dC = np.sqrt((cb_l - cb_r) ** 2 + (cr_l - cr_r) ** 2)
    edge_d = np.abs(sobel_mag(y_l) - sobel_mag(y_r))
    hp_d = np.abs((y_l - local_mean(y_l)) - (y_r - local_mean(y_r)))

    method = settings.get("method", "ycc")
    if method == "y":
        raw = np.abs(dY)
    elif method == "edge":
        raw = edge_d
    elif method == "combined":
        raw = np.maximum(np.sqrt(dY * dY + dC * dC), edge_d)
    else:
        raw = np.sqrt(dY * dY + dC * dC)

    hp = float(settings.get("highpass", 0.35))
    raw = (1.0 - hp) * raw + hp * np.maximum(raw, hp_d)
    gain = float(settings.get("gain", 2.5))
    mag = raw * gain

    yt = float(settings.get("y_thresh", 6.0)) / 255.0
    ct = float(settings.get("c_thresh", 8.0)) / 255.0
    et = float(settings.get("e_thresh", 0.035))
    floor = float(settings.get("floor", 2.0)) / 255.0

    hit_y = np.abs(dY) >= yt
    hit_c = dC >= ct
    hit_e = edge_d >= et
    hit_floor = mag >= floor

    cls = np.zeros(dY.shape, dtype=np.uint8)
    if method == "y":
        hit = hit_y & hit_floor
        cls[hit & (dY > 0)] = 1
        cls[hit & (dY <= 0)] = 2
    elif method == "edge":
        cls[hit_e & hit_floor] = 4
    elif method == "combined":
        hit = (hit_y | hit_c | hit_e) & hit_floor
        cls[hit & hit_e & ~hit_y & ~hit_c] = 4
        cls[hit & hit_c & ~hit_y & (cls == 0)] = 3
        cls[hit & (dY >= 0) & (cls == 0)] = 1
        cls[hit & (dY < 0) & (cls == 0)] = 2
    else:
        hit = (hit_y | hit_c) & hit_floor
        cls[hit & hit_c & ~hit_y] = 3
        cls[hit & (dY >= 0) & (cls == 0)] = 1
        cls[hit & (dY < 0) & (cls == 0)] = 2

    roi_box = None
    if settings.get("auto_roi", True):
        roi_box = find_green_board_box(live) or find_green_board_box(ref)
        if roi_box is not None:
            x0, y0, x1, y1 = roi_box
            inside = np.zeros(cls.shape, dtype=bool)
            inside[y0:y1, x0:x1] = True
            cls = np.where(inside, cls, 0)

    colors = np.array(
        [
            [0, 0, 0],
            [0, 199, 82],
            [213, 0, 0],
            [255, 214, 0],
            [41, 182, 246],
        ],
        dtype=np.float32,
    ) / 255.0
    overlay = live.copy()
    mask = cls > 0
    opacity = float(settings.get("opacity", 0.75))
    overlay[mask] = (1 - opacity) * live[mask] + opacity * colors[cls[mask]]
    if roi_box is not None:
        overlay = draw_box(overlay, roi_box)

    heat = np.clip(mag / 0.35, 0, 1)
    if roi_box is not None:
        x0, y0, x1, y1 = roi_box
        outside = np.ones(heat.shape, dtype=bool)
        outside[y0:y1, x0:x1] = False
        heat[outside] *= 0.12
    heat_rgb = np.stack([heat, heat * 0.18, np.zeros_like(heat)], axis=-1)
    if roi_box is not None:
        heat_rgb = draw_box(heat_rgb, roi_box, color=(0.2, 0.7, 1.0))

    mask_rgb = colors[cls]
    if roi_box is not None:
        mask_rgb = draw_box(mask_rgb, roi_box, color=(0.0, 0.85, 1.0))

    changed = int(mask.sum())
    if roi_box is not None:
        x0, y0, x1, y1 = roi_box
        total = max((x1 - x0) * (y1 - y0), 1)
    else:
        total = int(cls.size)
    return {
        "overlay": overlay,
        "mask": mask_rgb,
        "heat": heat_rgb,
        "live": live,
        "changed_px": changed,
        "changed_pct": 100.0 * changed / max(total, 1),
        "shift": (dx, dy),
        "peak": peak,
        "shape": (int(cls.shape[0]), int(cls.shape[1])),
        "roi": roi_box,
    }


def to_jpeg_bytes(rgb01: np.ndarray, quality: int = 85) -> bytes:
    arr = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    from io import BytesIO

    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
