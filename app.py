#!/usr/bin/env python3
"""SMT Change Detector — Flask UI for Raspberry Pi 3 B+ + Sony IMX219 (Camera Module V2)."""
from __future__ import annotations

import json
import os
import threading
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file
from PIL import Image

import detector

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REF_DIR = DATA / "references"
RES_DIR = DATA / "results"
CFG_PATH = DATA / "settings.json"
REF_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

# Sony IMX219 (Camera Module V2) — official native modes
CAMERA_NAME = "Sony IMX219 Camera Module V2"
SENSOR = "IMX219"
PREVIEW_SIZE = (640, 480)       # binned-ish preview; keeps the 1 GB Pi alive
INSPECT_SIZE = (3280, 2464)     # IMX219 full still (not HQ 4056×3040)

DEFAULTS = {
    "method": "ycc",
    "display": "overlay",
    "y_thresh": 6.0,
    "c_thresh": 8.0,
    "e_thresh": 0.035,
    "gain": 2.5,
    "floor": 2.0,
    "highpass": 0.35,
    "opacity": 0.75,
    "align": True,
    "auto_roi": True,
    "sku": "default",
    "inspect_w": INSPECT_SIZE[0],
    "inspect_h": INSPECT_SIZE[1],
}

lock = threading.Lock()
settings = dict(DEFAULTS)
last_result_meta = {}
last_images = {}  # name -> jpeg bytes

if CFG_PATH.exists():
    try:
        settings.update(json.loads(CFG_PATH.read_text()))
    except Exception:
        pass
# Always IMX219 native still size.
settings["inspect_w"] = INSPECT_SIZE[0]
settings["inspect_h"] = INSPECT_SIZE[1]
settings["camera"] = CAMERA_NAME
settings["sensor"] = SENSOR


def save_settings():
    CFG_PATH.write_text(json.dumps(settings, indent=2))


class Camera:
    def __init__(self):
        self.mode = "none"
        self.picam = None
        self.cv = None
        self._init()

    def _init(self):
        try:
            from picamera2 import Picamera2

            cam = Picamera2()
            self.preview_cfg = cam.create_preview_configuration(
                main={"size": PREVIEW_SIZE, "format": "RGB888"}
            )
            self.still_cfg = cam.create_still_configuration(
                main={"size": INSPECT_SIZE, "format": "RGB888"}
            )
            cam.configure(self.preview_cfg)
            cam.start()
            self.picam = cam
            self.mode = "picamera2"
            print(f"Camera: {CAMERA_NAME}  still={INSPECT_SIZE}", flush=True)
            return
        except Exception as e:
            print("picamera2 unavailable:", e, flush=True)
        try:
            import cv2

            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_SIZE[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_SIZE[1])
                self.cv = cap
                self.mode = "opencv"
                print("Camera: OpenCV VideoCapture(0)", flush=True)
                return
        except Exception as e:
            print("OpenCV camera unavailable:", e, flush=True)
        self.mode = "demo"
        print("Camera: demo pattern (no hardware)", flush=True)

    def preview_rgb(self) -> np.ndarray:
        if self.mode == "picamera2":
            arr = self.picam.capture_array("main")
            return arr
        if self.mode == "opencv":
            ok, frame = self.cv.read()
            if not ok:
                return self._demo()
            import cv2

            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._demo()

    def still_rgb(self, size=None) -> np.ndarray:
        size = size or (int(settings["inspect_w"]), int(settings["inspect_h"]))
        if self.mode == "picamera2":
            # IMX219 full still via mode switch (3280×2464). Never invent other sensor sizes.
            size = INSPECT_SIZE
            try:
                arr = self.picam.switch_mode_and_capture_array(self.still_cfg)
            except Exception as e:
                print("still switch failed, using preview frame:", e, flush=True)
                arr = self.picam.capture_array("main")
            img = Image.fromarray(arr)
            if img.size != size:
                img = img.resize(size, Image.BILINEAR)
            return np.array(img)
        if self.mode == "opencv":
            frames = []
            for _ in range(3):
                ok, frame = self.cv.read()
                if ok:
                    frames.append(frame)
            if not frames:
                return self._demo(size)
            import cv2

            frame = frames[-1]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize(size, Image.BILINEAR)
            return np.array(img)
        return self._demo(size)

    def _demo(self, size=None):
        w, h = size or PREVIEW_SIZE
        yy, xx = np.mgrid[0:h, 0:w]
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[..., 1] = 40
        img[(yy // 40 + xx // 40) % 2 == 0] = (30, 80, 40)
        # fake SMT pads
        for i in range(8):
            x = 80 + i * 60
            img[h // 2 - 8 : h // 2 + 8, x : x + 28] = (180, 180, 190)
        return img


camera = Camera()
app = Flask(__name__)
TOKEN = os.environ.get("SMT_TOKEN", "").strip()


@app.before_request
def _gate():
    if not TOKEN:
        return None
    if request.path == "/":
        q = request.args.get("token", "")
        if q == TOKEN:
            return None
    hdr = request.headers.get("X-SMT-Token", "")
    q = request.args.get("token", "")
    if hdr == TOKEN or q == TOKEN:
        return None
    return ("unauthorized — add ?token=… or header X-SMT-Token", 401)


def load_reference(sku: str | None = None) -> np.ndarray | None:
    sku = sku or settings.get("sku", "default")
    path = REF_DIR / f"{sku}.jpg"
    if not path.exists():
        path = REF_DIR / "default.jpg"
    if not path.exists():
        return None
    return np.array(Image.open(path).convert("RGB"))


def save_reference(rgb: np.ndarray, sku: str | None = None):
    sku = sku or settings.get("sku", "default")
    settings["sku"] = sku
    Image.fromarray(rgb).save(REF_DIR / f"{sku}.jpg", quality=92)
    save_settings()


@app.route("/")
def index():
    return render_template(
        "index.html",
        settings=settings,
        camera=camera.mode,
        camera_name=CAMERA_NAME,
        sensor=SENSOR,
        inspect_size=f"{INSPECT_SIZE[0]}×{INSPECT_SIZE[1]}",
    )


@app.route("/stream")
def stream():
    def gen():
        while True:
            try:
                rgb = camera.preview_rgb()
                buf = BytesIO()
                Image.fromarray(rgb).save(buf, format="JPEG", quality=70)
                jpg = buf.getvalue()
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
            except Exception:
                time.sleep(0.2)
            time.sleep(0.12)  # ~8 fps — kind to Pi 3

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or request.form.to_dict()
        for k in DEFAULTS:
            if k in data:
                val = data[k]
                if k in ("method", "display", "sku"):
                    settings[k] = str(val)
                elif k in ("align", "auto_roi"):
                    settings[k] = str(val).lower() in ("1", "true", "on", "yes")
                else:
                    settings[k] = type(DEFAULTS[k])(val)
        save_settings()
    return jsonify(settings)


@app.route("/api/preset/fine", methods=["POST"])
def preset_fine():
    settings.update(
        {
            "method": "combined",
            "display": "overlay",
            "y_thresh": 3.0,
            "c_thresh": 5.0,
            "e_thresh": 0.02,
            "gain": 4.0,
            "floor": 1.0,
            "highpass": 0.55,
            "opacity": 0.75,
        }
    )
    save_settings()
    return jsonify(settings)


busy = threading.Event()


def set_busy_led(on: bool):
    led = getattr(app, "gpio_led", None)
    if led is None:
        return
    try:
        led.on() if on else led.off()
    except Exception:
        pass


def do_capture_ref(sku: str | None = None) -> dict:
    sku = sku or settings.get("sku", "default")
    if busy.is_set():
        return {"ok": False, "error": "Busy"}
    busy.set()
    set_busy_led(True)
    try:
        with lock:
            rgb = camera.still_rgb()
            save_reference(rgb, sku)
            last_images["reference"] = detector.to_jpeg_bytes(rgb.astype(np.float32) / 255.0)
        return {"ok": True, "sku": sku, "shape": list(rgb.shape), "source": "capture"}
    finally:
        set_busy_led(False)
        busy.clear()


def do_inspect(source: str = "http") -> dict:
    sku = settings.get("sku", "default")
    ref = load_reference(sku)
    if ref is None:
        return {"ok": False, "error": "No reference. Capture one first."}
    if busy.is_set():
        return {"ok": False, "error": "Busy"}
    busy.set()
    set_busy_led(True)
    try:
        with lock:
            live = camera.still_rgb((ref.shape[1], ref.shape[0]))
            result = detector.inspect(ref, live, settings)
            disp = settings.get("display", "overlay")
            key = {"overlay": "overlay", "mask": "mask", "heat": "heat", "live": "live"}.get(disp, "overlay")
            jpg = detector.to_jpeg_bytes(result[key])
            last_images["result"] = jpg
            last_images["overlay"] = detector.to_jpeg_bytes(result["overlay"])
            last_images["heat"] = detector.to_jpeg_bytes(result["heat"])
            last_images["mask"] = detector.to_jpeg_bytes(result["mask"])
            out = RES_DIR / f"{sku}_{int(time.time())}.jpg"
            out.write_bytes(jpg)
            last_result_meta.clear()
            last_result_meta.update(
                {
                    "ok": True,
                    "camera": CAMERA_NAME,
                    "sensor": SENSOR,
                    "sku": sku,
                    "changed_px": result["changed_px"],
                    "changed_pct": round(result["changed_pct"], 3),
                    "shift": result["shift"],
                    "peak": round(result["peak"], 4),
                    "shape": result["shape"],
                    "file": out.name,
                    "source": source,
                    "roi": result.get("roi"),
                }
            )
        return dict(last_result_meta)
    finally:
        set_busy_led(False)
        busy.clear()


@app.route("/api/capture_ref", methods=["POST"])
def capture_ref():
    sku = (request.json or {}).get("sku") if request.is_json else request.form.get("sku")
    out = do_capture_ref(sku)
    return jsonify(out), (200 if out.get("ok") else 409)


@app.route("/api/inspect", methods=["POST"])
def inspect():
    out = do_inspect("http")
    return jsonify(out), (200 if out.get("ok") else (400 if "reference" in out.get("error", "") else 409))


@app.route("/api/last")
def api_last():
    return jsonify(last_result_meta)


@app.route("/image/<name>")
def image(name):
    if name == "reference":
        sku = settings.get("sku", "default")
        path = REF_DIR / f"{sku}.jpg"
        if path.exists():
            return send_file(path, mimetype="image/jpeg")
    blob = last_images.get(name)
    if blob:
        return send_file(BytesIO(blob), mimetype="image/jpeg")
    return ("", 404)


def start_gpio():
    """
    BCM pins (button to GND, internal pull-up, press = falling edge):
      SMT_GPIO_INSPECT  default 17  → inspect still
      SMT_GPIO_REF      default 27  → capture reference
      SMT_GPIO_LED      default 22  → busy LED (HIGH while capturing)
    Set any pin to -1 to disable that line.
    """
    pin_ins = int(os.environ.get("SMT_GPIO_INSPECT", "17"))
    pin_ref = int(os.environ.get("SMT_GPIO_REF", "27"))
    pin_led = int(os.environ.get("SMT_GPIO_LED", "22"))
    try:
        from gpiozero import Button, LED
    except Exception as e:
        print("GPIO off (gpiozero not available):", e, flush=True)
        return None

    def on_inspect():
        print("GPIO inspect", flush=True)
        threading.Thread(target=do_inspect, args=("gpio",), daemon=True).start()

    def on_ref():
        print("GPIO capture reference", flush=True)
        threading.Thread(target=do_capture_ref, daemon=True).start()

    handles = {}
    try:
        if pin_ins >= 0:
            btn = Button(pin_ins, pull_up=True, bounce_time=0.08)
            btn.when_pressed = on_inspect
            handles["inspect"] = btn
            print(f"GPIO inspect trigger: BCM {pin_ins} (to GND)", flush=True)
        if pin_ref >= 0:
            btn = Button(pin_ref, pull_up=True, bounce_time=0.08)
            btn.when_pressed = on_ref
            handles["ref"] = btn
            print(f"GPIO reference trigger: BCM {pin_ref} (to GND)", flush=True)
        if pin_led >= 0:
            led = LED(pin_led)
            led.off()
            app.gpio_led = led
            handles["led"] = led
            print(f"GPIO busy LED: BCM {pin_led}", flush=True)
    except Exception as e:
        print("GPIO init failed:", e, flush=True)
        return None
    return handles


def main():
    host = os.environ.get("SMT_HOST", "0.0.0.0")
    port = int(os.environ.get("SMT_PORT", "8080"))
    gpio = start_gpio()
    app.gpio_handles = gpio
    print(
        f"SMT UI  http://{host}:{port}  camera={camera.mode}  gpio={'on' if gpio else 'off'}",
        flush=True,
    )
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
