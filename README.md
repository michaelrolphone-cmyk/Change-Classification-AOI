# SMT Pi Inspector — Sony IMX219 (Camera Module V2)

Written for **Raspberry Pi 3 B+ (1 GB) + official Camera Module V2 (Sony IMX219)**.

Not HQ (IMX477), not Camera Module 3 (IMX708). Inspect size is locked to the IMX219 native still **3280×2464**.

- Live MJPEG preview at 640×480 (~8 fps)
- Inspect: real IMX219 still via `switch_mode_and_capture_array` at 3280×2464
- Same residual math as the simplified phone app: ΔY, ΔC, edge, gain, floor, local contrast
- Translation-only optional nudge (64×64 phase correlation). No rotation/scale, no classifiers.
- View on the Pi, on the LAN, or remotely (port-forward / VPN). Optional token.

## Pi 3 B+ setup (Bookworm)

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-flask python3-numpy python3-pil python3-libcamera python3-gpiozero
# Enable camera if needed
sudo raspi-config   # Interface Options → Camera → Enable, then reboot
```

Copy this folder to the Pi, e.g. `/home/pi/smt_pi`.

```bash
cd /home/pi/smt_pi
python3 app.py
```

Open `http://<pi-ip>:8080` from a phone or laptop on the same network.

Inspect stills are locked to the Sony IMX219 native frame **3280×2464**. The app will not request HQ (4056×3040) or Module 3 (4608×2592) sizes.

Expect several seconds per inspect on a Pi 3 B+ and a RAM spike. Emergency escape only: set `inspect_w`/`inspect_h` to `1640`×`1232` (IMX219 2×2 bin) if the Pi swaps — that is still an IMX219 mode, not a different camera.

## GPIO trigger (Pi 3 B+ header, BCM numbering)

Buttons are **active-low**: pin → button → **GND**. Internal pull-up is enabled. 80 ms debounce.

| Function | Default BCM | Header pin | Action |
|----------|-------------|------------|--------|
| Inspect | 17 | physical 11 | Full IMX219 still + difference |
| Capture reference | 27 | physical 13 | Save new golden frame |
| Busy LED | 22 | physical 15 | HIGH while a capture is running |

Disable a line with `-1`:

```bash
export SMT_GPIO_INSPECT=17
export SMT_GPIO_REF=27
export SMT_GPIO_LED=22
# export SMT_GPIO_REF=-1   # no reference button
python3 app.py
```

Do not press inspect while the LED is on; the second press is ignored (`Busy`).

## Remote access

Bind is `0.0.0.0:8080` by default.

Safer options:

- Tailscale / WireGuard onto the Pi, then open `http://<tailscale-ip>:8080`
- SSH tunnel: `ssh -L 8080:127.0.0.1:8080 pi@<pi>`
- Optional token:

```bash
export SMT_TOKEN=pick-a-long-secret
python3 app.py
# then http://<pi>:8080/?token=pick-a-long-secret
```

Do not put this port on the open internet without the token (or better, without a VPN).

## systemd

`sudo cp smt-inspector.service /etc/systemd/system/`
Edit the `WorkingDirectory` and `User` paths, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now smt-inspector
```

## Fixture notes

- Lock working distance and lighting; leave **X/Y nudge align** on only if the nest can sit a few pixels off.
- Capture a reference per SKU name.
- Use **SMT Fine** for 0402-class parts, then raise **Highlight floor** if the whole board lights up.
- Lighting: two-sided diffuse. Point glare on solder will false-trigger ΔY.
- White fixture background: auto board box treats white as not-the-board and inspects the green PCB only.

## Demo mode

If no camera is present, the server draws a fake board so you can test the UI on a laptop:

```bash
pip install flask numpy pillow
python3 app.py
```
