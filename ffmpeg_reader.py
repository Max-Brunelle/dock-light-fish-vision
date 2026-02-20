import subprocess
import numpy as np
import cv2
import time
import sys

W, H = 1280, 720

URL = "udp://0.0.0.0:5000?fifo_size=1000000&overrun_nonfatal=1"

cmd = [
    "ffmpeg",
    "-hide_banner",
    "-loglevel", "warning",
    "-i", URL,
    "-an",
    "-vf", f"scale={W}:{H}",     # force size, prevents reshape issues
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-"
]

print("Starting FFmpeg...")
print("Command:", " ".join(cmd))

p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)

frame_bytes = W * H * 3
frames = 0
t0 = time.time()
last_print = t0

try:
    while True:
        raw = p.stdout.read(frame_bytes)
        if len(raw) != frame_bytes:
            print(f"\nStopped: got {len(raw)} bytes (expected {frame_bytes}).")
            err = p.stderr.read().decode("utf-8", errors="ignore")
            print("FFmpeg stderr (tail):\n", err[-2000:])
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))
        frames += 1

        now = time.time()
        if now - last_print >= 1.0:
            fps = frames / (now - t0)
            print(f"Frames: {frames} | Avg FPS: {fps:.1f}")
            last_print = now

        cv2.imshow("Dock Feed (FFmpeg->Python)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    p.terminate()
    cv2.destroyAllWindows()
    print("Exited cleanly.")
