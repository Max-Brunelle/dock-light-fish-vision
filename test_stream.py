import cv2
import time

cap = cv2.VideoCapture("udp://@:5000")  # VLC URL

if not cap.isOpened():
    raise SystemExit("Failed to open stream. Is ffmpeg on the Pi running? Is the port 5000 correct?")

t0 = time.time()
frames = 0

while time.time() - t0 < 5:
    ret, frame = cap.read()
    if not ret:
        continue
    frames += 1

cap.release()
print("Frames read in 5s:", frames)
