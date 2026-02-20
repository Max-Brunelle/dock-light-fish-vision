import subprocess
import numpy as np
import cv2
import time

# Stream / decode config
W, H = 1280, 720
URL = "udp://0.0.0.0:5000?fifo_size=1000000&overrun_nonfatal=1"

cmd = [
    "ffmpeg",
    "-hide_banner",
    "-loglevel", "warning",
    "-i", URL,
    "-an",
    "-vf", f"scale={W}:{H}",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-"
]

# Detection tuning (dock)
WARMUP_SEC = 8
LEARNING_RATE_WARMUP = -1     # let MOG2 adapt during warmup
LEARNING_RATE_FROZEN = 0      # freeze background after warmup

MOTION_PIXELS_MIN = 800       # ignore frames with too little foreground

MIN_AREA = 250
MAX_AREA = 30000
MIN_ASPECT = 1.2
MAX_ASPECT = 8.0

PRESENT_STREAK_ON = 3         # require consecutive-ish frames to turn ON
PRESENT_STREAK_OFF = 1        # hysteresis to turn OFF (lower = less sticky)

COOLDOWN_SEC = 2.0            # min time between "FISH EVENT" logs

DEBUG_SHOW_MASKS = True       # shows roi_gray_eq + fg_mask windows
DEBUG_DRAW_REJECTED = True    # draw rejected blobs in yellow

# ROI: crop borders (cone dominates)
x0, y0 = int(0.08 * W), int(0.08 * H)
x1, y1 = int(0.92 * W), int(0.92 * H)

# Models / kernels (init once)
fgbg = cv2.createBackgroundSubtractorMOG2(history=1500, varThreshold=20, detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# CLAHE created ONCE (was previously inside loop)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def log_event(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


log_event("Starting FFmpeg decode...")
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)

frame_bytes = W * H * 3
frames = 0
t0 = time.time()
last_print = t0

# Warmup timer should start BEFORE loop and persist
start_time = time.time()

# Presence state
present_streak = 0
fish_present = 0
last_event_time = 0.0

try:
    while True:
        raw = p.stdout.read(frame_bytes)
        if len(raw) != frame_bytes:
            err = p.stderr.read().decode("utf-8", errors="ignore")
            log_event(f"Stream ended / decode error. stderr tail:\n{err[-1500:]}")
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3)).copy()
        frames += 1

        # ROI crop
        roi = frame[y0:y1, x0:x1]

        # Preprocess: grayscale -> mild blur -> CLAHE
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray_eq = clahe.apply(gray)

        # Background subtract: warmup then freeze
        elapsed = time.time() - start_time
        lr = LEARNING_RATE_WARMUP if elapsed < WARMUP_SEC else LEARNING_RATE_FROZEN
        mask = fgbg.apply(gray_eq, learningRate=lr)

        # Binarize + morphology cleanup
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        if DEBUG_SHOW_MASKS:
            cv2.imshow("roi_gray_eq", gray_eq)
            cv2.imshow("fg_mask", mask)

        # Motion "energy" gate
        motion_pixels = cv2.countNonZero(mask)
        if motion_pixels < MOTION_PIXELS_MIN:
            contours = []
        else:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Contour filtering
        fish_like_this_frame = 0

        for c in contours:
            area = cv2.contourArea(c)

            if area < MIN_AREA or area > MAX_AREA:
                if DEBUG_DRAW_REJECTED and area > 80:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x0 + x, y0 + y), (x0 + x + w, y0 + y + h), (0, 255, 255), 1)
                continue

            x, y, w, h = cv2.boundingRect(c)
            if h <= 0:
                continue

            aspect = w / float(h)
            if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                if DEBUG_DRAW_REJECTED:
                    cv2.rectangle(frame, (x0 + x, y0 + y), (x0 + x + w, y0 + y + h), (0, 255, 255), 1)
                continue

            # accepted blob
            fish_like_this_frame += 1
            cv2.rectangle(frame, (x0 + x, y0 + y), (x0 + x + w, y0 + y + h), (0, 255, 0), 2)

        # Temporal presence hysteresis (replaces deque window)
        if fish_like_this_frame > 0:
            present_streak += 1
        else:
            present_streak = max(0, present_streak - 1)

        if fish_present == 0:
            fish_present = 1 if present_streak >= PRESENT_STREAK_ON else 0
        else:
            fish_present = 0 if present_streak <= PRESENT_STREAK_OFF else 1

        # Debounced event logging
        now = time.time()
        if fish_present and (now - last_event_time) > COOLDOWN_SEC:
            last_event_time = now
            log_event(f"FISH EVENT (fish_like={fish_like_this_frame}, motion_pixels={motion_pixels})")

        # Overlay
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 1)
        cv2.putText(
            frame,
            f"fish_like: {fish_like_this_frame}  present: {int(fish_present)}  mpix:{motion_pixels}  lr:{lr}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        # FPS print 
        if time.time() - last_print >= 1.0:
            fps = frames / (time.time() - t0)
            print(f"Frames: {frames} | Avg FPS: {fps:.1f}")
            last_print = time.time()

        cv2.imshow("Dock Inference v1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    try:
        p.terminate()
    except Exception:
        pass
    cv2.destroyAllWindows()
    log_event("Exited cleanly.")
