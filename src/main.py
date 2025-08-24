import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

import RPi.GPIO as GPIO
import datetime

import shutil, subprocess

BASE_LOG_DIR = Path(os.environ.get("LOG_DIR", "/home/rizki/yolo/logs")).resolve()
SESSION_TS = datetime.datetime.now()
SESSION_STAMP = SESSION_TS.strftime("%Y-%m-%d_%H-%M-%S")
SESSION_DATE = datetime.datetime.now().strftime("%Y-%m-%d")
SESSION_DIR = BASE_LOG_DIR / SESSION_TS.strftime("%Y-%m-%d") / f"session_{SESSION_STAMP}"
SNAP_DIR = SESSION_DIR / "snapshots"
SESSION_DIR.mkdir(parents=True, exist_ok=True)
SNAP_DIR.mkdir(parents=True, exist_ok=True)


# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")',
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Validate model path
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found.')
    sys.exit(1)

# Load model and labels
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input source type
img_ext_list = ['.jpg','.jpeg','.png','.bmp']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext_list:
        source_type = 'image'
    elif ext.lower() in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Unsupported file extension: {ext}')
        sys.exit(1)
elif img_source.startswith('usb'):
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif img_source.startswith('picamera'):
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Invalid input source: {img_source}')
    sys.exit(1)

# Handle resolution
resize = False
if user_res:
    try:
        resW, resH = map(int, user_res.split('x'))
        resize = True
    except:
        print('Invalid resolution format. Use WxH (e.g., 640x480)')
        sys.exit(1)

# Video recording setup
# recorder = None
# if record:
#     if source_type not in ['video','usb']:
#         print('Recording only supported for video and camera sources.')
#         sys.exit(1)
#     if not user_res:
#         print('Resolution required for recording.')
#         sys.exit(1)

#     record_name = str(SESSION_DIR / f'record_{SESSION_STAMP}.mp4')
#     record_fps = 30
#     recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))
# Video recording setup
recorder = None
record_name_base = (SESSION_DIR / f"record_{SESSION_STAMP}")
record_fps = 30
record_mp4_path = str(record_name_base.with_suffix(".mp4"))
record_avi_path = None
NEED_CONVERT = False

if record:
    if source_type not in ['video','usb','picamera']:
        print('Recording only supported for video, camera, or picamera sources.')
        sys.exit(1)
    if not user_res:
        print('Resolution required for recording.')
        sys.exit(1)

    # 1) Coba langsung MP4 (H.264)
    fourcc_h264 = cv2.VideoWriter_fourcc(*'avc1')  # H.264 alias
    recorder = cv2.VideoWriter(record_mp4_path, fourcc_h264, record_fps, (resW, resH))

    if not recorder.isOpened():
        # 2) Fallback: rekam AVI (MJPG) lalu konversi setelah sesi
        print("OpenCV H.264 encoder unavailable. Falling back to AVI (MJPG) and will convert to MP4 at the end.")
        fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')
        record_avi_path = str(record_name_base.with_suffix(".avi"))
        recorder = cv2.VideoWriter(record_avi_path, fourcc_mjpg, record_fps, (resW, resH))
        NEED_CONVERT = True

    if not recorder.isOpened():
        print("ERROR: cannot open VideoWriter for recording.")
        sys.exit(1)

# Initialize image source
imgs_list = []
cap = None

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = glob.glob(os.path.join(img_source, '*'))
    imgs_list = [f for f in imgs_list if os.path.splitext(f)[1].lower() in img_ext_list]
elif source_type in ['video', 'usb']:
    cap_arg = usb_idx if source_type == 'usb' else img_source
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    config = cap.create_video_configuration(main={"size": (resW, resH)})
    cap.configure(config)
    cap.start()

# Bounding box colors
bbox_colors = [
    (164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
    (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)
]

# Performance tracking
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Relay setup
RELAY_PIN = 17
# RELAY_ON_DURATION = 5.0  # seconds
# RELAY_OFF_DURATION = 3.0  # seconds

RELAY_ON_DELAY_SEC  = float(os.environ.get("RELAY_ON_DELAY_SEC",  "0.0"))  # ada burung: tahan 1s baru ON
RELAY_OFF_DELAY_SEC = float(os.environ.get("RELAY_OFF_DELAY_SEC", "3.0"))  # tidak ada: tahan 3s baru OFF

GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.HIGH)  # Start with relay OFF

relay_active = False
# relay_last_change = time.time()

present_since = None   # waktu pertama kali terdeteksi (kontinu)
absent_since  = None   # waktu pertama kali tidak terdeteksi (kontinu)

# Logging setup
log_path = SESSION_DIR / f"relay_log_{SESSION_STAMP}.txt"
log_file = open(log_path, "a", buffering=1)  # line-buffered
log_file.write(
    f"\n\n{'='*50}\n"
    f"Session started: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"
    f"{'='*50}\n"
)
log_file.flush()
OBJECT_STABLE_SEC = float(os.environ.get("OBJECT_STABLE_SEC", "0.7"))  # butuh stabil 0.7s
SAVE_EVERY_SEC = float(os.environ.get("SAVE_EVERY_SEC", "5"))  # simpan tiap 0.5s saat detect
last_save_ts = 0.0
last_object_count = None
reported_object_count = None  # count yang sudah "disahkan"
pending_count = None
pending_since = 0.0

# Main processing loop
try:
    while True:
        t_start = time.perf_counter()

        # Read frame
        frame = None
        if source_type in ['image', 'folder']:
            if img_count >= len(imgs_list):
                break
            frame = cv2.imread(imgs_list[img_count])
            if frame is None:
                print(f"Failed to read image: {imgs_list[img_count]}")
                img_count += 1
                continue
            img_count += 1
        elif source_type == 'video':
            ret, frame = cap.read()
            if not ret:
                break
        elif source_type == 'usb':
            ret, frame = cap.read()
            if not ret or frame is None:
                break
        elif source_type == 'picamera':
            frame = cap.capture_array()
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                break

        # Resize if needed
        if resize and frame is not None:
            frame = cv2.resize(frame, (resW, resH))

        # Skip empty frames
        if frame is None or frame.size == 0:
            continue

        # Run inference
        results = model(frame, verbose=False, conf=min_thresh)
        detections = results[0].boxes
        object_count = 0
        bird_detected = False

        # Process detections
        for i in range(len(detections)):
            xyxy = detections[i].xyxy.cpu().numpy().squeeze()
            if xyxy.size < 4:
                continue

            xmin, ymin, xmax, ymax = map(int, xyxy[:4])
            class_idx = int(detections[i].cls.item())
            class_name = labels.get(class_idx, f"Class{class_idx}")
            conf = detections[i].conf.item()

            # Check if detection meets confidence threshold
            if class_idx == 0 and conf >= min_thresh:
                class_name = "bird"
                color = bbox_colors[class_idx % len(bbox_colors)]

                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                # Draw label
                label = f'{class_name}: {conf:.2f}'
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (xmin, ymin - 20), (xmin + w, ymin), color, -1)
                cv2.putText(frame, label, (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

                object_count += 1

                # Check for bird detection
                if "bird" in class_name.lower():
                    bird_detected = True

        if last_object_count is None or object_count != last_object_count:
           log_file.write(f"[{datetime.datetime.now()}] objects={object_count}\n")
           log_file.flush()
           last_object_count = object_count  

        current_time = time.time()
        if bird_detected:
            if present_since is None:
                present_since = current_time
            absent_since = None
        else:
            if absent_since is None:
                absent_since = current_time
            present_since = None

        # Relay control logic
        # current_time = time.time()
        # elapsed = current_time - relay_last_change

        # if bird_detected:
        #     if not relay_active and elapsed >= RELAY_OFF_DURATION:
        #         GPIO.output(RELAY_PIN, GPIO.LOW)  # Relay ON
        #         relay_active = True
        #         relay_last_change = current_time
        #         log_msg = f"[{datetime.datetime.now()}] Relay=ON Reason=Bird detected Objects={object_count}\n"
        #         log_file.write(log_msg)
        #         log_file.flush()
        #         print(log_msg.strip())

        #     if (current_time - last_save_ts) >= SAVE_EVERY_SEC:
        #         img_filename = str(SNAP_DIR / f"bird_{int(current_time)}.jpg")
        #         cv2.imwrite(img_filename, frame)
        #         last_save_ts = current_time
        # else:
        #     if relay_active and elapsed >= RELAY_ON_DURATION:
        #         GPIO.output(RELAY_PIN, GPIO.HIGH)  # Relay OFF
        #         relay_active = False
        #         relay_last_change = current_time
        #         log_msg = f"[{datetime.datetime.now()}]Relay=OFF Objects={object_count}\n"
        #         log_file.write(log_msg)
        #         log_file.flush()
        #         print(log_msg.strip())

        if bird_detected and (not relay_active) and (present_since is not None) and ((current_time - present_since) >= RELAY_ON_DELAY_SEC):
            GPIO.output(RELAY_PIN, GPIO.LOW)  # Relay ON (aktif)
            relay_active = True
            log_msg = f"[{datetime.datetime.now()}] Relay=ON Reason=Bird detected Objects={object_count}\n"
            log_file.write(log_msg); log_file.flush()
            print(log_msg.strip())

        # OFF: jika tidak terdeteksi terus-menerus >= RELAY_OFF_DELAY_SEC
        if (not bird_detected) and relay_active and (absent_since is not None) and ((current_time - absent_since) >= RELAY_OFF_DELAY_SEC):
            GPIO.output(RELAY_PIN, GPIO.HIGH)  # Relay OFF (non-aktif)
            relay_active = False
            log_msg = f"[{datetime.datetime.now()}] Relay=OFF Objects={object_count}\n"
            log_file.write(log_msg); log_file.flush()
            print(log_msg.strip())

        # Snapshot tetap mengikuti status deteksi (bukan status relay)
        if bird_detected and (current_time - last_save_ts) >= SAVE_EVERY_SEC:
            img_filename = str(SNAP_DIR / f"bird_{int(current_time)}.jpg")
            cv2.imwrite(img_filename, frame)
            last_save_ts = current_time

        # Display info
        fps_text = f'FPS: {avg_frame_rate:.2f}'
        count_text = f'Objects: {object_count}'
        relay_status = f'Relay: {"ON" if relay_active else "OFF"}'

        cv2.putText(frame, fps_text, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, count_text, (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, relay_status, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if relay_active else (0, 0, 255), 2)

        # Show output
        cv2.imshow('Bird Detection', frame)
        if record and recorder is not None:
            recorder.write(frame)

        # Handle keypress
        key = cv2.waitKey(1 if source_type in ['video', 'usb', 'picamera'] else 0)
        if key in [ord('q'), ord('Q')]:
            break
        elif key in [ord('s'), ord('S')]:
            cv2.waitKey(0)
        elif key in [ord('p'), ord('P')]:
            cv2.imwrite(f'capture_{int(time.time())}.png', frame)

        # Calculate FPS
        t_stop = time.perf_counter()
        frame_rate = 1.0 / (t_stop - t_start)
        frame_rate_buffer.append(frame_rate)
        if len(frame_rate_buffer) > fps_avg_len:
            frame_rate_buffer.pop(0)
        avg_frame_rate = np.mean(frame_rate_buffer)

except KeyboardInterrupt:
    print("\nProcess interrupted by user")


finally:
    def convert_to_mp4(in_path, out_path):
        ff = shutil.which("ffmpeg")
        if not ff:
            print("WARNING: ffmpeg not found; MP4 not created.")
            return False

    # Coba libx264 dulu, kalau gagal coba h264_v4l2m2m (hardware Pi)
        attempts = [
            [ff, "-y", "-i", in_path, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", out_path],
            [ff, "-y", "-i", in_path, "-c:v", "h264_v4l2m2m", "-pix_fmt", "yuv420p",
            "-an", "-movflags", "+faststart", out_path],
        ]
        for cmd in attempts:
            try:
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)
                return True
            except subprocess.CalledProcessError:
                continue
        print("ERROR: ffmpeg conversion failed with all encoders.")
        return False

    # Cleanup
    print(f'Log dan artefak sesi ini tersimpan di: {SESSION_DIR}, Average FPS: {avg_frame_rate:.2f}')

    if cap is not None:
        if source_type == 'picamera':
            cap.stop()
        else:
            cap.release()

    if recorder is not None:
        recorder.release()

    if record:
        if NEED_CONVERT and record_avi_path and os.path.exists(record_avi_path) and os.path.getsize(record_avi_path) > 0:
            ok = convert_to_mp4(record_avi_path, record_mp4_path)
            if ok:
                try:
                    os.remove(record_avi_path)
                except Exception as e:
                    print("WARN: failed to remove temp AVI:", e)
                print(f"MP4 ready for browser: {record_mp4_path}")
            else:
                print(f"Keep AVI at: {record_avi_path} (browser may not play it)")
        else:
            if os.path.exists(record_mp4_path):
                print(f"MP4 ready for browser: {record_mp4_path}") 

    cv2.destroyAllWindows()

    GPIO.output(RELAY_PIN, GPIO.HIGH)
    GPIO.cleanup()

    log_file.write(
    f"\n{'='*50}\n"
    f"Session ended: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"
    f"{'='*50}\n\n"
    )
    log_file.close()
    print("Resources released and log saved")