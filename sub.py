import cv2
import threading
import subprocess
import numpy as np
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw, ImageFont

VIDEO_PATH = "test2.mp4"

# Whisper GPU
model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16"
)

# 試用 Windows 常見中文字型
FONT_PATHS = [
    r"C:\Windows\Fonts\msjh.ttc",
    r"C:\Windows\Fonts\msjhbd.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\mingliu.ttc",
]


def load_font(size=32):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


font = load_font(32)
subtitle_segments = []
segments_lock = threading.Lock()


# =========================
# 音訊辨識 Thread
# =========================
def transcribe_audio():
    command = [
        "ffmpeg",
        "-i",
        VIDEO_PATH,
        "-vn",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-"
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    audio_data = process.stdout.read()
    process.wait()
    if not audio_data:
        return

    audio_np = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio_np, language=None)

    with segments_lock:
        subtitle_segments.clear()
        for seg in segments:
            text = seg.text.strip()
            try:
                start = float(seg.start)
                end = float(seg.end)
            except AttributeError:
                continue
            if text:
                subtitle_segments.append((start, end, text))


def get_text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        try:
            return draw.textsize(text, font=font)
        except AttributeError:
            return font.getsize(text)


def draw_subtitle(frame, subtitle):
    subtitle_height = 100
    subtitle_bg = np.zeros((subtitle_height, frame.shape[1], 3), dtype=np.uint8)
    pil_bar = Image.fromarray(cv2.cvtColor(subtitle_bg, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_bar)

    text = subtitle.strip()
    max_width = frame.shape[1] - 60
    lines = []
    current = ""
    for char in text:
        test = current + char
        width, _ = get_text_size(draw, test, font)
        if width > max_width and current:
            lines.append(current)
            current = char
        else:
            current = test
    if current:
        lines.append(current)

    line_sizes = [get_text_size(draw, line, font) for line in lines]
    line_height = max(size[1] for size in line_sizes) if line_sizes else 0
    total_height = line_height * len(lines) + 10 * (len(lines) - 1)

    rect_width = frame.shape[1]
    rect_height = subtitle_height
    draw.rectangle([0, 0, rect_width, rect_height], fill=(0, 0, 0))

    y = (subtitle_height - total_height) // 2
    for i, line in enumerate(lines):
        line_width, _ = get_text_size(draw, line, font)
        x = (frame.shape[1] - line_width) // 2
        draw.text((x, y + i * (line_height + 10)), line, font=font, fill=(255, 255, 255))

    subtitle_bar = cv2.cvtColor(np.array(pil_bar), cv2.COLOR_RGB2BGR)
    return np.vstack((frame, subtitle_bar))


def get_subtitle_for_time(time_sec):
    with segments_lock:
        for start, end, text in subtitle_segments:
            if start <= time_sec < end:
                return text
    return ""


# 啟動辨識執行緒：在背景執行，不阻塞主線程
print("正在辨識字幕，請稍候...")
transcribe_thread = threading.Thread(target=transcribe_audio, daemon=False)
transcribe_thread.start()

# 等待轉錄完成
transcribe_thread.join()
print("字幕辨識完成，開始播放影片")

# =========================
# 播影片
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

delay = int(1000 / fps)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    current_subtitle = get_subtitle_for_time(current_time)

    frame = draw_subtitle(frame, current_subtitle)
    cv2.imshow("AI Subtitle Player", frame)

    key = cv2.waitKey(delay)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()