import sqlite3, subprocess, os, json
from pathlib import Path
from tqdm import tqdm

# ── 설정 ───────────────────────────────────────────────
DB_PATH      = "data/monopoly.sqlite"
RAW_VIDEO_DIR = Path("data/raw_videos")   # video_id.mp4
WAV_OUT_DIR   = Path("data/wav")          # wav/<video_id>/<segment_id>.wav
FRAME_OUT_DIR = Path("data/frames")       # frames/<video_id>/<segment_id>/fXXXXXX.jpg
FPS          = 1                          # 프레임 추출 간격 (1 fps)

WAV_OUT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── ffmpeg helper ─────────────────────────────────────
def run_ffmpeg(cmd: list[str]):
    r = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[:500])

def extract_wav(video_file: Path, start: float, end: float, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-ss", str(start), "-to", str(end),
        "-i", str(video_file),
        "-ar", "16000", "-ac", "1",
        str(out_path)
    ])

def extract_frames(video_file: Path, start: float, end: float, out_dir: Path, fps: int = FPS):
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-ss", str(start), "-to", str(end),
        "-i", str(video_file),
        "-vf", f"fps={fps}",
        str(out_dir / "f%06d.jpg")
    ])

# ── 메인 전처리 ─────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

segments = conn.execute(
    "SELECT segment_id, video_id, start_time, end_time "
    "FROM speech_segments"
).fetchall()

for seg in tqdm(segments, desc="processing segments"):
    video_id   = seg["video_id"]
    segment_id = seg["segment_id"]
    start_t    = seg["start_time"]
    end_t      = seg["end_time"]

    video_file = RAW_VIDEO_DIR / f"{video_id}.mp4"
    if not video_file.exists():
        print(f"!! missing raw video {video_file}")
        continue

    # ❷ 오디오 추출
    wav_path = WAV_OUT_DIR / video_id / f"{segment_id}.wav"
    if not wav_path.exists():
        extract_wav(video_file, start_t, end_t, wav_path)

    # ❸ 프레임 추출
    frame_dir = FRAME_OUT_DIR / video_id / segment_id
    if not frame_dir.exists() or not any(frame_dir.iterdir()):
        extract_frames(video_file, start_t, end_t, frame_dir)

print("DONE.")
