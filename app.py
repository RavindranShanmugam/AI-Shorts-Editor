#!/usr/bin/env python3
"""
AI Short Editor — Streamlit Web App
Upload a long video, click a button, get short clips back. Powered by gemma3 (local).

Run with:
    streamlit run app.py
"""

import os
import json
import subprocess
import tempfile
import urllib.request
import urllib.error
from pathlib import Path

import streamlit as st

try:
    import whisper
except ImportError:
    st.error("Whisper not installed. Run: pip install openai-whisper")
    st.stop()


# ── Config ───────────────────────────────────────────────────────────────────
OLLAMA_HOST    = "http://localhost:11434"
OLLAMA_MODEL   = "gemma3"
SHORT_MAX_SECS = 60
SHORT_MIN_SECS = 15
SILENCE_DB     = -35
SILENCE_GAP    = 0.5


# ── Ollama helpers ─────────────────────────────────────────────────────────────
def ollama_is_running() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def ollama_list_models() -> list:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as r:
            data = json.loads(r.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def ollama_chat(prompt: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3}
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read()).get("response", "").strip()
    except urllib.error.URLError:
        return ""


# ── Core processing steps ──────────────────────────────────────────────────────
def extract_audio(video_path: str, output_path: str):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
           "-ar", "16000", "-ac", "1", output_path]
    subprocess.run(cmd, capture_output=True, text=True)


def transcribe(audio_path: str, whisper_model: str) -> dict:
    model = whisper.load_model(whisper_model)
    result = model.transcribe(audio_path, verbose=False)
    return {"segments": result.get("segments", []), "text": result.get("text", "")}


def detect_best_moments(transcript_data: dict, num_clips: int, model: str) -> list:
    segments_summary = [
        {"id": s["id"], "start": round(s["start"], 2),
         "end": round(s["end"], 2), "text": s["text"].strip()}
        for s in transcript_data["segments"]
    ]
    prompt = f"""You are a YouTube Shorts editor. Pick the {num_clips} most engaging moments for YouTube Shorts (15-60 seconds each).

IMPORTANT: Return ONLY a raw JSON array. No explanation, no markdown, no backticks.
Format:
[
  {{"clip_number": 1, "start": 12.5, "end": 47.2, "reason": "Strong hook", "suggested_title": "How to do X"}}
]

Transcript segments:
{json.dumps(segments_summary[:80], indent=2)}

Full transcript (first 2500 chars):
{transcript_data['text'][:2500]}

Return ONLY the JSON array now:"""

    raw = ollama_chat(prompt, model)
    if not raw:
        return []
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                raw = part
                break
    s, e = raw.find("["), raw.rfind("]")
    if s != -1 and e != -1:
        raw = raw[s:e + 1]
    try:
        moments = json.loads(raw)
        valid = []
        for m in moments:
            dur = m["end"] - m["start"]
            if dur < SHORT_MIN_SECS:
                m["end"] = m["start"] + SHORT_MIN_SECS
            if dur > SHORT_MAX_SECS:
                m["end"] = m["start"] + SHORT_MAX_SECS
            valid.append(m)
        return valid
    except json.JSONDecodeError:
        return []


def get_duration(video_path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def fallback_clips(duration: float, num_clips: int) -> list:
    seg = duration / num_clips
    return [{"clip_number": i + 1, "start": round(i * seg, 2),
             "end": round(min((i + 1) * seg, duration), 2),
             "reason": "Evenly spaced", "suggested_title": f"Clip_{i+1}"}
            for i in range(num_clips)]


def remove_silences(input_path: str, output_path: str):
    cmd = ["ffmpeg", "-y", "-i", input_path, "-af",
           f"silenceremove=start_periods=1:start_silence={SILENCE_GAP}:"
           f"start_threshold={SILENCE_DB}dB:stop_periods=-1:"
           f"stop_silence={SILENCE_GAP}:stop_threshold={SILENCE_DB}dB",
           output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        import shutil
        shutil.copy(input_path, output_path)


def trim_clip(input_path: str, start: float, end: float, output_path: str):
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", input_path,
           "-t", str(end - start), "-c:v", "libx264", "-c:a", "aac",
           "-preset", "fast", output_path]
    subprocess.run(cmd, capture_output=True, text=True)


def generate_srt(segments: list, clip_start: float, clip_end: float, output_srt: str):
    clip_segs = [s for s in segments if s["start"] >= clip_start and s["end"] <= clip_end + 1]

    def fmt(t):
        h, m, s, ms = int(t // 3600), int((t % 3600) // 60), int(t % 60), int((t % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(clip_segs, 1):
        rs = max(0, seg["start"] - clip_start)
        re = max(0, seg["end"] - clip_start)
        lines += [str(i), f"{fmt(rs)} --> {fmt(re)}", seg["text"].strip(), ""]
    with open(output_srt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def burn_captions(input_path: str, srt_path: str, output_path: str):
    safe_srt = str(srt_path).replace("\\", "/").replace(":", "\\:")
    cmd = ["ffmpeg", "-y", "-i", input_path, "-vf",
           f"subtitles='{safe_srt}':force_style='FontSize=20,PrimaryColour=&HFFFFFF,"
           f"OutlineColour=&H000000,Outline=2,Alignment=2'",
           "-c:a", "copy", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        import shutil
        shutil.copy(input_path, output_path)


def safe_filename(name: str) -> str:
    """Remove characters Windows forbids in filenames: < > : " / \\ | ? *"""
    import re
    cleaned = re.sub(r'[<>:"/\\|?*]', "", name)   # strip illegal chars
    cleaned = cleaned.replace(" ", "_").strip("._")
    return cleaned[:40] or "clip"


# ════════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="AI Short Editor", page_icon="🎬", layout="centered")

st.title("🎬 AI Short Editor")
st.caption("Upload a long video → get YouTube Shorts. Powered by gemma3, running locally on your machine.")

# Sidebar — settings
with st.sidebar:
    st.header("⚙️ Settings")
    num_clips = st.slider("Number of shorts", 1, 6, 3)
    whisper_model = st.selectbox(
        "Transcription quality",
        ["base", "small", "medium"],
        index=0,
        help="'base' is fast. 'small' is more accurate (better for accents). 'medium' is slowest/best."
    )
    do_captions = st.checkbox("Burn captions", value=True)
    do_silence = st.checkbox("Remove silences", value=True)

    st.divider()
    # System status
    st.subheader("System check")
    if ollama_is_running():
        models = ollama_list_models()
        if any("gemma3" in m for m in models):
            st.success("✅ Ollama + gemma3 ready")
        else:
            st.warning(f"⚠️ gemma3 not found. Models: {models or 'none'}")
            st.code("ollama pull gemma3")
    else:
        st.error("❌ Ollama not running. Open the Ollama app.")

# Main — upload
uploaded = st.file_uploader(
    "Upload your video",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    help="Start with a short video (2-4 min) for your first test."
)

if uploaded is not None:
    st.video(uploaded)

    if st.button("✨ Generate Shorts", type="primary", use_container_width=True):
        if not ollama_is_running():
            st.error("Ollama isn't running. Open the Ollama app and try again.")
            st.stop()

        # Work in a temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            video_path = str(tmp / uploaded.name)
            with open(video_path, "wb") as f:
                f.write(uploaded.getbuffer())

            progress = st.progress(0, text="Starting...")

            # 1. Extract audio
            progress.progress(10, text="🎬 Extracting audio...")
            audio_path = str(tmp / "audio.wav")
            extract_audio(video_path, audio_path)

            # 2. Transcribe
            progress.progress(25, text=f"🎙️ Transcribing ({whisper_model})... this can take a bit")
            transcript_data = transcribe(audio_path, whisper_model)

            # 3. Detect best moments
            progress.progress(50, text="🦙 Asking gemma3 to find the best moments...")
            moments = detect_best_moments(transcript_data, num_clips, OLLAMA_MODEL)
            if not moments:
                st.info("AI moment detection didn't return clips — using evenly spaced fallback.")
                moments = fallback_clips(get_duration(video_path), num_clips)

            # 4. Build each clip
            output_dir = Path.cwd() / "shorts"
            output_dir.mkdir(exist_ok=True)
            results = []

            total = len(moments)
            for idx, moment in enumerate(moments):
                pct = 50 + int((idx / total) * 45)
                n = moment["clip_number"]
                title = safe_filename(str(moment.get("suggested_title", f"clip_{n}")))
                progress.progress(pct, text=f"✂️ Building clip {n} of {total}: {title}")

                start, end = moment["start"], moment["end"]

                raw_clip = str(tmp / f"clip_{n}_raw.mp4")
                trim_clip(video_path, start, end, raw_clip)

                if do_silence:
                    clean_clip = str(tmp / f"clip_{n}_clean.mp4")
                    remove_silences(raw_clip, clean_clip)
                else:
                    clean_clip = raw_clip

                final_clip = str(output_dir / f"short_{n}_{title}.mp4")
                if do_captions:
                    srt_path = str(tmp / f"clip_{n}.srt")
                    generate_srt(transcript_data["segments"], start, end, srt_path)
                    burn_captions(clean_clip, srt_path, final_clip)
                else:
                    import shutil
                    shutil.copy(clean_clip, final_clip)

                # Read bytes for download + preview
                with open(final_clip, "rb") as f:
                    clip_bytes = f.read()
                results.append({
                    "title": title,
                    "reason": moment.get("reason", ""),
                    "duration": round(end - start, 1),
                    "bytes": clip_bytes,
                    "filename": Path(final_clip).name,
                })

            progress.progress(100, text="✅ Done!")

        # Show results
        st.success(f"Generated {len(results)} shorts! 🎉")
        for r in results:
            st.divider()
            st.subheader(f"📹 {r['title']}")
            st.caption(f"{r['duration']}s — {r['reason']}")
            st.video(r["bytes"])
            st.download_button(
                f"⬇️ Download {r['filename']}",
                data=r["bytes"],
                file_name=r["filename"],
                mime="video/mp4",
                use_container_width=True,
            )

else:
    st.info("👆 Upload a video to get started.")
