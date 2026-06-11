#!/usr/bin/env python3
"""
AI Short Editor — CLI tool for YouTube Shorts
Uses Whisper (transcription) + Ollama local LLM (moment detection) + ffmpeg (editing)
100% free, runs offline after setup.
"""

import os
import sys
import json
import subprocess
import argparse
import tempfile
import urllib.request
import urllib.error
from pathlib import Path

# ── Dependencies check ──────────────────────────────────────────────────────
try:
    import whisper
except ImportError:
    print("❌ whisper not installed. Run: pip install openai-whisper")
    sys.exit(1)


# ── Config ──────────────────────────────────────────────────────────────────
OLLAMA_HOST    = "http://localhost:11434"   # default Ollama address
OLLAMA_MODEL   = "gemma3"                   # change to llama3.2, mistral, etc.
SHORT_MAX_SECS = 60
SHORT_MIN_SECS = 15
SILENCE_DB     = -35
SILENCE_GAP    = 0.5
WHISPER_MODEL  = "base"                     # base / small / medium / large


# ── Ollama helpers ───────────────────────────────────────────────────────────
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


def ollama_chat(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Send a prompt to Ollama and return the response text."""
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
            data = json.loads(r.read())
            return data.get("response", "").strip()
    except urllib.error.URLError as e:
        print(f"   ❌ Ollama request failed: {e}")
        return ""


def check_ollama_setup(model: str):
    """Verify Ollama is running and the model is available."""
    print(f"\n🦙 Checking Ollama setup...")

    if not ollama_is_running():
        print("❌ Ollama is not running!")
        print("   Start it with:  ollama serve")
        print("   Or open the Ollama desktop app.")
        sys.exit(1)

    available = ollama_list_models()
    model_base = model.split(":")[0]
    matched = [m for m in available if model_base in m]

    if not matched:
        print(f"⚠️  Model '{model}' not found locally.")
        print(f"   Available models: {available or 'none pulled yet'}")
        print(f"\n   Pull it now with:")
        print(f"   ollama pull {model}")
        print(f"\n   Recommended free models:")
        print(f"   ollama pull llama3.2        ← fast, good quality (2GB)")
        print(f"   ollama pull llama3.1        ← smarter, slower (4GB)")
        print(f"   ollama pull mistral         ← great for structured JSON")
        print(f"   ollama pull phi3            ← lightweight, fast (2GB)")
        sys.exit(1)

    print(f"   ✅ Ollama running | Model: {model} | Available: {len(available)} model(s)")


# ── Step 1: Extract audio ────────────────────────────────────────────────────
def extract_audio(video_path: str, output_path: str) -> str:
    print(f"\n🎬 Extracting audio from: {Path(video_path).name}")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ ffmpeg error:\n{result.stderr}")
        sys.exit(1)
    print(f"   ✅ Audio extracted")
    return output_path


# ── Step 2: Transcribe with Whisper ─────────────────────────────────────────
def transcribe(audio_path: str) -> dict:
    print(f"\n🎙️  Transcribing with Whisper ({WHISPER_MODEL})...")
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_path, verbose=False)
    segments = result.get("segments", [])
    full_text = result.get("text", "")
    print(f"   ✅ {len(segments)} segments | {len(full_text.split())} words")
    return {"segments": segments, "text": full_text}


# ── Step 3: Detect best moments via Ollama ───────────────────────────────────
def detect_best_moments(transcript_data: dict, num_clips: int, model: str) -> list:
    print(f"\n🦙 Asking {model} to detect best {num_clips} moments...")

    segments_summary = [
        {"id": s["id"], "start": round(s["start"], 2),
         "end": round(s["end"], 2), "text": s["text"].strip()}
        for s in transcript_data["segments"]
    ]

    prompt = f"""You are a YouTube Shorts editor. Analyze this transcript and pick the {num_clips} most engaging moments for YouTube Shorts (15–60 seconds each).

Pick moments that have:
- A strong hook in the first 3 seconds
- A clear tip, story, or surprising fact
- Energy or emotion

IMPORTANT: Return ONLY a raw JSON array. No explanation. No markdown. No backticks.
Use exactly this format:
[
  {{
    "clip_number": 1,
    "start": 12.5,
    "end": 47.2,
    "reason": "Strong hook with actionable tip",
    "suggested_title": "How to do X fast"
  }}
]

Transcript segments:
{json.dumps(segments_summary[:80], indent=2)}

Full transcript (first 2500 chars):
{transcript_data['text'][:2500]}

Return ONLY the JSON array now:"""

    raw = ollama_chat(prompt, model=model)

    if not raw:
        return []

    # Strip any accidental markdown fences
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                raw = part
                break

    # Extract JSON array if surrounded by extra text
    start_idx = raw.find("[")
    end_idx = raw.rfind("]")
    if start_idx != -1 and end_idx != -1:
        raw = raw[start_idx:end_idx + 1]

    try:
        moments = json.loads(raw)
        # Clamp clip lengths to valid range
        valid = []
        for m in moments:
            dur = m["end"] - m["start"]
            if dur < SHORT_MIN_SECS:
                m["end"] = m["start"] + SHORT_MIN_SECS
            if dur > SHORT_MAX_SECS:
                m["end"] = m["start"] + SHORT_MAX_SECS
            valid.append(m)

        print(f"   ✅ {len(valid)} moments detected")
        for m in valid:
            print(f"      Clip {m['clip_number']}: {m['start']}s–{m['end']}s — {m['reason']}")
        return valid

    except json.JSONDecodeError as e:
        print(f"   ⚠️  Could not parse model response: {e}")
        print(f"   Raw (first 300 chars): {raw[:300]}")
        return []


# ── Step 4: Remove silences ──────────────────────────────────────────────────
def remove_silences(input_path: str, output_path: str):
    print(f"\n✂️  Removing silences...")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-af",
        f"silenceremove=start_periods=1:start_silence={SILENCE_GAP}:"
        f"start_threshold={SILENCE_DB}dB:"
        f"stop_periods=-1:stop_silence={SILENCE_GAP}:"
        f"stop_threshold={SILENCE_DB}dB",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        import shutil
        shutil.copy(input_path, output_path)
        print(f"   ⚠️  Silence removal failed, using original clip")
    else:
        print(f"   ✅ Silences removed")


# ── Step 5: Trim clip ────────────────────────────────────────────────────────
def trim_clip(input_path: str, start: float, end: float, output_path: str):
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", input_path,
        "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac",
        "-preset", "fast",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"   ⚠️  Trim failed: {result.stderr[-150:]}")
    else:
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"   ✅ Trimmed: {duration:.1f}s, {size_mb:.1f}MB")


# ── Step 6: Captions ─────────────────────────────────────────────────────────
def generate_srt(segments: list, clip_start: float, clip_end: float, output_srt: str):
    clip_segs = [s for s in segments if s["start"] >= clip_start and s["end"] <= clip_end + 1]

    def fmt(t):
        h, m, s, ms = int(t//3600), int((t%3600)//60), int(t%60), int((t%1)*1000)
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
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"subtitles='{safe_srt}':force_style='FontSize=20,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,Alignment=2'",
        "-c:a", "copy", output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        import shutil
        shutil.copy(input_path, output_path)
        print(f"   ⚠️  Caption burn failed (install ffmpeg full build). Saving without captions.")
    else:
        print(f"   ✅ Captions burned")


# ── Fallback: evenly spaced clips ────────────────────────────────────────────
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
    return [{"clip_number": i+1, "start": round(i*seg, 2),
             "end": round(min((i+1)*seg, duration), 2),
             "reason": "Evenly spaced fallback", "suggested_title": f"Clip_{i+1}"}
            for i in range(num_clips)]


# ── Main CLI ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="🎬 AI Short Editor (Ollama) — Free local AI to cut YouTube Shorts"
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("-n", "--num-clips", type=int, default=3)
    parser.add_argument("-o", "--output-dir", default="./shorts")
    parser.add_argument("-m", "--model", default=OLLAMA_MODEL,
                        help=f"Ollama model to use (default: {OLLAMA_MODEL})")
    parser.add_argument("--no-captions", action="store_true")
    parser.add_argument("--no-silence-removal", action="store_true")
    args = parser.parse_args()

    video_path = args.video
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    print("=" * 55)
    print("🎬  AI SHORT EDITOR  (Powered by Ollama — 100% Free)")
    print("=" * 55)
    print(f"📁 Input  : {video_path}")
    print(f"📂 Output : {output_dir}")
    print(f"🦙 Model  : {args.model}")
    print(f"🎯 Clips  : {args.num_clips}")
    print("=" * 55)

    # Verify Ollama is ready
    check_ollama_setup(args.model)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        audio_path = str(tmp / "audio.wav")
        extract_audio(video_path, audio_path)

        transcript_data = transcribe(audio_path)

        moments = detect_best_moments(transcript_data, args.num_clips, args.model)

        if not moments:
            print("\n⚠️  AI detection failed — using evenly spaced fallback clips")
            duration = get_duration(video_path)
            moments = fallback_clips(duration, args.num_clips)

        print(f"\n📦 Generating {len(moments)} shorts...\n")
        for moment in moments:
            n = moment["clip_number"]
            start, end = moment["start"], moment["end"]
            title = moment.get("suggested_title", f"clip_{n}").replace(" ", "_")[:40]

            print(f"── Clip {n}: {title} ──")

            raw_clip = str(tmp / f"clip_{n}_raw.mp4")
            trim_clip(video_path, start, end, raw_clip)

            if not args.no_silence_removal:
                clean_clip = str(tmp / f"clip_{n}_clean.mp4")
                remove_silences(raw_clip, clean_clip)
            else:
                clean_clip = raw_clip

            final_clip = str(output_dir / f"short_{n}_{title}.mp4")

            if not args.no_captions:
                srt_path = str(tmp / f"clip_{n}.srt")
                generate_srt(transcript_data["segments"], start, end, srt_path)
                burn_captions(clean_clip, srt_path, final_clip)
            else:
                import shutil
                shutil.copy(clean_clip, final_clip)
                print(f"   ✅ Saved: {Path(final_clip).name}")

        print("\n" + "=" * 55)
        print("✅  DONE! Your shorts:")
        for f in sorted(output_dir.glob("short_*.mp4")):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   📹 {f.name} ({size_mb:.1f}MB)")
        print(f"\n📂 Folder: {output_dir.resolve()}")
        print("=" * 55)


if __name__ == "__main__":
    main()
