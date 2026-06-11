# 🎬 AI Short Editor

Turn long-form videos into YouTube Shorts automatically — running **100% locally and free** using a local LLM (no API costs).

Upload a video, and the tool transcribes it, uses a local AI model to pick the most engaging moments, trims them, removes silences, burns in captions, and gives you ready-to-post short clips.

Built as part of my AI learning journey, exploring the difference between simple LLM scripts and tool-calling agents.

---

## ✨ What it does

- 🎙️ **Transcribes** your video locally with OpenAI Whisper
- 🦙 **Finds the best moments** using a local LLM (gemma3 via Ollama) — no API key, no cost
- ✂️ **Trims** each moment into a short clip
- 🔇 **Removes silences** automatically
- 📝 **Burns captions** directly onto the video
- ⬇️ **Download** finished shorts from a simple web interface

---

## 🧰 What's in this repo

| File | What it is |
|---|---|
| `app.py` | The main web app (Streamlit) — upload, click, download |
| `editor_ollama.py` | Command-line version of the same editor |
| `agent_example.py` | A minimal tool-calling AI agent (learning example — shows how an agent loop works) |
| `requirements.txt` | Python dependencies |

---

## ⚙️ Setup

### 1. Install Ollama and pull a model
Download [Ollama](https://ollama.com), then pull a model:
```bash
ollama pull gemma3
```

### 2. Install ffmpeg
ffmpeg handles all the video/audio work.
- **Windows:** `winget install ffmpeg` (or download from [ffmpeg.org](https://ffmpeg.org/download.html))
- **Mac:** `brew install ffmpeg`

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Web app (recommended)
```bash
streamlit run app.py
```
Your browser opens automatically. Upload a video, choose how many shorts you want, and click **Generate Shorts**.

> Tip: if the `streamlit` command isn't found, use `python -m streamlit run app.py`.

### Command line
```bash
python editor_ollama.py "path/to/your/video.mp4" -n 3
```

---

## 🛠️ Settings you can tweak

In the web app sidebar (or at the top of the scripts):

- **Number of shorts** — how many clips to generate (1–6)
- **Transcription quality** — `base` (fast) / `small` (better for accents) / `medium` (slowest, best)
- **Burn captions** — on/off
- **Remove silences** — on/off

---

## 💡 Notes

- Start with a **short video (2–4 min)** for your first run — transcription time scales with length.
- The first run downloads the Whisper model (one-time, ~1 min).
- Everything runs on your own machine. Nothing is uploaded to the cloud.

---

## 🧠 About the agent example

`agent_example.py` is a tiny, self-contained demo of an **AI agent with a tool-calling loop** — where the LLM decides which tools to call and in what order, looping until the task is done. It's separate from the video editor (which is a fixed-pipeline *script*, not an agent) and is included to illustrate the difference between the two approaches.

---

*Built locally. No API costs. Part of my ongoing AI journey.*
