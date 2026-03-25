````markdown
## Tested System Configuration

This project was tested on:

- **Operating System:** Ubuntu 22.04 LTS  
- **Python Version:** 3.10  
- **Ollama Version:** (latest as of test)  
- **LLM Model Used:** `llama3.2:1b`  
- **Whisper Backend:** faster-whisper  
- **Whisper Model Size:** small  
- **TTS Model:** Qwen3-TTS (CPU)  

> ⚠️ Runs fully on CPU. GPU is optional but not required.

---

## Features

- Voice input (Whisper)
- Keyboard input (fallback or manual mode)
- LLM via Ollama
- Speech output (Qwen TTS)
- Runtime mode switching:
  - `/voice`
  - `/keyboard`

---

## System Dependencies (Ubuntu 22.04)

### Install SoX (required for audio)
```bash
sudo apt update
sudo apt install sox libsox-fmt-all -y
````

Verify:

```bash
sox --version
```

---

## Setup Instructions

### 1. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

---

### 2. Create Environment

```bash
conda create -n llm_chat python=3.10 -y
conda activate llm_chat
```

---

### 3. Navigate to Project

```bash
cd llm_powered_social_robot
```

---

### 4. Install Python Dependencies

```bash
pip install numpy faster-whisper scipy torch sounddevice qwen-tts
```

---

### 5. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify:

```bash
ollama --version
```

Start service:

```bash
ollama serve
```

---

### 6. Pull Model

```bash
ollama pull llama3.2:1b
```

Test:

```bash
ollama run llama3.2:1b
```

---

## Prompt File

Ensure the system prompt file exists:

```
ameca_prompt.json
```

It must contain a valid JSON object:

```json
{
  "role": "system",
  "content": "..."
}
```

If missing or invalid, a fallback prompt will be used.

---

## Run the Application

```bash
python llm_chat.py
```

At startup, choose mode:

```
[v] voice
[k] keyboard
```

---

## Controls

* `exit`, `quit`, `stop` → end session
* `/voice` → switch to voice mode
* `/keyboard` → switch to keyboard mode

---

## Notes

* First run may be slow due to model loading.
* `flash-attn` warning can be ignored (performance only).
* TTS and Whisper run on CPU → expect latency.
* `llama3.2:1b` is lightweight → fast but less capable.

---

## Known Limitations

* No streaming responses (full response latency)
* Fixed recording window for speech input
* No voice activity detection (VAD)
* CPU-only inference is slow for real-time interaction

```
```
