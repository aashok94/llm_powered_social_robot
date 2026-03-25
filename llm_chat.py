# ==========================================
# Simple Voice/Keyboard Chat (Whisper + Ollama + Qwen3-TTS)
# ==========================================

import json
import time
import datetime

import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from ollama import Client
from qwen_tts import Qwen3TTSModel


LOG_FILE = "conversation_log.txt"

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llama3.2:1b"
WHISPER_MODEL = "small"
SAMPLE_RATE = 16000
LISTEN_DURATION = 5
MAX_TURNS = 6
PROMPT_FILE = "ameca_prompt.json"
TTS_SPEAKER = "Sohee"


# -----------------------------
# LOGGING
# -----------------------------
def log_line(text: str) -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")


# -----------------------------
# Speech to Text
# -----------------------------
class SpeechToText:
    def __init__(self):
        self.model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
            cpu_threads=8,
        )

    def listen(self, duration: int = LISTEN_DURATION) -> str:
        print("Listening...")
        start_record = time.time()

        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        end_record = time.time()
        record_time = end_record - start_record
        print(f"[TIME] Recording took: {record_time:.2f}s")
        log_line(f"[TIME] Recording took: {record_time:.2f}s")

        start_transcribe = time.time()
        segments, _ = self.model.transcribe(audio.flatten(), language="en")
        text = " ".join(seg.text for seg in segments).strip()

        end_transcribe = time.time()
        transcribe_time = end_transcribe - start_transcribe
        print(f"[TIME] Transcription took: {transcribe_time:.2f}s")
        log_line(f"[TIME] Transcription took: {transcribe_time:.2f}s")

        return text


# -----------------------------
# Text to Speech
# -----------------------------
class TextToSpeech:
    def __init__(self):
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device_map="cpu",
            dtype=torch.float32,
        )

    def speak(self, text: str) -> None:
        print("Assistant:", text)

        start_tts_gen = time.time()
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language="English",
            speaker=TTS_SPEAKER,
        )

        end_tts_gen = time.time()
        gen_time = end_tts_gen - start_tts_gen
        print(f"[TIME] TTS generation took: {gen_time:.2f}s")
        log_line(f"[TIME] TTS generation took: {gen_time:.2f}s")

        start_play = time.time()
        sd.play(wavs[0], sr)
        sd.wait()

        end_play = time.time()
        play_time = end_play - start_play
        print(f"[TIME] TTS playback took: {play_time:.2f}s")
        log_line(f"[TIME] TTS playback took: {play_time:.2f}s")


# -----------------------------
# LLM
# -----------------------------
class LLMChat:
    def __init__(self):
        self.client = Client(host=OLLAMA_HOST)
        self.messages = [self._load_system_message()]

    def _load_system_message(self) -> dict:
        try:
            with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                msg = json.load(f)

            if not isinstance(msg, dict):
                raise ValueError("Prompt JSON must be an object.")
            if msg.get("role") != "system":
                raise ValueError("Prompt JSON must have role='system'.")
            if "content" not in msg or not isinstance(msg["content"], str):
                raise ValueError("Prompt JSON must contain string field 'content'.")

            return msg

        except Exception as e:
            log_line(f"[ERROR] Failed to load prompt file: {e}")
            print(f"[WARN] Failed to load {PROMPT_FILE}. Using fallback prompt.")
            return {
                "role": "system",
                "content": "You are Ameca, a humanoid social robot. Be concise, clear, honest, and do not fabricate facts.",
            }

    def chat(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})

        system_msg = self.messages[0]
        recent = self.messages[1:][-MAX_TURNS * 2 :]
        self.messages = [system_msg] + recent

        try:
            start_llm = time.time()

            response = self.client.chat(
                model=MODEL_NAME,
                messages=self.messages,
            )

            end_llm = time.time()
            llm_time = end_llm - start_llm
            print(f"[TIME] LLM response took: {llm_time:.2f}s")
            log_line(f"[TIME] LLM response took: {llm_time:.2f}s")

            content = response["message"]["content"].strip()

        except Exception as e:
            log_line(f"[ERROR] Ollama chat failed: {e}")
            return "I am having trouble connecting to my language model."

        self.messages.append({"role": "assistant", "content": content})
        return content


# -----------------------------
# MAIN LOOP
# -----------------------------
def choose_mode() -> str:
    while True:
        mode = input("Choose mode: [v]oice or [k]eyboard: ").strip().lower()
        if mode in ("v", "k"):
            return mode
        print("Type 'v' or 'k'. That's not complicated.")


def main():
    stt = SpeechToText()
    tts = TextToSpeech()
    llm = LLMChat()

    mode = choose_mode()
    print("\nChat Ready.")
    print("Commands: /voice, /keyboard, exit\n")

    while True:
        try:
            if mode == "k":
                user_text = input("You: ").strip()
            else:
                user_text = stt.listen()
                if user_text:
                    print("You:", user_text)

            if not user_text:
                continue

            log_line(f"You: {user_text}")

            cmd = user_text.lower()

            if cmd in ["exit", "quit", "stop"]:
                log_line("Session ended.")
                break

            if cmd == "/voice":
                mode = "v"
                print("Switched to voice mode.")
                log_line("Switched to voice mode.")
                continue

            if cmd == "/keyboard":
                mode = "k"
                print("Switched to keyboard mode.")
                log_line("Switched to keyboard mode.")
                continue

            response = llm.chat(user_text)
            log_line(f"Assistant: {response}")

            if mode == "k":
                print("Assistant:", response)
            else:
                try:
                    tts.speak(response)
                except Exception as e:
                    log_line(f"[ERROR] TTS failed: {e}")
                    print("Assistant:", response)

        except KeyboardInterrupt:
            print("\nShutting down.")
            log_line("Session ended by KeyboardInterrupt.")
            break

        except Exception as e:
            log_line(f"[ERROR] Main loop failed: {e}")
            print("Something broke, but the loop survived.")


if __name__ == "__main__":
    main()