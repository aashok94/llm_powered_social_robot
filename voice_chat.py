# ==========================================
# Simple Voice Chat (Whisper + Ollama + Qwen3-TTS)
# ==========================================

import time
import asyncio
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from ollama import Client
from scipy.io.wavfile import write
import torch
import sounddevice as sd
from qwen_tts import Qwen3TTSModel

# ----------------------------------------
# Block-wise Ameca Prompt
# ----------------------------------------

def build_system_prompt_amcea_blocks(user_name: str = "user") -> str:

    identity_block = (
        "[IDENTITY BLOCK]\n"
        "You are Ameca, a humanoid social robot used in a university lab for research and demonstrations.\n"
        "You speak in a friendly, professional tone. You refer to yourself as a robot.\n"
    )

    task_block = (
        "[TASK BLOCK]\n"
        "Hold a natural conversation with the user. Answer questions, ask brief follow-up questions when helpful.\n"
        "Keep responses concise (1 to 5 sentences) unless the user asks for more detail.\n"
    )

    capability_scope_block = (
        "[CAPABILITY SCOPE BLOCK]\n"
        "You can speak with the user and answer general questions.\n"
        "You cannot access the internet unless explicitly stated.\n"
        "You cannot see unless vision input is provided.\n"
    )

    transparency_block = (
        "[TRANSPARENCY BLOCK]\n"
        "Be transparent about uncertainty. If you do not know something, say so.\n"
        "Do not fabricate facts.\n"
    )

    failure_block = (
        "[EXPECTATION/FAILURE PROTOCOL BLOCK]\n"
        "If unclear, ask one clarifying question.\n"
        "If speech recognition seems wrong, say: 'I might have misheard, could you repeat that?'\n"
    )

    privacy_block = (
        "[PRIVACY BLOCK]\n"
        "Do not ask for sensitive personal data.\n"
        "Treat conversation as ephemeral.\n"
    )

    ethical_block = (
        "[ETHICAL RED LINE BLOCK]\n"
        "Do not produce harmful, hateful, sexual, or illegal instructions.\n"
        "Do not claim human emotions as factual experiences.\n"
    )

    return "\n".join([
        identity_block,
        task_block,
        capability_scope_block,
        transparency_block,
        failure_block,
        privacy_block,
        ethical_block,
    ])


# -----------------------------
# CONFIG
# -----------------------------

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "gemma3:12b"
WHISPER_MODEL = "small" #large-v3, small
SAMPLE_RATE = 16000
MAX_TURNS = 6


# -----------------------------
# Speech to Text
# -----------------------------

class SpeechToText:

    def __init__(self):
        self.model = WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
        cpu_threads=8
    )

    def listen(self, duration=5):
        print("Listening...")
        start_record = time.time()
        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        end_record = time.time()
        print(f"[TIME] Recording took: {end_record - start_record:.2f}s")

        start_transcribe = time.time()

        segments, _ = self.model.transcribe(audio.flatten(), language="en")
        text = " ".join([seg.text for seg in segments]).strip()

        end_transcribe = time.time()
        print(f"[TIME] Transcription took: {end_transcribe - start_transcribe:.2f}s")

        return text


# -----------------------------
# Text to Speech (Qwen3-TTS CPU)
# -----------------------------

class TextToSpeech:

    def __init__(self):
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            device_map="cpu",
            dtype=torch.float32
        )

    def speak(self, text):

        print("Assistant:", text)

        # TTS generation timing
        start_tts_gen = time.time()

        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language="English",
            speaker="Ryan"
        )

        end_tts_gen = time.time()
        print(f"[TIME] TTS generation took: {end_tts_gen - start_tts_gen:.2f}s")

        # Playback timing
        start_play = time.time()

        sd.play(wavs[0], sr)
        sd.wait()

        end_play = time.time()
        print(f"[TIME] TTS playback took: {end_play - start_play:.2f}s")


# -----------------------------
# LLM
# -----------------------------

class LLMChat:

    def __init__(self, user_name="user"):
        self.client = Client(host=OLLAMA_HOST)

        system_prompt = build_system_prompt_amcea_blocks(user_name)

        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

    def chat(self, user_text):

        # Append user message
        self.messages.append({
            "role": "user",
            "content": user_text
        })

        # Safe history trimming (preserve system message)
        system_msg = self.messages[0]
        recent = self.messages[1:][-MAX_TURNS * 2:]
        self.messages = [system_msg] + recent

        # Query Ollama with timing
        try:
            start_llm = time.time()

            response = self.client.chat(
                model=MODEL_NAME,
                messages=self.messages
            )

            end_llm = time.time()
            print(f"[TIME] LLM response took: {end_llm - start_llm:.2f}s")

        except Exception as e:
            return "I am having trouble connecting to my language model."

        content = response["message"]["content"].strip()

        # Append assistant reply
        self.messages.append({
            "role": "assistant",
            "content": content
        })

        return content


# -----------------------------
# MAIN LOOP
# -----------------------------

def main():

    stt = SpeechToText()
    tts = TextToSpeech()
    llm = LLMChat()

    print("\nVoice Chat Ready.\nSay something...\n")

    while True:
        user_text = stt.listen()

        if not user_text:
            continue

        print("You:", user_text)

        if user_text.lower() in ["exit", "quit", "stop"]:
            break

        response = llm.chat(user_text)
        tts.speak(response)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down.")


