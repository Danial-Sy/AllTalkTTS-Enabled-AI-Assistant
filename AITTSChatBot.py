import os
import torch
import torchaudio
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import openai
import time
import threading
import random
import tkinter as tk
from tkinter import font, Label, Tk, Button
from PIL import Image, ImageTk, ImageSequence
import webrtcvad
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# --------------------------
# GIF player class
class MyLabel(Label):
    def __init__(self, master, filename):
        # load frames into a list
        im = Image.open(filename)
        seq = []
        try:
            while True:
                seq.append(im.copy())
                im.seek(len(seq))
        except EOFError:
            pass

        # frame delay (fallback 100ms)
        try:
            delay = im.info['duration']
        except KeyError:
            delay = 100

        # ensure we have the right container size
        master.update_idletasks()
        W = master.winfo_width() or 300
        H = master.winfo_height() or 300

        # build PhotoImage list, forcing exact same size
        self.frames = []
        for frame in seq:
            f = frame.convert('RGBA')
            f = f.resize((W, H), Image.Resampling.LANCZOS)
            self.frames.append(ImageTk.PhotoImage(f))

        # init label with first frame
        Label.__init__(self, master, image=self.frames[0], bg=master['bg'])

        self.idx = 1
        self.delay = delay or 100
        self.cancel = self.after(self.delay, self.play)

    def play(self):
        self.config(image=self.frames[self.idx])
        self.idx = (self.idx + 1) % len(self.frames)
        self.cancel = self.after(self.delay, self.play)

# --------------------------
# Set OpenAI API key 
openai.api_key = 'INSERT_KEY_HERE'

# --------------------------
# Define paths and settings
MODEL_DIR = r"C:\TTSModelWGPT\alltalk\alltalk_tts\models\lastfinetuned"
REFERENCE_WAV_REL = os.path.join("wavs", "CombinedAudioFile_00000024.wav")
INPUT_AUDIO_FILENAME = "user_input.wav"
RECORD_SAMPLE_RATE = 16000

STATIC_PNG_PATH = "Static.png"
ANIMATED_GIF_DICT = {
    1: "AINerd.gif",
    2: "AINeutral.gif",
    3: "AIThinking.gif"
}

# --------------------------
def record_audio_vad(vad_aggressiveness=2,
                     silence_duration_ms=2000,
                     frame_duration_ms=30,
                     sample_rate=16000):
    vad = webrtcvad.Vad(vad_aggressiveness)
    frame_size = int(sample_rate * frame_duration_ms / 1000)
    silence_frames = int(silence_duration_ms / frame_duration_ms)
    recorded = []
    silent = 0

    with sd.InputStream(samplerate=sample_rate,
                        channels=1,
                        dtype='int16',
                        blocksize=frame_size) as stream:
        print("Start speaking")
        while True:
            frame, overflow = stream.read(frame_size)
            if overflow:
                print("Buffer overflow")
            is_speech = vad.is_speech(frame.tobytes(), sample_rate)
            recorded.append(frame)
            silent = silent + 1 if not is_speech else 0
            if silent >= silence_frames:
                break

    audio = np.concatenate(recorded, axis=0)
    return (audio.astype(np.float32) / 32768.0)

def save_audio(filename, audio, sample_rate):
    wav.write(filename, sample_rate,
              (audio * 32767).astype(np.int16))

# --------------------------
def transcribe_audio(filename):
    with open(filename, "rb") as f:
        t = openai.Audio.transcribe("whisper-1", f)
    return t["text"]

def chat_with_gpt(prompt):
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are Danial, a helpful AI assistant. Keep your responses concise, no more than a few sentences."
            },
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message["content"]

# --------------------------
def load_finetuned_model():
    config = XttsConfig()
    config.load_json(os.path.join(MODEL_DIR, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config,
                          checkpoint_dir=MODEL_DIR,
                          vocab_path=os.path.join(MODEL_DIR, "vocab.json"),
                          use_deepspeed=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ref = os.path.join(MODEL_DIR, REFERENCE_WAV_REL)
    gpt_latent, speaker_emb = model.get_conditioning_latents(
        audio_path=[ref],
        gpt_cond_len=config.gpt_cond_len,
        max_ref_length=config.max_ref_len,
        sound_norm_refs=config.sound_norm_refs,
    )
    return model, config, gpt_latent, speaker_emb, device

def synthesize_finetuned(text,
                         model,
                         config,
                         gpt_latent,
                         speaker_emb):
    out = model.inference(
        text=text,
        language="en",
        gpt_cond_latent=gpt_latent,
        speaker_embedding=speaker_emb,
        temperature=0.75,
        length_penalty=config.length_penalty,
        repetition_penalty=config.repetition_penalty,
        top_k=config.top_k,
        top_p=config.top_p,
        speed=1.0,
        enable_text_splitting=True,
    )
    return out["wav"]

# --------------------------
class TTSChatApp:
    def __init__(self, master):
        self.master = master
        master.title("Danial TTS Chatbot")
        master.configure(bg="#f0f0f0")

        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=2)
        master.grid_rowconfigure(0, weight=1)

        # Left frame for static + GIF
        self.left_frame = tk.Frame(master, bg="#f0f0f0")
        self.left_frame.grid(row=0, column=0, sticky="nsew",
                             padx=10, pady=10)
        self.left_frame.bind("<Configure>", self.resize_image)

        # Right frame
        self.right_frame = tk.Frame(master, bg="#f0f0f0")
        self.right_frame.grid(row=0, column=1, sticky="nsew",
                              padx=10, pady=10)
        self.right_frame.grid_rowconfigure(1, weight=1)

        self.header = tk.Label(self.right_frame,
                               text="Danial GPT",
                               bg="#f0f0f0",
                               fg="#333333",
                               font=("Helvetica", 20, "bold"))
        self.header.grid(row=0, column=0, sticky="ew",
                         pady=(0,10))

        self.text_box = tk.Text(self.right_frame,
                                height=10,
                                width=40,
                                font=("Helvetica",12),
                                bg="white",
                                fg="#333333",
                                wrap=tk.WORD)
        self.text_box.grid(row=1, column=0,
                           sticky="nsew",
                           pady=(0,10))

        self.button = tk.Button(self.right_frame,
                                text="Speak",
                                command=self.start_process,
                                font=("Helvetica",14),
                                bg="#4CAF50",
                                fg="white",
                                activebackground="#45a049")
        self.button.grid(row=2, column=0, sticky="ew",
                         pady=(0,10))

        try:
            self.original_static = Image.open(STATIC_PNG_PATH)
        except Exception as e:
            print(f"Error loading static image: {e}")
            self.original_static = None

        self.static_img = self.get_scaled_static_image()
        self.label_image = tk.Label(self.left_frame,
                                    image=self.static_img,
                                    bg="#f0f0f0")
        self.label_image.pack(expand=True, fill="both")

        self.anim = None  # MyLabel instance

        print("Loading fine-tuned model...")
        (self.model,
         self.config,
         self.gpt_latent,
         self.speaker_emb,
         self.device) = load_finetuned_model()
        print("Model loaded.")
        self.append_text("Model loaded.\n")

    def get_scaled_static_image(self):
        if not self.original_static:
            return None
        w = self.left_frame.winfo_width() or 300
        h = self.left_frame.winfo_height() or 300
        try:
            method = Image.Resampling.LANCZOS
        except AttributeError:
            method = Image.LANCZOS
        img = self.original_static.copy()
        img.thumbnail((w, h), method)
        return ImageTk.PhotoImage(img)

    def resize_image(self, _):
        img = self.get_scaled_static_image()
        if img:
            self.static_img = img
            self.label_image.config(image=img)

    def append_text(self, msg):
        self.text_box.insert(tk.END, msg + "\n")
        self.text_box.see(tk.END)

    def start_process(self):
        self.button.config(state=tk.DISABLED)
        threading.Thread(target=self.process_speech,
                         daemon=True).start()

    def process_speech(self):
        self.master.after(0, lambda: self.append_text("Listening…"))
        audio = record_audio_vad(sample_rate=RECORD_SAMPLE_RATE)
        save_audio(INPUT_AUDIO_FILENAME, audio, RECORD_SAMPLE_RATE)

        self.master.after(0, lambda: self.append_text("Transcribing…"))
        user_text = transcribe_audio(INPUT_AUDIO_FILENAME)
        self.master.after(0, lambda: self.append_text(f"You said: {user_text}"))

        self.master.after(0, lambda: self.append_text("Generating response…"))
        ai_resp = chat_with_gpt(user_text)
        self.master.after(0, lambda: self.append_text(f"Danial says: {ai_resp}"))

        self.master.after(0, lambda: self.append_text("Synthesizing response…"))
        wav_out = synthesize_finetuned(
            ai_resp,
            self.model,
            self.config,
            self.gpt_latent,
            self.speaker_emb
        )

        self.master.after(0, self.start_animation)
        sd.play(np.array(wav_out), 24000)
        sd.wait()

        self.master.after(0, self.stop_animation)
        self.master.after(0, lambda: self.append_text("Done."))
        self.master.after(0, lambda: self.button.config(state=tk.NORMAL))

    def start_animation(self):
        # remove any previous animation
        if self.anim:
            try:
                self.anim.after_cancel(self.anim.cancel)
                self.anim.destroy()
            except:
                pass

        gif_path = ANIMATED_GIF_DICT[random.randint(1, 3)]
        # overlay the GIF player
        self.anim = MyLabel(self.left_frame, gif_path)
        self.anim.place(relx=0, rely=0, relwidth=1, relheight=1)

    def stop_animation(self):
        if self.anim:
            try:
                self.anim.after_cancel(self.anim.cancel)
                self.anim.destroy()
            except:
                pass
        # revert to static
        img = self.get_scaled_static_image()
        if img:
            self.static_img = img
            self.label_image.config(image=img)

def main():
    root = tk.Tk()
    root.geometry("800x600")
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    TTSChatApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
