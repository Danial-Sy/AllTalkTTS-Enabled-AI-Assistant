import os
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torchaudio
import numpy as np
import sounddevice as sd

# Path to fine-tuned model folder
model_dir = r"C:\TTSModelWGPT\alltalk\alltalk_tts\models\lastfinetuned"
reference_wav = os.path.join(model_dir, "wavs", "CombinedAudioFile_00000024.wav")

text = "Every single time my friends like john or nathan or anybody at school say something i DIE laughing"

# Load the config
config = XttsConfig()
config.load_json(f"{model_dir}/config.json")

# Load the model
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_dir=model_dir,
    vocab_path=f"{model_dir}/vocab.json",
    use_deepspeed=False,
)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Get speaker latents
gpt_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=[reference_wav],
    gpt_cond_len=config.gpt_cond_len,
    max_ref_length=config.max_ref_len,
    sound_norm_refs=config.sound_norm_refs,
)

# Run inference
output = model.inference(
    text=text,
    language="en",
    gpt_cond_latent=gpt_latent,
    speaker_embedding=speaker_embedding,
    temperature=0.75,
    length_penalty=config.length_penalty,
    repetition_penalty=config.repetition_penalty,
    top_k=config.top_k,
    top_p=config.top_p,
    speed=1.0,
    enable_text_splitting=True,
)

# Play audio
wav = output["wav"]
sample_rate = 24000  # XTTS default
sd.play(np.array(wav), sample_rate)
sd.wait()
