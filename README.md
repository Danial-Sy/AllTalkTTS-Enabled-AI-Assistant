# AllTalkTTS-Enabled-AI-Assistant
This repository uses a finetuned AllTalkTTS model, OpenAI’s Whisper transcription, and the OpenAI API to create a voice‑activated AI assistant. AllTalkTTS is based on the Coqui TTS engine and includes features such as a settings interface, low VRAM support, DeepSpeed integration, a narrator mode, model finetuning, custom models, and maintenance of generated audio files
github.com
Whisper is a general‑purpose speech recognition model trained on a large multilingual dataset that can handle multilingual speech recognition, translation and language identification
github.com
When combined with the OpenAI API for text generation, these components enable you to interact with the assistant using any voice; your speech is transcribed and then the assistant responds using a cloned voice.

Features

Voice cloning via AllTalkTTS: The finetuned model can mimic any voice and produce clear audio output.

Speech‑to‑text using Whisper: The assistant can transcribe multilingual audio and identify the spoken language

Natural language understanding using OpenAI’s text generation API: The assistant can process transcribed text and produce relevant responses.

Low‑resource options: AllTalkTTS supports low VRAM configurations and DeepSpeed for efficient generation

Flexible integration: Voice cloning and transcription capabilities can be accessed through a JSON API, allowing integration with other applications

Essentially this project provides an interface to incorporate all of these incredible tools into one barebones app, feel free to use it however you'd like!

# Acknowledgements
This project builds on the work of the AllTalkTTS and Whisper communities. AllTalkTTS extends the Coqui TTS engine with advanced features and Whisper is a multilingual speech recognition model trained on a diverse dataset.
The OpenAI API provides the conversational language model used to generate responses.
