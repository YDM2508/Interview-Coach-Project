import whisper
import os
import wave

model = whisper.load_model("tiny")
# result = ("audio.mp3")


class Config:
    channels = 2
    sample_width = 2
    sample_rate = 44100

def save_wav_file(file_path, wav_bytes):
    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(Config.channels)
        wav_file.setsampwidth(Config.sample_width)
        wav_file.setframerate(Config.sample_rate)
        wav_file.writeframes(wav_bytes)

def transcribe(file_path):
    # with open(file_path, 'rb') as audio_file:
    #     transcription = model.transcribe("whisper-1", audio_file)
    #     return transcription['text']

    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]

