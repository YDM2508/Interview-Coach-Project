import whisper

model = whisper.load_model("base")
result = model.transcribe("temp/audio.wav",fp16=False)
print(result["text"])