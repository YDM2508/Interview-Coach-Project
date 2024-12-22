import whisper
import os 

file_path = os.path.join(os.getcwd(), "audio.mp3")

# Check if the file exists before trying to transcribe it
if os.path.isfile(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    print(result["text"])
else:
    print(f"The file {file_path} does not exist.")
