import whisper
import datetime

model = whisper.load_model("base")

# Load audio and process
audio = whisper.load_audio("harvard.wav")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# Detect language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# Transcribe with timestamps (for SRT)
result = model.transcribe("harvard.wav")

# Save TXT file (raw text)
with open("harvard.txt", "w", encoding="utf-8") as txt_file:
    txt_file.write(result["text"])

# Save SRT file (subtitles with timestamps)
with open("harvard.srt", "w", encoding="utf-8") as srt_file:
    for idx, segment in enumerate(result["segments"]):
        start = datetime.timedelta(seconds=segment['start'])
        end = datetime.timedelta(seconds=segment['end'])
        srt_file.write(f"{idx+1}\n")
        srt_file.write(f"{start}.000 --> {end}.000\n")
        srt_file.write(f"{segment['text'].strip()}\n\n")