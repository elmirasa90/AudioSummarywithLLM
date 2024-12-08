import os
import torch
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa

# Transcription Engine
class TranscriptionEngine:
    def __init__(self, transcription_model, device):
        self.device = device

        # Load Wav2Vec2 for transcription
        print("Loading Wav2Vec2 model for transcription...")
        self.processor = Wav2Vec2Processor.from_pretrained(transcription_model)
        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained(transcription_model).to(device)

    def transcribe_audio(self, audio_path):
        try:
            audio, rate = librosa.load(audio_path, sr=16000)
            input_values = self.processor(audio, return_tensors="pt", sampling_rate=rate).input_values.to(self.device)
            logits = self.wav2vec2(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            return transcription
        except Exception as e:
            return f"Error transcribing {audio_path}: {e}"

# Main script for transcription
if __name__ == "__main__":
    transcription_model = "facebook/wav2vec2-base-960h"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    engine = TranscriptionEngine(transcription_model, device)

    input_directory = "D:\Speech Summerization\processed_audio_files"
    output_excel = "D:\Speech Summerization\Final_Experiments/transcriptions.xlsx"

    results = []  

    for file_name in os.listdir(input_directory):
        if file_name.endswith(".wav"):
            audio_path = os.path.join(input_directory, file_name)
            print(f"Processing {audio_path}...")

            # Transcription
            transcription = engine.transcribe_audio(audio_path)
            if transcription.startswith("Error"):
                print(transcription)
                continue

            # Append result to list
            results.append({"ID": os.path.splitext(file_name)[0], "Transcription": transcription})

    # Save results to Excel
    if results:
        df = pd.DataFrame(results)
        df.to_excel(output_excel, index=False)
        print(f"Transcription complete. Results saved in '{output_excel}'.")
    else:
        print("No transcriptions were generated.")
