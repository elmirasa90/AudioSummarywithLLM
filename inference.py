import argparse
import librosa
import torch
import os
from omegaconf import OmegaConf
from transformers import AutoTokenizer, T5ForConditionalGeneration
from model.audio_encoder import AudioEncoder
from utils import merge_prompt_tokens

class LLMSpeechTextInference:
    def __init__(self, config, audio_encoder_checkpoint, device):
        self.config = config
        self.device = device

        # Load audio encoder
        print("DEBUG: Loading audio encoder...")
        checkpoint = torch.load(audio_encoder_checkpoint, map_location="cpu")
        self.audio_encoder = AudioEncoder(config)
        self.audio_encoder.load_state_dict(checkpoint)
        self.audio_encoder.eval().to(self.device)
        print("DEBUG: Audio encoder loaded.")

        # Load LLM
        print("DEBUG: Loading FLAN-T5...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.llm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").eval().to(self.device)
        print("DEBUG: LLM loaded.")

    def generate_audio_response(self, audio, max_new_tokens=256):
        # Process audio through encoder
        print("DEBUG: Generating audio embeddings...")
        audio_tensor = torch.tensor(audio).float().unsqueeze(0).to(self.device)
        audio_embeds = self.audio_encoder(audio_tensor)

        # Merge prompt tokens
        print("DEBUG: Merging prompt embeddings...")
        prompt_emb_sequence = merge_prompt_tokens(
            inputs_embeds=audio_embeds,
            tokenizer=self.llm_tokenizer,
            embed_tokens=self.llm.get_input_embeddings(),
            device=self.device,
            target_dim=self.config.model.llm_embedding_channels
        )

        # Generate response
        print("DEBUG: Generating response...")
        with torch.no_grad():
            generate_ids = self.llm.generate(inputs_embeds=prompt_emb_sequence, max_new_tokens=max_new_tokens)

        response_text = self.llm_tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        return response_text[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--audio_encoder_checkpoint", type=str, required=True, help="Path to audio encoder checkpoint")
    parser.add_argument("--audio_file", type=str, required=True, help="Path to audio file")
    parser.add_argument("--gpu_idx", type=int, default=0, help="GPU index")
    args = parser.parse_args()

    # Debug paths
    print(f"DEBUG: Config Path: {args.config}")
    print(f"DEBUG: Checkpoint Path: {args.audio_encoder_checkpoint}")
    print(f"DEBUG: Audio File Path: {args.audio_file}")

    # Ensure files exist
    for path, name in [(args.config, "Config"), (args.audio_encoder_checkpoint, "Checkpoint"), (args.audio_file, "Audio File")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found at {path}")

    # Load configuration
    config = OmegaConf.load(args.config)

    # Initialize inference
    device = torch.device(f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu")
    inferencer = LLMSpeechTextInference(config, args.audio_encoder_checkpoint, device)

    # Load and process audio
    print("DEBUG: Loading audio...")
    audio, sr = librosa.load(args.audio_file, sr=config.audio.sampling_rate)

    # Generate summary
    print("DEBUG: Running inference...")
    response = inferencer.generate_audio_response(audio)
    print("Summary:")
    print(response)
