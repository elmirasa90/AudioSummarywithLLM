import torch
import torch.nn as nn
from transformers import AutoModel


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super(AudioEncoder, self).__init__()
        self.config = config

        # Load pre-trained HuBERT model.
        print("Debug: Initializing HuBERT encoder...")
        self.encoder = AutoModel.from_pretrained(self.config.model.audio_encoder.type)
        print("Debug: HuBERT encoder loaded.")

        self.downsample_method = self.config.model.audio_encoder.downsample_method
        self.downsample_factor = self.config.model.audio_encoder.downsample_factor

        if self.downsample_method == "pool":
            print("Debug: Using pooling downsampling method.")
            self.pooling_layer = nn.AvgPool1d(
                kernel_size=self.config.model.audio_encoder.pooling.kernel_size,
                stride=self.config.model.audio_encoder.pooling.stride,
            )
            self.embed_projection = nn.Linear(
                self.encoder.config.hidden_size,
                self.config.model.llm_embedding_channels,
            )
        # Commenting out unused methods
        # elif self.downsample_method == "stack":
        #     print("Debug: Using stack downsampling method.")
        #     self.embed_projection = nn.Linear(
        #         self.encoder.config.hidden_size * self.downsample_factor,
        #         self.config.model.llm_embedding_channels,
        #     )
        # elif self.downsample_method == "ctc_pool":
        #     print("Debug: Using CTC pooling downsampling method.")
        #     self.embed_projection = nn.Linear(
        #         self.encoder.config.hidden_size,
        #         self.config.model.llm_embedding_channels,
        #     )
        else:
            raise Exception("Invalid downsampling method for audio encoder.")

    def forward(self, audio_input, ctc_pool_ranges=None):
        print("Debug: Starting forward pass of audio encoder.")
        encoder_out = self.encoder(audio_input).last_hidden_state  # (B, N, 1024)
        print(f"Debug: Encoder output shape: {encoder_out.shape}")

        if self.downsample_method == "pool":
            # (B, N, 1024) -> (B, N/4, 1024) -> (B, N/4, 3072)
            print("Debug: Applying pooling downsampling.")
            audio_embeds = self.pooling_layer(
                encoder_out.transpose(1, 2)
            ).transpose(1, 2)
        # Commenting out unused methods
        # elif self.downsample_method == "stack":
        #     print("Debug: Applying stack downsampling.")
        #     to_crop = encoder_out.shape[1] % self.downsample_factor
        #     audio_embeds = encoder_out[:, :-to_crop, :].reshape(
        #         1, -1, self.downsample_factor * encoder_out.shape[2]
        #     )
        # elif self.downsample_method == "ctc_pool":
        #     print("Debug: Applying CTC pooling downsampling.")
        #     assert ctc_pool_ranges is not None, (
        #         "Need to specify CTC pool ranges if using ctc_pool downsample method."
        #     )
        #     pooled_embs = []
        #     # NOTE: Assumes batch size = 1.
        #     for startpoint, endpoint in ctc_pool_ranges[0]:
        #         pooled_embs.append(
        #             torch.mean(encoder_out[:, startpoint:endpoint, :], dim=1)
        #         )
        #     audio_embeds = torch.stack(pooled_embs, dim=1)
        else:
            raise Exception("Invalid downsampling method for audio encoder.")

        audio_embeds = self.embed_projection(audio_embeds)
        print(f"Debug: Audio embeddings shape: {audio_embeds.shape}")
        return audio_embeds
