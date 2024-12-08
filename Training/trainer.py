import os
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from datasets import load_from_disk, concatenate_datasets
from transformers import T5Tokenizer, T5ForConditionalGeneration

from model.audio_encoder import AudioEncoder
from utils import (
    batch_full_embed_sequence,
    collate_audio_batch,
    compute_num_audio_embeds,
    merge_prompt_tokens,
)
from writer import MyWriter


class Trainer():
    def __init__(self, args, config, device) -> None:
        self.args = args
        self.config = config

        self.run_name = args.run_name
        self.device = device

        # Set seed.
        torch.cuda.manual_seed(self.config.seed_everything)

        # Set up checkpointing and Tensorboard logging.
        self.checkpoint_save_dir = os.path.join(self.config.log.checkpoint_dir, self.run_name)
        self.log_dir = os.path.join(self.config.log.log_dir, self.run_name)

        os.makedirs(self.checkpoint_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = MyWriter(self.config, self.log_dir)

        self.get_dataloaders()
        print("Set up dataloaders.\n")

        # Audio encoder.
        self.audio_encoder = AudioEncoder(self.config)
        print("Loaded audio encoder.\n")

        # LLM tokenizer.
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load FLAN-T5 Base model.
        self.llm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").eval()
        self.llm.to(self.device)
        print("Loaded FLAN-T5 Base.\n")

        # Gradient accumulation interval.
        self.grad_accum_interval = self.config.train.grad_accum_interval

        # Number of epochs to train.
        self.num_epochs = self.config.train.epochs

        # Set up optimizer and learning rate scheduler.
        self.optimizer = torch.optim.AdamW(
            [
                {'params': self.audio_encoder.parameters()}
            ],
            lr=self.config.train.optimizer.lr,
            betas=(self.config.train.optimizer.beta1, self.config.train.optimizer.beta2),
        )
        self.lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            self.optimizer,
            total_iters=(self.num_epochs * len(self.train_dataloader) // self.grad_accum_interval),
            power=1.0,
        )

        # Load checkpoint if specified.
        if self.args.checkpoint_path:
            self.load_checkpoint(self.args.checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.audio_encoder.load_state_dict(checkpoint["audio_encoder"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.start_epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]

        if self.device == torch.device(f"cuda:{self.args.gpu_idx}"):
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda(self.args.gpu_idx)

        print(f"Loaded checkpoint from {checkpoint_path}.\n")

    def get_dataloaders(self):
        # Load train datasets and combine into one Dataset object.
        all_train_datasets = []
        for dataset_name in self.config.data.train_set:
            dataset_path = os.path.join(self.config.data.base_path, dataset_name)
            dataset = load_from_disk(dataset_path)
            dataset.set_format(type='torch')
            all_train_datasets.append(dataset)
        self.train_dataset = concatenate_datasets(all_train_datasets)

        # Load val datasets and combine into one Dataset object.
        all_val_datasets = []
        for dataset_name in self.config.data.val_set:
            dataset_path = os.path.join(self.config.data.base_path, dataset_name)
            dataset = load_from_disk(dataset_path)
            dataset.set_format(type='torch')
            all_val_datasets.append(dataset)
        self.val_dataset = concatenate_datasets(all_val_datasets)

        # Create dataloaders.
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=collate_audio_batch,
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=collate_audio_batch,
        )

    def train(self):
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch}")

            self.audio_encoder.train()
            self.optimizer.zero_grad()

            for batch_idx, (
                padded_audios,
                audio_len_samples,
                _,
                text_input_ids,
                response_input_ids,
                ctc_pool_ranges,
            ) in enumerate(tqdm(self.train_dataloader)):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    padded_audios = padded_audios.to(self.device)

                    # Compute audio embeddings.
                    padded_audio_embeds = self.audio_encoder(padded_audios, ctc_pool_ranges)

                    # If batch size = 1, no need to unpad by cropping.
                    unpadded_audio_embeds = padded_audio_embeds

                    # Generate embeddings for audio prompts.
                    (
                        batched_audio_prompt_sequences,
                        _,
                        _,
                        _,
                    ) = batch_full_embed_sequence(
                        all_audio_embeds=unpadded_audio_embeds,
                        all_text_input_ids=text_input_ids,
                        all_response_input_ids=response_input_ids,
                        tokenizer=self.tokenizer,
                        embed_tokens=None,  
                        device=self.device,
                        process_text=False,
                    )

                    # Forward pass for FLAN-T5.
                    llm_audio_output = self.llm(
                        inputs_embeds=batched_audio_prompt_sequences,
                        labels=response_input_ids,
                        return_dict=True,
                    )

                    # Compute loss.
                    total_loss = llm_audio_output.loss

                # Backpropagation.
                total_loss /= self.grad_accum_interval
                scaler.scale(total_loss).backward()

                # Weights update.
                if (
                    ((batch_idx + 1) % self.grad_accum_interval == 0) or
                    (batch_idx + 1 == len(self.train_dataloader))
                ):
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                self.step += 1

                # Logging.
                if self.step % self.config.log.log_interval == 0:
                    self.writer.log_training({"loss": total_loss.item()}, self.step)

            # Perform validation at end of epoch.
            self.validate(epoch)

    def validate(self, epoch):
        self.audio_encoder.eval()

        for batch_idx, (
            audio, _, texts, text_input_ids, response_input_ids, ctc_pool_ranges
        ) in enumerate(tqdm(self.val_dataloader)):
            with torch.no_grad():
                audio = audio.to(self.device)

                # Compute audio embeddings.
                audio_embeds = self.audio_encoder(audio, ctc_pool_ranges)

                # Generate embeddings for prompts.
                (
                    full_audio_prompt_sequence,
                    _,
                    _,
                    _,
                ) = batch_full_embed_sequence(
                    all_audio_embeds=audio_embeds,
                    all_text_input_ids=text_input_ids,
                    all_response_input_ids=response_input_ids,
                    tokenizer=self.tokenizer,
                    embed_tokens=None,  # FLAN-T5 handles embeddings internally
                    device=self.device,
                    process_text=False,
                )

                # Forward pass for FLAN-T5.
                llm_audio_output = self.llm(
                    inputs_embeds=full_audio_prompt_sequence,
                    labels=response_input_ids,
                    return_dict=True,
                )

                # Log losses.
                losses = {"ntp_loss": llm_audio_output.loss.item()}
                self.writer.log_validation(losses, self.step)

        print(f"Validation completed for epoch {epoch}.")  