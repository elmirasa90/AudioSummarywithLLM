seed_everything: 1234
data:
  base_path: ""
model:
  type: facebook/hubert-large-ls960-ft
  llm_embedding_channels: 3072
audio:
  sampling_rate: 16000
train:
  num_gpus: 1
  batch_size: 1
log:
  checkpoint_dir: checkpoints
  log_dir: logs
