# General-Purpose Speech Summarization with LLMs (Flan-T5-Base)
This study is based on the [this study](https://arxiv.org/abs/2406.05968). In the baseline framework the used LLM is MiniChat-3B, however to evaluate the performance of the proposed model on other LLMs, I change the MiniChat-3B to Flan-T5-Base. In fact, this model is and end-to-end model for generating summaries from audios without the nee for ASR.

This repository includes the Code for running the audio encoder and LLMs pipeline.

## Prerequisites
The prerequisites could be installed by install by running 
pip install -r requirements.txt

## Audio-Encoder
Audio-Encoder is necessary for running the model and due to its size, it has not included in the repository, but it could be downloaded from [this Google Drive link](https://drive.google.com/drive/folders/1o363nAqpyP80tivFNdjmyyoWGCLUeHZS). 

## Running the code
To run the code and generate summaries from audio, you need to run the inference.py. Make you you have the audio encoder in the same directory and also config_full.yaml and also the audio_encoder.py are run. </br></br>
*I used the inference code from the [baseline paper](https://arxiv.org/abs/2406.05968), however, I commneted the part of the code related to training the audio encoder, becuase I have not retrained the audio encoder and used the available one from the baseline framework.*








