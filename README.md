# Cacophony
Inference codebase for "Cacophony: An Improved Contrastive Audio-Text Model"

# Abstract
Despite recent improvements in audio-text modeling, audio-text contrastive models still lag behind their image-text counterparts in scale and performance. We propose a method to improve both the scale and the training of audio-text contrastive models. Specifically, we craft a large-scale audio-text dataset consisting of over 13,000 hours of text-labeled audio, aided by large language model (LLM) processing and audio captioning. Further, we employ an masked autoencoder (MAE) pre-pretraining phase with random patch dropout, which allows us to both scale unlabeled audio datasets and train efficiently with variable length audio. After MAE pre-pretraining of our audio encoder, we train a contrastive model with an auxiliary captioning objective. Our final model, which we name Cacophony, achieves state-of-the-art performance on audio-text retrieval tasks, and exhibits competitive results on other downstream tasks such as zero-shot classification.

# Requirements

# Evaluation
## Audio-Text Retrieval

## Zero-Shot Classification

## Audio Captioning

## HEAR Benchmark

# Acknowledgements