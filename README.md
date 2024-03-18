# Cacophony
Inference codebase for "Cacophony: An Improved Contrastive Audio-Text Model"

## Abstract
Despite recent improvements in audio-text modeling, audio-text contrastive models still lag behind their image-text counterparts in scale and performance. We propose a method to improve both the scale and the training of audio-text contrastive models. Specifically, we craft a large-scale audio-text dataset consisting of over 13,000 hours of text-labeled audio, aided by large language model (LLM) processing and audio captioning. Further, we employ an masked autoencoder (MAE) pre-pretraining phase with random patch dropout, which allows us to both scale unlabeled audio datasets and train efficiently with variable length audio. After MAE pre-pretraining of our audio encoder, we train a contrastive model with an auxiliary captioning objective. Our final model, which we name Cacophony, achieves state-of-the-art performance on audio-text retrieval tasks, and exhibits competitive results on other downstream tasks such as zero-shot classification.

<br>
<p align="center">
    <img src="./assets/training_block.png" width="500">
</p>
<br>

## Requirements
Jax and Flax are used for the model implementation. Tested on RTX 2080Ti, CUDA version 11.5, cuDNN version 8.2.1, cudatoolkit 11.3.1, and Python 3.8.17.

```bash
pip install requirements.txt
```

## Pretrained Models
We provide the following pretrained models on both stages of the Cacophony model:
### Stage 1: AudioMAE
[link](https://drive.google.com/), Model detail: 
* Audio Encoder: parameters
* Audio Decoder: parameters


### Stage 2: Cacophony
[link](https://drive.google.com/), Model detail:
* Audio Encoder: parameters,
* Text Encoder: parameters,


## Evaluation Results

The evaluation step involves [AudioCaps](https://audiocaps.github.io/), [Clotho](https://github.com/audio-captioning/clotho-dataset), [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html), [ESC-50](https://github.com/karolpiczak/ESC-50[), [TUT Acoustic Scene 2017](https://zenodo.org/records/400515) and [VGGSound-test](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) datasets. Since our model is trained on audio sampled at 16kHz, we first downsample all of the audio from the above datasets to match with the training stage.

### 1. Audio-Text Retrieval

We evaluate the model performance Audio-Text retrieval task using the [AudioCaps](https://audiocaps.github.io/) dataset and [Clotho](https://github.com/audio-captioning/clotho-dataset) dataset.

```bash
python eval_caco.py --task ar --model_path <path_to_model>
```

Reproducible results for the Audio-Text retrieval task are as follows:
<center>

| | |Text-to-Audio| | |Audio-to-Text| | 
|:------:|:------:|:------:|:-----:|:------:|:------:|:-----:|
| |R@1|R@5|R@10|R@1|R@5|R@10|
|AudioCaps |0.410| 0.753| 0.864|0.553|0.836|0.924|
|Clotho|0.202| 0.459| 0.588|0.265|0.541|0.762|

</center>

### 2. Zero-Shot Classification
We evaluate the model performance on the zero-shot classification task using the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html), [ESC-50](https://github.com/karolpiczak/ESC-50[),[TUT Acoustic Scene 2017](https://zenodo.org/records/400515) and [VGGSound-test](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) datasets. Note that, upon our evaluation, we found that some of the audio from the VGGSound-test dataset is not publicly available anymore, and we were unable to evaluate our model on the full dataset. Instead we evaluate on 12,722 samples.

```bash
python eval_caco.py --task zs --model_path <path_to_model>
```
<center>

| UrbanSound8K| ESC-50|TUTAS2017 |VGGSound-test| 
|:------:|:------:|:-----:|:------:|
|0.934| 0.771| 0.486|0.271|

</center>

### 3. Audio Captioning

```bash
python eval_caco.py --task caption --model_path <path_to_model>
```

### 4. HEAR Benchmark

Our environment does not support the HEAR benchmark, but we provide the code to run the benchmark in the `hear` directory. To successfully run the benchmark, follow the instructions in the `hear` directory.


#### *HEAR Benchmark Results
For easy comparison, we provide the numbers for the HEAR benchmark, comparing with [LAION-CLAP](https://arxiv.org/abs/2211.06687), [MS-CLAP](https://arxiv.org/abs/2206.04769), [WavCaps-CNN14](https://arxiv.org/abs/2303.17395), and [WavCaps-HTSAT](https://arxiv.org/abs/2303.17395) papers.

<center>

| Model| ESC50| Libri<br>Count| CREMAD| Gunshot|SC 5hr|SC Full|Vox<br>Lingua|Vocal<br>Imitation|NSynth<br>Pitch<br>5hr|NSynth<br>Pitch<br>50hr|GTZAN<br>Genre|GTZAN<br>Music<br>Speech|Beijing<br>Opera<br>Percussion|
|----------|:-------------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| [LAION-CLAP-fusion](https://arxiv.org/abs/2211.06687) |  0.964|0.625| 0.566| 0.914| 0.693 | 0.758| 0.264| 0.155| 0172| 0.376 | 0.842| 0.962| 0.962|
| [LAION-CLAP](https://arxiv.org/abs/2211.06687) |**0.971**|0.659|0.557|0.845|0.693|0.774|0.189| 0.151| 0.180| 0.423|0.838|0.969|0.953|
| [MS-CLAP](https://arxiv.org/abs/2206.04769) |0.930|0.649|0.547|0.798|0.511|0.626|0.236| 0.106|0.112|0.274| 0.818|**0.992**|0.932|
|[WavCaps-CNN14](https://arxiv.org/abs/2303.17395)|0.962| 0.646| 0.556|0.789|0.583|0.640|0.270|0.158| 0.140|0.324|**0.861**| **0.992**|0.957|
|[WavCaps-HTSAT](https://arxiv.org/abs/2303.17395)|0.961| 0.690| 0.595| **0.929**| 0.752|0.806|0.234| 0.168| 0.256| 0.548|0.847| 0.962| 0.958|
|Stage1: AudioMAE (Ours)|0.841| **0.754**|**0.661**| 0.893| **0.841**|**0.893**| **0.439**| 0.161| **0.700**|**0.827**|0.828| 0.985| 0.945|
|Stage2: Cacophony (Ours)|0.970| 0.660| 0.593|0.833|0.680|0.762| 0.262|**0.191**| 0.420|0.726|0.850| 0.985|**0.970**|
</center>

## Acknowledgements

We are immensely grateful to the Google TPU Research Cloud (TRC) for generously providing the computational resources vital to our project. 
Their support has been invaluable.


We thank the FreeSound team from Pompeu Fabra University for providing us with the scraping API.
We thank the University of Rochester Goergen Institute for Data Science (GIDS) seed funding program.
We thank LAION CLAP team for collecting open source datesets and generously sharing them with the research community.