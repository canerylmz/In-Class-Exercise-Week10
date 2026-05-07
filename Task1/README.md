# EE471 Week 10 - Task 1

This project implements the Hugging Face Transformers tasks from the in-class exercise:

- sentiment analysis
- zero-shot classification
- text generation
- fill-mask
- named entity recognition
- question answering validation
- summarization
- English to Turkish translation
- ViT image classification
- Whisper automatic speech recognition

The English to Turkish translation model is `Helsinki-NLP/opus-mt-tc-big-en-tr`.

## Recommended Python

Use the existing project conda Python 3.11 environment:

```powershell
..\..\.conda\python.exe --version
```

The global `python` command on this machine is Python 3.14, which is too new for many ML wheels. Python 3.11 is safer for PyTorch and Transformers.

## Install CPU Dependencies

From this folder:

```powershell
..\..\.conda\python.exe -m pip install -r requirements.txt
```

This installs CPU PyTorch from the PyTorch CPU wheel index plus the Hugging Face dependencies.

## Run Text Tasks

```powershell
..\..\.conda\python.exe task1.py
```

The first run downloads the models and may take several minutes.

To run only one text task:

```powershell
..\..\.conda\python.exe task1.py --task sentiment
..\..\.conda\python.exe task1.py --task zero-shot
..\..\.conda\python.exe task1.py --task generation
..\..\.conda\python.exe task1.py --task fill-mask
..\..\.conda\python.exe task1.py --task ner
..\..\.conda\python.exe task1.py --task qa
..\..\.conda\python.exe task1.py --task summarization
..\..\.conda\python.exe task1.py --task translation
```

## Run Image Classification

```powershell
..\..\.conda\python.exe task1.py --skip-text --image path\to\image.jpg
..\..\.conda\python.exe task1.py --task image --image path\to\image.jpg
```

The image model is `google/vit-base-patch16-224`.

## Run Speech Recognition

```powershell
..\..\.conda\python.exe task1.py --skip-text --audio path\to\audio.wav
..\..\.conda\python.exe task1.py --task asr --audio path\to\audio.wav
```

The ASR model is `openai/whisper-large-v3`. This model is large, so CPU inference can be slow.

## Run Everything

```powershell
..\..\.conda\python.exe task1.py --image path\to\image.jpg --audio path\to\audio.wav
```
