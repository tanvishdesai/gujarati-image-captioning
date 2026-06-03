# Gujarati Image Captioning

An end-to-end Deep Learning project for generating accurate image captions in the Gujarati language. This model is trained on standard datasets (like Flickr8k/Flickr30k) that have been translated to Gujarati, utilizing an Encoder-Decoder architecture to bridge computer vision and natural language processing.

## Architecture

- **Image Encoder**: Uses a pre-trained Convolutional Neural Network (CNN) to extract rich feature vectors from input images.
- **Text Decoder**: Utilizes Recurrent Neural Networks (RNNs/GRUs) combined with an Attention Mechanism to generate context-aware Gujarati sentences word-by-word.

## Features

- Custom tokenization and vocabulary generation for the Gujarati language.
- Multiple iterative model versions (`V2` to `V9`) reflecting architectural improvements (e.g., adding Attention, switching from LSTM to GRU).
- Evaluation scripts calculating BLEU scores to objectively measure translation/captioning quality (`BLEU_v9.py`).
- Inference pipeline for generating captions on unseen images.

## Getting Started

1. Ensure you have the required datasets in the `flickr8k` or appropriate directory.
2. The core training loops and model definitions can be run via the later version scripts (e.g., `python V9.py` or through Google Colab using `colab.py`).
3. To test the model on a single image, use `inference.py`.
