# Genre Classification with Neural Networks

## Overview

This code is designed for genre classification using neural networks. It leverages the TensorFlow and Hugging Face Transformers libraries for building, training, and evaluating a text classification model.

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Visualization](#visualization)

## Requirements

Make sure you have the following libraries installed:

```bash
pip install numpy matplotlib seaborn pandas tensorflow transformers
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/shreyuu/CODSOFT.git
cd MachineLearning/MOVIE_GENRE_CLASSIFICATION
```

2. Run the main script:

```bash
python main.py
```

## Data

The training and testing data are expected to be in specific formats as mentioned in the code (`train_data.txt` and `test_data_solution.txt`). Ensure that your data conforms to these requirements.

## Preprocessing

- The code performs text cleaning, removing stopwords, and converting text to lowercase.
- Tokenization of sentences and labels.
- Special character handling in labels.

## Model Architecture

The neural network model is a simple sequential model with the following layers:

1. Embedding layer
2. Global Average Pooling layer
3. Dense layers with ReLU activation and dropout for regularization
4. Softmax output layer

## Training

The model is compiled with sparse categorical cross-entropy loss and Adam optimizer. It is trained on the provided training data for a specified number of epochs.

```python
history = model.fit(train_padded_seq, train_label_seq, epochs=2, validation_data=(val_padded_seq, val_label_seq))
```

## Results

The training history is visualized using matplotlib to assess the model's performance in terms of accuracy and loss.

## Visualization

Matplotlib is used to plot training and validation accuracy as well as training and validation loss.
