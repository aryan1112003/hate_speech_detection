# Hate Speech Detection 🛡️

## Overview

This project aims to detect hate speech in textual data using Natural Language Processing (NLP) and deep learning techniques. Leveraging transformer-based models, it classifies text inputs as either hateful or non-hateful, providing a tool to combat online abuse and promote safer digital spaces.

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [License](#license)

## Project Structure

```
hate_speech_detection/
├── app.py
├── contextual_sentence_abuse_dataset.csv
├── dataset1.csv
├── saved_model/
├── results/
│   └── checkpoint-1500/
├── logs/
├── certificate.pem
├── rng_state.pth
├── scheduler.pt
├── trainer_state.json
```

- `app.py`: Main application script for inference.
- `contextual_sentence_abuse_dataset.csv` & `dataset1.csv`: Datasets used for training and evaluation.
- `saved_model/`: Directory containing the trained model.
- `results/`: Checkpoints and training artifacts.
- `logs/`: Training logs.
- `certificate.pem`: SSL certificate for secure connections.
- `rng_state.pth`, `scheduler.pt`, `trainer_state.json`: Files related to training state and scheduling.

## Dataset

The project utilizes two primary datasets:

1. **Contextual Sentence Abuse Dataset**: Contains labeled sentences indicating the presence or absence of hate speech.

2. **Dataset1**: An additional dataset to enhance model robustness.

*Note: Ensure that the datasets are preprocessed appropriately before training.*

## Model Architecture

The model is built upon transformer-based architectures, specifically fine-tuning pre-trained models like BERT for the task of hate speech classification. This approach allows the model to understand contextual nuances in text, leading to more accurate predictions.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/aryan1112003/hate_speech_detection.git
   cd hate_speech_detection
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   *Ensure that `requirements.txt` includes all necessary packages such as `transformers`, `torch`, `scikit-learn`, etc.*

## Usage

1. **Running the Application**:

   The `app.py` script serves as the main entry point for the application. To run the application:

   ```bash
   python app.py
   ```

   *This will start the application, allowing you to input text and receive predictions on whether the input constitutes hate speech.*

2. **Inference Example**:

   Within the application, you can input sentences, and the model will output predictions indicating the likelihood of the text being hateful.

## Evaluation

The model's performance is evaluated using standard classification metrics:

- **Accuracy**: Measures the overall correctness of the model.

- **Precision**: Indicates the proportion of positive identifications that were actually correct.

- **Recall**: Measures the proportion of actual positives that were correctly identified.

- **F1-Score**: Harmonic mean of precision and recall, providing a balance between the two.

*Evaluation results and metrics can be found in the `logs/` directory.*

