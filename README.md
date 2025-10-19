# Email Spam Detection

## Overview

This project develops a machine learning model to detect spam emails using a dataset of email texts. It preprocesses email content, removes duplicates, tokenizes text, and trains a Bidirectional LSTM model with TensorFlow/Keras to classify emails as spam or non-spam (binary classification). The project includes data visualization and model evaluation.

## Dataset

- **Source**: Custom dataset (`emails.csv`) with preprocessed version (`cleaned_emails.csv`).
- **Description**: Contains 5,728 email records (5,695 after duplicate removal), with columns:
  - `text`: Email content with subjects.
  - `spam`: Binary label (1 for spam, 0 for non-spam).
- **Size**: Original: 8,745 KB; Cleaned: 16,301 KB.

## Features

- **Data Preprocessing**: Removes duplicates, cleans text (removes non-word characters, extra spaces), and applies tokenization with stopword removal.
- **Model**: Uses a Bidirectional LSTM with Embedding, Dense, and Dropout layers for spam classification.
- **Evaluation**: Splits data into 60% training, 20% validation, and 20% test sets; reports accuracy on the test set.

## Requirements

- Python 3.x
- Libraries (listed in `requirements.txt`):
  ```
  tensorflow
  matplotlib
  pandas
  re
  nltk
  ```
- Optional: GPU support (e.g., NVIDIA) with CUDA and cuDNN for faster training.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/JedAbidi/Email-Spam-Detection.git
   cd email-spam-detection
   ```

2. **Set Up**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset**:
   - Ensure `emails.csv` (8,745 KB) is in the project directory.
   - The notebook generates `cleaned_emails.csv` (16,301 KB) during execution.

## Usage

1. **Run the Notebook**:
   ```bash
   jupyter notebook Email_Spam_Detection.ipynb
   ```

2. **Key Steps**:
   - **Load Data**: Reads `emails.csv` and displays column info.
   - **Preprocessing**: Removes duplicates, cleans text, tokenizes, and removes stopwords.
   - **Model Training**: Trains the LSTM model with early stopping on validation data.
   - **Evaluation**: Reports test accuracy (e.g., ~98.68% in sample run).

3. **Outputs**:
   - Cleaned dataset saved as `cleaned_emails.csv`.
   - Test accuracy printed after model evaluation.

## Project Structure

```
email-spam-detection/
├── .idea                  # IDE configuration folder
├── cleaned_emails.csv    # Preprocessed dataset (16,301 KB)
├── Email_Spam_Detection.ipynb  # Main Jupyter Notebook
├── emails.csv            # Original dataset (8,745 KB)
├── requirements.txt      # Dependencies
├── README.md             # This file
```

## Results

- Removed 33 duplicates, reducing the dataset to 5,695 emails.
- Achieved a test accuracy of approximately 98.68% with the LSTM model after 4 epochs (early stopping applied).
- Text cleaning and tokenization successfully prepared data for model input.

## Future Improvements

- Add advanced text preprocessing (e.g., stemming, lemmatization).
- Experiment with other models (e.g., CNN, Transformers).
- Implement cross-validation for robust performance metrics.
- Optimize model hyperparameters for better efficiency.

## Acknowledgments

- [NLTK](https://www.nltk.org) for text processing tools.
- [TensorFlow](https://www.tensorflow.org) for the deep learning framework.
- The open-source community for supporting libraries.
