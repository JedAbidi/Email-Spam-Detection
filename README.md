
## Implementation Steps

### 1. Data Preprocessing
- **Duplicate Removal**: Removed 33 duplicate entries
- **Text Cleaning**: 
  - Removed non-word characters using regex
  - Converted to lowercase
  - Removed extra whitespaces
- **Tokenization**: Split text into individual words using NLTK
- **Stopword Removal**: Filtered out common English stopwords

### 2. Feature Engineering
- **Text Vectorization**: Converted tokens to integer sequences using Keras Tokenizer
- **Padding**: Ensured uniform sequence length with post-padding

### 3. Model Architecture
- **Embedding Layer**: 128-dimensional word embeddings
- **Bidirectional LSTM**: 64 units with dropout regularization
- **Dense Layers**: 64 and 32 units with ReLU activation
- **Output Layer**: Single neuron with sigmoid activation for binary classification

### 4. Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Early Stopping**: Monitored validation loss with patience=3
- **Batch Size**: 32
- **Train/Validation/Test Split**: 60%/20%/20%

## Results

- **Training Accuracy**: ~100%
- **Validation Accuracy**: 98.95%
- **Test Accuracy**: 98.68%

## Dependencies

```txt
tensorflow>=2.0.0
pandas
nltk
scikit-learn
matplotlib

Installation & Usage
Clone the repository:

bash
git clone https://github.com/yourusername/spam-detection.git
cd spam-detection
Install required packages:

bash
pip install tensorflow pandas nltk scikit-learn matplotlib
Download NLTK data:

python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
Open and run the Jupyter notebook Untitled4 (1).ipynb to train and evaluate the model.

Code Implementation Highlights
Text Cleaning Function
python
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.lower()
Model Architecture
python
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
