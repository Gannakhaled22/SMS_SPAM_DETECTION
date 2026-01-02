# SMS_SPAM_DETECTION
# SMS Spam Detection

A simple, clear Google Colab notebook for detecting spam in SMS messages. The notebook loads data, cleans and preprocesses text, builds and evaluates models, and includes notes for deployment.

## Quick start
1. Open the notebook in Google Colab (link below) and run cells top to bottom.
2. Mount Google Drive and place the dataset at: `/content/drive/MyDrive/spam.csv` (or change the file path in the notebook).
3. Run the install cell that runs:
   ```
   !pip install tensorflow flask scikit-learn pandas numpy flask-ngrok
   ```
4. The notebook downloads required NLTK data (punkt, punkt_tab, stopwords).

Notebook permalink:
https://github.com/Gannakhaled22/SMS_SPAM_DETECTION/blob/d18f9628bbbb6565bc0dfcefa506f67893474655/sms/sms_spam_detection_colab.ipynb

## Requirements
- Python 3.x (Colab recommended)
- Libraries used (installed in notebook): TensorFlow, Keras, scikit-learn, pandas, numpy, nltk, Flask, flask-ngrok

## What the notebook does (high level)
- Load and inspect the SMS dataset (spam.csv).
- Clean data and drop unused columns.
- Text preprocessing (tokenization, stopword removal, basic text cleaning).
- Convert text to numerical features (TF-IDF or Keras Tokenizer → sequences/embeddings).
- Train a classifier (examples in the notebook: a simple Keras model; scikit-learn models can be used).
- Evaluate with accuracy, precision, recall, and confusion matrix.
- Notes about saving the model and serving via Flask + ngrok.

## Features used (what’s extracted / computed)
- Raw text (SMS message)
- Label: ham / spam
- Tokenization (NLTK)
- Stopwords removal
- Text cleaning (lowercasing, removing non-alphanumerics)
- Vector representation options:
  - TF-IDF vectors (scikit-learn)
  - Tokenizer + embedding layers (Keras/TensorFlow)
- Train/test split and evaluation metrics:
  - Accuracy, Precision, Recall, F1-score, Confusion Matrix

## Tips & next steps
- Try different models: Logistic Regression, SVM, simple LSTM, or transformer-based models.
- Improve preprocessing: lemmatization, spelling correction, emoji handling.
- Address class imbalance with oversampling/undersampling or class weights.
- Save the best model and serve it using a small Flask API (notebook includes hints).


