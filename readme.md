# Email Spam Checker

This is my AI-powered spam detection tool built with Python and machine learning.

![Spam Detection](spam_detected.png)

## What It Does

My application uses machine learning model to detect spam emails with high accuracy. It analyzes text patterns and identifies suspicious words commonly found in spam messages.

## Features

- **AI-Powered Detection**: Uses Naive Bayes classification trained on 5,572 real messages
- **Visual Confidence Indicators**: Shows prediction confidence with progress bars
- **Suspicious Word Highlighting**: Identifies which words triggered spam detection
- **Easy Testing**: Built-in example buttons for quick demonstrations
- **Real-Time Analysis**: Instant predictions with detailed explanations

## How It Works

The model was trained on the SMS Spam Collection dataset from Kaggle, learning to recognize patterns in spam vs non-spam messages using:

- **TF-IDF Vectorization**: Converts text into numerical features
- **Naive Bayes Classification**: Probabilistic model effective for text classification
- **5,572 Training Examples**: Real spam and non-spam messages

## The language and libraries used.

- Python 3.x
- Streamlit (Web interface)
- scikit-learn (Machine learning)
- pandas (Data handling)

## Installation

1. Clone this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
streamlit run spam_detector.py
```

Then:
1. Paste an email into the text box, or click an example button
2. Click "Check for Spam"
3. View the results with confidence scores and highlighted suspicious words

## Screenshots

### Spam Detected
![Spam Detection](spam_detected.png)

### Safe Email
![Safe Email](safe_email.png)

## What I have Learned

- Training machine learning models with real datasets
- Text preprocessing and feature extraction with TF-IDF
- Building interactive web applications with Streamlit
- Implementing classification algorithms
- Understanding spam detection patterns

## Future Improvements I would like to make

- Add support for email attachments analysis
- Implement additional ML models for comparison
- Add user feedback to improve accuracy
- Create API for integration with email clients

## Author

Jon - Career changer learning AI and cybersecurity

## Dataset

SMS Spam Collection from Kaggle (5,572 messages)