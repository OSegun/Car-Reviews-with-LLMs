# 🚗 Car-Reviews-with-LLMs

## Project Overview

An AI prototype application leveraging pre-trained Hugging Face Large Language Models (LLMs). This application can address diverse customer inquiries including sentiment analysis of car reviews, language translation, question answering, and text summarization.


## 🌟 Task

This project prototype demonstrates various NLP/AI capabilities through four key tasks:

1. **Sentiment Analysis**: Classifies car reviews as positive or negative
2. **Language Translation**: Translates English car reviews to Spanish for international customers
3. **Question Answering**: Extracts specific information from car reviews based on customer queries
4. **Text Summarization**: Condenses lengthy car reviews into concise summaries

## 🔍 Implementation Details

### Task 1: Sentiment Analysis

Uses the `distilbert-base-uncased-finetuned-sst-2-english` model to classify car review sentiment:

- **Input**: Five car reviews from the `car_reviews.csv` dataset
- **Processing**: Applies the pre-trained sentiment analysis model to determine if each review is positive or negative
- **Output**: Binary sentiment classification (0 for negative, 1 for positive)
- **Evaluation**: Accuracy score and F1 score to assess classification performance

```python
# Example code snippet for sentiment analysis
model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
predicted_labels = model(reviews)
predictions = [1 if label['label'] == "POSITIVE" else 0 for label in predicted_labels]
```

### Task 2: Language Translation

Uses the `Helsinki-NLP/opus-mt-en-es` model to translate English car reviews to Spanish:

- **Input**: First two sentences of the first review in the dataset
- **Processing**: Applies the pre-trained English-to-Spanish translation model
- **Output**: Spanish translation of the input text
- **Evaluation**: BLEU score to assess translation quality against reference translations

```python
# Example code snippet for translation
translate_model = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
translated_review = translate_model(first_two_sentences)[0]["translation_text"]
```

### Task 3: Question Answering

Uses the `deepset/minilm-uncased-squad2` model to answer specific questions about car reviews:

- **Input**: Car review text as context and a specific question ("What did he like about the brand?")
- **Processing**: Extracts the answer from the context using the QA model
- **Output**: Extracted text answering the specific question
- **Evaluation**: Manual assessment of answer relevance and accuracy

```python
# Example code snippet for question answering
model_QA = AutoModelForQuestionAnswering.from_pretrained("deepset/minilm-uncased-squad2")
tokenizer = AutoTokenizer.from_pretrained("deepset/minilm-uncased-squad2")
token = tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = model_QA(**token)
```

### Task 4: Text Summarization

Uses the `cnicu/t5-small-booksum` model to create concise summaries of car reviews:

- **Input**: Full text of a car review
- **Processing**: Generates a condensed summary with specified maximum length
- **Output**: Summary of approximately 50-55 tokens
- **Evaluation**: Manual assessment of summary quality and comprehensiveness

```python
# Example code snippet for summarization
model_sum = pipeline("summarization", model="cnicu/t5-small-booksum")
summarized_text = model_sum(review_text, max_length=55)[0]["summary_text"]
```

## 💻 Technical Requirements/Tools

- Python 3.11
- PyTorch
- Transformers library by Hugging Face
- Pandas for data handling
- Evaluate library for metrics calculation

Required packages:
```
torch
transformers
pandas
evaluate
```

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/car-reviws-with-llms
   cd car-reviews-with-llms
   ```

2. **Install the dependencies**:
   ```bash
   torch
   transformers
   pandas
   ```

3. **Download the data**:
   Ensure the following files are in the `data/` directory:
   - `car_reviews.csv`: Contains car reviews for sentiment analysis
   - `reference_translations.txt`: Contains reference translations for BLEU score calculation

4. **Run the Jupyter notebook**:
   ```bash
   jupyter notebook notebook.ipynb
   ```

## 📊 Evaluation Metrics

The project uses the following metrics to evaluate model performance:

- **Sentiment Analysis**: Accuracy (80%) and F1 score (0.85)
- **Translation**: BLEU score (0.65) against reference translations
- **Question Answering**: Manual verification of extracted answers
- **Summarization**: Ensuring summaries contain key information in the target length

## 🔮 Future Improvements

1. **Multi-language Support**: Expand translation capabilities to more languages beyond Spanish
2. **Fine-tuning Models**: Adapt pre-trained models with automotive domain-specific data
3. **Interactive Interface**: Develop a user-friendly web or mobile interface for the chatbot
4. **Intent Recognition**: Add better classification of customer intent for more accurate responses
5. **Sentiment Analysis Improvements**: Fine-tune models specifically for automotive reviews to improve accuracy

