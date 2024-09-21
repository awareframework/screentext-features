import csv
import re
import json
import os
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
#from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import jieba
from langid.langid import LanguageIdentifier, model
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
#nltk.download('maxent_ne_chunker')
nltk.download('words')
#nltk.download('maxent_ne_chunker_tab')

# Ensure the 'punkt' and 'averaged_perceptron_tagger_eng' resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

def is_chinese(char):
    return '\u4e00' <= char <= '\u9fff'

def count_sentences(english_text, chinese_text):
    # Count English sentences (split by newlines or sentence-ending punctuation)
    english_sentences = re.split(r'(?<=[.!?])\s+|\n+', english_text.strip())
    english_sentences = [s for s in english_sentences if s.strip()]
    # print(f"English sentences: {len(english_sentences)}")
    # print(f"English sentences: {english_sentences}")
    
    # Count Chinese sentences (split by newlines or any punctuation)
    chinese_sentences = re.split(r'[。！？\.\!?]\s*|\n+', chinese_text.strip())
    chinese_sentences = [s for s in chinese_sentences if s.strip()]
    # print(f"Chinese sentences: {len(chinese_sentences)}")
    # print(f"Chinese sentences: {chinese_sentences}")
    
    # Combine counts
    total = len(english_sentences) + len(chinese_sentences)
    # print(f"Total sentences: {total}")
    return total

def extract_named_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_linguistic_features():
    # Load the text data from a file
    with open('step2_data/clean_screentext.json', 'r', encoding='utf-8') as f:
        text_data = f.read()

    data = json.loads(text_data)
    features = {}
    full_text = ' '.join(data)
    
    # Language detection
    lang, confidence = identifier.classify(full_text)
    features['detected_language'] = lang
    features['language_detection_confidence'] = confidence
    
    features['contains_english'] = bool(re.search(r'[a-zA-Z]', full_text))
    features['contains_chinese'] = bool(re.search(r'[\u4e00-\u9fff]', full_text))
    
    # Separate Chinese and English text
    chinese_text = ''.join(char for char in full_text if is_chinese(char))
    english_text = ''.join(char for char in full_text if char.isascii())
    

    # print(f"Full text: {full_text[:100]}...")  # Print first 100 chars
    # print(f"Chinese text length: {len(chinese_text)}")
    # print(f"English text length: {len(english_text)}")

    # Character counts
    features['num_chinese_chars'] = len(chinese_text)
    features['num_english_chars'] = len(re.findall(r'[a-zA-Z]', english_text))
    features['num_digits'] = len(re.findall(r'\d', full_text))
    features['num_punctuation_marks'] = len(re.findall(r'[^\w\s]', full_text))
    
    # English-specific features
    if features['contains_english']:
        english_words = word_tokenize(english_text)
        english_words = [word.lower() for word in english_words if word.isalpha()]
        features['english_word_count'] = len(english_words)
        
        if english_words:
            # cast to 2 decimal places
            features['avg_english_word_length'] = round(sum(len(word) for word in english_words) / len(english_words), 2)
            
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in english_words if word not in stop_words]
            
            word_freq = FreqDist(filtered_words)
            features['top_5_english_words'] = ', '.join([word for word, _ in word_freq.most_common(5)])
    
    # Chinese-specific features
    if features['contains_chinese']:
        chinese_words = jieba.lcut(chinese_text)
        chinese_words = [word for word in chinese_words if word.strip()]
        features['chinese_word_count'] = len(chinese_words)
        
        if chinese_words:
            chinese_word_freq = FreqDist(chinese_words)
            features['top_5_chinese_words'] = ', '.join([word for word, _ in chinese_word_freq.most_common(5)])
    
    # Use the count_sentences function
    sentence_count = count_sentences(english_text, chinese_text) or 1
    # print(f"Final sentence count: {sentence_count}")
    features['sentence_count'] = sentence_count

    # Other features (URLs, emails, time, emojis)
    features['contains_url'] = bool(re.search(r'(https?:\/\/)?[\w\-]+(\.[\w\-]+)+[\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#]', full_text))
    features['contains_email'] = bool(re.search(r'[^@]+@[^@]+\.[^@]+', full_text))
    features['contains_time'] = bool(re.search(r'\d{1,2}:\d{2}', full_text))
    features['contains_emoji'] = bool(re.search(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', full_text))
    
    # Lexical diversity
    all_words = []
    print("English words in locals:", 'english_words' in locals())
    print("Chinese words in locals:", 'chinese_words' in locals())
    if 'english_words' in locals() and english_words:
        all_words.extend(english_words)
    if 'chinese_words' in locals() and chinese_words:
        all_words.extend(chinese_words)
    unique_words = set(all_words)
    features['lexical_diversity'] = len(unique_words) / len(all_words) if all_words else 0

    # Part-of-speech tagging for English
    if features['contains_english']:
        pos_tags = nltk.pos_tag(english_words)
        pos_counts = Counter(tag for word, tag in pos_tags)
        features['most_common_pos'] = pos_counts.most_common(3)

        # Named Entity Recognition
        named_entities = extract_named_entities(english_text)
        features['named_entities'] = [ent[0] for ent in named_entities]
        features['named_entity_types'] = [ent[1] for ent in named_entities]

    # Text complexity
    if features['contains_english']:
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(english_text)
        features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(english_text)

    # Sentiment analysis
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = sentiment_analyzer.polarity_scores(full_text)
    features['sentiment_compound'] = sentiment_scores['compound']
    features['sentiment_positive'] = sentiment_scores['pos']
    features['sentiment_negative'] = sentiment_scores['neg']
    features['sentiment_neutral'] = sentiment_scores['neu']

    # Emotion detection (using TextBlob for simplicity)
    blob = TextBlob(full_text)
    features['subjectivity'] = blob.sentiment.subjectivity

    
    return features

def save_features(filename='step3_data/linguistic_features.csv'):
    features = extract_linguistic_features()
    fieldnames = list(features.keys())
    #Check if the step3_data directory exists
    if not os.path.exists("step3_data"):
        os.makedirs("step3_data")

    # Create or overwrite the file with headers
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(features)

    print("Advanced linguistic features extracted and saved to CSV file.")