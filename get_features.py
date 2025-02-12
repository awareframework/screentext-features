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
from langid import classify as lang_classify  # simple language classifier
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
import spacy
import argparse
import yaml
# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

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
    # Attempt to load tokenizers/punkt_tab/english/
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')


def combine_text(df):
    """Combine text from all rows in a DataFrame.

    Splits each text by the '||' delimiter, removes duplicates, and omits empty strings.

    Args:
        df (pandas.DataFrame): DataFrame containing a 'text' column with textual data.

    Returns:
        list: A list of unique text segments.
    """
    all_segments = [segment.strip() for text in df['text'] for segment in text.split('||')]
    return list(set(segment for segment in all_segments if segment))


def is_chinese(char):
    """Check if a given character is a Chinese character.

    Args:
        char (str): A single character string.

    Returns:
        bool: True if the character is Chinese, False otherwise.
    """
    return '\u4e00' <= char <= '\u9fff'


def count_sentences(english_text, chinese_text):
    """Count the total number of sentences in English and Chinese texts.

    Splits text using common sentence-ending punctuation and newlines.

    Args:
        english_text (str): Text assumed to be in English.
        chinese_text (str): Text assumed to be in Chinese.

    Returns:
        int: Total number of sentences.
    """
    # Count English sentences
    english_sentences = re.split(r'(?<=[.!?])\s+|\n+', english_text.strip())
    english_sentences = [s for s in english_sentences if s.strip()]
    
    # Count Chinese sentences
    chinese_sentences = re.split(r'[。！？\.\!?]\s*|\n+', chinese_text.strip())
    chinese_sentences = [s for s in chinese_sentences if s.strip()]
    
    return len(english_sentences) + len(chinese_sentences)


def extract_named_entities(text):
    """Extract named entities from a text using spaCy.

    Args:
        text (str): The input text.

    Returns:
        list: A list of tuples, where each tuple contains the entity text and its label.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


def extract_linguistic_features(input_file):
    """Extract linguistic features from a JSONL file containing session data.

    The function processes each JSON object (session) in the input file, splits the
    screen text logs by "||" to obtain segments, and computes a variety of linguistic
    metrics, including language detection, lexical diversity, part-of-speech tagging,
    named entity recognition, sentiment analysis, and text complexity.

    Args:
        input_file (str): Path to the JSONL file containing session data.

    Returns:
        dict: A dictionary containing the extracted linguistic features.
    """
    session_texts = []

    # Read and process each session (each line)
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                session = json.loads(line)
                logs = session.get("screen_text_logs", [])
                for log in logs:
                    text = log.get("text", "").strip()
                    if text:
                        segments = [seg.strip() for seg in text.split("||") if seg.strip()]
                        session_texts.extend(segments)

    # Combine all segments into one full text for analysis.
    full_text = " ".join(session_texts)
    features = {}

    # Language detection using langid.
    lang, confidence = lang_classify(full_text)
    features['detected_language'] = lang
    features['language_detection_confidence'] = confidence

    features['contains_english'] = bool(re.search(r'[a-zA-Z]', full_text))
    features['contains_chinese'] = bool(re.search(r'[\u4e00-\u9fff]', full_text))

    # Lexical diversity
    all_words = re.findall(r'\w+', full_text)
    unique_words = set(all_words)
    features['lexical_diversity'] = len(unique_words) / len(all_words) if all_words else 0

    # For texts that contain English, compute additional features.
    if features['contains_english']:
        english_text = full_text

        # Part-of-speech tagging
        pos_tags = nltk.pos_tag(re.findall(r'\w+', english_text))
        pos_counts = Counter(tag for word, tag in pos_tags)
        features['most_common_pos'] = pos_counts.most_common(3)

        # Named Entity Recognition (NER)
        named_entities = extract_named_entities(english_text)
        features['named_entities'] = [ent[0] for ent in named_entities]
        features['named_entity_types'] = [ent[1] for ent in named_entities]

        # Text complexity metrics (Flesch Reading Ease and Flesch-Kincaid Grade)
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(english_text)
        features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(english_text)

    # Sentiment analysis using VADER
    sentiment_scores = SentimentIntensityAnalyzer().polarity_scores(full_text)
    features['sentiment_compound'] = sentiment_scores['compound']
    features['sentiment_positive'] = sentiment_scores['pos']
    features['sentiment_negative'] = sentiment_scores['neg']
    features['sentiment_neutral'] = sentiment_scores['neu']

    # Emotion analysis using TextBlob
    blob = TextBlob(full_text)
    features['subjectivity'] = blob.sentiment.subjectivity

    return features


def save_features(input_file=None, output_file=None):
    """
    Extract linguistic features from the input file and save them to the output file.
    The output is saved in CSV format by default, but supports JSON and JSONL formats based
    on the provided file extension. In addition, the features are always saved to a corresponding
    YAML file regardless of the output format provided.

    Args:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path to the output file. The output format is determined by the file extension:
            - .csv or unspecified: outputs as CSV with each feature as a column.
            - .json: outputs as formatted JSON.
            - .jsonl: outputs as JSON Lines.
            The base name of the output file is used to also save a YAML file.
    """
    features = extract_linguistic_features(input_file)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    ext = os.path.splitext(output_file)[1].lower()

    if ext == ".json":
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(features, f, indent=4)
        print("Features extracted and saved to (JSON):", output_file)
    elif ext == ".jsonl":
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(features) + "\n")
        print("Features extracted and saved to (JSONL):", output_file)
    else:
        # Default to CSV format: write keys as header columns and their values as one row.
        import csv
        safe_features = {}
        for key, value in features.items():
            if isinstance(value, (list, dict)):
                safe_features[key] = json.dumps(value)
            else:
                safe_features[key] = value

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(safe_features.keys()))
            writer.writeheader()
            writer.writerow(safe_features)
        print("Features extracted and saved to (CSV):", output_file)

    # Always also save to YAML format.
    yaml_output_file = os.path.splitext(output_file)[0] + ".yaml"
    with open(yaml_output_file, 'w', encoding='utf-8') as yaml_file:
        yaml.dump(features, yaml_file, default_flow_style=False, allow_unicode=True)
    print("Features also extracted and saved to (YAML):", yaml_output_file)

def main():
    """
    Extract linguistic features from session data files and save them as JSON files.
    Uses a base input directory (default "step1_data") and base output directory (default "step2").
    If a participant is provided, it processes only that participant folder;
    otherwise, it processes all participant folders found in the base input directory.
    """
    parser = argparse.ArgumentParser(
        description="Extract and save linguistic features from session data files."
    )
    parser.add_argument('--base_input_dir', type=str, default='step1_data',
                        help='Base input directory containing participant folders (default: step1_data)')
    parser.add_argument('--base_output_dir', type=str, default='step2_data',
                        help='Base output directory for extracted linguistic features (default: step2_data)')
    parser.add_argument('--participant', '-p', type=str, default=None,
                        help='Participant folder name to process. If not specified, process all participants.')
    parser.add_argument('--input_filename', type=str, default='clean_input.jsonl',
                        help='Input file name inside each participant folder (default: clean_input.jsonl)')
    parser.add_argument('--output_filename', type=str, default='linguistic_features.json',
                        help='Output file name for the extracted features (default: linguistic_features.json)')
    args = parser.parse_args()

    if args.participant:
        input_file = os.path.join(args.base_input_dir, args.participant, args.input_filename)
        output_dir = os.path.join(args.base_output_dir, args.participant)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, args.output_filename)
        features = extract_linguistic_features(input_file)
        save_features(features, output_file)
    else:
        # Process all participant folders under the base input directory
        for participant in os.listdir(args.base_input_dir):
            participant_dir = os.path.join(args.base_input_dir, participant)
            if os.path.isdir(participant_dir):
                input_file = os.path.join(participant_dir, args.input_filename)
                output_dir = os.path.join(args.base_output_dir, participant)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, args.output_filename)
                features = extract_linguistic_features(input_file)
                save_features(features, output_file)


if __name__ == "__main__":
    main()

    #attention
    #session time, sd, avg, reading speed estmation (words / min)
    # count of session
    # word embeeding file
    #a text input for parameter: session thershold,, time-window, n top words for xyz