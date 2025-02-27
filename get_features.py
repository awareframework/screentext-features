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
# Increase max_length to handle large texts (since we have sufficient RAM)
nlp.max_length = 15000000  # Set to ~15M chars to handle texts like the current one (14.1M chars)

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

# Maximum size of text to process at once (in characters)
MAX_TEXT_CHUNK_SIZE = 10000000  # 10MB

# Maximum number of tokens to process for NER and other intensive operations
MAX_TOKENS_FOR_INTENSIVE_ANALYSIS = 100000

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
    # Safety check - limit the text size for NER processing
    if len(text) > MAX_TOKENS_FOR_INTENSIVE_ANALYSIS:
        # If text is too large, truncate it
        text = text[:MAX_TOKENS_FOR_INTENSIVE_ANALYSIS]
        print(f"Warning: Text truncated to {MAX_TOKENS_FOR_INTENSIVE_ANALYSIS} characters for NER analysis.")
    
    try:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    except Exception as e:
        print(f"Error during NER processing: {e}")
        return []


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
    print(f"Processing file: {input_file}")
    session_texts = []
    total_file_size = os.path.getsize(input_file)
    print(f"File size: {total_file_size/1024/1024:.2f} MB")

    # Read and process each session (each line)
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        session = json.loads(line)
                        logs = session.get("screen_text_logs", [])
                        for log in logs:
                            text = log.get("text", "").strip()
                            if text:
                                segments = [seg.strip() for seg in text.split("||") if seg.strip()]
                                session_texts.extend(segments)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse JSON line: {e}")
                        continue
    except Exception as e:
        print(f"Error reading input file: {e}")
        return {}  # Return empty dict if file reading fails

    # Check if we have any valid text to process
    if not session_texts:
        print("Warning: No valid text segments found.")
        return {
            'error': 'No valid text segments found in input file',
            'detected_language': 'unknown',
            'lexical_diversity': 0
        }

    # Combine all segments into one full text for analysis, but split into chunks if too large
    full_text = " ".join(session_texts)
    features = {}
    
    print(f"Total text length: {len(full_text)} characters")

    # Language detection using langid.
    # Only use a sample of the text if it's very large
    lang_sample = full_text[:100000] if len(full_text) > 100000 else full_text
    try:
        lang, confidence = lang_classify(lang_sample)
        features['detected_language'] = lang
        features['language_detection_confidence'] = confidence
    except Exception as e:
        print(f"Error during language detection: {e}")
        features['detected_language'] = 'unknown'
        features['language_detection_confidence'] = 0

    features['contains_english'] = bool(re.search(r'[a-zA-Z]', full_text[:10000]))
    features['contains_chinese'] = bool(re.search(r'[\u4e00-\u9fff]', full_text[:10000]))

    # Lexical diversity - using a sample if text is very large
    sample_for_diversity = full_text[:500000] if len(full_text) > 500000 else full_text
    try:
        all_words = re.findall(r'\w+', sample_for_diversity)
        unique_words = set(all_words)
        features['lexical_diversity'] = len(unique_words) / len(all_words) if all_words else 0
        features['total_words'] = len(all_words)
        features['unique_words'] = len(unique_words)
    except Exception as e:
        print(f"Error calculating lexical diversity: {e}")
        features['lexical_diversity'] = 0
        features['total_words'] = 0
        features['unique_words'] = 0

    # For texts that contain English, compute additional features.
    if features['contains_english']:
        # Use a sample of the text if it's very large
        english_text = full_text[:500000] if len(full_text) > 500000 else full_text
        
        try:
            # Part-of-speech tagging - use a smaller sample if needed
            pos_sample = english_text[:200000] if len(english_text) > 200000 else english_text
            pos_tokens = re.findall(r'\w+', pos_sample)
            # Limit number of tokens for POS tagging
            if len(pos_tokens) > 100000:
                pos_tokens = pos_tokens[:100000]
                print(f"Warning: Limited to 100,000 tokens for POS tagging")
            
            pos_tags = nltk.pos_tag(pos_tokens)
            pos_counts = Counter(tag for word, tag in pos_tags)
            features['most_common_pos'] = pos_counts.most_common(3)
        except Exception as e:
            print(f"Error during POS tagging: {e}")
            features['most_common_pos'] = []

        try:
            # Named Entity Recognition (NER) - use a limited sample
            ner_sample = english_text[:100000] if len(english_text) > 100000 else english_text
            named_entities = extract_named_entities(ner_sample)
            features['named_entities'] = [ent[0] for ent in named_entities][:100]  # Limit to top 100 entities
            features['named_entity_types'] = [ent[1] for ent in named_entities][:100]
        except Exception as e:
            print(f"Error during NER: {e}")
            features['named_entities'] = []
            features['named_entity_types'] = []

        try:
            # Text complexity metrics - use a sample if needed
            complexity_sample = english_text[:100000] if len(english_text) > 100000 else english_text
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(complexity_sample)
            features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(complexity_sample)
        except Exception as e:
            print(f"Error calculating text complexity: {e}")
            features['flesch_reading_ease'] = 0
            features['flesch_kincaid_grade'] = 0

    # Sentiment analysis using VADER - use a sample if needed
    try:
        sentiment_sample = full_text[:100000] if len(full_text) > 100000 else full_text
        sentiment_scores = SentimentIntensityAnalyzer().polarity_scores(sentiment_sample)
        features['sentiment_compound'] = sentiment_scores['compound']
        features['sentiment_positive'] = sentiment_scores['pos']
        features['sentiment_negative'] = sentiment_scores['neg']
        features['sentiment_neutral'] = sentiment_scores['neu']
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        features['sentiment_compound'] = 0
        features['sentiment_positive'] = 0
        features['sentiment_negative'] = 0
        features['sentiment_neutral'] = 0

    # Emotion analysis using TextBlob - use a sample if needed
    try:
        emotion_sample = full_text[:100000] if len(full_text) > 100000 else full_text
        blob = TextBlob(emotion_sample)
        features['subjectivity'] = blob.sentiment.subjectivity
    except Exception as e:
        print(f"Error during emotion analysis: {e}")
        features['subjectivity'] = 0

    print("Feature extraction completed successfully")
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