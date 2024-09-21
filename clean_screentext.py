import os
import json
import re
import pandas as pd

# clean the screentext and save a list of unique text as a json file
def clean_screentext():
    input_file = 'step1_data/selected_screentext.json'
    with open(input_file, 'r') as file:
        screentext_data = json.load(file)

    # Convert to DataFrame first
    df = pd.DataFrame(screentext_data)

    # Print column names
    print("DataFrame columns:", df.columns)
    print("DataFrame head:")
    print(df.head())

    df = remove_rect_df(df)
    
   # Split the text by '||', clean each segment, and rejoin
    df['text'] = df['text'].apply(
        lambda x: '||'.join([segment.strip() for segment in x.split('||')]))

    # Split text and get unique segments
    unique_segments = combine_text(df)

    #Check if the step2_data directory exists
    if not os.path.exists("step2_data"):
        os.makedirs("step2_data")
    #save the list as a json file
    with open(os.path.join("step2_data", "clean_screentext.json"), 'w') as file:
        json.dump(unique_segments, file)

# Remove the Rect() from the text
def remove_rect(text):
    #split the text by all occurences of ***Rect()
    text = re.sub(r'\*\*\*Rect\([^)]*\)', '', text)
    return text

# Remove the Rect() from the df
def remove_rect_df(df):
    df['text'] = df['text'].apply(remove_rect)
    return df

# Combine text from all rows as a list of strings, with no duplicates
def combine_text(df):
    # Split all texts by "||" and flatten the resulting list
    all_segments = [segment.strip() for text in df['text'] for segment in text.split('||')]
    # Remove empty strings and return unique segments
    return list(set(segment for segment in all_segments if segment))


if __name__ == "__main__":
    clean_screentext()
