from archive.clean_screentext import clean_screentext
from merge_by_session import merge_by_session
from get_features import save_features
#select screentext, clean it, and extract linguistic features then save the features to a file
if __name__ == "__main__":
    input_file = 'step1_data/clean_df.json'
    #output_file = 'step3_data/linguistic_features.json'
    clean_screentext()
    #merge_by_session()
    #save_features()