from get_screentext import get_screentext_between_time
from clean_screentext import clean_screentext
from get_features import save_features
#select screentext, clean it, and extract linguistic features then save the features to a file
if __name__ == "__main__":
    input_file = 'data/screentext.json'
    output_file = 'step1_data/selected_screentext.json'
    start_time = 20240912001327
    end_time = 20240912132730
    get_screentext_between_time(input_file, output_file, start_time, end_time)
    clean_screentext()
    save_features()