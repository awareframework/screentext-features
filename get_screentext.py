import pandas as pd
import os

# a function that get screentext between 2 timestamps. 
# Taking input file path, output file path, start and end time as input
# convert the start and end time to unix timestamp and filter the screentext between the start and end timestamp
# The timestamp in json is 1724248833871.0, which is in milliseconds
# %Y: Year with century (e.g., 2024)
# %m: Month as a zero-padded decimal number (01-12)
# %d: Day of the month as a zero-padded decimal number (01-31)
# %H: Hour (24-hour clock) as a zero-padded decimal number (00-23)
# %M: Minute as a zero-padded decimal number (00-59)
# %S: Second as a zero-padded decimal number (00-59)
# input time can be in the format of 20240822000000 
def get_screentext_between_time(input_file, output_file, start_time, end_time):
    #read screentext.json
    df = pd.read_json(input_file, convert_dates=False)
    print(start_time)
    print(end_time)
    #convert start and end time to unix timestamp
    start_time = pd.to_datetime(str(start_time), format='%Y%m%d%H%M%S').timestamp()*1000
    end_time = pd.to_datetime(str(end_time), format='%Y%m%d%H%M%S').timestamp()*1000
    #convert the start and end timestamp from utc+10 to standard timestamp
    start_time = start_time - 36000000
    end_time = end_time - 36000000
    print(start_time)
    print(end_time)
    #filter screentext between start and end time
    df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

    #     {
    #     "_id": 243405,
    #     "timestamp": 1724317200779,
    #     "device_id": "6c826e40-158d-40f4-8ee5-7ca5e0d11cd4",
    #     "class_name": "android.widget.TextView",
    #     "package_name": "com.android.systemui",
    #     "text": "19:00||",
    #     "user_action": 0,
    #     "event_type": 2048
    # },
    #only keep the necessary columns: timestamp, package_name, and text
    df = df[['timestamp', 'package_name', 'text']]

    #Make sure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    #save filtered screentext into output file
    df.to_json(output_file, orient='records')
    #save to csv
    df.to_csv(output_file.replace('.json', '.csv'), index=False)

if __name__ == "__main__":
    input_file = 'data/screentext.json'
    output_file = 'step1_data/selected_screentext.json'
    start_time = 2024091200132701
    end_time = 2024091200132703
    get_screentext_between_time(input_file, output_file, start_time, end_time)


