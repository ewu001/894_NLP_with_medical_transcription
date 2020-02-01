import pandas as pd
import os

def target_encoding(row):
    if (row.Target == 2 or row.Target == 3):
        row.Target = 2
    return row

def extract_process_csv(read_path, export_path):

    df = pd.read_csv(read_path)
    df = df.apply(target_encoding, axis=1)

    returndf = df[["transcription", "Target"]]
    processeddf = returndf[returndf['Target'] !=0]
    processeddf.to_csv(export_path, index=False)
    return

if __name__ == '__main__':
    read_path = os.path.join(os.path.dirname(__file__), "../warehouse/medical_transcription_data_overall.csv")
    export_path = os.path.join(os.path.dirname(__file__), "../warehouse/keyword_target_data.csv")
    extract_process_csv(read_path, export_path)
    print("Finished load keyword target data")