import pandas as pd

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


extract_process_csv('warehouse/medical_transcription_data_overall.csv',"warehouse/keyword_target_data.csv")