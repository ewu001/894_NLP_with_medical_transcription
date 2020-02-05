import pandas as pd
import os

from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import string

def target_encoding(row):
    if (row.Target == 2 or row.Target == 3):
        row.Target = 2
    return row

def remove_html(text):
    # remove HTML tags using BeautifulSoup library

    soup = BeautifulSoup(text, 'html.parser')
    html_free = soup.get_text()
    return html_free

def remove_punctuation(text):
    # remove punctuation based on string library

    punc_free = "".join([ item for item in text if item not in string.punctuation])
    return punc_free

def text_lemmalization(text, lemmatizer):
    # Perform lemmalization on text using NLTK WordNetLemmatizer package

    lemma_text = [lemmatizer.lemmatize(item) for item in text]
    return lemma_text

def extract_transform_load_csv(read_path, export_path):
    # Main ETL pipeline
    print('Start extracting ')
    #tokenizer = RegexpTokenizer(r'\w+')
    df = pd.read_csv(read_path)
    df = df.apply(target_encoding, axis=1)

    returndf = df[["transcription", "Target"]]
    processeddf = returndf[returndf['Target'] !=0]
    print('Start pre-processing ')
    #processeddf['transcription'] = processeddf['transcription'].apply(lambda x: remove_html(str(x).lower()))
    #processeddf['transcription'] = processeddf['transcription'].apply(lambda x: remove_punctuation(str(x).lower()))
    processeddf['transcription'] = processeddf['transcription'].apply(lambda x: str(x).split(' '))
    processeddf['transcription'] = processeddf['transcription'].apply(lambda x: text_lemmalization(x, WordNetLemmatizer()))
    processeddf['transcription'] = processeddf['transcription'].apply(lambda x: "".join([item + " " for item in x]))

    processeddf.to_csv(export_path, index=False)
    return

if __name__ == '__main__':

    read_path = os.path.join(os.path.dirname(__file__), "../warehouse/medical_transcription_data_overall.csv")
    export_path = os.path.join(os.path.dirname(__file__), "../warehouse/keyword_target_data.csv")
    extract_transform_load_csv(read_path, export_path)
    print("Finished load keyword target data")