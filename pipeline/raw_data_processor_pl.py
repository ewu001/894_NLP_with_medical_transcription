import pandas as pd
import numpy as np
import os
import glob

CLASSES = [1, 2, 3]

def target_encoding(row):
    '''
    Things to note
    Pulmonary Function Test --> 3
    Biopsy --> 3
    Autopsy --> 3
    Bariatrics --> 3
    Insertion --> 3
    Preoperative --> 3
    Postoperative --> 3
    chiropractic --> 3
    speech therapy --> 3
    otitis media= 3
    shoulder pain = 3
    pulmonary test = 3
    adenocarcinoma = 3
    esophagogastrectomy= 3
    Well-Child Check =3
    sports= 3
    knee pain = 3


    
    Shortness of Breathe --> 1
    Chest Pain --> 1
    Heart Catheterization=1
    cardioversion =1  
    cardiac catheterization=1
    angina = 1
    psych consult =1
    sepsis= 1
    fever=1
    kill=1
    hallucination=1
    discharge summary = 1
    admitted = 1
    admit = 1
    cardiac arrest = 1
    respiratory failure = 1
    
    '''
    if ("pulmonary function test" in str(row.description).lower()) or ("pulmonary function test" in str(row.transcription).lower()) or ("pulmonary function test" in str(row.medical_specialty).lower()):
        row.Target = 3
    elif ("biopsy" in str(row.description).lower()) or ("biopsy" in str(row.transcription).lower()) or ("biopsy" in str(row.medical_specialty).lower()):
        row.Target = 3
    elif ("autopsy" in str(row.description).lower()) or ("autopsy" in str(row.transcription).lower()) or ("autopsy" in str(row.medical_specialty).lower()):
        row.Target = 3
    elif ("bariatrics" in str(row.description).lower()) or ("bariatrics" in str(row.transcription).lower()) or ("bariatrics" in str(row.medical_specialty).lower()):
        row.Target = 3
    elif ("insertion" in str(row.description).lower()) or ("insertion" in str(row.transcription).lower()) or ("insertion" in str(row.medical_specialty).lower()):
        row.Target = 3
    elif ("preoperative" in str(row.description).lower()) or ("preoperative" in str(row.transcription).lower()) or ("preoperative" in str(row.medical_specialty).lower()):
        row.Target = 3
    elif ("postoperative" in str(row.description).lower()) or ("postoperative" in str(row.transcription).lower()) or ("postoperative" in str(row.medical_specialty).lower()):
        row.Target = 3
    elif ("chiropractic" in str(row.description).lower()) or ("chiropractic" in str(row.transcription).lower()) or ("chiropractic" in str(row.medical_specialty).lower()):
        row.Target = 3
    elif ("speech therapy" in str(row.description).lower()) or ("speech therapy" in str(row.transcription).lower()) or ("speech therapy" in str(row.medical_specialty).lower()):
        row.Target = 3 
    elif ("otitis media" in str(row.description).lower()) or ("otitis media" in str(row.transcription).lower()) or ("otitis media" in str(row.medical_specialty).lower()):
        row.Target = 3
    elif ("shoulder pain" in str(row.description).lower()) or ("shoulder pain" in str(row.transcription).lower()) or ("shoulder pain" in str(row.medical_specialty).lower()):
        row.Target = 3
    elif ("pulmonary test" in str(row.description).lower()) or ("pulmonary test" in str(row.transcription).lower()) or ("pulmonary test" in str(row.medical_specialty).lower()):
        row.Target = 3      
    elif ("adenocarcinoma" in str(row.description).lower()) or ("adenocarcinoma" in str(row.transcription).lower()) or ("adenocarcinoma" in str(row.medical_specialty).lower()):
        row.Target = 3         
    elif ("esophagogastrectomy" in str(row.description).lower()) or ("esophagogastrectomy" in str(row.transcription).lower()) or ("esophagogastrectomy" in str(row.medical_specialty).lower()):
        row.Target = 3  
    elif ("well-child check" in str(row.description).lower()) or ("well-child check" in str(row.transcription).lower()) or ("well-child check" in str(row.medical_specialty).lower()):
        row.Target = 3  
    elif ("sports" in str(row.description).lower()) or ("sports" in str(row.transcription).lower()) or ("sports" in str(row.medical_specialty).lower()):
        row.Target = 3  
        
        
    if ("shortness of breath" in str(row.description).lower()) or ("shortness of breath" in str(row.transcription).lower()) or ("shortness of breath" in str(row.medical_specialty).lower()):
        row.Target = 1
    elif ("chest pain" in str(row.description).lower()) or ("chest pain" in str(row.transcription).lower()) or ("chest pain" in str(row.medical_specialty).lower()):
        row.Target = 1
    elif ("heart catheterization" in str(row.description).lower()) or ("heart catheterization" in str(row.transcription).lower()) or ("heart catheterization" in str(row.medical_specialty).lower()):
        row.Target = 1
    elif ("cardioversion" in str(row.description).lower()) or ("cardioversion" in str(row.transcription).lower()) or ("cardioversion" in str(row.medical_specialty).lower()):
        row.Target = 1  
    elif ("cardiac catheterization" in str(row.description).lower()) or ("cardiac catheterization" in str(row.transcription).lower()) or ("cardiac catheterization" in str(row.medical_specialty).lower()):
        row.Target = 1 
    elif ("angina" in str(row.description).lower()) or ("angina" in str(row.transcription).lower()) or ("angina" in str(row.medical_specialty).lower()):
        row.Target = 1 
    elif ("psych consult" in str(row.description).lower()) or ("psych consult" in str(row.transcription).lower()) or ("psych consult" in str(row.medical_specialty).lower()):
        row.Target = 1 
    elif ("sepsis" in str(row.description).lower()) or ("sepsis" in str(row.transcription).lower()) or ("sepsis" in str(row.medical_specialty).lower()):
        row.Target = 1 
    elif ("fever" in str(row.description).lower()) or ("fever" in str(row.transcription).lower()) or ("fever" in str(row.medical_specialty).lower()):
        row.Target = 1 
    elif ("kill" in str(row.description).lower()) or ("kill" in str(row.transcription).lower()) or ("kill" in str(row.medical_specialty).lower()):
        row.Target = 1 
    elif ("hallucination" in str(row.description).lower()) or ("hallucination" in str(row.transcription).lower()) or ("hallucination" in str(row.medical_specialty).lower()):
        row.Target = 1 
    elif ("discharge summary" in str(row.description).lower()) or ("discharge summary" in str(row.transcription).lower()) or ("discharge summary" in str(row.medical_specialty).lower()):
        row.Target = 1 
    elif ("admitted" in str(row.description).lower()) or ("admitted" in str(row.transcription).lower()) or ("admitted" in str(row.medical_specialty).lower()):
        row.Target = 1 
    elif ("admit" in str(row.description).lower()) or ("admit" in str(row.transcription).lower()) or ("admit" in str(row.medical_specialty).lower()):
        row.Target = 1 
    elif ("cardiac arrest" in str(row.description).lower()) or ("cardiac arrest" in str(row.transcription).lower()) or ("cardiac arrest" in str(row.medical_specialty).lower()):
        row.Target = 1 
    elif ("respiratory failure" in str(row.description).lower()) or ("respiratory failure" in str(row.transcription).lower()) or ("respiratory failure" in str(row.medical_specialty).lower()):
        row.Target = 1 
    return row

def concatenate_labeled_data(path, classes):

    all_files = glob.glob(path + "/*.csv")
    li = []

    for filename in all_files:
        df = pd.read_csv(filename)
        df = df[df['Target'].isin(classes)]
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True, sort=True)
    return frame

def merge_data(raw, labeled, target_encoding, classes):
    # Argument: 
    # raw: raw dataframe scrapped from web
    # labeled: data manually labeled
    # target_encoding: functional parameter to pass the transformation process
    # classes: list of unique class to encode
    # This function will first exclude labeled data from raw data, apply transformation, and then merge back to labeled data

    raw['Target'] = np.zeros((raw.shape[0],1))
    df_nodup = raw[~raw['transcription'].isin(labeled['transcription'])]

    df_nodup = df_nodup.apply(target_encoding, axis=1)
    df_labeled_by_logic = df_nodup[df_nodup['Target'].isin(classes)]
    merge_df = pd.concat([df_labeled_by_logic, labeled], sort=True)
    return merge_df


if __name__ == '__main__':

    raw_data_path = os.path.join(os.path.dirname(__file__), "../lake/medical_transcription_data_raw.csv")
    rawdf = pd.read_csv(raw_data_path)

    hand_labeled_path = os.path.join(os.path.dirname(__file__), "../lake/hand_labeled")
    labeled_df = concatenate_labeled_data(hand_labeled_path, CLASSES)

    print('extracted raw data and hand labeled data from lake')

    merge_df = merge_data(rawdf, labeled_df, target_encoding, CLASSES)

    export_file_path = os.path.join(os.path.dirname(__file__), '../warehouse/medical_transcription_data_overall.csv')
    merge_df.to_csv(export_file_path)
    print('transformed data shape: ', merge_df.shape)
    print('loaded processed data to warehouse successfully')