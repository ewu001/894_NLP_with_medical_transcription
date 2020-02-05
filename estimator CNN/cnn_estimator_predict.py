import pickle
import tensorflow as tf
import pandas as pd
import keras
import utility
import pathlib
import argparse


MAX_SEQUENCE_LENGTH = 500
export_model_path = 'cnnmodel_dir/export/exporter/'

def accuracy_percentage(x, y):
    return (100.0 * len(set(x) & set(y))) / len(set(x) | set(y))

if __name__ == '__main__':
    # parse command line argument for hyper parameter input
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_id',
        help='ID of the export trained model',
        required=True
    )

    args, _ = parser.parse_known_args()
    hparams = args.__dict__
    model_id = hparams.pop('model_id')


    urgent=[
    'This 5-year-old male presents to Children’s Hospital Emergency Department by the mother with "have asthma." Mother states he has been wheezing and coughing. They saw their primary medical doctor. He was evaluated at the clinic, given the breathing treatment and discharged home, was not having asthma, prescribed prednisone and an antibiotic. They told to go to the ER if he got worse. He has had some vomiting and some abdominal pain. His peak flows on the morning are normal at 150, but in the morning, they were down to 100 and subsequently decreased to 75 over the course of the day.',
    'This is a 78-year-old male who has prostate cancer with metastatic disease to his bladder and in several locations throughout the skeletal system including the spine and shoulder. The patient has had problems with hematuria in the past, but the patient noted that this episode began yesterday, and today he has been passing principally blood with very little urine. The patient states that there is no change in his chronic lower back pain and denies any incontinence of urine or stool. The patient has not had any fever. There is no abdominal pain and the patient is still able to pass urine. The patient has not had any melena or hematochezia. There is no nausea or vomiting. The patient has already completed chemotherapy and is beyond treatment for his cancer at this time. The patient is receiving radiation therapy, but it is targeted to the bones and intended to give symptomatic relief of his skeletal pain and not intended to treat and cure the cancer. The patient is not enlisted in hospice, but the principle around the patient’s current treatment management is focusing on comfort care measures.'
    ]
    non_urgent=[
    'Mr. XYZ is 41 years of age, who works for Chevron and lives in Angola. He was playing basketball in Angola back last Wednesday, Month DD, YYYY, when he was driving toward the basket and felt a pop in his posterior leg. He was seen locally and diagnosed with an Achilles tendon rupture. He has been on crutches and has been nonweightbearing since that time. He had no pain prior to his injury. He has had some swelling that is mild. He has just been on aspirin a day due to his traveling time. Pain currently is minimal.',
    'This is a 56-year-old female who fell on November 26, 2007 at 11:30 a.m. while at work. She did not recall the specifics of her injury but she thinks that her right foot inverted and subsequently noticed pain in the right ankle. She describes no other injury at this time.'
    ]
    # Tokenize and pad sentences using same mapping used in the trained estimator model
    #abspath = pathlib.Path('tokenizer.pickle').absolute()
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    #tokenizer = pickle.load( open( "tokenizer.pickled", "rb" ) )

    # load eval data to predict
    #(eval_text, eval_label) = utility.load_eval_data_for_pred("warehouse/store/", {'1.0': 1, '2.0': 0})
    request = urgent + non_urgent

    requests_tokenized = tokenizer.texts_to_sequences(request)
    requests_tokenized = tf.keras.preprocessing.sequence.pad_sequences(requests_tokenized,maxlen=MAX_SEQUENCE_LENGTH)

    #print(requests_tokenized.tolist())
    # JSON format the requests
    request_data = requests_tokenized.tolist()

   # print(request_data[439, 493])

    predict_fn = tf.contrib.predictor.from_saved_model(export_model_path+model_id)
    predictions = predict_fn({"input": request_data})
    prediction_label = predictions['dense_1'].argmax(axis=-1)
    print(predictions)
    print("Expected output: 1 1 0 0. 1 for urgent and 0 for non urgent")
    print("Predicted labels for request: ", prediction_label)



