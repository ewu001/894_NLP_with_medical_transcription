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

    # Tokenize and pad sentences using same mapping used in the trained estimator model
    #abspath = pathlib.Path('tokenizer.pickle').absolute()
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    #tokenizer = pickle.load( open( "tokenizer.pickled", "rb" ) )

    # load eval data to predict
    (eval_text, eval_label) = utility.load_eval_data_for_pred("warehouse/store/", {'1.0': 1, '2.0': 0})


    requests_tokenized = tokenizer.texts_to_sequences(eval_text)
    requests_tokenized = tf.keras.preprocessing.sequence.pad_sequences(requests_tokenized,maxlen=MAX_SEQUENCE_LENGTH)

    #print(requests_tokenized.tolist())
    # JSON format the requests
    request_data = requests_tokenized.tolist()

    predict_fn = tf.contrib.predictor.from_saved_model(export_model_path+model_id)
    predictions = predict_fn({"input": request_data})
    prediction_label = predictions['dense_1'].argmax(axis=-1)
    print("Accuracy on evaluation: ",accuracy_percentage(prediction_label, eval_label))



