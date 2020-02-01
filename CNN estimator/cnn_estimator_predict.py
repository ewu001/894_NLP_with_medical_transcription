import pickle
import tensorflow as tf
import pandas as pd
import keras
import utility


MAX_SEQUENCE_LENGTH = 500
export_model_path = 'cnnmodel_dir/export/exporter/1580528520'

def accuracy_percentage(x, y):
    return (100.0 * len(set(x) & set(y))) / len(set(x) | set(y))

if __name__ == '__main__':
    # Tokenize and pad sentences using same mapping used in the trained estimator model
    tokenizer = pickle.load( open( "cnn_tokenizer.pickled", "rb" ) )

    # load eval data to predict
    (eval_text, eval_label) = utility.load_eval_data_for_pred("warehouse/store/", {'1.0': 1, '2.0': 0})


    requests_tokenized = tokenizer.texts_to_sequences(eval_text)
    requests_tokenized = tf.keras.preprocessing.sequence.pad_sequences(requests_tokenized,maxlen=MAX_SEQUENCE_LENGTH)

    # JSON format the requests
    request_data = {'instances':requests_tokenized.tolist()}

    predict_fn = tf.contrib.predictor.from_saved_model(export_model_path)
    predictions = predict_fn(
        {"input": requests_tokenized.tolist()})
    prediction_label = predictions['dense_1'].argmax(axis=-1)
    print(accuracy_percentage(prediction_label, eval_label))



