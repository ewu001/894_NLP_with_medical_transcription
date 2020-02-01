import os
import shutil
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import datetime

from keras.preprocessing import sequence, text
import estimator_model
import keras

CLASSES = {'1.0': 1, '2.0': 0}
VOCAB_FILE_PATH = None  # set in train_and_eval function
PADWORD = "ZYXW"
MAX_SEQUENCE_LENGTH = 500
TOP_K = 50000
EMBEDDING_DIM = 200
EVAL_INTERVAL = 50

def load_data(path):
    columns = ('transcription', 'Target')
    train_df = pd.read_csv(path+'training_dataset.csv', names=columns)
    eval_df = pd.read_csv(path+'evaluation_dataset.csv', names=columns)

    train_Y = train_df['Target'].iloc[1:].map(CLASSES)
    eval_Y = eval_df['Target'].iloc[1:].map(CLASSES)

    print(train_Y.unique())
    one_hot_train_Y = tf.keras.utils.to_categorical(train_Y)
    one_hot_eval_Y = tf.keras.utils.to_categorical(eval_Y)

    print(one_hot_train_Y.shape)
    print(one_hot_eval_Y.shape)

    return((list(train_df['transcription'].iloc[1:].astype(str)), one_hot_train_Y),
            (list(eval_df['transcription'].iloc[1:].astype(str)), one_hot_eval_Y))


def get_embedding(embedding_path):
    # This function will read the pretrained embedding from file and prepare embedding look up

    embedding_matrix_all = {}

    # prepare embedding matrix
    with open(embedding_path, encoding="utf8") as e_file:
        for line in e_file:
            values = line.split()
            word = values[0]
            coefficient = np.asarray(values[1:], dtype='float32')
            embedding_matrix_all[word] = coefficient
    return embedding_matrix_all

def get_sentence_level_embedding(word_index, matrix, embedding_dim):
    # This function will get word to word vector mapping from embedding look up 

    num_words = min(len(word_index)+1, TOP_K)
    embedding_matrix = np.zeros((num_words, embedding_dim))

    # prepare embedding matrix
    for word, index in word_index.items():
        if index > TOP_K:
            continue
        else:
            embedding_vector = matrix.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    return embedding_matrix

def input_function(texts, labels, tokenizer, batch_size, mode):
    # Transform sentence to sequence of integers
    sentence_token = tokenizer.texts_to_sequences(texts)

    # Pad the sequence to max value. 
    sentence_token = tf.keras.preprocessing.sequence.pad_sequences(sentence_token, maxlen=MAX_SEQUENCE_LENGTH)


    if mode == tf.estimator.ModeKeys.TRAIN:
        # loop indefinitely
        num_epochs = None 
        shuffle = True
    else:
        num_epochs = 1
        shuffle = False

    return tf.compat.v1.estimator.inputs.numpy_input_fn(
        x=sentence_token, y=labels, batch_size=batch_size, num_epochs=num_epochs, shuffle=shuffle
    )

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    return auc

def cnn_estimator(model_dir, config, learning_rate, embedding_dim, word_index=None, embedding_path=None):

    input_dim = min(len(word_index)+1, TOP_K)


    if embedding_path != None:
        matrix = get_embedding(embedding_path)
        embedding_matrix = get_sentence_level_embedding(word_index, matrix, embedding_dim)
    else:
        embedding_matrix = None
    
    cnnmodel = estimator_model.cnn_model(input_dim, MAX_SEQUENCE_LENGTH, learning_rate, embedding_dim,
                                embedding=embedding_matrix, word_index=word_index)


    adamOptimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    cnnmodel.compile(optimizer=adamOptimizer, loss='binary_crossentropy', clipnorm=1.0, metrics=['acc', tf.keras.metrics.AUC()])
    estimator = tf.keras.estimator.model_to_estimator(keras_model=cnnmodel, model_dir=model_dir, config=config)
    return estimator

def train_and_evaluate(output_dir, hparams):
    print("learning rate: ", hparams['learning_rate'])
    # Main orchastrator of training and evaluation by calling models from estimator_model.py
    shutil.rmtree('model_dir', ignore_errors = True)
    tf.compat.v1.summary.FileWriterCache.clear()

    # Set log configuration, export to local file
    date_string = datetime.datetime.now().strftime("%m%d_%H%M")
    filename = 'training log/train_estimator_log_' + date_string + '.txt'
    logging.basicConfig(filename=filename, level=20)

    ((train_text, train_label), (eval_text, eval_label)) = load_data("warehouse/store/")


    # Create vocabulary from training corpus 
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_text)

    # Create estimator config
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=EVAL_INTERVAL,
                                        log_step_count_steps = 20,
                                        save_summary_steps = 50
                                    )
    estimator = cnn_estimator(output_dir, run_config,
                                hparams['learning_rate'],
                                EMBEDDING_DIM,
                                word_index=tokenizer.word_index,
                                embedding_path=hparams['embedding_path'])

    train_steps = hparams['num_epochs'] * len(train_text) / hparams['batch_size']
    train_spec = tf.estimator.TrainSpec(input_fn=input_function(train_text, train_label, tokenizer,
            hparams['batch_size'], mode=tf.estimator.ModeKeys.TRAIN), max_steps=train_steps)

    eval_spec = tf.estimator.EvalSpec(input_fn=input_function(eval_text, eval_label, tokenizer,
            hparams['batch_size'], mode=tf.estimator.ModeKeys.EVAL), steps=None, 
            start_delay_secs=10,
            throttle_secs = EVAL_INTERVAL)  # evaluate every N seconds

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    return True