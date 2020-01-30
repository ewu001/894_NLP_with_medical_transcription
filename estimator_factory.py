import tensorflow as tf
import pandas as pd
import numpy as np

from keras.preprocessing import text
from keras.optimizers import Adam
import estimator_model

VOCAB_FILE_PATH = None  # set in train_and_eval function
PADWORD = "ZYXW"
MAX_SEQUENCE_LENGTH = 500
TOP_K = 25000

def load_data(path):
    columns = ('transcription', 'Target')
    train_df = pd.read_csv(path+'training_dataset.csv', names=columns)
    eval_df = pd.read_csv(path+'evaluation_dataset.csv', names=columns)

    return((list(train_df['transcription']), np.array(train_df['Target']),
            list(eval_df['transcription']), np.array(eval_df['Target'])))

def vectorize_sentences(text_tensor):
    # remove punctuation
    sentences = tf.regex_replace(text_tensor, '[[:punct:]]', ' ')

    # split to component words
    words = tf.string_split(sentences)
    words = tf.sparse_tensor_to_dense(words, default_value=PADWORD)

    # create lookup table
    table = tf.contrib.lookup.index_table_from_file(
        vocabulary_file = VOCAB_FILE_PATH,
        num_oov_buckets = 0,
        vocab_size=None,
        default_value = 0,  # for OOV
        key_column_index = 0,
        value_column_index = 1,
        delimiter = ','
    )
    numbers = table.lookup(words)
    return numbers

def pad(text, label):
    # Get rid of oov indices of value zero
    oov_indices = tf.where(tf.not_equal(text, tf.zeros_like(text)))
    print(oov_indices)
    remove_oov = tf.gather(text, oov_indices)
    print(remove_oov)
    remove_oov = tf.squeeze(remove_oov, axis=1)
    print(remove_oov)

    # pad sequence based on max length with zero, then slice and return
    pad_sentence = tf.pad(remove_oov, [[0, 0], [MAX_SEQUENCE_LENGTH, 0]])
    return (pad_sentence[-MAX_SEQUENCE_LENGTH:], label)

def get_embedding(embedding_path):
    # This function will read the pretrained embedding from file and prepare embedding look up

    embedding_matrix_all = {}

    # prepare embedding matrix
    with open(embedding_path) as e_file:
        for line in e_file:
            values = line.split()
            word = values[0]
            coefficient = np.asarray(values[1:], dtype='float32')
            embedding_matrix_all[word] = coefficient
    return embedding_matrix_all

def get_sentence_level_embedding(word_index, matrix, embedding_dim):
    # This function will get word to word vector mapping from embedding look up 

    num_words = min(len(word_index), TOP_K)
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

def input_function(texts, labels, batch_size, mode):
    text_tensor = tf.constant(texts)
    text_tensor = vectorize_sentences(text_tensor)
    
    dataset = tf.data.Dataset.from_tensor_slices((text_tensor, labels))

    dataset = dataset.map(pad)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # loop indefinitely
        num_epochs = None 
        dataset = dataset.shuffle(buffer_size=50000)
    else:
        num_epochs = 1

    dataset = dataset.repeat(num_epochs).batch(batch_size)
    return dataset


def cnn_estimator(model_dir, config, learning_rate, word_index, embedding_path=None, embedding_dim=200):
    optimizer = Adam(lr=learning_rate)
    input_dim = min(len(word_index), TOP_K)


    if embedding_path != None:
        matrix = get_embedding(embedding_path)
        embedding_matrix = get_sentence_level_embedding(word_index, matrix, embedding_dim)
    else:
        embedding_matrix = None
    
    model = estimator_model.cnn_model(input_dim, MAX_SEQUENCE_LENGTH, learning_rate, 
                                embedding=embedding_matrix, word_index=word_index)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
    estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir, config=config)
    return estimator

def train_and_evaluate(output_dir, hparams):
    # Main orchastrator of training and evaluation by calling models from estimator_model.py
    
    return True