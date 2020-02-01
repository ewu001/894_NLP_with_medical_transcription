import pandas as pd
import tensorflow as tf
import numpy as np

def load_data(path, classes):
    columns = ('transcription', 'Target')
    train_df = pd.read_csv(path+'training_dataset.csv', names=columns)
    eval_df = pd.read_csv(path+'evaluation_dataset.csv', names=columns)

    train_Y = train_df['Target'].iloc[1:].map(classes)
    eval_Y = eval_df['Target'].iloc[1:].map(classes)

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


def get_sentence_level_embedding(word_index, matrix, embedding_dim, vocab_size):
    # This function will get word to word vector mapping from embedding look up 

    num_words = min(len(word_index)+1, vocab_size)
    embedding_matrix = np.zeros((num_words, embedding_dim))

    # prepare embedding matrix
    for word, index in word_index.items():
        if index > vocab_size:
            continue
        else:
            embedding_vector = matrix.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    return embedding_matrix

def auc(y_true, y_pred):
    # Customized definition of AUC performance metric
    auc = tf.metrics.auc(y_true, y_pred)[1]
    return auc
