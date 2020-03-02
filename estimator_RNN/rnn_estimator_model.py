import tensorflow as tf
from keras import models
from keras.layers import Dense, Dropout, Embedding, LSTM, MaxPooling1D, GlobalAveragePooling1D, Bidirectional


def lstm_model(input_dim, input_length, learning_rate, dropout_rate, embedding_dim, lstm_units, embedding=None, word_index=None):


    # input_length is MAX_SEQUENCE_LENGTH
    # input dim is num_features

    #md_input = tf.keras.layers.Input(shape=(input_length,))
    model = tf.keras.models.Sequential()


    if embedding is not None:
        # Freeze embedding weights
        #is_embedding_trainable = False

        model.add(tf.keras.layers.Embedding(input_dim = input_dim,
                            output_dim = embedding_dim,
                            input_length = input_length,
                            weights = [embedding],
                            trainable = True))
    else:
        model.add(tf.keras.layers.Embedding(input_dim = input_dim, output_dim=embedding_dim,
                            input_length = input_length))
    
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units,return_sequences=True,dropout=dropout_rate),merge_mode='concat'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(2,activation='softmax'))

    return model

def GRU_model(input_dim, input_length, learning_rate, dropout_rate, embedding_dim, gru_units, embedding=None, word_index=None):


    # input_length is MAX_SEQUENCE_LENGTH
    # input dim is num_features

    #md_input = tf.keras.layers.Input(shape=(input_length,))
    model = tf.keras.models.Sequential()


    if embedding is not None:
        # Freeze embedding weights
        #is_embedding_trainable = False

        model.add(tf.keras.layers.Embedding(input_dim = input_dim,
                            output_dim = embedding_dim,
                            input_length = input_length,
                            weights = [embedding],
                            trainable = True))
    else:
        model.add(tf.keras.layers.Embedding(input_dim = input_dim, output_dim=embedding_dim,
                            input_length = input_length))
    
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_units,return_sequences=True,dropout=dropout_rate)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(2,activation='softmax'))

    return model