import tensorflow as tf
from keras import models
from keras.layers import Dense, Dropout, Embedding, LSTM, MaxPooling1D, GlobalAveragePooling1D, Bidirectional

'''
Ruchika's code
def lstm_model(input_dim, input_length, learning_rate, embedding_dim, lstm_units=32, embedding=None, word_index=None):


    # input_length is MAX_SEQUENCE_LENGTH
    # input dim is num_features

    input = tf.keras.layers.Input(shape=(input_length,))
    model = tf.keras.models.Sequential()
    #num_features = min(len(word_index), TOP_K)

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
    
    model = Bidirectional (LSTM (lstm_units,return_sequences=True,dropout=0.50),merge_mode='concat')(model)
    #model = TimeDistributed(Dense(100,activation='relu'))(model)
    model = tf.keras.layers.Flatten()(model)
    model = Dense(100,activation='relu')(model)
    output = Dense(2,activation='softmax')(model)
    #output = Dense(1, activation='sigmoid')(model) since our target is binary - urgent -1 and non-urgent-0
    model = tf.keras.Model(input,output)
    return model
'''

# Ethan's model code
def lstm_model(input_dim, input_length, learning_rate, embedding_dim, lstm_units=32, embedding=None, word_index=None):


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
    
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units,return_sequences=True,dropout=0.50),merge_mode='concat'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    model.add(tf.keras.layers.Dense(2,activation='softmax'))

    return model