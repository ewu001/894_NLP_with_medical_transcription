import tensorflow as tf
from keras import models
from keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, GlobalAveragePooling1D


def cnn_model_basic(input_dim, input_length, learning_rate, embedding_dim, filters, dropout_rate, kernel_size, pool_size, strides, padding_type, growth_rate, nn_nodes,
                embedding=None, word_index=None):

    # input_length is MAX_SEQUENCE_LENGTH
    # input dim is num_features

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
    model.add(tf.keras.layers.Conv1D(filters=filters,
                    kernel_size = kernel_size,
                    padding = padding_type,
                    activation = 'relu'))
    

    model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size))
    model.add(tf.keras.layers.Conv1D(filters = filters * growth_rate,
                        kernel_size = kernel_size,
                        padding = padding_type,
                        activation = 'relu'))

    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(nn_nodes, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    return model

def cnn_model_2(input_dim, input_length, learning_rate, embedding_dim, filters, dropout_rate, kernel_size, pool_size, strides, padding_type, growth_rate, nn_nodes,
                embedding=None, word_index=None):

    # input_length is MAX_SEQUENCE_LENGTH
    # input dim is num_features

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
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.Conv1D(filters=filters,
                    kernel_size = kernel_size,
                    strides = strides,
                    padding = padding_type))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size))
    model.add(tf.keras.layers.Conv1D(filters = filters * growth_rate,
                        kernel_size = kernel_size,
                        strides = strides,
                        padding = padding_type))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(nn_nodes, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate = dropout_rate))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    return model

def cnn_model_3(input_dim, input_length, learning_rate, embedding_dim, filters, dropout_rate, kernel_size, pool_size, strides, padding_type, growth_rate, nn_nodes,
                embedding=None, word_index=None):

    # input_length is MAX_SEQUENCE_LENGTH
    # input dim is num_features

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
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.Conv1D(filters=filters,
                    kernel_size = kernel_size,
                    strides = strides,
                    padding = padding_type))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Conv1D(filters=int(filters / 2) ,
                    kernel_size = 1, strides = 1))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size))

    model.add(tf.keras.layers.Conv1D(filters = filters * growth_rate,
                        kernel_size = kernel_size,
                        strides = strides,
                        padding = padding_type))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    
    model.add(tf.keras.layers.Conv1D(filters=filters,
                    kernel_size = 1, strides = 1))

    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(nn_nodes, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate = dropout_rate))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    return model