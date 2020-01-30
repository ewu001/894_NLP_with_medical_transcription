from keras import models
from keras.layers import Dense, Dropout, Embedding, Conv1D, MaxPooling1D, GlobalAveragePooling1D


def cnn_model(input_dim, input_length, learning_rate, filters=32, dropout_rate=0.2, embedding_dim=200, kernel_size=3, pool_size=2,
                embedding=None, word_index=None):

    # input_length is MAX_SEQUENCE_LENGTH
    # input dim is num_features

    model = models.Sequential()
    #num_features = min(len(word_index), TOP_K)

    if embedding != None:
        # Freeze embedding weights
        is_embedding_trainable = False

        model.add(Embedding(input_dim = input_dim,
                            output_dim = embedding_dim,
                            input_length = input_length,
                            weights = [embedding],
                            trainable = is_embedding_trainable))
    else:
        model.add(Embedding(input_dim = input_dim, output_dim=embedding_dim,
                            input_length = input_length))
    model.add(Dropout(rate=dropout_rate))
    model.add(Conv1D(filters=filter,
                    kernel_size = kernel_size,
                    activation = 'relu',
                    padding = 'same'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters = filters * 2,
                        kernel_size = kernel_size,
                        activation='relu',
                        padding = 'same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate = dropout_rate))
    model.add(Dense(2, activation = 'sigmoid'))

    return model
