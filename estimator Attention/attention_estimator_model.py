import tensorflow as tf

def attention_model(input_dim, input_length, learning_rate, embedding_dim, lstm_units=32, embedding=None, word_index=None):

    # input_length is MAX_SEQUENCE_LENGTH
    # input dim is num_features

    md_input = tf.keras.layers.Input(shape=[input_length], dtype='int32')

    # Create embedding layer
    if embedding is not None:
        # Freeze embedding weights
        #is_embedding_trainable = False

        embed_layer = tf.keras.layers.Embedding(input_dim = input_dim,
                            output_dim = embedding_dim,
                            input_length = input_length,
                            weights = [embedding],
                            trainable = False)(md_input)
    else:
        embed_layer = tf.keras.layers.Embedding(input_dim = input_dim, output_dim=embedding_dim,
                            input_length = input_length)(md_input)
    
    activations = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(embed_layer)

    # Compute attention for each input time step
    # Learn a probability distribution over each  step.
    # Reshape to match LSTM's output shape, so that we can do element-wise multiplication  with LSTM output.
    # Stack attention network on top of LSTM layer
    # Reconstruct output layer to match many-to-one structure

    attention = tf.keras.layers.Dense(1, activation='tanh')(activations)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax')(attention)
    attention = tf.keras.layers.RepeatVector(lstm_units)(attention)
    attention = tf.keras.layers.Permute([2, 1])(attention)
    attention_vector = tf.keras.layers.concatenate([attention, activations])

    LSTM_2 = tf.keras.layers.LSTM(lstm_units, return_sequences=False)(attention_vector)
    output = tf.keras.layers.Dense(2, activation='softmax', name='output')(LSTM_2)
    model = tf.keras.Model(inputs=md_input, outputs=output)

    return model
