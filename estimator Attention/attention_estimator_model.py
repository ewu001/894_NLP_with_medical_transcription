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


'''
Model summary: 
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 500)]        0
__________________________________________________________________________________________________
embedding (Embedding)           (None, 500, 200)     4319000     input_1[0][0]
__________________________________________________________________________________________________
lstm (LSTM)                     (None, 500, 32)      29824       embedding[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 500, 1)       33          lstm[0][0]
__________________________________________________________________________________________________
flatten (Flatten)               (None, 500)          0           dense[0][0]
__________________________________________________________________________________________________
activation (Activation)         (None, 500)          0           flatten[0][0]
__________________________________________________________________________________________________
repeat_vector (RepeatVector)    (None, 32, 500)      0           activation[0][0]
__________________________________________________________________________________________________
permute (Permute)               (None, 500, 32)      0           repeat_vector[0][0]
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 500, 64)      0           permute[0][0]
                                                                 lstm[0][0]
__________________________________________________________________________________________________
lstm_1 (LSTM)                   (None, 32)           12416       concatenate[0][0]
__________________________________________________________________________________________________
output (Dense)                  (None, 2)            66          lstm_1[0][0]
==================================================================================================
Total params: 4,361,339
Trainable params: 42,339
Non-trainable params: 4,319,000
__________________________________________________________________________________________________
'''