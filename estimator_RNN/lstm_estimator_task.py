import argparse
import lstm_estimator_factory
import tempfile

if __name__ == '__main__':
    # parse command line argument for hyper parameter input
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_dir',
        help='location to write checkpoints and export trained model',
        required=True
    )
    parser.add_argument(
        '--model_version',
        help='Specify the version of the recurrent model with attention to train. [lstm_attention, gru_attention] ',
        required=True
    )
    parser.add_argument(
        '--embedding_path',
        help='Optional, path to the embedding location'
    )
    parser.add_argument(
        '--embedding_dim',
        help='Optional, the dimension of the embedding to be used, if pre trained embedding is specified, should match with the embedding',
        type=int,
        default=200
    )
    parser.add_argument(
        '--num_epochs',
        help='Number of epochs to go through the data, default to 10',
        default=10,
        type=int
    )
    parser.add_argument(
        '--batch_size',
        help='number of records to read during each training step, default to 100',
        default=100,
        type=int
    )
    parser.add_argument(
        '--learning_rate',
        help='learning rate for gradient descent, required for model train',
        default=0.00001,
        type=float
    )
    parser.add_argument(
        '--dropout_rate',
        help='dropout rate used inside the convolutional network to regularize model from being overfit',
        default=0.2,
        type=float
    )
    parser.add_argument(
        '--vocab_size',
        help='set the size limit of text corpus used on the tokenization for sentence vocabulary',
        default=50000,
        type=int
    )
    parser.add_argument(
        '--max_sequence_length',
        help='set the limit of maximum sequence length used for padding',
        default=300,
        type=int
    )
    parser.add_argument(
        '--rnn_units',
        help='Specifies the dimension of the RNN weights',
        type=int
    )


    args, _ = parser.parse_known_args()
    hparams = args.__dict__
    output_dir = hparams.pop('output_dir')

    lstm_estimator_factory.MAX_SEQUENCE_LENGTH = hparams.pop('max_sequence_length')
    lstm_estimator_factory.VOCAB_SIZE= hparams.pop('vocab_size')
    lstm_estimator_factory.EMBEDDING_DIM= hparams.pop('embedding_dim')
    

    # Initialize the training and evaluation
    lstm_estimator_factory.train_and_evaluate(output_dir, hparams)