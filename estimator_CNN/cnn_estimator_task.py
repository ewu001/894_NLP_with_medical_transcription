import argparse
import cnn_estimator_factory
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
        help='Specify the version of the convolutional model to train, [cnn_base, cnn_2, cnn_3]',
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
        '--filters',
        help='number of filters to be used in the convolutional network',
        default=32,
        type=int
    )
    parser.add_argument(
        '--kernel_size',
        help='size of the kernels to be used in the convolutional network, by default it uses 3x3 kernel',
        default=3,
        type=int
    )
    parser.add_argument(
        '--pool_size',
        help='size of the pooling layer to be used in the convolutional network, by default it uses 2x2 pool',
        default=2,
        type=int
    )
    parser.add_argument(
        '--strides',
        help='stride of the filters that applies to the input in layer via convolution operation',
        default=1,
        type=int
    )
    parser.add_argument(
        '--padding_type',
        help='choose between same and valid to specify the padding type for applying filters to the input',
        default='same'
    )
    parser.add_argument(
        '--fc_layer_nodes',
        help='Specify the number of nodes in the fully connected neural network layer after flatten the convolution feature map',
        type=int
    )
    parser.add_argument(
        '--growth_rate',
        help='Specify the growth rate to the increase of the filter numbers used between different stages',
        type=int
    )

#, pool_size


    args, _ = parser.parse_known_args()
    hparams = args.__dict__
    output_dir = hparams.pop('output_dir')

    cnn_estimator_factory.MAX_SEQUENCE_LENGTH = hparams.pop('max_sequence_length')
    cnn_estimator_factory.VOCAB_SIZE= hparams.pop('vocab_size')
    cnn_estimator_factory.EMBEDDING_DIM= hparams.pop('embedding_dim')
    

    # Initialize the training and evaluation
    cnn_estimator_factory.train_and_evaluate(output_dir, hparams)