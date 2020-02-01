import argparse
import estimator_factory
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

    args, _ = parser.parse_known_args()
    hparams = args.__dict__
    output_dir = hparams.pop('output_dir')

    # Initialize the training and evaluation
    print(tempfile.gettempdir())
    estimator_factory.train_and_evaluate(output_dir, hparams)