# python3 scripts/build_classifier.py --train_size 0.50

from simpletransformers.classification import ClassificationModel
import pandas as pd
import argparse

from utils import split_train_test

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', default=0.75, type=float,
                        help='What is the size of the training set?')
    args = parser.parse_args()

    # Generate Training set
    split_train_test(train_size = args.train_size)
    train_df = pd.read_csv('data/train.tsv', sep='\t', encoding='latin')

    # Create a ClassificationModel
    model = ClassificationModel('bert', 'bert-base-cased', use_cuda=False, args={'overwrite_output_dir': True})

    # Train the model
    model.train_model(train_df, output_dir = 'models')
