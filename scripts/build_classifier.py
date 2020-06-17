# python3 scripts/build_classifier.py --train_size 0.50

from simpletransformers.classification import ClassificationModel
import pandas as pd
import argparse

from utils import split_train_test

model_args={
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    "config": {
        "output_hidden_states": True
    }
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', default=0.75, type=float,
                        help='What is the size of the training set?')
    args = parser.parse_args()

    # Generate Training set
    split_train_test(train_size = args.train_size)
    train_df = pd.read_csv('data/train.tsv', sep='\t', encoding='latin')

    # Create a ClassificationModel
    model = ClassificationModel('distilbert', 'distilbert-base-multilingual-cased', args=model_args, use_cuda = False)

    # Train the model
    model.train_model(train_df)
