# python3 scripts/build_classifier.py --train_size 0.50

import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from utils import split_train_test

random_state = 123

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', default=0.75, type=float,
                        help='What is the size of the training set?')
    args = parser.parse_args()

    # Generate Training set
    split_train_test(train_size = args.train_size)

    train_df = pd.read_csv('data/train.tsv', sep='\t', encoding='latin')
    train_embedding = np.load('data/train_embeddings.npy')

    # Create a ClassificationModel
    clf_model = LogisticRegression(random_state=123)

    # Train the model
    clf_model.fit(train_embedding, train_df['label'])

    joblib.dump(clf_model, 'models/clf_logreg.pkl')
