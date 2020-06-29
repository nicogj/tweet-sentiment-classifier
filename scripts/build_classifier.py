# python3 scripts/build_classifier.py --train_size 0.50

import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from utils import split_train_test

random_state = 123

def train_model(train_df, train_embeddings, args):

    X = train_embeddings
    y = train_df.label

    if args.reg_norm == 'l1':
        if train_df.shape[0] < 10e4:
            clf = LogisticRegression(solver='liblinear', max_iter=args.max_iter, C=C, penalty='l1').fit(X, y)
        else:
            clf = LogisticRegression(solver='saga', max_iter=args.max_iter, C=C, penalty='l1').fit(X, y)
    else:
        clf = LogisticRegression(solver='lbfgs', max_iter=args.max_iter, C=args.reg, penalty=args.reg_norm).fit(X, y)
    print('Training set accuracy: {}'.format(clf.score(X, y)))
    return clf


def test_model(clf, test_df, test_embeddings):
    test_pred = clf.predict(test_embeddings)

    correct = test_pred==test_df['label']
    wrong = test_pred!=test_df['label']
    tp = sum(correct & test_df['label'])
    tn = sum(correct & ~test_df['label'])
    fn = sum(wrong & test_df['label'])
    fp = sum(wrong & ~test_df['label'])
    t = sum(correct)

    print('got %s out of %s correct, accuracy rate is %s' %
                (t,
                test_df.shape[0],
                t/test_df.shape[0]))
    print('precision is %s, recall is %s' %
         (tp / (tp+fp), tp / (tp + fn)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', default=0.75, type=float,
                        help='What is the size of the training set?')
    parser.add_argument('--max_iter', type=int, default=200,
                        help='number of max iterations for model fitting')
    parser.add_argument('--reg', type=float, default=1.,
                        help='inverse regularization stregnth, smaller values specify stronger regularization.')
    parser.add_argument('--reg_norm', type=str, default='l2',
                        help='regularization norm')
    args = parser.parse_args()

    # Generate Training set
    split_train_test(train_size = args.train_size)

    # Create and Train Model
    train_df = pd.read_csv('data/train.tsv', sep='\t', encoding='latin')
    train_embeddings = np.load('data/train_embeddings.npy')

    clf_model = train_model(train_df, train_embeddings, args)
    joblib.dump(clf_model, 'models/clf_logreg.pkl')

    # Test Model:
    test_df = pd.read_csv('data/test.tsv', sep='\t', encoding='latin')
    test_embeddings = np.load('data/test_embeddings.npy')

    test_model(clf_model, test_df, test_embeddings)

        #
        # # Predict
        # preds, model_outputs, all_embedding_outputs, all_layer_hidden_states = model.predict(['So happy test.', 'This is depressing test.'])
        # print(preds)
