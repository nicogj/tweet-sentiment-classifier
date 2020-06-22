# python3 scripts/evaluate_model.py models/checkpoint-20-epoch-1 --evalrows 1000

import pandas as pd
import numpy as np
from simpletransformers.classification import ClassificationModel
import sklearn
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('modelpath', type=str, help='Where is the model you want to evaluate?')
    args = parser.parse_args()

    # Load in model
    model = ClassificationModel('distilbert', args.modelpath, use_cuda=False)

    # Load in Test
    test_df = pd.read_csv('data/test.tsv', sep='\t', encoding='latin')
    test_embedding = np.load('data/test_embeddings.npy')

    # Evaluate the model
    results, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)
    print(results)

    # Predict
    preds, model_outputs, all_embedding_outputs, all_layer_hidden_states = model.predict(['So happy test.', 'This is depressing test.'])
    print(preds)
