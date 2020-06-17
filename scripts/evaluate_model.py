# python3 scripts/evaluate_model.py models/checkpoint-20-epoch-1 --evalrows 1000

import pandas as pd
import numpy as np
from simpletransformers.classification import ClassificationModel
import sklearn
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('modelpath', type=str, help='Where is the model you want to evaluate?')
    parser.add_argument('--evalrows', type=int, default=1000, help='How many rows do you want to evaluate on?')
    args = parser.parse_args()

    # Load in model
    model = ClassificationModel('bert', args.modelpath, use_cuda=False)

    # Load in Test
    test_df = pd.read_csv('data/test.tsv', sep='\t', encoding='latin', nrows=args.evalrows)

    # Evaluate the model
    results, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)
    print(results)

    # Predict
    preds, model_outputs, all_embedding_outputs, all_layer_hidden_states = model.predict(['So happy test.', 'This is depressing test.'])
    print(preds)
