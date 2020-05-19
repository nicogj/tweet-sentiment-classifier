from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging

# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
train_df = pd.read_csv('data/train.tsv', sep='\t', encoding='latin')

test_df = pd.read_csv('data/test.tsv', sep='\t', encoding='latin')

# Create a ClassificationModel
model = ClassificationModel('bert', 'bert-base-cased', use_cuda=False, args={'overwrite_output_dir': True})

# Train the model
model.train_model(train_df, output_dir = 'models')
