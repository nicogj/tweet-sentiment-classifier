#Usage: python3 src/py/tweet_embedding.py 20200101 topic_name
import sys
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import argparse

model = SentenceTransformer('distiluse-base-multilingual-cased')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default = 100, type = int, help='batch size')
    args = parser.parse_args()

    print("Reading in data")
    corpus = pd.read_csv('data/labeled_data.tsv', sep='\t')['text'].values

    print("Creating Embeddings")
    embeddings = model.encode(corpus, show_progress_bar=True, batch_size=args.batch_size)

    print("Exporting")
    np.save('data/labeled_embeddings.npy', np.array(embeddings))

    print("Done !")
