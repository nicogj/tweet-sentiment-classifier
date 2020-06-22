# python3 scripts/data_setup.py data/training_1600000_processed_noemoticon.csv

import pandas as pd
import argparse
from tqdm.auto import tqdm

from utils import clean_for_content

def readin_data(path):

    if path == 'data/training_1600000_processed_noemoticon.csv':
        df = pd.read_csv('data/training_1600000_processed_noemoticon.csv', encoding='latin', header=None)
        df = df[[0,5]]
        df.columns = ['label', 'text']
        df['label'] = [0 if x==0 else 1 for x in df['label']]
        df['lang'] = 'en'

    df['text'] = [clean_for_content(text, lang) for text, lang in tqdm(zip(df['text'], df['lang']), total=df.shape[0])]

    df = df[df['text']!=''].reset_index(drop=True)

    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('labeled_datafile', help='where is the labeled data located?')
    args = parser.parse_args()

    df = readin_data(args.labeled_datafile)

    df.to_csv('data/labeled_data.tsv', sep='\t', index=False)
