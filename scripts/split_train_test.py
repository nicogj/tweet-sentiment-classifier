from sklearn.model_selection import train_test_split
import pandas as pd

TEST_SIZE = 0.25
random_state = 123

if __name__ == '__main__':
    # Read in data
    df = pd.read_csv('data/training.1600000.processed.noemoticon.csv', encoding='latin', header=None)
    df = df[[0,5]]
    df.columns = ['label', 'text']
    df['label'] = [0 if x==0 else 1 for x in df['label']]

    df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=random_state)

    print("TRAIN size:", df_train.shape[0])
    print("TEST size:", df_test.shape[0])

    df_train[['text', 'label']].to_csv('data/train.tsv', sep='\t', index=False)
    df_test[['text', 'label']].to_csv('data/test.tsv', sep='\t', index=False)
