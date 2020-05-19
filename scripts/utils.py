from sklearn.model_selection import train_test_split
import pandas as pd

random_state = 123

def split_train_test(train_size):

    df = pd.read_csv('data/training_1600000_processed_noemoticon.csv', encoding='latin', header=None)
    df = df[[0,5]]
    df.columns = ['labels', 'text']
    df['labels'] = [0 if x==0 else 1 for x in df['labels']]

    df_train, df_test = train_test_split(df, test_size=1-train_size, random_state=random_state)

    print("TRAIN size:", df_train.shape[0])
    print("TEST size:", df_test.shape[0])

    df_train[['text', 'labels']].to_csv('data/train.tsv', sep='\t', index=False)
    df_test[['text', 'labels']].to_csv('data/test.tsv', sep='\t', index=False)
