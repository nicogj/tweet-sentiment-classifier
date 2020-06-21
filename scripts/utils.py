from sklearn.model_selection import train_test_split
import pandas as pd

random_state = 123

def clean_for_content(string):

    string = string.lower()

    string = re.sub(r'\bhttps?\:\/\/[^\s]+', ' ', string) #remove websites

    # Classic replacements:
    string = re.sub(r'\&gt;', ' > ', string)
    string = re.sub(r'\&lt;', ' < ', string)
    string = re.sub(r'<\s?3', ' ❤ ', string)
    string = re.sub(r'\@\s', ' at ', string)
    string = re.sub(r'(\&(amp)?|amp;)', ' and ', string)
    string = re.sub(r'(\bw\/?\b)', ' with ', string)
    string = re.sub(r'\brn\b', ' right now ', string)

    string = re.sub(r'\s+', ' ', string).strip()

    return string

def split_train_test(train_size):

    df = pd.read_csv('data/labeled_data.tsv', sep='\t')

    df_train, df_test = train_test_split(df, test_size=1-train_size, random_state=random_state)

    print("TRAIN size:", df_train.shape[0])
    print("TEST size:", df_test.shape[0])

    df_train[['text', 'labels']].to_csv('data/train.tsv', sep='\t', index=False)
    df_test[['text', 'labels']].to_csv('data/test.tsv', sep='\t', index=False)
