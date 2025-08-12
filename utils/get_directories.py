import pandas as pd

root_dir = '../dataset/maestro/'

csv_dir = root_dir + 'maestro-v3.0.0.csv'
metadata = pd.read_csv(csv_dir)

train_split = metadata[
    ((metadata['year'] == 2018) | (metadata['year'] == 2017)) &
    (metadata['split'] == 'train')
]

test_split = metadata[
    ((metadata['year'] == 2018) | (metadata['year'] == 2017)) &
    (metadata['split'] == 'validation')
]

print(test_split.head())