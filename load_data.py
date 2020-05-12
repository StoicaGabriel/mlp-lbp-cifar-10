import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Despachetare baza de date dupa batch (documentatie cifar-10)
def unpickle(file: str) -> dict:
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')
    return dct


def load_data():
    train_batch_root = 'cifar-10-batches-py/data_batch_'
    data_batches = {}
    print("Getting training data batches...")
    # Loop pentru load pe fiecare batch de antrenare
    for i in range(1, 6):
        print(f"Getting batch {i}")
        batch = unpickle(train_batch_root + str(i))
        # numpy arrays din datele binare data si labels din batch -> usor de prelucrat
        train_data = np.array(batch[b'data'])
        train_labels = np.array(batch[b'labels'])
        # DataFrame pentru consistenta
        train_data_df = pd.DataFrame()
        # Calcul grayscale din RGB direct
        train_data_df['grayscale'] = [
            item[0:1024] * 0.2989 + item[1024:2048] * 0.5870 + item[2048:3072] * 0.1140
            for item in train_data
        ]
        train_data_df['labels'] = train_labels

        data_batches[f'batch{i}'] = train_data_df
    # Concatenare a DataFrame-urilor din dictionarul data_batches intr-unul singur
    train_df = pd.DataFrame()
    print("Concat'ing training data batches...")
    for i in range(1, 6):
        train_df = pd.concat([train_df, data_batches[f'batch{i}']], ignore_index=True)

    # Load data pentru datele de testare
    print("Getting test data batch...")
    test_batch_root = 'cifar-10-batches-py/test_batch'
    batch = unpickle(test_batch_root)
    test_data = np.array(batch[b'data'])
    test_labels = np.array(batch[b'labels'])
    test_df = pd.DataFrame()
    test_df['grayscale'] = [
        item[0:1024] * 0.2989 + item[1024:2048] * 0.5870 + item[2048:3072] * 0.1140
        for item in test_data
    ]
    test_df['labels'] = test_labels

    # Pastreaza doar forma grayscale + labels din dataframe
    train_df['grayscale'] = train_df['grayscale'].apply(lambda x: np.reshape(x, (32, 32)))
    test_df['grayscale'] = test_df['grayscale'].apply(lambda x: np.reshape(x, (32, 32)))

    return [train_df, test_df]


if __name__ == '__main__':
    # Print de verificare
    train, test = load_data()
    plt.imshow(train['grayscale'].loc[144], cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.show()
