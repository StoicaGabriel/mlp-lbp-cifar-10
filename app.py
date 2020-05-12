import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from lbp import generate_lbp
from time import time


# TODO: operation timing
if __name__ == '__main__':
    start = time()
    try:
        print("Reading existing data files...")
        train_df = pd.read_pickle('train_data')
        test_df = pd.read_pickle('test_data')
    except:
        print("Data files non existent, attempting to create...")
        generate_lbp()
        train_df = pd.read_pickle('train_data')
        test_df = pd.read_pickle('test_data')
    end = time()

    # Amestecare aleatoare a datelor (se foloseste Stochastic Gradient ca optimizer pentru MLP
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    print(f"Operation successful in {end - start} seconds")

    print("Generating training data as lists...")
    start = time()
    X_train = train_df['features'].tolist()
    y_train = train_df.loc[:, 'labels']

    X_test = test_df['features'].tolist()
    y_test = test_df.loc[:, 'labels']
    end = time()
    print(f"Operation successful in {end - start} seconds")

    # mlpclassifier = MLPClassifier(alpha=0.4, hidden_layer_sizes=(500,))
    # mlpclassifier.fit(X_train, y_train)
    # y_pred = mlpclassifier.predict(X_test)
    # print(f" Classifier accuracy score: {accuracy_score(y_test, y_pred) * 100}%")
    # print(mlpclassifier.loss_curve_)

    # Partea de machine learning
    print("Starting model training...")
    for n_hid in [100, 2500, 5000]:
        print(f"\t{n_hid} neurons")
        for lr in [0.1, 10, 30]:
            start = time()
            print(f"learning rate {lr}")
            print("Fitting classifier...")
            mlpclassifier = MLPClassifier(alpha=lr, hidden_layer_sizes=(n_hid,))
            mlpclassifier.fit(X_train, y_train)
            print("Generating predictions...")
            y_pred = mlpclassifier.predict(X_test)
            print(f" Classifier accuracy score: {accuracy_score(y_test, y_pred) * 100}%")
            end = time()
            print(f"Network trained in {end - start} seconds")
    pass
