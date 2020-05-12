from load_data import load_data
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Raza de calcul lbp (vecinatatea) si numarul de puncte din vecinatate
# Vecinatatea de 8 puncte este cea mai buna pentru imaginile cifar-10.
radius = 1
points = 8 * radius


# Deprecated
def lbp_deprecated():
    train_df, test_df = load_data()
    print("Generating lbp feature vectors...")

    # Aplica lbp pe fiecare imagine din dataset si stocheaza rezultatul intr-o coloana noua
    train_df['lbp'] = train_df['grayscale'].apply(
        lambda x: local_binary_pattern(x, points, radius, 'uniform')
    )
    train_df = train_df.drop(columns=['grayscale'])

    test_df['lbp'] = test_df['grayscale'].apply(
        lambda x: local_binary_pattern(x, points, radius, 'uniform')
    )
    test_df = test_df.drop(columns=['grayscale'])

    # Print pentru verificare
    lbpimage = train_df['lbp'].loc[144]
    plt.imshow(lbpimage, cmap='gray', interpolation='bicubic')
    plt.show()

    # Coloana noua ce contine histograma fiecarei imagini ca features
    train_df['features'] = train_df['lbp'].apply(lambda x: hist(x))
    test_df['features'] = test_df['lbp'].apply(lambda x: hist(x))

    # Convert la liste
    print("Generating training data as lists...")
    X_train = train_df['features'].tolist()
    y_train = train_df.loc[:, 'labels']

    X_test = test_df['features'].tolist()
    y_test = test_df.loc[:, 'labels']

    return X_train, y_train, X_test, y_test


# Scoate histograma unei imagini lbp ce va fi folosita ca features
def hist(lbp_image):
    # Normalizare automata cu normed=True
    # Nr de bins se ia ca nr de puncte + 2 (inclusiv limitele inf si sup)
    his = np.histogram(lbp_image, normed=True, bins=points + 2, range=(0, points + 2))
    return his[0]


# genereaza histogramele pentru lbp realizat pe regiuni de imagine (4 la numar)
def lbp_by_pieces(lbp_image, regions=4):
    # Vector de bucati de imagine
    pieces = []
    # Vector de histograme lbp
    vectors = []
    # Se calculeaza bucatile de imagine
    for i in range(regions):
        for j in range(regions):
            pieces.append(lbp_image[8 * i: 8 * (i + 1), 8 * j: 8 * (j + 1)])

    # Se calculeaza imaginea lbp pentru fiecare bucata de imagine in parte + histograma aferenta
    for piece in pieces:
        lbp = local_binary_pattern(piece, points, radius, 'uniform')
        his, _ = np.histogram(lbp, normed=True, bins=points + 2, range=(0, points + 2))
        vectors.append(his)

    return np.concatenate(vectors)


# Aplica LBP
def generate_lbp():
    train_df, test_df = load_data()
    print("Generating lbp feature vectors...")

    # Partea cu impartire pe 4 regiuni a fiecarei imagini - foarte costisitoare computational
    train_df['features'] = train_df['grayscale'].apply(lambda x: lbp_by_pieces(x))
    test_df['features'] = test_df['grayscale'].apply(lambda x: lbp_by_pieces(x))
    train_df = train_df.drop(columns=['grayscale'])
    test_df = test_df.drop(columns=['grayscale'])

    print("Outputing to pickle files")
    train_df.to_pickle('train_data')
    test_df.to_pickle('test_data')


if __name__ == '__main__':
    generate_lbp()
