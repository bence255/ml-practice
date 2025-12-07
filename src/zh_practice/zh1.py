import numpy as np
import seaborn
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

def main():
    # Olvassa be a digits beépített adatállományt és írassa ki a legfontosabb jellemzőit
    # (rekordok száma, attribútumok száma és osztályok száma). (3 pont)
    digits = load_digits()
    digits_df = load_digits(as_frame=True)
    n = digits.data.shape[0]  # number of records
    p = digits.data.shape[1]  # number of attributes
    k = digits.target_names.shape[0]

    print(f"Number of records:{n}")
    print(f"Number of attributes:{p}")
    print(f"Number of classes:{k}")


    # Készítsen többdimenziós vizualizációt a mátrix ábra segítségével (pairplot). (4 pont)
    # PCA a vizualizációhoz
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(digits.data)

    df_pca = pd.DataFrame(pca_data, columns=["PCA1", "PCA2", "PCA3"])
    df_pca["target"] = digits.target

    # Pairplot
    seaborn.pairplot(df_pca, hue="target")
    plot = seaborn.pairplot(df_pca, hue="target")
    plot.savefig("pairplot.png", dpi=300)

    # Particionálja az adatállományt 80% tanító és 20% tesztállományra. Keverje össze a
    # rekordokat és a véletlenszám-generátort inicializálja az idei évvel. (3 pont)
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=2024
    )


    # Végezzen felügyelt tanítást az alábbi modellekkel és beállításokkal: döntési fa (4
    # mélység, entrópia homogenitási kritérium), logisztikus regresszió (liblinear
    # solverrel) és neurális háló (1 rejtett réteg 4 neuronnal, logisztikus aktivációs
    # függvény). A teszt score alapján hasonlítsa össze az illesztett modelleket,
    # melyeket nyomtasson ki. (10 pont)
    tree = DecisionTreeClassifier(max_depth=4, criterion="entropy")
    logistic = LogisticRegression(solver="liblinear")
    neural = MLPClassifier(hidden_layer_sizes=4, activation="logistic")

    tree.fit(X_train, y_train)
    logistic.fit(X_train, y_train)
    neural.fit(X_train, y_train)

    tree_score = tree.score(X_test, y_test)
    logistic_score = logistic.score(X_test, y_test)
    neural_score = neural.score(X_test, y_test)

    print(f"tree score: {tree_score}")
    print(f"logistic score: {logistic_score}")
    print(f"neural score: {neural_score}")


    # Számolja ki az 5. pont legjobb modelljére a teszt
    # tévesztési mátrixot. (4 pont)
    predicted = logistic.predict(X_test)
    logistic_confusion_matrix = confusion_matrix(y_test, predicted)


    # Ábrázolja a tévesztési mátrixot. (3 pont)
    disp = ConfusionMatrixDisplay(logistic_confusion_matrix, display_labels=digits.target_names)
    disp.plot(cmap=plt.cm.Greens)
    plt.savefig("confusion.png", dpi=300)


    # Végezzen nemfelügyelt tanítást a K-közép módszerrel az input attribútumokon.
    # Határozza meg az optimális klaszterszámot 30-ig a DB indexszel. Az optimális
    # klaszterszám mellett vizualizálja a klasztereket egy pontdiagrammon, ahol a két
    # koordináta egy 2 dimenziós PCA eredménye. (13 pont)
    Max_K = 30
    DB = np.zeros((Max_K - 2))
    best_k = 0
    for i in range(Max_K - 2):
        K = i + 2
        kmeans_cluster = KMeans(n_clusters=K, random_state=2025)
        kmeans_cluster.fit(X)
        y_pred = kmeans_cluster.labels_
        DB[i] = davies_bouldin_score(X, y_pred)

    best_k = np.argmin(DB) + 2
    print(f"Optimal number of clusters: {best_k}")
    # Visualization of DB scores
    fig = plt.figure(4)
    plt.title("Davies-Bouldin score curve")
    plt.xlabel("Number of clusters")
    plt.ylabel("DB index")
    plt.plot(np.arange(2, Max_K), DB, color="blue", marker=".", markersize=10)
    plt.vlines([4, 8], ymin=0.6, ymax=0.9, colors="red")
    plt.savefig("db_score.png", dpi=300)

    pca = PCA(n_components=2, random_state=2025)
    pca_data = pca.fit_transform(X)

    K = 8
    kmeans8 = KMeans(n_clusters=K, random_state=2024)  # instance of KMeans class
    kmeans8.fit(X)  #  fiting cluster model for X
    y_pred = kmeans8.predict(X)  #  predicting cluster label
    centers = kmeans8.cluster_centers_  #  centroids

    fig = plt.figure(6)
    plt.title("Scatterplot of datapoints with 8 clusters")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.scatter(pca_data[:, 0], pca_data[:, 1], s=50, c=y_pred)
    plt.savefig("cluster.png", dpi=300)

if __name__ == "__main__":
    main()