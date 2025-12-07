from urllib.request import urlopen

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, \
    auc, roc_curve, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

def main() -> None:
    # Olvassa be a banknote_authentication.txt adatállományt a
    # https://arato.inf.unideb.hu/ispany.marton/MachineLearning/Datasets/ URL címről és
    # írassa ki a legfontosabb jellemzőit, úm. rekordok száma, attribútumok száma és
    # osztályok száma. (3 pont)
    url = "https://arato.inf.unideb.hu/ispany.marton/MachineLearning/Datasets/banknote_authentication.txt"
    raw_data = urlopen(url)
    data = np.loadtxt(raw_data, delimiter=",")

    classes_set = set()
    for i in data:
        classes_set.add(i[-1])

    print(f"number of records: {data.shape[0]}")
    print(f"Number of attributes: {data.shape[1]}")
    print(f"Number of classes: {classes_set.__len__()}")


    # Csináljon DataFrame-t az adatokból és ábrázolja őket parallel_coordinates ábrával
    # ahol színezze a két osztálytt kékkel és pirossal. (3 pont)
    df = pd.DataFrame(data=data)
    pd.plotting.parallel_coordinates(df, class_column=df.columns[-1], color=['blue','red'])
    plt.savefig("pandas.png", dpi=300)

    # Partícionálja az adatállományt 70% tanító és 30% tesztállományra. Keverje össze a
    # rekordokat és a véletlenszám-generátort inicializálja az idei évvel. (2 pont)
    x = data[:, :-1]
    y = data[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2025, shuffle=True, test_size=0.3)


    # Végezzen felügyelt tanítást az alábbi modellekkel és beállításokkal: döntési fa (6
    # mélység, Gini homogenitási kritérium), logisztikus regresszió (alapbeállítással) és
    # Gauss-féle naív Bayes. A teszt pontosság alapján hasonlítsa össze az illesztett
    # modelleket, nyomtassa ki a legjobb teszt pontosságát. (10 pont)
    tree = DecisionTreeClassifier(max_depth=6, criterion="gini", random_state=2025)
    logistic = LogisticRegression(random_state=2025)
    bayes = GaussianNB()

    tree.fit(x_train,y_train)
    logistic.fit(x_train,y_train)
    bayes.fit(x_train,y_train)

    print(f"tree score: {tree.score(x_test,y_test)}")
    print(f"logistic score: {logistic.score(x_test,y_test)}")
    print(f"bayes score: {bayes.score(x_test,y_test)}")


    # Számolja ki a 4. pont legjobb modelljére a teszt tévesztési mátrixot, amelyet
    # jelenítsen is meg egyben. (4 pont)
    pred_log = logistic.predict(x_test)
    cm_logreg_train = confusion_matrix(y_test, pred_log)

    disp = ConfusionMatrixDisplay(cm_logreg_train)
    disp.plot(cmap=plt.cm.Greens)
    plt.title("Confusion matrix for training dataset (logistic regression)")
    plt.savefig("confusion_matrix.png", dpi=300)


    # Rajzolja ki a 4. pont legjobb modelljének ROC görbéjét. (4 pont)
    proba_logreg = logistic.predict_proba(x_test)
    fpr_logreg, tpr_logreg, _ = roc_curve(y_test, proba_logreg[:, 1])
    auc_logreg = auc(fpr_logreg, tpr_logreg)
    disp = RocCurveDisplay(
        fpr=fpr_logreg,
        tpr=tpr_logreg,
        roc_auc=auc_logreg,
        estimator_name="Logistic regression",
    )
    disp.plot()
    plt.title("ROC curve for test dataset (logistic regression)")
    plt.savefig("roc_curve.png", dpi=300)


    # Végezzen nemfelügyelt tanítást a K-közép módszerrel a tanító állomány input
    # attribútumain. Határozza meg az optimális klaszterszámot 30-ig való kereséssel a DB
    # index alapján a teszt állományon. (7 pont)
    Max_K = 31  # maximum cluster number
    SSE = np.zeros((Max_K - 2))  # array for sum of squares errors
    DB = np.zeros((Max_K - 2))  # array for Davies Bouldin indeces
    for i in range(Max_K - 2):
        K = i + 2
        kmeans1 = KMeans(n_clusters=K, random_state=2025)
        kmeans1.fit(x)
        y_pred = kmeans1.labels_
        SSE[i] = kmeans1.inertia_
        DB[i] = davies_bouldin_score(x, y_pred)
    best_k = np.argmin(DB) + 2
    print(f"Optimal number of clusters: {best_k}")

    # K=2 klaszterszám mellett vizualizálja a klasztereket egy pontdiagramon, ahol a két
    # koordináta egy 2 dimenziós PCA eredménye. (7 pont)
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(x)
    K = 2
    kmeans4 = KMeans(n_clusters=K, random_state=2025)  # instance of KMeans class
    kmeans4.fit(x)  # fiting cluster model for X
    y_pred = kmeans4.predict(x)  # predicting cluster label

    plt.figure(5)
    plt.title("Scatterplot of datapoints with 2 clusters")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.scatter(pca_data[:, 0], pca_data[:, 1], s=50, c=y_pred)
    plt.savefig("pca_bank_notes.png",dpi=300)


    print()
if __name__ == "__main__":
    main()