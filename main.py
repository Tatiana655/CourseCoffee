import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.

pathHQ = "C:\\Users\\Yawor\\OneDrive\\Рабочий стол\\dataset\\dataset\\HQ_Coffee\\"
pathLQ = "C:\\Users\\Yawor\\OneDrive\\Рабочий стол\\dataset\\dataset\\LQ_Coffee\\"
pathAQ = "C:\\Users\\Yawor\\OneDrive\\Рабочий стол\\dataset\\dataset\\AQ_Coffee\\"

if __name__ == '__main__':
    print_hi('PyCharm')
    dataHQ = []
    dataLQ = []
    dataAQ = []
    all_data = []
    Nose_data = []
    ans = []
    for filename in os.listdir(pathHQ):
        all_data.append(np.loadtxt(pathHQ + filename))
        ans.append(1)

    for filename in os.listdir(pathAQ):
        all_data.append(np.loadtxt(pathAQ + filename))
        ans.append(2)

    for filename in os.listdir(pathLQ):
        all_data.append(np.loadtxt(pathLQ + filename))
        ans.append(3)
    print("Количество образцов:", len(all_data))
    all_data = np.array([features_2d.flatten() for features_2d in all_data])
    scaler = StandardScaler()
    scaler.fit(all_data)
    Bcancer_scaled = scaler.transform(all_data)

    n = 13
    pca = PCA(n_components=n)
    pca.fit(Bcancer_scaled)
    pca_bcancer = pca.transform(Bcancer_scaled)

    component = np.array([pca_bcancer[:, i] for i in range(n)]).transpose()
    y = np.array(ans)
    X_train, X_test, y_train, y_test = train_test_split(
        component, y, stratify=y, test_size=0.1, random_state=42
    )
    n_neighbors = 4
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            #("nca", NeighborhoodComponentsAnalysis()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')),
        ]
    )
    ''' тут типа посмотреть что работает'''
    '''
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    y_res = clf.predict(X_test)
    print(y_res)
    #print(clf.predict_proba(X_test))
    print(y_test)
    print(sum(y_test==y_res)/len(y_test))
    print(score)'''

    '''глобальная кросс валидация'''
    _scoring = ['accuracy']
    results = cross_validate(estimator=clf,
                             X=component,
                             y=y,
                             cv = 5,
                             scoring=_scoring,
                             return_train_score=True)

    print("Training Accuracy scores", results['train_accuracy'])
    print("Mean Training Accuracy", results['train_accuracy'].mean() * 100)
    print("Validation Accuracy scores", results['test_accuracy'])
    print("Mean Validation Accuracy", results['test_accuracy'].mean() * 100)


    '''тут плот собственных чисел
        plt.style.use("ggplot")
    val = pca.explained_variance_
    print(val)
    plt.plot(val, marker='o')
    plt.xlabel("Eigenvalue number")
    plt.ylabel("Eigenvalue size")
    plt.title("Scree Plot")
    plt.semilogy()
    plt.show()
    '''

    '''тут плот трех первых компонент
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111,
                         projection='3d')
    color=["r","g","b"]
    for l in np.unique(y):
        ix = np.where(y == l)
        ax.scatter(Xax[ix],
                   Yax[ix],
                   Zax[ix],
                   c=color[l-1],
                   s=60,
                   label=l)

    ax.set_xlabel("PC1",
                  fontsize=12)
    ax.set_ylabel("PC2",
                  fontsize=12)
    ax.set_zlabel("PC3",
                  fontsize=12)

    ax.view_init(30, 125)
    ax.legend()
    plt.title("3D PCA plot")
    plt.show()
    '''
