import Database
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import Database

def get_cross_validation_score(flageig=False):
    pathHQ = "C:\\Users\\Yawor\\OneDrive\\Рабочий стол\\dataset\\dataset\\HQ_Coffee\\"
    pathLQ = "C:\\Users\\Yawor\\OneDrive\\Рабочий стол\\dataset\\dataset\\LQ_Coffee\\"
    pathAQ = "C:\\Users\\Yawor\\OneDrive\\Рабочий стол\\dataset\\dataset\\AQ_Coffee\\"
    dataH = Database.get_all_data(pathHQ, 1)
    dataA = Database.get_all_data(pathAQ, 2)
    dataL = Database.get_all_data(pathLQ, 3)
    all_data = dataH[0] + dataA[0] + dataL[0]
    ans = dataH[1] + dataA[1] + dataL[1]
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
            # ("nca", NeighborhoodComponentsAnalysis()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')),
        ]
    )
    '''кроссвалидация'''
    _scoring = ['accuracy']
    results = cross_validate(estimator=clf,
                             X=component,
                             y=y,
                             cv=5,
                             scoring=_scoring,
                             return_train_score=True)

    print("Training Accuracy scores", results['train_accuracy'])
    print("Mean Training Accuracy", results['train_accuracy'].mean() * 100)
    print("Validation Accuracy scores", results['test_accuracy'])
    print("Mean Validation Accuracy", results['test_accuracy'].mean() * 100)
    if flageig is True:
        plt.style.use("ggplot")
        val = pca.explained_variance_
        print(val)
        plt.plot(val, marker='o')
        plt.xlabel("Eigenvalue number")
        plt.ylabel("Eigenvalue size")
        plt.title("Scree Plot")
        plt.semilogy()
        plt.show()



