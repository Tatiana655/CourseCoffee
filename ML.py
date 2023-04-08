import numpy as np
import Database
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis

class Classification:
    def __init__(self, database):
        self.eps = 0.1
        self.dataBase = database
        self.n_comp = 13
        self.n_neigh = 4

    def get_prediction(self, sample):
        # check_sample + send to brocker
        data, ans = self.dataBase.get_all_data()
        data = np.array([features_2d.flatten() for features_2d in data])
        y = np.array(ans)
        scaler = StandardScaler()
        scaler.fit(data)
        Bcancer_scaled = scaler.transform(data)
        pca = PCA(n_components=self.n_comp)
        pca.fit(Bcancer_scaled)
        pca_bcancer = pca.transform(Bcancer_scaled)
        component = np.array([pca_bcancer[:, i] for i in range(self.n_comp)]).transpose()
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                # ("nca", NeighborhoodComponentsAnalysis()),
                ("knn", KNeighborsClassifier(n_neighbors=self.n_neigh, weights='distance')),
            ]
        )
        clf.fit(data, y)
        # clf.predict_proba(X_test) # тут типа вероятности, пока не решили надо оно или нет
        return clf.predict(sample)


