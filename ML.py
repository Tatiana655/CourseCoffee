import numpy as np
import Database
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis


class Model:
    def __init__(self, database):
        self.eps = 0.1
        self.dataBase = database
        self.n_comp = 13
        self.n_neigh = 4
        self.model = None

    def get_model(self):
        # check_sample + send to brocker
        data = self.dataBase.data
        ans = self.dataBase.ans
        data = np.array([features_2d.flatten() for features_2d in data])
        y = np.array(ans)
        clf = Pipeline(
            [
                #("scaler1", StandardScaler()),
                ("normalizer1", Normalizer()),
                ('pca', PCA(self.n_comp)),
                ("scaler2", StandardScaler()),
                # ("nca", NeighborhoodComponentsAnalysis()),
                ("knn", KNeighborsClassifier(n_neighbors=self.n_neigh, weights='distance')),
            ]
        )
        clf.fit(data, y)
        self.model = clf
        return True

    def get_prediction(self, sample):
        # clf.predict_proba(X_test) # тут типа вероятности, пока не решили надо оно или нет
        return self.model.predict(sample)
