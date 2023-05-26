import numpy as np
import Database
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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
                ("scaler1", StandardScaler()),
                #("flatten", Flatten())
                ("normalizer", Normalizer()),
                ('pca', PCA(self.n_comp)),
                ("scaler", StandardScaler()),
                # ("nca", NeighborhoodComponentsAnalysis()),
                ("knn", KNeighborsClassifier(n_neighbors=self.n_neigh, weights='distance')),
            ]
        )
        clf.fit(data, y)
        self.model = clf
        return True

    def get_prediction(self, sample):
        # clf.predict_proba(X_test) # тут типа вероятности, пока не решили надо оно или нет
        sample = sample.flatten()
        return self.model.predict(sample)

    def get_proba_prediction(self, sample):
        sample = sample.flatten()
        return self.model.predict_proba([sample])

    def grid_search(self):
        clf = Pipeline(
            [
                ("scaler1", StandardScaler()),
                ("normalizer", Normalizer()),
                ('pca', PCA()),
                ("scaler", StandardScaler()),
                # ("nca", NeighborhoodComponentsAnalysis()),
                ("knn", KNeighborsClassifier(n_neighbors=self.n_neigh, weights='distance')),
            ]
        )
        parameters = {
            'pca__n_components': [3, 5, 7, 11, 13],
            'clf___neighbors': [2, 3, 4, 5, 6],
        }
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        gs = GridSearchCV(clf, parameters, cv=kf, n_jobs=-1, verbose=1)
        all_data = self.dataBase.data
        ans = self.dataBase.ans
        all_data = np.array([features_2d.flatten() for features_2d in all_data])
        gs.fit(all_data, ans)

        print("Best score: %0.3f" % gs.best_score_)
        print("Best parameters set:")
        best_parameters = gs.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))