import Database
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import ML


def get_cross_validation_score(model, flageig=False):
    all_data = model.dataBase.data
    ans = model.dataBase.ans
    all_data = np.array([features_2d.flatten() for features_2d in all_data])

    clf = model.model

    _scoring = ['accuracy']
    results = cross_validate(estimator=clf,
                             X=all_data,
                             y=ans,
                             cv=5,
                             scoring=_scoring,
                             return_train_score=True)

    print("Training Accuracy scores", results['train_accuracy'])
    print("Mean Training Accuracy", results['train_accuracy'].mean() * 100)
    print("Validation Accuracy scores", results['test_accuracy'])
    print("Mean Validation Accuracy", results['test_accuracy'].mean() * 100)



