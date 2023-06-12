import matplotlib.pyplot as plt

import Metrics
import ML
import Database
import numpy as np


def get_nosed_data(all_data, p):
    return all_data * np.random.uniform(1 - p, 1 + p, (len(all_data), len(all_data[0]), len(all_data[0][0])))


if __name__ == '__main__':
    database = Database.Database()
    model1 = ML.Model(database)
    print("модель создана:", model1.get_model())
    Metrics.get_cross_validation_score(model1)
    all_data = model1.dataBase.data
    for p in [0.025, 0.25]:
        nosed_data = get_nosed_data(all_data, p)
        print(np.shape(nosed_data))
        prediction = []
        for el in nosed_data:
            print(np.shape(el))
            prediction.append(model1.get_proba_prediction(el))

        prediction = np.array(prediction)
        arr = np.amax(prediction, -1)
        plt.plot(arr, "o", label="noise=" + str(p))
        plt.ylim([0, 1])
    plt.title("max probability")
    plt.legend()
    plt.show()

    nosed_data025 = get_nosed_data(all_data, 0.025)
    nosed_data25 = get_nosed_data(all_data, 0.25)
    prediction = []

    for el in nosed_data025:
        prediction.append(model1.get_proba_prediction(el))
    idch = len(prediction)
    for el in nosed_data25:
        prediction.append(model1.get_proba_prediction(el))
    prediction = np.array(prediction)
    arr = np.amax(prediction, -1)
    time_arr = [sum(arr[0:i]) / (i) for i in range(1, len(arr))]
    # plt.plot(range(1, iddef + 1), time_arr[0:iddef], "o", label="noise=" + str(0))
    plt.plot(range(1, idch + 1), time_arr[0:idch], "o", label="noise=" + str(0.025))
    plt.plot(range(idch + 1, len(time_arr)), time_arr[idch:-1], "o", label="noise=" + str(0.25))
    # plt.ylim([0, 1])
    plt.title("sum(max(probas))/len(probas)")
    plt.legend()
    plt.show()

    for p in [0.025, 0.25]:
        nosed_data = get_nosed_data(all_data, p)
        prediction = []
        for el in nosed_data:
            prediction.append(model1.get_proba_prediction(el))
        prediction = np.array(prediction)
        arr = abs(2 * np.amax(prediction, -1) - 1 - np.amin(prediction, -1))
        plt.plot(arr, "o", label="noise=" + str(p))
        plt.ylim([0, 1])
    plt.title("minimal difference")
    plt.legend()
    plt.show()
