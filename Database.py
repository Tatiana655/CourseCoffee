import numpy as np
import os


def get_samples(path_files, quality):
    all_data = []
    ans = []
    for filename in os.listdir(path_files):
        all_data.append(np.loadtxt(path_files + filename))
        ans.append(quality)
    return [all_data, ans]


HQ = "HQ_Coffee\\"
AQ = "AQ_Coffee\\"
LQ = "LQ_Coffee\\"


class Database:
    def __init__(self, path="data\\"):
        self.data = []
        # тут пока чтение из файла, видимо.
        self.ans = []
        print("Creation of database")
        dataH = get_samples(path + HQ, 1)
        dataA = get_samples(path + AQ, 2)
        dataL = get_samples(path + LQ, 3)
        self.data = dataH[0] + dataA[0] + dataL[0]
        self.ans = dataH[1] + dataA[1] + dataL[1]

    def add_sample(self, filename, answer):
        print("check sample and add if ok else msg + save data in broker")
        self.data.append(np.loadtxt(filename))
        self.ans.append(answer)

    def refresh(self, path="data\\HQ_Coffee\\"):
        self.data = []
        # тут пока чтение из файла, видимо.
        self.ans = []
        print("Refreshing of database")
        dataH = get_samples(path + HQ, 1)
        dataA = get_samples(path + AQ, 2)
        dataL = get_samples(path + LQ, 3)
        self.data = dataH[0] + dataA[0] + dataL[0]
        self.ans = dataH[1] + dataA[1] + dataL[1]

    def delete_sample(self, num):
        print("ask again: Are you sure? and delete.")
        self.data.pop(num)
        self.ans.pop(num)
