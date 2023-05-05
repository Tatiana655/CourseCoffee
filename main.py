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
import Metrics

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.

pathHQ = "C:\\Users\\Yawor\\OneDrive\\Рабочий стол\\dataset\\dataset\\HQ_Coffee\\"
pathLQ = "C:\\Users\\Yawor\\OneDrive\\Рабочий стол\\dataset\\dataset\\LQ_Coffee\\"
pathAQ = "C:\\Users\\Yawor\\OneDrive\\Рабочий стол\\dataset\\dataset\\AQ_Coffee\\"

if __name__ == '__main__':
    print_hi('PyCharm')
    Metrics.get_cross_validation_score()

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
