import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_data(name) -> tuple:
    data = pd.read_csv(name, header=None)
    first = data.iloc[:, 0]
    rest = data.iloc[:, 1:] / 255
    res_data = (first, rest)
    return res_data


def show_image(vector):
    image = np.array(vector).reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
