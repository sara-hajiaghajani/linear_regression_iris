import pandas as pd
import numpy as np

dataset = pd.read_csv('iris_dataset.csv')


def train(x, t):
    X_setosa = x[:40]
    T_setosa = t[:40]
    XT_setosa = X_setosa.transpose()
    A_setosa = np.linalg.inv(np.dot(XT_setosa, X_setosa))
    w_setosa = np.dot(np.dot(A_setosa, XT_setosa), T_setosa)
    print("w for setosa:\n", w_setosa)

    X_versicolor = x[50:90]
    T_versicolor = t[50:90]
    XT_versicolor = X_versicolor.transpose()
    A_versicolor = np.linalg.inv(np.dot(XT_versicolor, X_versicolor))
    w_versicolor = np.dot(np.dot(A_versicolor, XT_versicolor), T_versicolor)
    print("w for versicolor:\n", w_versicolor)

    X_virginica = x[100:140]
    T_virginica = t[100:140]
    XT_virginica = X_virginica.transpose()
    A_virginica = np.linalg.inv(np.dot(XT_virginica, X_virginica))
    w_virginica = np.dot(np.dot(A_virginica, XT_virginica), T_virginica)
    print("w for virginica:\n", w_virginica)

    return w_setosa, w_versicolor, w_virginica


def test(w_setosa, w_versicolor, w_virginica, x, t):
    SSE_setosa = 0
    X_setosa = x[40:50]
    T_setosa = t[40:50]
    t_setosa = np.dot(X_setosa, w_setosa)
    for i in range(len(t_setosa)):
        SSE_setosa += ((T_setosa[i] - t_setosa[i])**2)
    SSE_setosa = (1/2)*SSE_setosa
    print("sum of square error for setosa:", SSE_setosa)

    SSE_versicolor = 0
    X_versicolor = x[90:100]
    T_versicolor = t[90:100]
    t_versicolor = np.dot(X_versicolor, w_versicolor)
    for i in range(len(t_versicolor)):
        SSE_versicolor += ((T_versicolor[i] - t_versicolor[i])**2)
    SSE_versicolor = (1/2)*SSE_versicolor
    print("sum of square error for versicolor:", SSE_versicolor)

    SSE_virginica = 0
    X_virginica = x[140:150]
    T_virginica = t[140:150]
    t_virginica = np.dot(X_virginica, w_virginica)
    for i in range(len(t_virginica)):
        SSE_virginica += ((T_virginica[i] - t_virginica[i])**2)
    SSE_virginica = (1/2)*SSE_virginica
    print("sum of square error for virginica:", SSE_virginica)


if __name__ == '__main__':
    df = dataset.copy()
    bias = []
    for i in range(150):
        bias.append(1)
    df.insert(0, "bias", bias)

    x_df = df.iloc[:, [0, 1, 2, 4]]
    t_df = df.iloc[:, [3]]

    x = x_df.to_numpy()
    t = t_df.to_numpy()

    w_setosa, w_versicolor, w_virginica= train(x, t)
    test(w_setosa, w_versicolor, w_virginica, x, t)

