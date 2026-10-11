import numpy as np

def preparar_matrices(dataframe):
    """
    Construye la matriz X y el vector y
    """
    y = dataframe["altura_hijo"].values.reshape(-1, 1)
    n = len(dataframe)
    ones = np.ones((n, 1))
    x_features = dataframe[["es_hombre", "altura_padre", "altura_madre"]].values
    X = np.hstack((ones, x_features))
    return X, y


def estimar_mco(X, y):
    """
    Calcula los coeficientes beta mediante la ecuacion: (X^T * X)^(-1) * X^T * y
    """
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta


def evaluar_modelo(X, y, beta):
    """
    Calcula las metricas de rendimiento MSE, RMSE y R^2
    """
    y_pred = X @ beta
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)

    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return mse, rmse, r2