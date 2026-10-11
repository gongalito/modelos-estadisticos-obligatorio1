#PROBLEMA 2
import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
SEED = 42
GRADOS = [1, 2, 3]
ALPHAS = np.logspace(-3, 5, 60)


# Datos
def cargar_boston():
    url = "http://lib.stat.cmu.edu/datasets/boston"
    raw = pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)
    cols = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS",
            "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    df = pd.DataFrame(np.hstack([raw.values[::2, :], raw.values[1::2, :2]]), columns=cols)
    df["MEDV"] = raw.values[1::2, 2]
    return df


def exploratorio(df):
    print(f"Dimensiones: {df.shape} | valores faltantes: {df.isna().sum().sum()}\n")
    print(df.describe().T.round(2), "\n")
    print("Correlacion con MEDV:")
    print(df.corr()["MEDV"].drop("MEDV").sort_values().round(2).to_string(), "\n")
    c = df.drop(columns="MEDV").corr().abs()
    pares = c.where(np.triu(np.ones(c.shape), 1).astype(bool)).stack().sort_values(ascending=False)
    print("Pares de atributos mas correlacionados:")
    print(pares.head(5).round(2).to_string(), "\n")
    print(f"Casas con MEDV = 50 (dato censurado): {(df.MEDV == 50).sum()}\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(df.MEDV, bins=30, edgecolor="k", alpha=0.7)
    axes[0].set(title="Distribución de MEDV", xlabel="MEDV (miles USD)", ylabel="frecuencia")
    axes[1].scatter(df.LSTAT, df.MEDV, s=10, alpha=0.6)
    axes[1].set(title=f"LSTAT vs MEDV (r = {df.LSTAT.corr(df.MEDV):.2f})", xlabel="LSTAT", ylabel="MEDV")
    axes[2].scatter(df.RM, df.MEDV, s=10, alpha=0.6, color="tab:orange")
    axes[2].set(title=f"RM vs MEDV (r = {df.RM.corr(df.MEDV):.2f})", xlabel="RM", ylabel="MEDV")
    plt.tight_layout(); plt.savefig("exploratorio.png", dpi=150); plt.close()


# Modelos
def modelo(grado, estimador):
    """estandarizar -> potencias -> estandarizar -> regresion.
    La primera estandarizacion centra las variables y reduce la colinealidad entre x y x^2;
    la segunda hace que la penalizacion de Ridge trate a todos los atributos por igual."""
    if grado == 1:
        return make_pipeline(StandardScaler(), estimador)
    return make_pipeline(StandardScaler(), PolynomialFeatures(grado, include_bias=False),
                         StandardScaler(), estimador)


def mse(m, X, y):
    return mean_squared_error(y, m.predict(X))


def main(rapido):
    df = cargar_boston()
    exploratorio(df)

    X, y = df.drop(columns="MEDV"), df["MEDV"]
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    print(f"Entrenamiento: {len(X_tr)} | Validacion: {len(X_va)}\n")

    # Minimos cuadrados, grados 1 a 3
    filas, ols = [], {}
    for d in GRADOS:
        m = modelo(d, LinearRegression()).fit(X_tr, y_tr)
        ols[d] = m
        Z = m[:-1].transform(X_tr)
        cv = -cross_val_score(modelo(d, LinearRegression()), X_tr, y_tr, cv=kf,
                              scoring="neg_mean_squared_error")
        filas.append({"modelo": "Lineal" if d == 1 else f"Grado {d}", "p": Z.shape[1],
                      "rango Z": np.linalg.matrix_rank(Z), "MSE train": mse(m, X_tr, y_tr),
                      "MSE val": mse(m, X_va, y_va), "MSE CV": cv.mean(),
                      "R2 val": r2_score(y_va, m.predict(X_va)), "max|beta|": np.abs(m[-1].coef_).max()})
    tabla_ols = pd.DataFrame(filas).set_index("modelo")
    print("Minimos cuadrados sin regularizar:\n", tabla_ols.round(2), "\n")

    # Ridge y lambda por CV de 5 folds dentro de entrenamiento
    filas, ridge, curvas = [], {}, {}
    for d in GRADOS:
        gs = GridSearchCV(modelo(d, Ridge()), {"ridge__alpha": ALPHAS}, cv=kf,
                          scoring="neg_mean_squared_error", n_jobs=-1).fit(X_tr, y_tr)
        b = gs.best_estimator_
        ridge[d], curvas[d] = b, -gs.cv_results_["mean_test_score"]
        filas.append({"modelo": f"Ridge grado {d}", "lambda": gs.best_params_["ridge__alpha"],
                      "MSE train": mse(b, X_tr, y_tr), "MSE val": mse(b, X_va, y_va),
                      "MSE CV": -gs.best_score_, "R2 val": r2_score(y_va, b.predict(X_va)),
                      "max|beta|": np.abs(b[-1].coef_).max()})
    tabla_ridge = pd.DataFrame(filas).set_index("modelo")
    print("Ridge:\n", tabla_ridge.round(2), "\n")

    p3 = ols[3].predict(X_va)
    print(f"Grado 3 sin regularizar: predicciones en validacion entre {p3.min():.0f} y {p3.max():.0f}\n")

    # Figura resumen (la del informe)
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    ax = axes[0]
    ax.plot(GRADOS, tabla_ols["MSE val"], "o-", color="tab:red", label="Validación, sin regularizar")
    ax.plot(GRADOS, tabla_ols["MSE CV"], "o--", color="tab:red", alpha=0.5, label="CV, sin regularizar")
    ax.plot(GRADOS, tabla_ridge["MSE val"], "s-", color="tab:green", label="Validación, Ridge")
    ax.plot(GRADOS, tabla_ridge["MSE CV"], "s--", color="tab:green", alpha=0.5, label="CV, Ridge")
    ax.set(yscale="log", xticks=GRADOS, xlabel="Grado del polinomio", ylabel="MSE (escala log)",
           title="(a) MSE vs. grado")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    for d, col in zip(GRADOS, ["tab:blue", "tab:orange", "tab:purple"]):
        ax.plot(ALPHAS, curvas[d], color=col, label=f"grado {d}")
        i = np.argmin(curvas[d]); ax.plot(ALPHAS[i], curvas[d][i], "o", color=col)
    ax.set(xscale="log", yscale="log", ylim=(8, 300), xlabel="λ", ylabel="MSE de CV",
           title="(b) Ridge: MSE de CV vs. λ")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.scatter(y_va, ols[2].predict(X_va), s=14, alpha=0.6, marker="x", color="tab:red",
               label=f"Grado 2 sin reg. (MSE {mse(ols[2], X_va, y_va):.1f})")
    ax.scatter(y_va, ridge[2].predict(X_va), s=14, alpha=0.7, color="tab:green",
               label=f"Grado 2 + Ridge (MSE {mse(ridge[2], X_va, y_va):.1f})")
    ax.scatter(y_va, ols[1].predict(X_va), s=10, alpha=0.4, color="gray",
               label=f"Lineal (MSE {mse(ols[1], X_va, y_va):.1f})")
    ax.plot([0, 55], [0, 55], "k--", lw=1)
    ax.set(xlim=(0, 55), ylim=(-5, 60), xlabel="MEDV real", ylabel="MEDV predicho",
           title="(c) Predicciones en validación")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("figura_problema2.png", dpi=200, bbox_inches="tight"); plt.close()
    print("Figuras guardadas: exploratorio.png, figura_problema2.png")

    if rapido:
        return

    # Robustez: se repite todo el procedimiento con 30 particiones
    print("\nRobustez en 30 particiones (tarda un par de minutos)...")
    alphas_r = np.logspace(-3, 5, 30)
    rob = {k: [] for k in ["Lineal", "Grado 2", "Grado 3", "Ridge grado 1", "Ridge grado 2", "Ridge grado 3"]}
    for s in range(30):
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=s)
        kf_s = KFold(5, shuffle=True, random_state=s)
        for d in GRADOS:
            m = modelo(d, LinearRegression()).fit(Xtr, ytr)
            rob["Lineal" if d == 1 else f"Grado {d}"].append(mse(m, Xva, yva))
            gs = GridSearchCV(modelo(d, Ridge()), {"ridge__alpha": alphas_r}, cv=kf_s,
                              scoring="neg_mean_squared_error", n_jobs=-1).fit(Xtr, ytr)
            rob[f"Ridge grado {d}"].append(mse(gs, Xva, yva))
    rob = pd.DataFrame(rob)
    resumen = rob.agg(["mean", "median", "std", "min", "max"]).T
    resumen.columns = ["media", "mediana", "desvio", "min", "max"]
    print(resumen.round(1))
    print(f"Ridge grado 3 le gana a Ridge grado 2 en "
          f"{(rob['Ridge grado 3'] < rob['Ridge grado 2']).mean():.0%} de las particiones")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapido", action="store_true", help="omite la seccion de robustez")
    main(parser.parse_args().rapido)
