import numpy as np
import matplotlib.pyplot as plt

from estandarizacion import estandarizar_y_limpiar_datos
from estimacion_modelo import preparar_matrices, estimar_mco, evaluar_modelo


def graficar_resultados(df_train, df_test, beta):
    """
    Genera la grafica del modelo
    """
    plt.figure(figsize=(11, 7))

    x_train_padres = (df_train["altura_padre"] + df_train["altura_madre"]) / 2
    x_test_padres = (df_test["altura_padre"] + df_test["altura_madre"]) / 2

    mask_m_train = df_train["es_hombre"] == 1
    mask_f_train = df_train["es_hombre"] == 0
    mask_m_test = df_test["es_hombre"] == 1
    mask_f_test = df_test["es_hombre"] == 0

    # Entrenamiento
    plt.scatter(
        x_train_padres[mask_m_train], df_train.loc[mask_m_train, "altura_hijo"],
        color="royalblue", marker="o", s=50, alpha=0.7, label="Train - Hombres", edgecolors="k"
    )
    plt.scatter(
        x_train_padres[mask_f_train], df_train.loc[mask_f_train, "altura_hijo"],
        color="deepskyblue", marker="^", s=50, alpha=0.7, label="Train - Mujeres", edgecolors="k"
    )

    # Test
    plt.scatter(
        x_test_padres[mask_m_test], df_test.loc[mask_m_test, "altura_hijo"],
        color="firebrick", marker="o", s=60, alpha=0.9, label="Test - Hombres", edgecolors="k"
    )
    plt.scatter(
        x_test_padres[mask_f_test], df_test.loc[mask_f_test, "altura_hijo"],
        color="crimson", marker="^", s=60, alpha=0.9, label="Test - Mujeres", edgecolors="k"
    )

    # Rectas
    x_min = min(x_train_padres.min(), x_test_padres.min())
    x_max = max(x_train_padres.max(), x_test_padres.max())
    x_linea = np.linspace(x_min, x_max, 100)

    b0 = beta[0][0]
    b_hombre = beta[1][0]
    b_padres = beta[2][0] + beta[3][0]

    y_mujeres = b0 + (b_padres * x_linea)
    y_hombres = (b0 + b_hombre) + (b_padres * x_linea)

    plt.plot(x_linea, y_hombres, color="blue", linestyle="--", linewidth=2, label="Ajuste Hombres")
    plt.plot(x_linea, y_mujeres, color="magenta", linestyle="--", linewidth=2, label="Ajuste Mujeres")

    plt.title("Ajuste del Modelo Lineal", fontsize=14, fontweight="bold")
    plt.xlabel("Altura Media de los Padres (cm)", fontsize=12)
    plt.ylabel("Altura del Hijo/a (cm)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


def main():

    # Cargar y estandarizar datos (los csv tienen formatos distintos)
    df, total_validos = estandarizar_y_limpiar_datos()
    print(f"Total de datos limpios y válidos: {total_validos}")
 
    # Split 80/20 para estimacion
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * 0.2)

    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]

    # Preparacion de matrices y estimacion MCO
    X_train, y_train = preparar_matrices(train_df)
    X_test, y_test = preparar_matrices(test_df)

    beta = estimar_mco(X_train, y_train)

    # Evaluacion en Split 80/20
    ecm_tr, rmse_tr, r2_tr = evaluar_modelo(X_train, y_train, beta)
    ecm_ts, rmse_ts, r2_ts = evaluar_modelo(X_test, y_test, beta)

    # Resultados
    print("\n" + "=" * 50)
    print("  ECUACIÓN DEL MODELO DE REGRESIÓN LINEAL")
    print("=" * 50)
    print(f"Altura_Hijo = {beta[0][0]:.2f} + ({beta[1][0]:.2f} * es_hombre) + "
          f"({beta[2][0]:.2f} * Altura_Padre) + ({beta[3][0]:.2f} * Altura_Madre)\n")

    print("\n" + "=" * 50)
    print("  MÉTRICAS DE RENDIMIENTO (SPLIT 80/20)")
    print("=" * 50)
    print(f"TRAIN -> R²: {r2_tr:.3f} | RMSE: {rmse_tr:.2f} cm | ECM: {ecm_tr:.2f}")
    print(f"TEST  -> R²: {r2_ts:.3f} | RMSE: {rmse_ts:.2f} cm | ECM: {ecm_ts:.2f}")

    # Grafica
    graficar_resultados(train_df, test_df, beta)


if __name__ == "__main__":
    main()