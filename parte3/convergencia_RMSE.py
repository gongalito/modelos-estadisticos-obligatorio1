import matplotlib.pyplot as plt
import numpy as np

from estandarizacion import estandarizar_y_limpiar_datos
from estimacion_modelo import estimar_mco, preparar_matrices


def ejecutar_experimento_convergencia(df, repeticiones_lista, k=5):
    """
    Evalua como converge el RMSE promedio a medida que aumenta el numero de repeticiones
    """
    rmse_promedios = []

    print("Ejecutando experimento de convergencia (K-Fold Repetido)...")

    for n_rep in repeticiones_lista:
        todos_los_rmse = []

        for rep in range(n_rep):
            np.random.seed(rep)
            indices_cv = np.random.permutation(len(df))
            folds = np.array_split(indices_cv, k)

            for i in range(k):
                test_idx = folds[i]
                train_idx = np.hstack([folds[j] for j in range(k) if j != i])

                tr_fold = df.iloc[train_idx]
                ts_fold = df.iloc[test_idx]

                X_tr, y_tr = preparar_matrices(tr_fold)
                X_ts, y_ts = preparar_matrices(ts_fold)

                beta_fold = estimar_mco(X_tr, y_tr)
                y_pred_fold = X_ts @ beta_fold

                rmse_k = np.sqrt(np.mean((y_ts - y_pred_fold) ** 2))
                todos_los_rmse.append(rmse_k)

        rmse_prom = np.mean(todos_los_rmse)
        rmse_promedios.append(rmse_prom)
        print(
            f"• Repeticiones: {n_rep:4d} | Pruebas totales: {n_rep * k:5d} | RMSE Promedio: {rmse_prom:.3f} cm"
        )

    return rmse_promedios


def graficar_convergencia(repeticiones_lista, rmse_promedios):
    """
    Genera y guarda la grafica
    """
    plt.figure(figsize=(9, 5))
    plt.plot(
        repeticiones_lista,
        rmse_promedios,
        marker="o",
        color="crimson",
        linewidth=2,
        markersize=6,
    )

    # Linea asintotica tomada del ultimo valor 
    valor_asintotico = rmse_promedios[-1]
    plt.axhline(
        y=valor_asintotico,
        color="black",
        linestyle="--",
        alpha=0.7,
        label=f"Valor Asintótico ({valor_asintotico:.2f} cm)",
    )

    plt.xscale("log")
    plt.title(
        "Convergencia del RMSE Promedio según el Número de Repeticiones",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Número de Repeticiones", fontsize=11)
    plt.ylabel("RMSE Promedio Global (cm)", fontsize=11)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Guardar la grafica
    nombre_archivo = "convergencia_rmse.png"
    plt.savefig(nombre_archivo, dpi=300)
    print(f"\nGráfica guardada como '{nombre_archivo}'")
    plt.show()


def main():
    df, total_validos = estandarizar_y_limpiar_datos()

    # se pueden modificar las repeticiones, a mayor cantidad de repeticiones mayor tiempo de computo (ejecuta en CPU)
    repeticiones_lista = [1, 5, 10, 50, 100, 500, 1000, 5000, 50000]
    k_folds = 5

    rmse_promedios = ejecutar_experimento_convergencia(df, repeticiones_lista, k=k_folds)
    graficar_convergencia(repeticiones_lista, rmse_promedios)


if __name__ == "__main__":
    main()
