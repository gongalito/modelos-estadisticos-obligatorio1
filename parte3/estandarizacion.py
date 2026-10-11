import pandas as pd

"""

 Funcion que estandariza los datos de los csv, puesto que estan en distinto formato.
 Elimina datos que no especifican sexo
 Agrega la variable es_hombre en los datos

"""
def estandarizar_y_limpiar_datos(path_df1="Alturas.csv", path_df2="Alturas_clase.csv"):

    df1 = pd.read_csv(path_df1)
    df2 = pd.read_csv(path_df2)

    # Mapeo de columnas para estandarizar
    cols_df1 = {
        "(Altura estudiante) Altura de la persona (estudiante del curso) en cm": "altura_hijo",
        "(sexo) Sexo biológico de la persona": "sexo",
        "(Altura padre) Altura padre en cm": "altura_padre",
        "(Altura madre) Altura madre en cm": "altura_madre",
    }

    cols_df2 = {
        "Altura estudiante del curso (en cm)": "altura_hijo",
        "sexo": "sexo",
        "Altura papá (en cm)": "altura_padre",
        "Altura mamá (en cm)": "altura_madre",
    }

    columnas_deseadas = ["altura_hijo", "sexo", "altura_padre", "altura_madre"]

    df1 = df1.rename(columns=cols_df1)[columnas_deseadas]
    df2 = df2.rename(columns=cols_df2)[columnas_deseadas]

    # Concatenar conjuntos
    df = pd.concat([df1, df2], ignore_index=True)

    # Filtrar filas sin datos completos
    df = df.dropna(subset=columnas_deseadas).copy()

    # Normalizar texto de sexo
    df["sexo"] = df["sexo"].astype(str).str.strip().str.capitalize()
    df = df[df["sexo"].isin(["Masculino", "Femenino"])].copy()

    # Variable Dummy: 1 para Masculino, 0 para Femenino
    df["es_hombre"] = (df["sexo"] == "Masculino").astype(int)

    cantidad_validos = len(df)

    return df, cantidad_validos