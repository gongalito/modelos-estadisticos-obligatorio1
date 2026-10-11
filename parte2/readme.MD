
* **`main.py`**: Carga el conjunto de datos Boston Housing, hace el análisis exploratorio y ajusta el modelo lineal, los modelos polinomiales hasta grado 3 y Ridge sobre las características polinomiales (con $\lambda$ elegido por validación cruzada de 5 particiones sobre el conjunto de entrenamiento). Imprime las tablas de resultados y guarda las figuras `exploratorio.png` y `figura_problema2.png` (la del informe). Al final repite todo el procedimiento con 30 particiones distintas para verificar que las conclusiones no dependen de la partición; esa parte tarda un par de minutos y se puede omitir con `--rapido`. Utiliza semillas fijas para garantizar la reproducibilidad de los resultados.


Los datos se descargan de la fuente original (http://lib.stat.cmu.edu/datasets/boston), ya que `scikit-learn` eliminó `load_boston` a partir de la versión 1.2.

---

### Ejecucion:

```bash
git clone https://github.com/gongalito/modelos-estadisticos-obligatorio1.git

cd modelos-estadisticos-obligatorio1

cd parte2

pip install -r requirements.txt

python main.py
```
