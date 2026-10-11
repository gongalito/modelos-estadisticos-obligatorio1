
* **`main.py`**: Ejecuta la estimación del modelo lineal y genera la gráfica con la relación estimada. Utiliza una semilla aleatoria (*seed*) para garantizar la reproducibilidad de los resultados.

* **`convergencia_RMSE.py`**: Corre una simulación repetida de Validación Cruzada para estimar el valor asintótico del $\text{RMSE}$ promedio. Su objetivo es obtener una métrica más robusta y realista del error del modelo. Por defecto ejecuta hasta $50.000$ repeticiones (configurable en el código). Al ejecutarse sobre CPU, el tiempo de cómputo para corridas grandes puede ser elevado.

---

### Ejecucion:

```bash
git clone https://github.com/gongalito/modelo-lineal-estatura.git

cd modelo-lineal-estatura

pip install -r requirements.txt

python main.py

python convergencia_RMSE.py
```
