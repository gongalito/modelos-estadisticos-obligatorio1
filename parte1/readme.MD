* **`main.py`**: Simula muestras de la variable aleatoria con densidad $f_X(x)=\frac{1}{2}(1+\theta x)$ en $[-1,1]$, suponiendo $\theta=-1/3$, mediante el método de la transformada inversa. Estima $\theta$ maximizando la log-verosimilitud con Golden search sobre $[-1,1]$. Para una muestra de tamaño 10 grafica la verosimilitud y guarda la figura `verosimilitud_n10.png`. Luego, para tamaños de muestra $n=10$, $100$ y $1000$, estima $\theta$ en 50 muestras independientes, imprime la tabla de sesgo, varianza y MSE, y guarda los histogramas de las estimaciones en `histogramas_solicitados.png`. Utiliza una semilla fija para garantizar la reproducibilidad de los resultados.

---

### Ejecución:

```bash
git clone https://github.com/gongalito/modelos-estadisticos-obligatorio1.git

cd modelos-estadisticos-obligatorio1

cd parte1

pip install -r requirements.txt

python main.py
```