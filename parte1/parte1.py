import numpy as np
import matplotlib
import matplotlib.pyplot as plt

THETA = -1/3
rng = np.random.default_rng(1234)

def simular(n, theta=THETA, rng=rng):
    u = rng.uniform(size=n)
    if abs(theta) < 1e-12:
        return 2*u - 1
    return (-1 + np.sqrt((1-theta)**2 + 4*theta*u)) / theta

def log_verosimilitud(theta, x):
    return np.sum(np.log(1 + theta*x)) - len(x)*np.log(2)

def golden_search(f, a, b, tol=1e-6):
    phi = (np.sqrt(5) - 1) / 2
    c, d = b - phi*(b-a), a + phi*(b-a)
    while b - a > tol:
        if f(c) > f(d): b, d = d, c; c = b - phi*(b-a)
        else:           a, c = c, d; d = a + phi*(b-a)
    return (a+b)/2

def emv(x):
    return golden_search(lambda t: log_verosimilitud(t, x), -1, 1)

# Parte 2
x10 = simular(10)
grid = np.linspace(-0.999, 0.999, 500)
L = [np.exp(log_verosimilitud(t, x10)) for t in grid]
th = emv(x10)
plt.figure(figsize=(6,4))
plt.plot(grid, L); plt.axvline(th, ls="--", c="r", label=f"EMV = {th:.3f}")
plt.axvline(THETA, ls=":", c="k", label="θ real = -1/3")
plt.xlabel("θ"); plt.ylabel("L(θ)"); plt.legend(); plt.title("Verosimilitud, n = 10")
plt.tight_layout(); plt.savefig("verosimilitud_n10.png", dpi=120)
print("EMV muestra n=10:", th)

# Partes 3 y 4
def experimento(n, M):
    est = np.array([emv(simular(n)) for _ in range(M)])
    sesgo = est.mean() - THETA
    var = est.var()
    mse = np.mean((est - THETA)**2)
    return est, sesgo, var, mse

# --- Ejecución ---
M = 50
res = {}

print(f"Resultados para M = {M} réplicas:")
print(f"{'n':>6} {'Sesgo':>10} {'Varianza':>10} {'MSE':>10}")

for n in (10, 100, 1000):
    est, s, v, m = experimento(n, M)
    res[n] = est
    print(f"{n:>6} {s:>10.4f} {v:>10.4f} {m:>10.4f}")

# Graficar los histogramas para cada tamaño de muestra n
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
for a, n in zip(ax, (10, 100, 1000)):
    a.hist(res[n], bins=15, color="skyblue", edgecolor="black")
    a.axvline(
        THETA, color="r", linestyle="--", label=f"θ real = {THETA:.3f}"
    )
    a.set_title(f"n = {n}")
    a.set_xlim(-1.05, 1.05)
    a.set_xlabel("θ estimado")
    a.legend()

plt.tight_layout()
plt.savefig("histogramas_solicitados.png", dpi=120)
plt.show()