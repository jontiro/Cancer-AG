import numpy as np
import matplotlib.pyplot as plt

# Patrones de aprendizaje y objetivos
P = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
T = np.array([-1, 1, 1, -1])
Q = P.shape[1]
n1 = 30
ep = 1

# Inicialización de pesos y umbrales
W1 = ep * (2 * np.random.rand(n1, 2) - 1)
b1 = ep * (2 * np.random.rand(n1, 1) - 1)
W2 = ep * (2 * np.random.rand(1, n1) - 1)
b2 = ep * (2 * np.random.rand() - 1)
alfa = 0.01

emedio = np.zeros(10000)

def tansig(x):
    return np.tanh(x)

for Epocas in range(10000):
    suma = 0
    for q in range(Q):
        entrada = P[:, q].reshape(-1, 1)
        a1 = tansig(np.dot(W1, entrada) + b1)
        a2 = tansig(np.dot(W2, a1) + b2)
        e = T[q] - a2
        s2 = -2 * (1 - a2 ** 2) * e
        s1 = (1 - a1 ** 2) * np.dot(W2.T, s2)
        W2 = W2 - alfa * s2 * a1.T
        b2 = b2 - alfa * s2
        W1 = W1 - alfa * np.dot(s1, entrada.T)
        b1 = b1 - alfa * s1
        suma += e ** 2
    emedio[Epocas] = suma / Q

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(emedio)
plt.title('Error cuadrático medio')

# Verificación de la respuesta de la multicapa
a = np.zeros(Q)
for q in range(Q):
    entrada = P[:, q].reshape(-1, 1)
    a1 = tansig(np.dot(W1, entrada) + b1)
    a[q] = tansig(np.dot(W2, a1) + b2)
print(a)

# Frontera de decisión
u = np.linspace(-2, 2, 100)
v = np.linspace(-2, 2, 100)
z = np.zeros((len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        entrada = np.array([[u[i]], [v[j]]])
        a1 = tansig(np.dot(W1, entrada) + b1)
        z[i, j] = tansig(np.dot(W2, a1) + b2)

plt.subplot(1, 2, 2)
plt.contour(u, v, z.T, levels=[-0.9, 0, 0.9], linewidths=2)
plt.axis([-0.5, 1.5, -0.5, 1.5])
plt.plot(P[0, [0, 3]], P[1, [0, 3]], 'ro')
plt.plot(P[0, [1, 2]], P[1, [1, 2]], 'bo')
plt.title('Frontera de decisión')
plt.show()