import random
import numpy as np

# Patrones de aprendizaje y objetivos

P = [[0, 0, 1, 1],  # Entrada 1
     [0, 1, 0, 1]]  # Entrada 2

T = [[-1, 1, 1, -1]]  # Salida esperada (XOR)
Q = len[P[0]]  # Número de patrones

n1 = 30;  # Número de neuronas en la capa oculta
ep = 1 # Ventana de valores iniciales

# Valores iniciales de pesos y umbrales

W1 = ep * (2 * np.random.rand(n1, 2) - 1)  # Pesos capa oculta
b1 = ep * (2 * np.random.rand(n1, 1) - 1)  # Umbrales capa oculta
W2 = ep * (2 * np.random.rand(1, n1) - 1)  # Pesos capa salida
b2 = ep * (2 * np.random.rand() - 1)  # Umbral capa salida
alpha = 0.01  # Tasa de aprendizaje

for i in range(1, 10000):
    for j in range(1, Q):
        a1 = np.tanh(np.dot(W1, np.array([P[0][j], P[1][j]])) + b1)
        a2[j] = np.tanh(np.dot(W2, a1) + b2)
        e = T[0][j] - a2[j]
        s2 = -2 * (1 - a2[j] ** 2) * e
        s1 = (1 - a1 ** 2) * np.dot(W2.T, s2)

        W2 = W2 - alpha * s2 * a1.T
        b2 = b2 - alpha * s2
        W1 = W1 - alpha * np.dot(s1, np.array([[P[0][j]], [P[1][j]]]).T
        b1 = b1 - alpha * s1
        sm = sum(e ** 2)
    e_medio [i] = sm / Q






