import random
import string

# --- CONFIGURACIÓN ---
OBJETIVO = "INTELIGENCIA ARTIFICIAL"
POBLACION_TAMANO = 100  # Número de individuos
MUTACION_PROB = 0.1  # 10% de probabilidad de mutar
GENES = string.ascii_uppercase + " "  # Letras posibles (A-Z y espacio)


# --- 1. DEFINICIÓN DEL INDIVIDUO ---
class Individuo:
    def __init__(self, cromosoma):
        self.cromosoma = cromosoma
        self.fitness = self.calcular_fitness()

    @classmethod
    def crear_aleatorio(cls):
        # Crea una cadena de letras al azar del mismo largo que el objetivo
        genes_aleatorios = [random.choice(GENES) for _ in range(len(OBJETIVO))]
        return cls(genes_aleatorios)

    def calcular_fitness(self):
        # Cuenta cuántas letras coinciden exactamente con el objetivo
        score = 0
        for i in range(len(self.cromosoma)):
            if self.cromosoma[i] == OBJETIVO[i]:
                score += 1
        return score


# --- 2. FUNCIONES DEL ALGORITMO ---

def seleccion(poblacion):
    # TORNEO: Tomamos 3 al azar y devolvemos el mejor
    aspirantes = random.sample(poblacion, 3)
    aspirantes.sort(key=lambda x: x.fitness, reverse=True)
    return aspirantes[0]


def cruce(padre1, padre2):
    # PUNTO DE CORTE: Cortamos los genes y los mezclamos
    punto_corte = random.randint(1, len(OBJETIVO) - 1)

    genes_hijo = padre1.cromosoma[:punto_corte] + padre2.cromosoma[punto_corte:]
    return Individuo(genes_hijo)


def mutacion(individuo):
    # Cambiamos un gen al azar por otro nuevo
    genes = individuo.cromosoma[:]
    for i in range(len(genes)):
        if random.random() < MUTACION_PROB:
            genes[i] = random.choice(GENES)
    return Individuo(genes)


# --- 3. BUCLE PRINCIPAL (MAIN) ---

def main():
    generation = 1
    found = False

    # Crear población inicial
    poblacion = [Individuo.crear_aleatorio() for _ in range(POBLACION_TAMANO)]

    print(f"Objetivo: {OBJETIVO}\n")

    while not found:
        # Ordenar por fitness (el mejor primero)
        poblacion.sort(key=lambda x: x.fitness, reverse=True)

        # El mejor de la generación actual
        mejor = poblacion[0]
        print(f"Gen {generation}: {''.join(mejor.cromosoma)} (Fitness: {mejor.fitness})")

        # Si el fitness es igual al largo del objetivo, ganamos
        if mejor.fitness == len(OBJETIVO):
            found = True
            break

        # --- NUEVA GENERACIÓN ---
        nueva_generacion = []

        # Elitismo: Pasamos al 10% de los mejores directamente a la siguiente ronda
        # Esto asegura que nunca perdamos la mejor solución encontrada
        s = int((10 * POBLACION_TAMANO) / 100)
        nueva_generacion.extend(poblacion[:s])

        # El resto (90%) se crea por cruce y mutación
        s = int((90 * POBLACION_TAMANO) / 100)
        for _ in range(s):
            padre1 = seleccion(poblacion)
            padre2 = seleccion(poblacion)
            hijo = cruce(padre1, padre2)
            hijo = mutacion(hijo)
            nueva_generacion.append(hijo)

        poblacion = nueva_generacion
        generation += 1

    print(f"\n¡Éxito! Frase encontrada en la generación {generation}")


if __name__ == '__main__':
    main()