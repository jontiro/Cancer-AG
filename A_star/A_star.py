import os
import time
from collections import deque
from queue import PriorityQueue

# Vatiables globales
matriz = []
solucion = []
x, y = 0, 0  # Posición inicial
nodos_visitados = 0
longitud_ruta = 0
costo_actual = 0
# Diccionario de direcciones
direcciones = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "O": (0, -1)
}

# Diccionario A*
direcciones_star = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "O": (0, -1),
    "NE": (-1, 1),
    "NO": (-1, -1),
    "SE": (1, 1),
    "SO": (1, -1)
}

# Colores para la salida
RED = "\033[31m"
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"


# Funciones

def encontrar_inicio():
    for i in range(len(matriz)):
        for j in range(len(matriz[0])):
            if matriz[i][j] == 2:  # Encontrar la posición de inicio
                return i, j
    return None  # Si no se encuentra el inicio

def encontrar_fin():
    for i in range(len(matriz)):
        for j in range(len(matriz[0])):
            if matriz[i][j] == 3:  # Encontrar la posición de fin
                return i, j
    return None  # Si no se encuentra el fin


def cambiar_laberinto(teclado):
    global matriz, x, y, a, b, visitados, solucion

    base_dir = os.path.dirname(__file__)
    filename = os.path.join(base_dir, "laberintos", f"{teclado}.txt")
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            traduccion = {'#': 1, '.': 0, 'S': 2, 'G': 3, ',': 5, '~': 6}
            matriz = [
                [traduccion[valor] for valor in line.strip()]
                for line in lines
            ]

        columnas = len(matriz[0]) if matriz else 0
        filas = len(matriz) if matriz else 0
        visitados = [[False for _ in range(columnas)] for _ in range(filas)]

        #   print(f"Laberinto cargado desde {filename}")
        #   for i in range(len(matriz)):
        #       print(" ".join(str(c) for c in matriz[i]))

        inicio = encontrar_inicio()
        fin = encontrar_fin()
        if inicio is not None and fin is not None:
            x, y = inicio
            a, b = fin

            print("Laberinto cargado correctamente")
            solucion = matriz
        else:
            print("No se encontró el punto de inicio 'S' en el laberinto")
            matriz = []
    except FileNotFoundError:
        print(f"Archivo no encontrado")


def costos(celda):
    if celda == 0:  # Camino normal
        return 1
    elif celda == 2:  # Inicio
        return 0
    elif celda == 3:  # Meta
        return 0
    elif celda == 5:  # Camino ,
        return 5
    elif celda == 6:  # Camino ~
        return 10


def imprimir_laberinto(matriz):
    for fila in matriz:
        for celda in fila:
            if celda == 0:
                print("." + RESET, end=" ")
            elif celda == 1:
                print(GREEN + "#" + RESET, end=" ")
            elif celda == 2:
                print(YELLOW + "S" + RESET, end=" ")
            elif celda == 3:
                print(RED + "G" + RESET, end=" ")
            elif celda == 4:
                print(MAGENTA + "." + RESET, end=" ")
            elif celda == 5:
                print("," + RESET, end=" ")
            elif celda == 6:
                print("~" + RESET, end=" ")
        print()


def imprimir_resultados(n):
    global nodos_visitados, longitud_ruta, costo_actual, inicio, fin
    print(YELLOW + f"Longitud de ruta: {longitud_ruta - n}" + RESET)
    print(CYAN + f"Nodos visitados: {nodos_visitados - 1}" + RESET)
    print(GREEN + f"Costo total: {costo_actual}" + RESET)
    print(MAGENTA + f"Tiempo: {(fin - inicio) * 1000:.3f} ms" + RESET)


# Busquedas - DFS, BFS, UCS

def dfs(x, y):
    global nodos_visitados, longitud_ruta, costo_actual

    if matriz[x][y] == 3:
        print("Solucion encontrada")
        imprimir_laberinto(solucion)
        return True

    visitados[x][y] = True
    if matriz[x][y] == 2:
        solucion[x][y] = 2
    else:
        solucion[x][y] = 4  # Marca el camino

    for direccion in direcciones:
        nx, ny = x + direcciones[direccion][0], y + direcciones[direccion][1]
        if 0 <= nx < len(matriz) and 0 <= ny < len(matriz[0]) and matriz[nx][ny] != 1 and not visitados[nx][ny]:
            nodos_visitados += 1
            longitud_ruta += 1
            costo_actual += costos(matriz[nx][ny])
            if dfs(nx, ny):
                return True  # Si se encontró solución, termina

    solucion[x][y] = 0  # Desmarca si no es parte del camino final
    costo_actual -= costos(matriz[x][y])
    longitud_ruta -= 1
    return False


def bfs(start_x, start_y):
    global nodos_visitados, longitud_ruta, costo_actual

    cola = deque()
    cola.append((start_x, start_y))
    visitados[start_x][start_y] = True
    parent = {}  # Mantener el rastro del camino
    parent[(start_x, start_y)] = None
    encontrado = False
    end_pos = None

    # Buscar el camino
    while cola and not encontrado:
        x, y = cola.popleft()

        if matriz[x][y] == 3:
            end_pos = (x, y)
            goal_x, goal_y = x, y
            encontrado = True
            break

        for direccion in direcciones:
            nx, ny = x + direcciones[direccion][0], y + direcciones[direccion][1]
            if (0 <= nx < len(matriz) and 0 <= ny < len(matriz[0]) and matriz[nx][ny] != 1 and not visitados[nx][ny]):
                nodos_visitados += 1
                visitados[nx][ny] = True
                parent[(nx, ny)] = (x, y)
                cola.append((nx, ny))

    # Reconstruir el camino si se encontró la meta
    if encontrado:
        print("Solución encontrada")
        path = []
        current = end_pos
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        longitud_ruta = len(path)
        for pos in path:
            costo_actual += costos(matriz[pos[0]][pos[1]])
            solucion[pos[0]][pos[1]] = 4
            if pos == (start_x, start_y):
                solucion[pos[0]][pos[1]] = 2
            if pos == (goal_x, goal_y):
                solucion[pos[0]][pos[1]] = 3

        imprimir_laberinto(solucion)

        return True
    else:
        print(RED + "No se encontró solución" + RESET)
        return False


def ucs(start_x, start_y):
    global nodos_visitados, longitud_ruta, costo_actual

    cola = PriorityQueue()
    cola.put((0, (start_x, start_y)))
    visitados[start_x][start_y] = True
    parent = {}
    parent[(start_x, start_y)] = None
    encontrado = False
    end_pos = None

    while not cola.empty() and not encontrado:
        costo_actual, (x, y) = cola.get()

        if matriz[x][y] == 3:
            end_pos = (x, y)
            encontrado = True
            break

        for direccion in direcciones:
            nx, ny = x + direcciones[direccion][0], y + direcciones[direccion][1]
            if (0 <= nx < len(matriz) and 0 <= ny < len(matriz[0]) and matriz[nx][ny] != 1 and not visitados[nx][ny]):
                nodos_visitados += 1
                visitados[nx][ny] = True
                parent[(nx, ny)] = (x, y)
                nuevo_costo = costo_actual + costos(matriz[nx][ny])
                cola.put((nuevo_costo, (nx, ny)))

    if encontrado:
        print("Solución encontrada")
        path = []
        current = end_pos
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        longitud_ruta = len(path)
        for pos in path:
            solucion[pos[0]][pos[1]] = 4
            if pos == (start_x, start_y):
                solucion[pos[0]][pos[1]] = 2
            if pos == end_pos:
                solucion[pos[0]][pos[1]] = 3

        imprimir_laberinto(solucion)

        return True

def heuristica_man(x, y, goal_x, goal_y):
    # Distancia Manhattan
    return abs(x - goal_x) + abs(y - goal_y)

def a_star_man(start_x, start_y, goal_x, goal_y):
    global nodos_visitados, longitud_ruta, costo_actual

    cola = PriorityQueue()
    cola.put((0, (start_x, start_y)))
    parent = {}
    parent[(start_x, start_y)] = None
    g_cost = {(start_x, start_y): 0}
    visitados[start_x][start_y] = True
    encontrado = False
    end_pos = None

    while not cola.empty():
        _, (x, y) = cola.get()

        if matriz[x][y] == 3:
            end_pos = (x, y)
            encontrado = True
            break

        for direccion in direcciones:
            nx, ny = x + direcciones_star[direccion][0], y + direcciones_star[direccion][1]
            if (0 <= nx < len(matriz) and 0 <= ny < len(matriz[0]) and matriz[nx][ny] != 1):
                nuevo_g = g_cost[(x, y)] + costos(matriz[nx][ny])
                if not visitados[nx][ny] or nuevo_g < g_cost.get((nx, ny), float('inf')):
                    nodos_visitados += 1
                    g_cost[(nx, ny)] = nuevo_g # g calcula el costo desde el inicio hasta el nodo actual.
                    f = nuevo_g + heuristica_man(nx, ny, goal_x, goal_y)  # Aplica heuristica. donde h es la heurística que estima el costo desde el nodo actual hasta la meta.
                    cola.put((f, (nx, ny)))
                    parent[(nx, ny)] = (x, y)
                    visitados[nx][ny] = True

    if encontrado:
        print("Solución encontrada")
        path = []
        current = end_pos
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        longitud_ruta = len(path)
        for pos in path:
            costo_actual += costos(matriz[pos[0]][pos[1]])
            solucion[pos[0]][pos[1]] = 4
            if pos == (start_x, start_y):
                solucion[pos[0]][pos[1]] = 2
            if pos == (goal_x, goal_y):
                solucion[pos[0]][pos[1]] = 3

        imprimir_laberinto(solucion)
        return True
    else:
        print(RED + "No se encontró solución" + RESET)
        return False

def heuristica_euc(x, y, goal_x, goal_y):
    # Distancia Euclidiana
    return ((x - goal_x) ** 2 + (y - goal_y) ** 2) ** 0.5

def a_star_euc(start_x, start_y, goal_x, goal_y):
    global nodos_visitados, longitud_ruta, costo_actual

    cola = PriorityQueue()
    cola.put((0, (start_x, start_y)))
    parent = {}
    parent[(start_x, start_y)] = None
    g_cost = {(start_x, start_y): 0}
    visitados[start_x][start_y] = True
    encontrado = False
    end_pos = None

    while not cola.empty():
        _, (x, y) = cola.get()

        if matriz[x][y] == 3:
            end_pos = (x, y)
            encontrado = True
            break

        for direccion in direcciones_star:
            nx, ny = x + direcciones_star[direccion][0], y + direcciones_star[direccion][1]
            if (0 <= nx < len(matriz) and 0 <= ny < len(matriz[0]) and matriz[nx][ny] != 1):
                nuevo_g = g_cost[(x, y)] + costos(matriz[nx][ny])
                if not visitados[nx][ny] or nuevo_g < g_cost.get((nx, ny), float('inf')):
                    nodos_visitados += 1
                    g_cost[(nx, ny)] = nuevo_g
                    f = nuevo_g + heuristica_euc(nx, ny, goal_x, goal_y)  # Aplica heuristica
                    cola.put((f, (nx, ny)))
                    parent[(nx, ny)] = (x, y)
                    visitados[nx][ny] = True

    if encontrado:
        print("Solución encontrada")
        path = []
        current = end_pos
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        longitud_ruta = len(path)
        for pos in path:
            costo_actual += costos(matriz[pos[0]][pos[1]])
            solucion[pos[0]][pos[1]] = 4
            if pos == (start_x, start_y):
                solucion[pos[0]][pos[1]] = 2
            if pos == (goal_x, goal_y):
                solucion[pos[0]][pos[1]] = 3

        imprimir_laberinto(solucion)
        return True
    else:
        print(RED + "No se encontró solución" + RESET)
        return False


# Menu principal
bandera = True
laberinto = "default"
while bandera:
    cambiar_laberinto(laberinto)  # Reiniciar el laberinto
    nodos_visitados = 0
    longitud_ruta = 0
    costo_actual = 0
    print(YELLOW + "=== Menu principal ===" + RESET)
    print("1- Resolver con DFS")
    print("2- Resolver con BFS")
    print("3- Resolver con UCS")
    print("4- Resolver con A* (Manhattan)")
    print("5- Resolver con A* (Euclidiana)")
    print("6- Cambiar laberinto")
    print("7- Salir")

    opcion_menu = input(BLUE + "Ingrese una opcion: " + RESET)

    while opcion_menu not in ["1", "2", "3", "4", "5", "6", "7"]:
        print(RED + "Opcion invalida" + RESET)
        opcion_menu = input("Ingrese una opcion: ")

    if opcion_menu == "1":
        inicio = time.perf_counter()
        if dfs(x, y):
            fin = time.perf_counter()

            print("== DFS ==")
            imprimir_resultados(1)
        else:
            print(RED + "No se encontro solucion" + RESET)

    elif opcion_menu == "2":
        inicio = time.perf_counter()
        if bfs(x, y):
            fin = time.perf_counter()

            print("==BFS ==")
            imprimir_resultados(2)
        else:
            print(RED + "No se encontro solucion" + RESET)
    elif opcion_menu == "3":
        inicio = time.perf_counter()
        if ucs(x, y):
            fin = time.perf_counter()

            print("== UCS ==")
            imprimir_resultados(2)
        else:
            print(RED + "No se encontro solucion" + RESET)

    elif opcion_menu == "4":
        inicio = time.perf_counter()
        if a_star_man(x, y, a, b):
            fin = time.perf_counter()

            print("== A* (Manhattan) ==")
            imprimir_resultados(2)
        else:
            print(RED + "No se encontro solucion" + RESET)

    elif opcion_menu == "5":
        inicio = time.perf_counter()
        if a_star_euc(x, y, a, b):
            fin = time.perf_counter()

            print("== A* (Euclidiana) ==")
            imprimir_resultados(2)
        else:
            print(RED + "No se encontro solucion" + RESET)

    elif opcion_menu == "6":
        teclado = input(BLUE + "Ingrese el nombre del archivo del laberinto (sin .txt): " + RESET)
        laberinto = teclado
        cambiar_laberinto(laberinto)

    elif opcion_menu == "7":
        bandera = False
        print(GREEN + "Saliendo..." + RESET)
