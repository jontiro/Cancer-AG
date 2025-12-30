#Trabajarás con un laberinto con celdas de distinto costo:
#•. → camino libre (costo = 1)
#•, → terreno difícil (costo = 5)
#•~ → agua/pantano (costo = 10)
#•# → pared (inaccesible)
#•S → inicio
#•G → meta
import os
import time
import heapq

# Vatiables globales
matriz = []
solucion = []
x, y = 0, 0  # Posición inicial
nodos_visitados = 0
longitud_ruta = 0
# Diccionario de direcciones
direcciones = {
    "N": (-1, 0),
    "S": (1, 0),
    "E": (0, 1),
    "O": (0, -1)
}

# Colores para la salida
RED = "\033[31m"
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"


# Funciones

def encontrar_inicio():
    for i in range(len(matriz)):
        for j in range(len(matriz[0])):
            if matriz[i][j] == 2:  # Encontrar la posición de inicio
                return i, j
    return None  # Si no se encuentra el inicio


def cambiar_laberinto(teclado):
    global matriz, x, y, visitados, solucion

    base_dir = os.path.dirname(__file__)
    filename = os.path.join(base_dir, "laberintos", f"{teclado}.txt")
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            traduccion = {'#': 1, '.': 0, 'S': 2, 'G': 3}
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
        if inicio is not None:
            x, y = inicio
            print("Laberinto cargado correctamente")
            solucion = matriz
        else:
            print("No se encontró el punto de inicio 'S' en el laberinto")
            matriz = []
    except FileNotFoundError:
        print(f"Archivo no encontrado")


def imprimir_laberinto(matriz):
    for fila in matriz:
        for celda in fila:
            if celda == 0:
                print("." + RESET, end=" ")
            elif celda == 1:
                print(GREEN + "#" + RESET, end=" ")
            elif celda == 2:
                print(BLUE + "S" + RESET, end=" ")
            elif celda == 3:
                print(YELLOW + "G" + RESET, end=" ")
            elif celda == 4:
                print(RED + "." + RESET, end=" ")
        print()


# Busquedas

def dfs(x, y):
    global nodos_visitados, longitud_ruta

    if matriz[x][y] == 3:
        print("Solucion encontrada")
        imprimir_laberinto(solucion)
        return True

    visitados[x][y] = True
    solucion[x][y] = 4  # Marca el camino

    for direccion in direcciones:
        nx, ny = x + direcciones[direccion][0], y + direcciones[direccion][1]
        if 0 <= nx < len(matriz) and 0 <= ny < len(matriz[0]) and matriz[nx][ny] != 1 and not visitados[nx][ny]:
            nodos_visitados += 1
            longitud_ruta += 1
            if dfs(nx, ny):
                return True  # Si se encontró solución, termina

    solucion[x][y] = 0  # Desmarca si no es parte del camino final
    longitud_ruta -= 1
    return False


from collections import deque


def bfs(start_x, start_y):
    global nodos_visitados, longitud_ruta

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
            solucion[pos[0]][pos[1]] = 4
            if pos == (goal_x, goal_y):
                solucion[pos[0]][pos[1]] = 3

        for i in range(len(solucion)):
            for j in range(len(solucion[0])):
                if solucion[i][j] == 0:
                    print("." + RESET, end=" ")
                elif solucion[i][j] == 1:
                    print("#" + RESET, end=" ")
                elif solucion[i][j] == 2:
                    print(YELLOW + "S" + RESET, end=" ")
                elif solucion[i][j] == 3:
                    print(RED + "G" + RESET, end=" ")
                elif solucion[i][j] == 4:
                    print(GREEN + "." + RESET, end=" ")
            print()

        return True
    else:
        print("No se encontró solución")
        return False


def ucs(x, y):
    global matriz, solucion, nodos_visitados, longitud_ruta
    # Costos según símbolo
    simbolo_costo = {0: 1, ',': 5, '~': 10, 1: float('inf'), 2: 1, 3: 1}
    filas, columnas = len(matriz), len(matriz[0])
    visitados = [[False for _ in range(columnas)] for _ in range(filas)]
    parent = {}
    costo_min = {}
    pq = []
    # Buscar posición de meta
    goal = None
    for i in range(filas):
        for j in range(columnas):
            if matriz[i][j] == 3:
                goal = (i, j)
    heapq.heappush(pq, (0, (x, y)))
    costo_min[(x, y)] = 0
    parent[(x, y)] = None
    while pq:
        costo_actual, (cx, cy) = heapq.heappop(pq)
        if visitados[cx][cy]:
            continue
        visitados[cx][cy] = True
        nodos_visitados += 1
        if (cx, cy) == goal:
            # Reconstruir camino
            path = []
            current = (cx, cy)
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            longitud_ruta = len(path)
            for pos in path:
                solucion[pos[0]][pos[1]] = 4
                if pos == goal:
                    solucion[pos[0]][pos[1]] = 3
            print(YELLOW + f"Nodos visitados: {nodos_visitados}" + RESET)
            print(YELLOW + f"Longitud de la ruta: {longitud_ruta}" + RESET)
            for i in range(len(solucion)):
                for j in range(len(solucion[0])):
                    if solucion[i][j] == 0:
                        print("." + RESET, end=" ")
                    elif solucion[i][j] == 1:
                        print("#" + RESET, end=" ")
                    elif solucion[i][j] == 2:
                        print(YELLOW + "S" + RESET, end=" ")
                    elif solucion[i][j] == 3:
                        print(RED + "G" + RESET, end=" ")
                    elif solucion[i][j] == 4:
                        print(GREEN + "." + RESET, end=" ")
                print()
            return True
        # Explorar vecinos
        for dx, dy in direcciones.values():
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < filas and 0 <= ny < columnas:
                simbolo = matriz[nx][ny]
                # Si es pared, no se puede pasar
                if simbolo == 1:
                    continue
                # Si ya fue visitado, ignorar
                if visitados[nx][ny]:
                    continue
                # Calcular costo
                costo_vecino = simbolo_costo.get(simbolo, 1)
                nuevo_costo = costo_actual + costo_vecino
                if (nx, ny) not in costo_min or nuevo_costo < costo_min[(nx, ny)]:
                    costo_min[(nx, ny)] = nuevo_costo
                    parent[(nx, ny)] = (cx, cy)
                    heapq.heappush(pq, (nuevo_costo, (nx, ny)))
    print("No se encontró solución")
    return False


# Menu principal
bandera = True
laberinto = "default"
while bandera:
    cambiar_laberinto(laberinto)  # Reiniciar el laberinto
    nodos_visitados = 0
    longitud_ruta = 0

    print(YELLOW + "=== Menu principal ===" + RESET)
    print("1- Resolver con DFS")
    print("2- Resolver con BFS")
    print("3- Resolver con UCS")
    print("4- Cambiar laberinto")
    print("5- Salir")

    opcion_menu = input(BLUE + "Ingrese una opcion: " + RESET)

    while opcion_menu not in ["1", "2", "3", "4"]:
        print(RED + "Opcion invalida" + RESET)
        opcion_menu = input("Ingrese una opcion: ")

    if opcion_menu == "1":
        inicio = time.perf_counter()
        dfs(x, y)
        fin = time.perf_counter()

        print("== DFS ==")
        print(f"Longitud de ruta: {longitud_ruta - 1}")
        print(f"Nodos visitados: {nodos_visitados - 1}")
        print(f"Tiempo: {(fin - inicio) * 1000:.3f} ms")

    elif opcion_menu == "2":
        inicio = time.perf_counter()
        bfs(x, y)
        fin = time.perf_counter()

        print("==BFS ==")
        print(f"Longitud de ruta: {longitud_ruta - 2}")
        print(f"Nodos visitados: {nodos_visitados - 1}")
        print(f"Tiempo: {(fin - inicio) * 1000:.3f} ms")

    elif opcion_menu == "3":
        teclado = input(BLUE + "Ingrese el nombre del archivo del laberinto (sin .txt): " + RESET)
        laberinto = teclado
        cambiar_laberinto(laberinto)

    elif opcion_menu == "4":
        bandera = False
        print(GREEN + "Saliendo..." + RESET)
