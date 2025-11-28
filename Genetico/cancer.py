import pandas as pd
import numpy as np
import pygad
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- PASO 1: GENERAR EL DATASET PROCESADO ---
print("Procesando datos crudos...")
# Cargar datos originales
df_raw = pd.read_csv('dataset/datos.csv')

# Guardar nombres de genes ANTES de transponer (FIX)
gene_names = df_raw['ID'].values

# Transponer
df_transposed = df_raw.set_index('ID').transpose()

# Crear etiquetas
labels = []
for pid in df_transposed.index:
    if pid.startswith('N'):
        labels.append(0)  # Sano
    else:
        labels.append(1)  # Cancer (Cualquier tipo para simplificar a Binario)

df_transposed['Target'] = labels

# --- PASO 2: PREPARAR DATOS PARA EL AG ---
print("Configurando Algoritmo Genético...")
# Separar X (Genes) y y (Diagnóstico)
X = df_transposed.drop(columns=['Target']).values
y = df_transposed['Target'].values

# Escalar datos
scaler = StandardScaler()
X_full = scaler.fit_transform(X)
y_full = y

print(f"Dataset preparado:")
print(f"  - Total de genes: {X_full.shape[1]}")
print(f"  - Total de pacientes: {X_full.shape[0]}")
print(f"  - Distribución: {np.bincount(y_full)} (sanos, cáncer)\n")

# --- PASO 3: DEFINIR LA INTELIGENCIA DEL AG ---
# Problema: Selección de Características.
# El cromosoma será una lista de 0s y 1s.
# 1 = Usar este Gen para diagnosticar.
# 0 = Ignorar este Gen.

def fitness_func(ga_instance, solution, solution_idx):
    """
    Función de fitness que evalúa qué tan buenos son los genes seleccionados.
    
    Parámetros:
    - solution: Cromosoma (array de 0s y 1s)
    - 1 = Usar este gen, 0 = Ignorar este gen
    
    Retorna:
    - fitness: Precisión del modelo - penalización por complejidad
    """
    # Seleccionar solo las columnas donde el gen es 1
    features_selected = np.where(solution == 1)[0]

    if len(features_selected) == 0:
        return 0  # Si no selecciona nada, fitness es 0

    # Reducir el dataset a esos genes
    X_train_reduced = X_train[:, features_selected]
    X_test_reduced = X_test[:, features_selected]

    # Entrenar un modelo rápido (Random Forest)
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=2,
        random_state=42,
        n_jobs = -1         # Usar todos los núcleos disponibles
    )
    clf.fit(X_train_reduced, y_train)

    # Predecir y evaluar
    predictions = clf.predict(X_test_reduced)
    accuracy = accuracy_score(y_test, predictions)

    # Penalización por complejidad (favorece menos genes)
    penalty = len(features_selected) / X.shape[1]
    fitness = accuracy - 0.05 * penalty

    return max(0, fitness)


# Callback para monitorear la evolución generación por generación
def on_generation(ga_instance):
    """
    Se ejecuta después de cada generación para mostrar el progreso.
    """
    generation = ga_instance.generations_completed
    solution, fitness, _ = ga_instance.best_solution()
    num_genes_selected = np.sum(solution == 1)
    
    print(f"Generación {generation:3d} | "
          f"Mejor Fitness: {fitness:.4f} | "
          f"Genes Seleccionados: {num_genes_selected:4d} | "
          f"Precisión Estimada: {fitness*100:.2f}%")


# --- PASO 4: CONFIGURAR Y CORRER ---
num_genes = X.shape[1]  # Total de genes

print("="*70)
print("CONFIGURACIÓN DEL ALGORITMO GENÉTICO")
print("="*70)
print(f"Espacio de búsqueda: {num_genes} genes")
print(f"Generaciones: 20")
print(f"Tamaño de población: 10 individuos")
print(f"Padres por cruzamiento: 4")
print(f"Padres conservados: 1")
print(f"Tipo de selección: Steady State Selection (SSS)")
print(f"Tipo de cruzamiento: Single Point")
print(f"Tipo de mutación: Random")
print(f"Porcentaje de mutación: 1%")
print(f"Modelo evaluador: Random Forest (10 árboles, profundidad=2)")
print("\nCRITERIOS DE EVALUACIÓN:")
print("  - Precisión del modelo (accuracy)")
print("  - Penalización por complejidad (favorece menos genes)")
print("  - Fitness = Accuracy - 0.05 * (genes_seleccionados / total_genes)")
print("\nCÓMO FUNCIONA LA EVOLUCIÓN:")
print("  1. Población inicial: 10 soluciones aleatorias (cromosomas)")
print("  2. Evaluación: Cada cromosoma se evalúa con la función fitness")
print("  3. Selección: Se eligen los 4 mejores padres (SSS)")
print("  4. Cruzamiento: Los padres se cruzan en un punto aleatorio")
print("  5. Mutación: 1% de los genes mutan aleatoriamente (0 -> 1 o 1 -> 0)")
print("  6. Nueva generación: Se repite el proceso 20 veces")
print("="*70)
print("\nINICIANDO EVOLUCIÓN...\n")

ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=4,
                       fitness_func=fitness_func,
                       sol_per_pop=10,
                       num_genes=num_genes,
                       init_range_low=0,
                       init_range_high=2,  # Genera valores en [0, 2), que con gene_type=int da 0 o 1
                       gene_type=int,
                       parent_selection_type="sss",
                       keep_parents=1,
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=1,
                       on_generation=on_generation,
                       random_seed=42)

ga_instance.run()

# --- PASO 5: RESULTADOS ---
solution, solution_fitness, _ = ga_instance.best_solution()
selected_genes_indices = np.where(solution == 1)[0]
selected_genes_names = gene_names[selected_genes_indices]  # FIX: Usar gene_names guardado

print("\n" + "="*70)
print("RESULTADOS FINALES")
print("="*70)
print(f"Fitness Final: {solution_fitness:.4f}")
print(f"Número de Genes seleccionados: {len(selected_genes_names)}")

# Validar que se hayan seleccionado genes
if len(selected_genes_names) == 0:
    print("\n⚠️  ADVERTENCIA: No se seleccionó ningún gen!")
    print("Esto puede deberse a:")
    print("  - Penalización muy alta")
    print("  - Pocas generaciones")
    print("  - Población inicial sin genes activos")
    print("\nPrueba aumentar: num_generations, sol_per_pop o reducir la penalización")
else:
    print(f"Reducción de dimensionalidad: {(1 - len(selected_genes_names)/num_genes)*100:.1f}%")

    # Validar con el conjunto de prueba
    X_test_final = X_test[:, selected_genes_indices]
    clf_final = RandomForestClassifier(
        n_estimators=100,
        max_depth=2,
        random_state=42,
        n_jobs=-1
    )
    clf_final.fit(X_train[:, selected_genes_indices], y_train)
    pred_final = clf_final.predict(X_test_final)
    accuracy_final = accuracy_score(y_test, pred_final)

    print(f"\nVALIDACIÓN EN CONJUNTO DE PRUEBA:")
    print(f"Precisión real: {accuracy_final * 100:.2f}%")

    print(f"\nPRIMEROS 20 GENES CLAVE ENCONTRADOS:")
    for i, gene in enumerate(selected_genes_names[:20], 1):
        print(f"  {i:2d}. {gene}")

    if len(selected_genes_names) > 20:
        print(f"  ... y {len(selected_genes_names) - 20} genes más")

print("\n" + "="*70)
print("ANÁLISIS DE EVOLUCIÓN:")
print("="*70)
print(f"Convergencia alcanzada en generación: {ga_instance.best_solution_generation}")
print(f"Fitness inicial promedio vs final:")
print(f"  - Inicial: {ga_instance.best_solutions_fitness[0]:.4f}")
print(f"  - Final: {ga_instance.best_solutions_fitness[-1]:.4f}")
if ga_instance.best_solutions_fitness[0] > 0:
    mejora = ((ga_instance.best_solutions_fitness[-1] - ga_instance.best_solutions_fitness[0]) / ga_instance.best_solutions_fitness[0] * 100)
    print(f"  - Mejora: {mejora:.2f}%")
else:
    print(f"  - Mejora: N/A (fitness inicial era 0)")
print("="*70)

ga_instance.plot_fitness()

