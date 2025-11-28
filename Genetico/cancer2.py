import pandas as pd
import numpy as np
import pygad
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- PASO 1: GENERAR EL DATASET PROCESADO ---
print("Procesando datos crudos...")
df_raw = pd.read_csv('dataset/datos.csv')

# Guardar nombres de genes ANTES de transponer
gene_names = df_raw['ID'].values

# Transponer
df_transposed = df_raw.set_index('ID').transpose()

# Crear etiquetas
labels = []
for pid in df_transposed.index:
    if pid.startswith('N'):
        labels.append(0)  # Sano
    else:
        labels.append(1)  # Cancer

df_transposed['Target'] = labels

# --- PASO 2: PREPARAR DATOS PARA EL AG ---
print("Configurando Algoritmo Genético...")
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


# --- PASO 3: FUNCIÓN FITNESS ---
def fitness_func(ga_instance, solution, solution_idx):
    """Evalúa la combinación de genes seleccionados"""
    features_selected = np.where(solution == 1)[0]

    if len(features_selected) == 0:
        return 0

    X_reduced = X_full[:, features_selected]

    # Entrenar con todos los datos
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1  # Usar todos los 16 hilos
    )
    clf.fit(X_reduced, y_full)

    # Evaluar con todos los datos
    predictions = clf.predict(X_reduced)
    accuracy = accuracy_score(y_full, predictions)

    # Penalización por complejidad
    penalty = len(features_selected) / X_full.shape[1]
    fitness = accuracy - 0.05 * penalty

    return max(0, fitness)


def on_generation(ga_instance):
    """Monitorea el progreso"""
    generation = ga_instance.generations_completed
    solution, fitness, _ = ga_instance.best_solution()
    num_genes_selected = np.sum(solution == 1)

    print(f"Generación {generation:3d} | "
          f"Fitness: {fitness:.4f} | "
          f"Genes: {num_genes_selected:4d}")


# --- PASO 4: CONFIGURAR Y EJECUTAR ---
num_genes = X_full.shape[1]

print("="*70)
print("CONFIGURACIÓN DEL ALGORITMO GENÉTICO")
print("="*70)
print(f"Genes totales: {num_genes}")
print(f"Pacientes: {X_full.shape[0]}")
print(f"Generaciones: 100")
print(f"Población: 10 individuos")
print(f"CPU: 16 hilos (n_jobs=-1)")
print(f"Random Forest: 100 árboles, profundidad 2")
print("="*70)
print("\nINICIANDO EVOLUCIÓN...\n")

ga_instance = pygad.GA(
    num_generations=100,
    num_parents_mating=4,
    fitness_func=fitness_func,
    sol_per_pop=16,                 # Tamaño de población. Optimo = numero de hilos del cpu.
    num_genes=num_genes,
    init_range_low=0,
    init_range_high=2,
    gene_type=int,
    parent_selection_type="sss",
    keep_parents=1,
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=1,
    on_generation=on_generation,
    random_seed=42,
    #parallel_processing=["thread", 16] # Usar 16 hilos
)

ga_instance.run()

# --- PASO 5: RESULTADOS ---
solution, solution_fitness, _ = ga_instance.best_solution()
selected_genes_indices = np.where(solution == 1)[0]
selected_genes_names = gene_names[selected_genes_indices]

print("\n" + "=" * 70)
print("RESULTADOS FINALES")
print("=" * 70)
print(f"Fitness: {solution_fitness:.4f}")
print(f"Genes seleccionados: {len(selected_genes_names)}")

if len(selected_genes_names) > 0:
    print(f"Reducción: {(1 - len(selected_genes_names) / num_genes) * 100:.1f}%")

    # Validación con todos los datos
    X_final = X_full[:, selected_genes_indices]
    clf_final = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    clf_final.fit(X_final, y_full)
    pred_final = clf_final.predict(X_final)
    accuracy_final = accuracy_score(y_full, pred_final)

    print(f"\nPrecisión en TODOS los pacientes: {accuracy_final * 100:.2f}%")

    # ← NUEVO: Calcular importancia de cada gen
    importances = clf_final.feature_importances_
    gene_importance = list(zip(selected_genes_names, importances))
    gene_importance_sorted = sorted(gene_importance, key=lambda x: x[1], reverse=True)

    # ← MOSTRAR TODOS LOS GENES ORDENADOS POR IMPORTANCIA
    print(f"\n{'=' * 70}")
    print(f"TODOS LOS {len(selected_genes_names)} GENES SELECCIONADOS (ordenados por importancia):")
    print(f"{'=' * 70}")
    print(f"{'#':<5} {'Gen':<20} {'Importancia':<15}")
    print("-" * 70)

    for i, (gene, importance) in enumerate(gene_importance_sorted, 1):
        print(f"{i:<5} {gene:<20} {importance:.6f}")

    print("=" * 70)

    # ← GUARDAR EN ARCHIVO CSV
    results_df = pd.DataFrame(gene_importance_sorted, columns=['Gen', 'Importancia'])
    results_df['Ranking'] = range(1, len(results_df) + 1)
    results_df = results_df[['Ranking', 'Gen', 'Importancia']]
    results_df.to_csv('genes_seleccionados_cancer.csv', index=False)
    print(f"\n✅ Resultados guardados en: genes_seleccionados_cancer.csv")

else:
    print("\n⚠️  No se seleccionaron genes")

print("\n" + "=" * 70)
print("EVOLUCIÓN:")
print("=" * 70)
print(f"Mejor generación: {ga_instance.best_solution_generation}")
print(f"Fitness inicial: {ga_instance.best_solutions_fitness[0]:.4f}")
print(f"Fitness final: {ga_instance.best_solutions_fitness[-1]:.4f}")
print("=" * 70)

ga_instance.plot_fitness()
