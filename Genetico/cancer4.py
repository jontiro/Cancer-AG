import pandas as pd
import numpy as np
import pygad
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

# --- VARIABLES GLOBALES PARA MULTIPROCESAMIENTO ---
# Es necesario definirlas fuera para que los procesos hijos las vean
X_full = None
y_full = None


def fitness_func(ga_instance, solution, solution_idx):
    """
    Esta función ahora correrá en paralelo en diferentes núcleos.
    """
    features_selected = np.where(solution == 1)[0]

    # Penalización fuerte si no selecciona nada
    if len(features_selected) == 0:
        return 0

    # Reducir dataset
    X_reduced = X_full[:, features_selected]

    # --- CAMBIO CLAVE 1: n_jobs=1 ---
    # Como PyGAD ya nos está paralelizando, el Random Forest debe ser
    # mononúcleo para no saturar el sistema.
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=1  # IMPORTANTE: 1 solo trabajo por individuo
    )

    clf.fit(X_reduced, y_full)
    predictions = clf.predict(X_reduced)
    accuracy = accuracy_score(y_full, predictions)

    # Penalización por complejidad (queremos menos genes)
    penalty = len(features_selected) / X_full.shape[1]
    fitness = accuracy - (0.05 * penalty)

    return max(0, fitness)


def on_generation(ga_instance):
    generation = ga_instance.generations_completed
    solution, fitness, _ = ga_instance.best_solution()
    num_genes_selected = np.sum(solution == 1)
    print(f"Gen {generation:3d} | Fitness: {fitness:.4f} | Genes: {num_genes_selected:4d}")


if __name__ == '__main__':
    # --- PASO 1: CARGAR Y PROCESAR DATOS (LÓGICA MULTICLASE) ---
    print("Procesando datos y creando etiquetas multiclase...")
    df_raw = pd.read_csv('dataset/datos.csv')  # Asegúrate que la ruta sea correcta
    gene_names = df_raw['ID'].values
    df_transposed = df_raw.set_index('ID').transpose()

    print("Generando etiquetas corregidas...")
    labels = []

    for pid in df_transposed.index:
        if pid.startswith('N'):
            labels.append(0)  # Sano
        elif pid.startswith('1S'):
            labels.append(1)  # Colon Etapa I
        elif pid.startswith('2S'):
            labels.append(2)  # Colon Etapa II
        elif pid.startswith('3S'):
            labels.append(3)  # Colon Etapa III
        elif pid.startswith('4S'):
            labels.append(4)  # Colon Etapa IV
        elif pid.startswith('PC'):
            labels.append(5)  # Próstata
        elif pid.startswith('TB') or pid.startswith('S'):
            labels.append(6)  # Páncreas
        else:
            labels.append(-1)  # Desconocido

    df_transposed['Target'] = labels

    # Validación rápida: Imprimimos cuántos hay de cada uno para asegurar que está bien
    print("Conteo de clases detectadas:", df_transposed['Target'].value_counts().to_dict())

    # Opcional: verificar que todas las muestras fueron etiquetadas
    if -1 in labels:
        print("ADVERTENCIA: Algunas muestras no pudieron ser etiquetadas.")
        # Considera filtrar las muestras no etiquetadas si es necesario
        # df_transposed = df_transposed[df_transposed['Target'] != -1]

    print(f"Etiquetas creadas para {len(np.unique(df_transposed['Target']))} clases.")

    X = df_transposed.drop(columns=['Target']).values
    y = df_transposed['Target'].values

    scaler = StandardScaler()
    X_full = scaler.fit_transform(X)
    y_full = y

    print(f"Datos listos: {X_full.shape}")

    # --- PASO 2: CONFIGURACIÓN AG ---
    num_genes = X_full.shape[1]
    sol_per_pop = 32
    num_parents_mating = 8

    ga_instance = pygad.GA(
        num_generations=300,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        init_range_low=0,
        init_range_high=2,
        gene_type=int,
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=1,
        on_generation=on_generation,
        random_seed=42,
        parallel_processing=["process", 16]
    )

    print("\nINICIANDO EVOLUCIÓN MULTINÚCLEO...")
    start_time = time.time()

    ga_instance.run()

    end_time = time.time()
    print(f"\nTiempo total: {end_time - start_time:.2f} segundos")

    # --- PASO 3: RESULTADOS ---
    solution, solution_fitness, _ = ga_instance.best_solution()
    selected_indices = np.where(solution == 1)[0]
    selected_genes = gene_names[selected_indices]

    print(f"\nMejor Fitness: {solution_fitness:.4f}")
    print(f"Genes seleccionados: {len(selected_genes)}")

    # Validación final con el mejor subconjunto de genes
    if len(selected_genes) > 0:
        X_final = X_full[:, selected_indices]
        clf_final = RandomForestClassifier(n_estimators=500, max_depth=15, n_jobs=-1, random_state=42)
        clf_final.fit(X_final, y_full)
        acc = accuracy_score(y_full, clf_final.predict(X_final))
        print(f"Precisión Final sobre el set de entrenamiento: {acc * 100:.2f}%")

        # Guardar resultados con importancia de características
        importances = clf_final.feature_importances_
        res = pd.DataFrame({'Gen': selected_genes, 'Importancia': importances})
        res = res.sort_values('Importancia', ascending=False)
        res.to_csv('genes_top_multiclase.csv', index=False)
        print("Guardado en genes_top_multiclase.csv")
    else:
        print("No se seleccionaron genes.")
