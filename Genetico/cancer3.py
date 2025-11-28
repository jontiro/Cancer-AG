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
        n_estimators=100,  # Bajé un poco esto para agilizar la evolución
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
    # --- PASO 1: CARGAR DATOS ---
    print("Procesando datos...")
    df_raw = pd.read_csv('dataset/datos.csv')  # Asegúrate que la ruta sea correcta
    gene_names = df_raw['ID'].values

    df_transposed = df_raw.set_index('ID').transpose()

    labels = []
    for pid in df_transposed.index:
        if pid.startswith('N'):
            labels.append(0)
        else:
            labels.append(1)

    df_transposed['Target'] = labels

    X = df_transposed.drop(columns=['Target']).values
    y = df_transposed['Target'].values

    scaler = StandardScaler()
    X_full = scaler.fit_transform(X)
    y_full = y

    print(f"Datos listos: {X_full.shape}")

    # --- PASO 2: CONFIGURACIÓN AG ---
    num_genes = X_full.shape[1]

    # Sincronizamos la población con tus hilos (16)
    # Aumentamos sol_per_pop para darle más trabajo a la CPU
    sol_per_pop = 32  # Múltiplo de 16 (2 individuos por núcleo por generación)

    ga_instance = pygad.GA(
        num_generations=100,
        num_parents_mating=8,  # Aumentado proporcionalmente
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
        # --- CAMBIO CLAVE 2: Multiprocesamiento ---
        # "process": Usa núcleos reales (salta el bloqueo de Python)
        # 16: Usa tus 16 hilos lógicos
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

    # Validación final (Aquí sí usamos n_jobs=-1 porque es una sola ejecución)
    if len(selected_genes) > 0:
        X_final = X_full[:, selected_indices]
        clf_final = RandomForestClassifier(n_estimators=500, max_depth=15, n_jobs=-1, random_state=42)
        clf_final.fit(X_final, y_full)
        acc = accuracy_score(y_full, clf_final.predict(X_final))
        print(f"Precisión Final: {acc * 100:.2f}%")

        # Guardar resultados
        importances = clf_final.feature_importances_
        res = pd.DataFrame({'Gen': selected_genes, 'Importancia': importances})
        res = res.sort_values('Importancia', ascending=False)
        res.to_csv('genes_top_cpu.csv', index=False)
        print("Guardado en genes_top_cpu.csv")