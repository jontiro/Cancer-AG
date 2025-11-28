# Selección de Genes Relacionados con Cáncer usando Algoritmos Genéticos

## 📋 Descripción General

Este proyecto implementa un **Algoritmo Genético (AG)** combinado con **Random Forest** para identificar los genes más relevantes en la clasificación de diferentes tipos y etapas de cáncer. El sistema analiza datos de expresión génica de pacientes y busca el subconjunto óptimo de genes que mejor discriminan entre múltiples clases de cáncer.

---

## 🎯 Objetivo

Seleccionar automáticamente los genes más informativos para:
- Diferenciar pacientes sanos de pacientes con cáncer
- Clasificar diferentes tipos de cáncer (colorrectal, próstata, páncreas)
- Identificar diferentes etapas del cáncer colorrectal (I, II, III, IV)

---

## 📊 Dataset

### Estructura de Datos

El archivo `dataset/datos.csv` contiene:
- **Filas**: Genes (identificados por ID)
- **Columnas**: Pacientes organizados en grupos

### Grupos de Pacientes (Total: 217 muestras)

| Código | Descripción | Cantidad | Clase |
|--------|-------------|----------|-------|
| `N` | Controles Sanos (Normal) | 50 | 0 |
| `1S` | Cáncer Colorrectal - Etapa I | 25 | 1 |
| `2S` | Cáncer Colorrectal - Etapa II | 25 | 2 |
| `3S` | Cáncer Colorrectal - Etapa III | 25 | 3 |
| `4S` | Cáncer Colorrectal - Etapa IV | 25 | 4 |
| `PC` | Cáncer de Próstata (HSPC y CRPC) | 36 | 5 |
| `TB/S` | Cáncer de Páncreas | 6 | 6 |

---

## 🧬 ¿Qué Mide el Código?

### 1. **Expresión Génica**
- Mide los niveles de expresión de miles de genes en diferentes muestras de pacientes
- Cada gen puede estar sobre-expresado o sub-expresado en casos de cáncer

### 2. **Clasificación Multiclase**
- Identifica genes que distinguen entre 7 clases diferentes:
  - Personas sanas vs. pacientes con cáncer
  - Entre diferentes tipos de cáncer
  - Entre diferentes etapas de progresión del cáncer

### 3. **Biomarcadores Potenciales**
- Los genes seleccionados son candidatos a biomarcadores para:
  - Diagnóstico temprano
  - Clasificación del tipo de cáncer
  - Determinación del estadio de la enfermedad

---

## 🔬 ¿Genes Cancerosos o Sanos?

El algoritmo **NO clasifica genes como "cancerosos" o "sanos"**, sino que:

✅ **Identifica genes diferencialmente expresados**: Genes cuya expresión varía significativamente entre:
- Pacientes sanos y enfermos
- Diferentes tipos de cáncer
- Diferentes etapas de cáncer

✅ **Encuentra biomarcadores**: Genes que pueden servir como indicadores de:
- Presencia de cáncer
- Tipo específico de cáncer
- Progresión de la enfermedad

---

## 🧮 Parámetros del Algoritmo Genético

### Configuración de la Población

```python
num_generations = 300           # Número de generaciones evolutivas
sol_per_pop = 32               # Tamaño de la población (individuos)
num_parents_mating = 8         # Padres seleccionados para reproducción
keep_parents = 2               # Padres que pasan directamente a la siguiente generación
```

### Operadores Genéticos

- **Selección**: `"sss"` (Steady State Selection)
- **Cruce**: `"single_point"` (Cruce de un punto)
- **Mutación**: `"random"` con 1% de genes mutados
- **Tipo de Gen**: Binario (0 o 1)
  - `1` = Gen seleccionado
  - `0` = Gen no seleccionado

### Procesamiento Paralelo

```python
parallel_processing = ["process", 16]  # 16 núcleos en paralelo
```

---

## 📈 Función de Fitness

### Fórmula

```
Fitness = Accuracy - (0.05 × Penalty)
```

Donde:
- **Accuracy**: Precisión del clasificador Random Forest con los genes seleccionados
- **Penalty**: Proporción de genes seleccionados (genes_seleccionados / total_genes)

### Componentes

1. **Accuracy (Precisión)**
   - Mide qué tan bien los genes seleccionados clasifican las 7 clases
   - Entrenamiento con Random Forest (100 árboles, profundidad máxima 10)

2. **Penalización por Complejidad**
   - Factor: 0.05 (5%)
   - Objetivo: Favorecer soluciones con menos genes
   - Principio de parsimonia: Menos genes = Modelo más interpretable

3. **Penalización Extrema**
   - Si no se selecciona ningún gen: Fitness = 0

### Ejemplo de Cálculo

Si seleccionamos 100 genes de 10,000 totales y obtenemos 95% de precisión:

```
Penalty = 100 / 10,000 = 0.01
Fitness = 0.95 - (0.05 × 0.01) = 0.95 - 0.0005 = 0.9495
```

---

## 🌲 Random Forest: Rol Dual

### 1. Durante la Evolución (Fitness)
```python
RandomForestClassifier(
    n_estimators=100,      # 100 árboles
    max_depth=10,          # Profundidad máxima 10
    n_jobs=1,              # 1 núcleo (AG ya paraleliza)
    random_state=42
)
```

### 2. Validación Final
```python
RandomForestClassifier(
    n_estimators=500,      # 500 árboles (más precisión)
    max_depth=15,          # Mayor profundidad
    n_jobs=-1,             # Todos los núcleos disponibles
    random_state=42
)
```

---

## 📊 Preprocesamiento de Datos

### 1. Transposición
```
Original: Genes × Pacientes
Procesado: Pacientes × Genes
```

### 2. Etiquetado Automático
Basado en el prefijo del ID del paciente:
- `N*` → Clase 0 (Sano)
- `1S*` → Clase 1 (Colorrectal I)
- `2S*` → Clase 2 (Colorrectal II)
- etc.

### 3. Normalización
```python
StandardScaler()  # Media 0, Desviación Estándar 1
```
- Elimina diferencias de escala entre genes
- Mejora el rendimiento del clasificador

---

## 🎲 Estadísticas Relevantes

### Métricas por Generación

Durante la ejecución, se muestra:
```
Gen   1 | Fitness: 0.8234 | Genes:  523
Gen   2 | Fitness: 0.8456 | Genes:  412
Gen   3 | Fitness: 0.8567 | Genes:  389
...
Gen 300 | Fitness: 0.9512 | Genes:  127
```

- **Gen**: Número de generación actual
- **Fitness**: Mejor fitness en esa generación
- **Genes**: Cantidad de genes seleccionados

### Métricas Finales

```
Mejor Fitness: 0.9512
Genes seleccionados: 127
Precisión Final sobre el set de entrenamiento: 96.78%
Tiempo total: 1234.56 segundos
```

---

## 📁 Resultados

### Archivo de Salida: `genes_top_multiclase.csv`

Contiene dos columnas:
1. **Gen**: ID del gen seleccionado
2. **Importancia**: Importancia del gen según Random Forest

Ejemplo:
```csv
Gen,Importancia
GENE_12345,0.0523
GENE_67890,0.0487
GENE_11111,0.0456
...
```

### Interpretación de Importancia

- **Importancia > 0.05**: Genes altamente relevantes
- **Importancia 0.01-0.05**: Genes moderadamente relevantes
- **Importancia < 0.01**: Genes de apoyo

---

## 🔄 Flujo del Algoritmo

```
1. CARGA DE DATOS
   ├─ Leer datos.csv
   ├─ Transponer matriz
   └─ Crear etiquetas multiclase

2. PREPROCESAMIENTO
   ├─ Normalización (StandardScaler)
   └─ Validación de clases

3. INICIALIZACIÓN AG
   ├─ Crear población inicial (32 individuos)
   └─ Cada individuo = Vector binario de genes

4. EVOLUCIÓN (300 generaciones)
   │
   ├─ Para cada individuo (en paralelo):
   │  ├─ Seleccionar genes donde gen[i] = 1
   │  ├─ Entrenar Random Forest
   │  ├─ Calcular accuracy
   │  └─ Aplicar penalización
   │
   ├─ Selección de padres (8 mejores)
   ├─ Cruce (single point)
   ├─ Mutación (1% de genes)
   └─ Nueva generación

5. VALIDACIÓN FINAL
   ├─ Genes seleccionados de la mejor solución
   ├─ Entrenar Random Forest robusto
   ├─ Calcular importancia de cada gen
   └─ Guardar resultados

6. EXPORTACIÓN
   └─ genes_top_multiclase.csv
```

---

## 🚀 Ejecución

### Requisitos

```bash
pip install pandas numpy pygad scikit-learn
```

### Comando

```bash
python cancer4.py
```

### Tiempo Estimado

- Con 16 núcleos: ~20-40 minutos
- Con 8 núcleos: ~40-80 minutos
- Con 4 núcleos: ~80-160 minutos

---

## 💡 Comparación con Diferentes Grupos (N1 vs N50)

### ¿Se puede entrenar con diferentes grupos?

**SÍ, es totalmente posible y recomendable.** Aquí hay varias estrategias:

### 1. **Análisis Intra-Grupo**
Entrenar AG solo con pacientes de un tipo:
```python
# Solo pacientes sanos (N)
df_filtered = df_transposed[df_transposed.index.str.startswith('N')]

# Solo Cáncer Colorrectal Etapa IV
df_filtered = df_transposed[df_transposed.index.str.startswith('4S')]
```

**Utilidad**: Identificar variabilidad genética dentro del mismo grupo

### 2. **Análisis Comparativo Binario**
Comparar dos grupos específicos:
```python
# Sanos (N) vs. Cáncer Colorrectal Etapa I (1S)
df_comparison = df_transposed[
    df_transposed.index.str.startswith('N') | 
    df_transposed.index.str.startswith('1S')
]
```

**Utilidad**: Genes específicos para detección temprana

### 3. **Análisis por Etapas**
Comparar progresión del cáncer:
```python
# Etapa I vs. Etapa IV
stages = df_transposed[
    df_transposed.index.str.startswith('1S') | 
    df_transposed.index.str.startswith('4S')
]
```

**Utilidad**: Genes relacionados con la progresión tumoral

### 4. **Análisis por Tipo de Cáncer**
```python
# Colorrectal vs. Próstata
cancer_types = df_transposed[
    df_transposed.index.str.contains(r'^[1-4]S') | 
    df_transposed.index.str.startswith('PC')
]
```

**Utilidad**: Biomarcadores específicos de cada tipo de cáncer

---

## 🧪 Variaciones Genéticas Dependientes del Cáncer

### Genes que Varían Entre Grupos

El AG actual **YA identifica estos genes**, pero puedes hacer análisis más específicos:

### Estrategia 1: Análisis Diferencial
```python
# Entrenar 3 AGs independientes:
# 1. Sanos vs. Todos los cánceres
# 2. Colorrectal vs. Próstata vs. Páncreas
# 3. Etapas I-II vs. Etapas III-IV

# Comparar genes seleccionados
genes_comunes = set(genes_ag1) & set(genes_ag2) & set(genes_ag3)
genes_especificos_tipo = set(genes_ag2) - genes_comunes
genes_especificos_etapa = set(genes_ag3) - genes_comunes
```

### Estrategia 2: One-vs-Rest
```python
# Para cada clase, entrenar AG binario:
# - Clase 0 (Sanos) vs. Resto
# - Clase 1 (Colorrectal I) vs. Resto
# - Clase 2 (Colorrectal II) vs. Resto
# etc.
```

### Estrategia 3: Análisis Jerárquico
```python
# Nivel 1: Sano vs. Cáncer
# Nivel 2: Tipo de cáncer (entre cancerosos)
# Nivel 3: Etapa del cáncer (dentro de cada tipo)
```

---

## 📊 Tipos de Genes Identificados

### 1. **Genes Pan-Cáncer**
- Seleccionados para múltiples tipos de cáncer
- Relacionados con procesos tumorales generales
- Ejemplos: Regulación del ciclo celular, apoptosis

### 2. **Genes Tipo-Específicos**
- Alta importancia para un tipo de cáncer específico
- Baja importancia para otros tipos
- Ejemplos: Biomarcadores específicos de próstata

### 3. **Genes de Progresión**
- Varían entre etapas tempranas y avanzadas
- Relacionados con metástasis e invasión
- Útiles para pronóstico

### 4. **Genes de Diagnóstico Temprano**
- Diferencia Etapa I vs. Sanos
- Alto valor clínico
- Candidatos para screening

---

## ⚠️ Consideraciones Importantes

### Limitaciones

1. **Sobreajuste**: El modelo se valida sobre los mismos datos de entrenamiento
   - **Solución**: Implementar validación cruzada

2. **Datos Desbalanceados**: Solo 6 muestras de cáncer de páncreas
   - **Impacto**: Menor precisión para esta clase

3. **Alta Dimensionalidad**: Miles de genes, pocas muestras
   - **Riesgo**: Correlaciones espurias

### Mejoras Recomendadas

```python
# 1. Validación cruzada
from sklearn.model_selection import cross_val_score

# 2. Conjunto de prueba independiente
X_train, X_test, y_train, y_test = train_test_split(...)

# 3. Balanceo de clases
from imblearn.over_sampling import SMOTE

# 4. Análisis estadístico adicional
from scipy.stats import mannwhitneyu  # Para validar diferencias
```

---

## 📚 Interpretación Biológica

### Genes Seleccionados: ¿Qué Significan?

1. **Sobre-expresión**: Gen produce más proteína de lo normal
2. **Sub-expresión**: Gen produce menos proteína de lo normal
3. **Biomarcador**: Patrón de expresión asociado con enfermedad

### Pasos Siguientes

1. **Validación experimental**: Confirmar hallazgos en laboratorio
2. **Búsqueda bibliográfica**: ¿Genes ya reportados en literatura?
3. **Análisis de rutas**: ¿Qué vías biológicas están afectadas?
4. **Validación clínica**: Probar en nuevas cohortes de pacientes

---

## 📞 Soporte y Contribuciones

### Modificar Parámetros

```python
# Más generaciones para mejor convergencia
num_generations = 500

# Población más grande para mayor diversidad
sol_per_pop = 50

# Ajustar penalización (mayor = menos genes)
fitness = accuracy - (0.10 * penalty)  # En vez de 0.05
```

### Experimentar con Clasificadores

```python
# Probar otros algoritmos
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
```

---

## 📄 Licencia

Proyecto académico para Inteligencia Artificial

---

## 🔗 Referencias

- **PyGAD**: https://pygad.readthedocs.io/
- **Scikit-learn**: https://scikit-learn.org/
- **Feature Selection**: Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection.

---

## 📝 Notas Finales

Este código representa un enfoque de **selección de características mediante metaheurística** aplicado a datos biomédicos. Los genes identificados son **candidatos** que requieren validación experimental antes de cualquier aplicación clínica.

**Autor**: Jonathan  
**Fecha**: Noviembre 2025  
**Curso**: Inteligencia Artificial

