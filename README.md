# T48A-E03A
E50A Examen del tercer parcial

# Ejercicio de Machine Learning para Soporte TI

Este ejercicio utiliza un dataset **ficticio** de 200 tickets de soporte TI para practicar cinco enfoques de Machine Learning:

- **Vecino más cercano (K-NN)** – Clasificación por similitud.
- **K-Means** – Agrupamiento no supervisado.
- **Perceptrón** – Clasificación (binaria/multiclase) lineal.
- **Perceptrón Multicapa (MLP)** – Clasificación no lineal.
- **Mapas Autoorganizados (SOM)** – Proyección/visualización no supervisada.

> **Nota:** Los datos se generaron con semilla aleatoria `seed=42` para asegurar reproducibilidad. Son de demostración y no reflejan incidentes reales.

---

## 1. Estructura del dataset
Archivo: `tickets_ficticios_TI.csv`

Columnas:
- `ticket_id` (string): Identificador del ticket.
- `categoria` (string): Red | Software | Hardware.
- `severidad` (string): Baja | Media | Alta.
- `tiempo_resolucion_horas` (float): Tiempo estimado/real de resolución.
- `descripcion` (string): Texto breve del incidente.

Ejemplo de filas:
```
TKT-1000,Hardware,Alta,17.4,Servidor físico sin encender tras actualización eléctrica.
TKT-1008,Red,Alta,36.4,Caída total de VPN corporativa. Usuarios sin acceso remoto.
TKT-1014,Software,Media,25.3,Aplicación ERP se congela al guardar pedidos.
```

---

## 2. Requisitos del entorno
- Python 3.10+
- Paquetes:
  - `pandas`, `numpy`, `scikit-learn`
  - Opcional para SOM: `minisom` o implementación propia

Instalación rápida (si fuera necesario):
```bash
pip install pandas numpy scikit-learn minisom matplotlib
```

---

## 3. Objetivos de aprendizaje
1. **Preparar datos**: codificación/normalización de variables.
2. **Entrenar y evaluar** modelos de clasificación (K-NN, Perceptrón, MLP).
3. **Realizar clustering** con K-Means y analizar sus centroides.
4. **Visualizar patrones** con SOM (o alternativa) para detección de grupos.
5. **Documentar hallazgos** con métricas y gráficas.

---

## 4. Flujo de trabajo sugerido

### 4.1 Cargar y explorar datos
```python
import pandas as pd

df = pd.read_csv('tickets_ficticios_TI.csv')
print(df.head())
print(df['categoria'].value_counts())
print(df['severidad'].value_counts())
```

### 4.2 Preprocesamiento
- Convertir `severidad` y `categoria` a etiquetas numéricas cuando sea necesario.
- Escalar `tiempo_resolucion_horas`.

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Mapear severidad a números


# Etiquetas de salida (clasificación por categoria)


```

### 4.3 Clasificación con K-NN
```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


print('Exactitud K-NN:', knn.score(X_test, y_test))
```

### 4.4 Clasificación con Perceptrón
```python
from sklearn.linear_model import Perceptron


print('Exactitud Perceptrón:', perc.score(X_test, y_test))
```

### 4.5 Clasificación con MLP
```python
from sklearn.neural_network import MLPClassifier


print('Exactitud MLP:', mlp.score(X_test, y_test))
```

### 4.6 Clustering con K-Means
```python
from sklearn.cluster import KMeans


print('Tamaño de clusters:', pd.Series(labels).value_counts().to_dict())
print('Centroides:', kmeans.cluster_centers_)
```

### 4.7 SOM

```python
from minisom import MiniSom


```
Alternativa sin librerías externas: proyectar con **PCA** o **t-SNE** para visualizar clusters.
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


plt.figure(figsize=(6,5))
plt.scatter(Z[:,0], Z[:,1], c=y, cmap='viridis', s=30)
plt.title('Proyección PCA de tickets (color = categoría)')
plt.xlabel('Componente 1'); plt.ylabel('Componente 2')
plt.show()
```

---

## 5. Métricas y reporte
- **Clasificación**: exactitud, matriz de confusión, F1 por clase.
```python
from sklearn.metrics import confusion_matrix, classification_report


print(confusion_matrix(y_test, pred_mlp))
print(classification_report(y_test, pred_mlp, target_names=le_cat.classes_))
```
- **Clustering**: inercia (SSE), silhouette (si aplica).
```python
from sklearn.metrics import silhouette_score
print('Silhouette:', silhouette_score(X, labels))
```

---

## 6. Extensiones sugeridas
1. Añadir campos: `SLA`, `impacto_usuarios`, `equipo_afectado`, `estado`.
2. Clasificar urgencia (alta/no) con Perceptrón.
3. Entrenar MLP con texto del ticket (TF-IDF) + metadatos.
4. Detectar anomalías de tiempo de resolución con **Isolation Forest**.

---

## 7. Cómo ejecutar rápidamente
```bash
# 1) Clona o descarga los archivos
#  - tickets_ficticios_TI.csv
#  - notebook o script que contenga el flujo (opcional)

# 2) Instala dependencias
pip install pandas numpy scikit-learn minisom matplotlib

# 3) Ejecuta tu script/notebook
python tu_script.py
# o abre el notebook en Jupyter/Lab
```

---

## 8. Licencia
Uso educativo y demostrativo. Puedes reutilizar y modificar libremente el material.

---


_Última actualización: 2025-11-27T23:19:35_

