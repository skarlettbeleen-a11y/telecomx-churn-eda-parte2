# Telecom X – Parte 2: Predicción de Cancelación (Churn)

## 1. Contexto y objetivo
Telecom X presenta una tasa relevante de cancelación de clientes (churn). El objetivo de este proyecto es construir modelos predictivos de clasificación capaces de anticipar qué clientes tienen mayor probabilidad de cancelar sus servicios, permitiendo priorizar acciones de retención y apoyar decisiones estratégicas basadas en datos.

Este proyecto corresponde a la **Parte 2** del Challenge Telecom X (ONE/Alura Latam), utilizando el dataset ya tratado en la **Parte 1 (ETL + EDA)**.

---

## 2. Estructura del proyecto
- `TelecomX_Parte2_Churn.ipynb` : Notebook principal con carga de datos tratados, preparación, modelado, evaluación e interpretación.
- `datos_tratados.csv` : Dataset tratado exportado desde la Parte 1 (base para el modelamiento).
- (Opcional) Carpeta `imgs/` o `figures/` : Visualizaciones exportadas (si decides guardar gráficos).

---

## 3. Preparación de datos (Parte 2)
### 3.1 Variable objetivo
Se construyó una variable binaria:
- `Churn_bin`: 0 = No cancela, 1 = Cancela

### 3.2 Eliminación de columnas irrelevantes
Se eliminaron identificadores únicos como `customerID` para evitar ruido y posibles fugas, ya que no aportan valor predictivo.

### 3.3 Codificación de variables categóricas
Se aplicó **One-Hot Encoding** a variables categóricas usando `pd.get_dummies(..., drop_first=True)` para transformar el dataset a un formato numérico compatible con algoritmos de Machine Learning.

### 3.4 Separación de entrenamiento y prueba
Se utilizó `train_test_split` con estratificación para mantener la proporción de churn:
- División típica: train/test (ej. 80/20 o 70/30, según notebook)
- `stratify=y` para preservar el desbalance de clases en ambos conjuntos.

### 3.5 Normalización/Estandarización
- Se aplicó estandarización **solo** en modelos sensibles a escala (Regresión Logística) mediante un `Pipeline` con `StandardScaler`.
- Para modelos basados en árboles (Random Forest), no se aplicó normalización, ya que no depende de la escala de las variables.

---

## 4. Modelos entrenados
Se entrenaron al menos dos modelos de clasificación:
1. **Regresión Logística (con estandarización)**
2. **Random Forest (sin estandarización)**

---

## 5. Evaluación de desempeño
Los modelos se evaluaron en el conjunto de prueba con:
- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusión
- Classification report

Además, se realizó una comparación crítica considerando el objetivo de negocio: detectar clientes con riesgo de churn (frecuentemente se prioriza recall de la clase positiva).

---

## 6. Interpretación e insights
Se interpretaron variables relevantes mediante:
- **Importancia de variables** en Random Forest (`feature_importances_`)
- **Coeficientes** (magnitud absoluta) en Regresión Logística

Entre los factores más influyentes se destacan:
- Antigüedad del cliente (`customer_tenure`)
- Variables de gasto (`account_Charges_Total`, `account_Charges_Monthly`, `Cuentas_Diarias`)
- Tipo de contrato (`account_Contract_*`)
- Tipo de internet (ej. fibra óptica)
- Método de pago (ej. electronic check)

---

## 7. Cómo ejecutar el proyecto
### 7.1 Requisitos
Python 3.x y las siguientes librerías:
- pandas
- numpy
- matplotlib
- scikit-learn

### 7.2 Instalación (si aplica)
En Colab normalmente no es necesario instalar, pero en entorno local:
```bash
pip install pandas numpy matplotlib scikit-learn
