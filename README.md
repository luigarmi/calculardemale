# Proyecto final - Prediccion clinica multiclase

Aplicacion web construida con Flask que permite realizar predicciones individuales y por lotes utilizando dos modelos entrenados (regresion logistica y red neuronal MLP) sobre el dataset clinico **DEMALE-HSJM 2025**. El flujo completo incluye entrenamiento, persistencia de los modelos, metricas de referencia y una interfaz web responsiva.

## Caracteristicas
- Carga dinamica de variables, metadatos y metricas directamente desde el backend.
- Prediccion individual con probabilidades por clase (3 clases: 1, 2 y 3).
- Prediccion por lotes leyendo archivos `.csv`, `.xlsx` o `.xls` y calculo automatico de matriz de confusion multiclase.
- Vista previa de las primeras filas procesadas y reporte de metricas globales (exactitud, precision, recall y F1 ponderados).
- Script de entrenamiento parametrizable (`--dataset`) para reutilizar modelos previos de la carpeta `data/`.

## Estructura de carpetas
```
final_project/
├── app.py
├── train_models.py
├── requirements.txt
├── data/
│   ├── balanced_normalized_dataset_covid_19_hiv.xlsx
│   └── DEMALE-HSJM_2025_data.xlsx
├── models/
│   ├── logistic_regression.joblib
│   ├── mlp_classifier.joblib
│   └── metrics.json
├── static/
│   ├── app.js
│   └── styles.css
└── templates/
    └── index.html
```

## Instalación y ejecución
1. (Opcional) Crear y activar un entorno virtual.
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Entrenar y guardar los modelos (solo la primera vez o cuando se actualice el dataset):
   ```bash
   python train_models.py          # usa el dataset DEMALE por defecto
   # opcional: python train_models.py --dataset covid_hiv
   ```
4. Levantar el servidor Flask:
   ```bash
   python app.py
   ```
5. Abrir `http://127.0.0.1:5000` en el navegador.

## Dataset y columnas requeridas
Para predicciones por lotes es obligatorio que el archivo incluya todas las columnas predictoras y la columna objetivo `diagnosis`. Las variables esperadas en orden son:

```
male
female
age
urban_origin
rural_origin
homemaker
student
professional
merchant
agriculture_livestock
various_jobs
unemployed
hospitalization_days
body_temperature
fever
headache
dizziness
loss_of_appetite
weakness
myalgias
arthralgias
eye_pain
hemorrhages
vomiting
abdominal_pain
chills
hemoptysis
edema
jaundice
bruises
petechiae
rash
diarrhea
respiratory_difficulty
itching
hematocrit
hemoglobin
red_blood_cells
white_blood_cells
neutrophils
eosinophils
basophils
monocytes
lymphocytes
platelets
AST (SGOT)
ALT (SGPT)
ALP (alkaline_phosphatase)
total_bilirubin
direct_bilirubin
indirect_bilirubin
total_proteins
albumin
creatinine
urea
diagnosis
```

El dataset combina variables binarias (0 o 1) y variables continuas con magnitudes originales (ej. edad, temperatura corporal, valores de laboratorio). Para predicciones individuales introduce valores segun estos rangos.

## Endpoints principales
- `GET /api/status` — lista modelos disponibles, orden de variables, clases y metricas de referencia.
- `POST /api/predict/individual` — recibe JSON `{ "model": "...", "features": { ... } }` y devuelve clase predicha mas probabilidades.
- `POST /api/predict/batch` — recibe formulario multipart con `model` y `file`, calcula metricas, matriz de confusion (como imagen base64) y muestra una vista previa de las predicciones.

## Notas
- Si se reemplaza el dataset o se ajustan hiperparametros, ejecutar de nuevo `python train_models.py` para regenerar los artefactos.
- El frontend genera los campos de forma automatica con base en `metrics.json`, por lo que agregar/quitar variables en el entrenamiento se refleja inmediatamente en la interfaz tras volver a entrenar.
- Para reutilizar el proyecto con otro dataset, copiar el archivo a `data/` y entrenar usando `python train_models.py --dataset <opcion>` tras registrar la configuracion dentro del diccionario `DATASETS`.
