# 🏢 HR Employee Attrition Analysis

## Objetivo
Identificar los factores que más influyen en la rotación de empleados 
y cuantificar el impacto financiero para una empresa de ~1,500 personas.

## Dataset
- **Fuente:** IBM HR Analytics — Kaggle
- **Link:** https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
- **Registros:** 1,470 empleados · 35 variables originales

## Herramientas
`Python` · `Pandas` · `NumPy` · `Seaborn` · `Matplotlib` · `Power BI`

## Hallazgos Clave
- Tasa de rotación global: **16.1%**
- El departamento de **Ventas** tiene la mayor rotación (20.6%)
- Empleados con **horas extra** tienen 3x más probabilidad de rotar (Sin horas extra **10.4%** Vs Con horas extra **30.5%**)
- La rotación ocurre principalmente en los **primeros 2 años**
- **Costo total estimado: $17,594,904 USD**

## Estructura del Proyecto

├── WA_Fn-UseC_-HR-Employee-Attrition.csv  ← Dataset original
├── Configuración del entorno e importación.py ← Análisis completo
├── hr_attrition_clean.csv                  ← Dataset limpio
├── hr_costo_rotacion.csv                   ← Dataset costos
├── hr_kpis_summary                         ← Dataset resumen
├── dashboard_preview.pbix                  ← Dashboard Power BI
└── img/
    └── eda_rotacion_empleados
    └── dashboard_preview.pdf


## Dashboard Power BI
<img width="999" height="520" alt="image" src="https://github.com/user-attachments/assets/e507df81-f6f2-4005-af50-f512b0236ee3" />
<img width="1404" height="750" alt="image" src="https://github.com/user-attachments/assets/e30c1b96-bd81-4100-8f45-fdd38e39e382" />

