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
- El departamento de **Ventas** tiene la mayor rotación (~21%)
- Empleados con **horas extra** tienen 3x más probabilidad de rotar
- La rotación ocurre principalmente en los **primeros 2 años**
- **Costo total estimado: ~$XX millones USD**

## Estructura del Proyecto
```
├── WA_Fn-UseC_-HR-Employee-Attrition.csv  ← Dataset original
├── hr_attrition_analysis.ipynb             ← Análisis completo
├── hr_attrition_clean.csv                  ← Dataset limpio
├── hr_costo_rotacion.csv                   ← Dataset costos
├── dashboard_powerbi.pbix                  ← Dashboard Power BI
└── img/
    └── dashboard_preview.png
```

## Dashboard Power BI
![Dashboard Preview](img/dashboard_preview.png)