# ============================================================
# PROYECTO 1: Análisis de Rotación de Empleados

# Autor: Víctor Bancayán Capuñay

# Dataset: IBM HR Analytics - Kaggle

# Objetivo: Identificar factores que predicen la rotación y
#            calcular el costo financiero del problema
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Configuración visual global
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'sans-serif'
sns.set_palette("Set2")
sns.set_style("whitegrid")

# ── Carga de datos ──────────────────────────────────────────
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

print(f"Shape: {df.shape}")
print(f"\nColumnas:\n{df.columns.tolist()}")
print(f"\nPrimeras filas:\n{df.head(3)}")



# ============================================================
# PASO 2: Auditoría de calidad
# ============================================================

def auditoria_datos(df):
    """
    Genera un reporte de calidad del dataset:
    - % nulos por columna
    - Tipo de dato
    - Cardinalidad (valores únicos)
    - Detección de columnas constantes (sin variación = inútiles)
    """
    reporte = pd.DataFrame({
        'tipo_dato': df.dtypes,
        'nulos': df.isnull().sum(),
        'pct_nulos': (df.isnull().sum() / len(df) * 100).round(2),
        'valores_unicos': df.nunique(),
        'ejemplo_valor': df.iloc[0]
    })
    
    # Columnas constantes: mismo valor en todas las filas (no aporta info, solo formato)
    cols_constantes = [col for col in df.columns if df[col].nunique() == 1]
    
    print("=" * 60)
    print("REPORTE DE CALIDAD DEL DATASET")
    print("=" * 60)
    print(f"\nTotal filas     : {df.shape[0]:,}")
    print(f"Total columnas  : {df.shape[1]}")
    print(f"Celdas nulas    : {df.isnull().sum().sum():,}")
    print(f"Duplicados      : {df.duplicated().sum():,}")
    print(f"\nColumnas constantes (eliminar): {cols_constantes}")
    print(f"\n{reporte.to_string()}")
    
    return reporte, cols_constantes

reporte, cols_constantes = auditoria_datos(df)






# ============================================================
# PASO 3: Limpieza y feature engineering
# ============================================================

df_clean = df.copy()


# ── 3.1 Eliminar columnas sin valor analítico ────────────────
cols_eliminar = cols_constantes + ['EmployeeNumber']
df_clean.drop(columns=cols_eliminar, inplace=True)
print(f"Columnas eliminadas: {cols_eliminar}")
print(f"Columnas restantes : {df_clean.shape[1]}")


# ── 3.2 Convertir variable objetivo a binario ────────────────
# Attrition: 'Yes' = 1 (se fue), 'No' = 0 (se quedó)
df_clean['Attrition_bin'] = (df_clean['Attrition'] == 'Yes').astype(int)
tasa_rotacion = df_clean['Attrition_bin'].mean() * 100
print(f"\nTasa de rotación global: {tasa_rotacion:.1f}%")


# ── 3.3 Feature Engineering: nuevas variables de negocio ─────

# Años en la empresa vs. años en el cargo actual
# Ratio cercano a 1 = estancado en el mismo puesto
df_clean['ratio_antiguedad_cargo'] = np.where(
    df_clean['YearsAtCompany'] > 0,
    df_clean['YearsInCurrentRole'] / df_clean['YearsAtCompany'],
    0
).round(2)

# Ingreso mensual relativo al promedio de su departamento
# Valores < 1 = gana menos que el promedio de su área
df_clean['ingreso_vs_dpto'] = df_clean.groupby('Department')['MonthlyIncome'].transform(
    lambda x: x / x.mean()
).round(2)

# Tiempo desde la última promoción (riesgo de fuga si > 3 años)
df_clean['riesgo_sin_promo'] = (df_clean['YearsSinceLastPromotion'] > 3).astype(int)

# Categoría de balance vida-trabajo (recode más descriptivo)
wlb_map = {1: 'Malo', 2: 'Regular', 3: 'Bueno', 4: 'Excelente'}
df_clean['WorkLifeBalance_cat'] = df_clean['WorkLifeBalance'].map(wlb_map)

print("\nNuevas variables creadas:")
print(df_clean[['ratio_antiguedad_cargo', 'ingreso_vs_dpto', 
                  'riesgo_sin_promo', 'WorkLifeBalance_cat']].head())




# ============================================================
# PASO 4: EDA orientado a decisiones de negocio (GRÁFICOS)
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Análisis de Rotación de Empleados — IBM HR Dataset', 
             fontsize=16, fontweight='bold', y=1.02)



# ── Gráfico 1: Tasa de rotación por departamento ─────────────
rot_dpto = df_clean.groupby('Department')['Attrition_bin'].mean() * 100
rot_dpto.sort_values().plot(kind='barh', ax=axes[0,0], color='#E05C5C')
axes[0,0].set_title('Tasa de Rotación por Departamento (%)')
axes[0,0].set_xlabel('% Rotación')
for i, v in enumerate(rot_dpto.sort_values()):
    axes[0,0].text(v + 0.3, i, f'{v:.1f}%', va='center', fontweight='bold')



# ── Gráfico 2: Rotación por nivel de satisfacción laboral ────
sat_rot = df_clean.groupby('JobSatisfaction')['Attrition_bin'].mean() * 100
sat_labels = {1:'Muy baja', 2:'Baja', 3:'Alta', 4:'Muy alta'}
sat_rot.index = sat_rot.index.map(sat_labels)
sat_rot.plot(kind='bar', ax=axes[0,1], color='#5C8EE0', edgecolor='white')
axes[0,1].set_title('Rotación por Satisfacción Laboral (%)')
axes[0,1].set_xlabel('Nivel de Satisfacción')
axes[0,1].set_ylabel('% Rotación')
axes[0,1].tick_params(axis='x', rotation=30)



# ── Gráfico 3: Distribución de ingresos (rotó vs. se quedó) ──
df_clean[df_clean['Attrition_bin']==1]['MonthlyIncome'].plot(
    kind='hist', ax=axes[0,2], alpha=0.6, label='Rotó', color='#E05C5C', bins=25)
df_clean[df_clean['Attrition_bin']==0]['MonthlyIncome'].plot(
    kind='hist', ax=axes[0,2], alpha=0.6, label='Se quedó', color='#5CE07A', bins=25)
axes[0,2].set_title('Distribución de Ingreso Mensual')
axes[0,2].set_xlabel('Ingreso Mensual (USD)')
axes[0,2].legend()



# ── Gráfico 4: Overtime y rotación ───────────────────────────
ot_rot = df_clean.groupby('OverTime')['Attrition_bin'].mean() * 100
ot_rot.plot(kind='bar', ax=axes[1,0], color=['#5CE07A', '#E05C5C'], edgecolor='white')
axes[1,0].set_title('Rotación: Con vs. Sin Horas Extra (%)')
axes[1,0].set_xticklabels(['Sin horas extra', 'Con horas extra'], rotation=0)
axes[1,0].set_ylabel('% Rotación')
for i, v in enumerate(ot_rot):
    axes[1,0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')



# ── Gráfico 5: Heatmap — correlación variables numéricas ─────
cols_num = ['Attrition_bin', 'Age', 'MonthlyIncome', 'YearsAtCompany',
            'JobSatisfaction', 'WorkLifeBalance', 'YearsSinceLastPromotion',
            'DistanceFromHome', 'NumCompaniesWorked']
corr = df_clean[cols_num].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=axes[1,1], annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, linewidths=0.5)
axes[1,1].set_title('Correlaciones con Rotación')



# ── Gráfico 6: Antigüedad vs. tasa de rotación (curva de riesgo)
rot_años = df_clean.groupby('YearsAtCompany')['Attrition_bin'].mean() * 100
axes[1,2].plot(rot_años.index, rot_años.values, color='#E05C5C', 
               linewidth=2.5, marker='o', markersize=4)
axes[1,2].fill_between(rot_años.index, rot_años.values, alpha=0.2, color='#E05C5C')
axes[1,2].set_title('Curva de Riesgo: Rotación por Años en la Empresa')
axes[1,2].set_xlabel('Años en la empresa')
axes[1,2].set_ylabel('% Rotación')
axes[1,2].axvline(x=1, color='gray', linestyle='--', alpha=0.7, label='Año crítico')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('eda_rotacion_empleados.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado como 'eda_rotacion_empleados.png'")


# ============================================================
# PASO 5: Cuantificar el impacto en $
# ============================================================

# Supuestos estándar de industria para costo de reemplazo
COSTO_REEMPLAZO_FACTOR = {
    'bajo':   0.50,   # 50% del salario anual (posiciones junior)
    'medio':  1.00,   # 100% del salario anual (posiciones medias)
    'alto':   2.00    # 200% del salario anual (posiciones senior/especialistas)
}


# Clasificar empleados por nivel de ingreso
def clasificar_nivel(ingreso_mensual):
    if ingreso_mensual < 3000:
        return 'bajo'
    elif ingreso_mensual < 8000:
        return 'medio'
    else:
        return 'alto'

df_clean['nivel_costo'] = df_clean['MonthlyIncome'].apply(clasificar_nivel)


# Calcular costo por empleado que rotó
df_rotacion = df_clean[df_clean['Attrition_bin'] == 1].copy()
df_rotacion['salario_anual'] = df_rotacion['MonthlyIncome'] * 12
df_rotacion['factor_costo'] = df_rotacion['nivel_costo'].map(COSTO_REEMPLAZO_FACTOR)
df_rotacion['costo_reemplazo'] = df_rotacion['salario_anual'] * df_rotacion['factor_costo']


# Resumen ejecutivo
costo_total = df_rotacion['costo_reemplazo'].sum()
costo_promedio = df_rotacion['costo_reemplazo'].mean()
empleados_rotaron = len(df_rotacion)

print("=" * 55)
print("   IMPACTO FINANCIERO DE LA ROTACIÓN — RESUMEN")
print("=" * 55)
print(f"  Empleados que rotaron   : {empleados_rotaron:,}")
print(f"  Costo promedio/empleado : ${costo_promedio:,.0f} USD")
print(f"  COSTO TOTAL ESTIMADO    : ${costo_total:,.0f} USD")
print("=" * 55)


# Desglose por departamento
print("\nCosto por Departamento:")
costo_dpto = df_rotacion.groupby('Department').agg(
    empleados=('Attrition_bin', 'count'),
    costo_total=('costo_reemplazo', 'sum'),
    costo_promedio=('costo_reemplazo', 'mean')
).sort_values('costo_total', ascending=False)
costo_dpto['costo_total'] = costo_dpto['costo_total'].apply(lambda x: f'${x:,.0f}')
costo_dpto['costo_promedio'] = costo_dpto['costo_promedio'].apply(lambda x: f'${x:,.0f}')
print(costo_dpto.to_string())





# ============================================================
# PASO 6: Exportar datasets limpios para Power BI
# ============================================================


# Dataset principal limpio
df_clean.to_csv('hr_attrition_clean.csv', index=False, encoding='utf-8-sig')



# Dataset solo rotación con costos (para el dashboard financiero)
df_rotacion[['Department', 'JobRole', 'MonthlyIncome', 'salario_anual',
             'costo_reemplazo', 'nivel_costo', 'YearsAtCompany',
             'JobSatisfaction', 'OverTime']].to_csv(
    'hr_costo_rotacion.csv', index=False, encoding='utf-8-sig')



# Tabla resumen para KPI cards en Power BI
resumen_kpis = pd.DataFrame({
    'KPI': ['Total Empleados', 'Tasa de Rotación (%)', 
            'Empleados Rotaron', 'Costo Total Rotación (USD)',
            'Costo Promedio por Empleado (USD)'],
    'Valor': [len(df_clean), round(tasa_rotacion, 1),
              empleados_rotaron, round(costo_total, 0),
              round(costo_promedio, 0)]
})
resumen_kpis.to_csv('hr_kpis_summary.csv', index=False, encoding='utf-8-sig')

print("✅ Archivos exportados:")
print("   → hr_attrition_clean.csv")
print("   → hr_costo_rotacion.csv")
print("   → hr_kpis_summary.csv")