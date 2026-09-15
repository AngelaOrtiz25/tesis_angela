# pipeline_adveccion_lluvia_automatico.py
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# CONFIGURACIÓN PARA LLUVIA
# -------------------------
DATA_CSV = r"C:/Users/yanel/OneDrive/Desktop/thesis/data/datos_meteorologicos_rutaaeroprotec.csv"
SAMPLING_MIN = 10
HORIZON_MIN = 60

# PARÁMETROS PARA LLUVIA
BLOCK_MIN = 60      # Bloques grandes (3 horas)
B_BOOT = 200

# VELOCIDAD DE ADVECCIÓN (misma que para otras variables)
V_ADVECCION_FIJA = 3.19  # m/s

# UMBRALES QUE SE AJUSTARÁN AUTOMÁTICAMENTE
UMBRAL_LLUVIA_DETECTABLE = 0.1  # mm/h - mínimo detectable (fijo)
# Los otros umbrales se calcularán automáticamente

# -------------------------
# Waypoints
# -------------------------
lats = [
    16.73060865, 16.7226902, 16.7133594, 16.70487338, 16.69703024,
    16.68941651, 16.68168956, 16.67422877, 16.66781532, 16.66485815,
    16.65701353, 16.6509456, 16.64648601, 16.64423022, 16.64515588,
    16.64196802, 16.63295291, 16.62635893, 16.62150832, 16.62270243,
    16.61895367, 16.61437756, 16.60970223, 16.60538093, 16.59665288,
    16.5870874, 16.58306009, 16.57996364, 16.57169214, 16.563521
]
lons = [
    -93.16834428, -93.17307527, -93.17585234, -93.18056029, -93.17821738,
    -93.17172057, -93.16535898, -93.1586827, -93.15091448, -93.14151255,
    -93.13531481, -93.12728524, -93.1182207, -93.10822403, -93.09804463,
    -93.08846331, -93.08458424, -93.07729853, -93.06907784, -93.05910324,
    -93.04960764, -93.04055895, -93.03943496, -93.04170614, -93.03908702,
    -93.0368152, -93.02916719, -93.02017621, -93.01496569, -93.013548
]

locations = [f"Punto {i}" for i in range(1, len(lats)+1)]

# -------------------------
# FUNCIONES GEOMÉTRICAS
# -------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

def bearing_deg(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlambda)
    br = math.degrees(math.atan2(x, y))
    return (br + 360) % 360

# -------------------------
# 1) CARGAR DATOS Y ANALIZAR DISTRIBUCIÓN
# -------------------------
print("="*70)
print("PROCESANDO DATOS DE PRECIPITACIÓN")
print("="*70)

df = pd.read_csv(DATA_CSV)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Verificar qué columna de lluvia tenemos
if 'rain_1h' in df.columns:
    rain_column = 'rain_1h'
    print("✓ Usando columna: 'rain_1h'")
elif 'rain_3h' in df.columns:
    rain_column = 'rain_3h'
    print("✓ Usando columna: 'rain_3h' (convertiendo a mm/h)")
else:
    raise ValueError("No se encontró columna de lluvia (rain_1h o rain_3h)")

# Extraer datos de lluvia por ubicación
rain_data_dict = {}
for loc in locations:
    sub_df = df[df['location'] == loc][['timestamp', rain_column]]
    if not sub_df.empty:
        rain_data_dict[loc] = sub_df.set_index('timestamp')[rain_column]
    else:
        print(f"⚠️  Advertencia: No hay datos para {loc}")

# Crear DataFrame con todos los datos
df_rain_all = pd.DataFrame(rain_data_dict)
df_rain_all = df_rain_all.sort_index()

# Si es rain_3h, convertir a mm/h aproximado
if rain_column == 'rain_3h':
    print("  Convirtiendo rain_3h a mm/h (dividiendo por 3)")
    df_rain_all = df_rain_all / 3

print(f"\nForma del dataset: {df_rain_all.shape}")
print(f"Período: {df_rain_all.index[0]} a {df_rain_all.index[-1]}")
print(f"Intervalo: {SAMPLING_MIN} minutos")

# -------------------------
# ANÁLISIS AUTOMÁTICO DE DISTRIBUCIÓN DE LLUVIA
# -------------------------
print("\n" + "="*50)
print("ANÁLISIS AUTOMÁTICO DE DISTRIBUCIÓN DE LLUVIA")
print("="*50)

# Obtener todos los valores de lluvia
all_rain_values = df_rain_all.values.flatten()
valid_rain_values = all_rain_values[~np.isnan(all_rain_values)]

# Estadísticas básicas
total_samples = len(valid_rain_values)
samples_with_rain = (valid_rain_values > UMBRAL_LLUVIA_DETECTABLE).sum()
percentage_with_rain = (samples_with_rain / total_samples) * 100 if total_samples > 0 else 0

print(f"Total de muestras: {total_samples:,}")
print(f"Muestras con lluvia (>0.1 mm/h): {samples_with_rain:,} ({percentage_with_rain:.1f}%)")

if samples_with_rain > 0:
    rain_only_values = valid_rain_values[valid_rain_values > UMBRAL_LLUVIA_DETECTABLE]
    
    print(f"\nESTADÍSTICAS DE LLUVIA (cuando hay precipitación):")
    print(f"  Mínima: {np.min(rain_only_values):.3f} mm/h")
    print(f"  Máxima: {np.max(rain_only_values):.3f} mm/h")
    print(f"  Media: {np.mean(rain_only_values):.3f} mm/h")
    print(f"  Mediana: {np.median(rain_only_values):.3f} mm/h")
    print(f"  Percentil 25: {np.percentile(rain_only_values, 25):.3f} mm/h")
    print(f"  Percentil 75: {np.percentile(rain_only_values, 75):.3f} mm/h")
    print(f"  Percentil 90: {np.percentile(rain_only_values, 90):.3f} mm/h")
    print(f"  Percentil 95: {np.percentile(rain_only_values, 95):.3f} mm/h")
    
    # CALCULAR UMBRALES ADAPTATIVOS
    # Basado en percentiles de los datos cuando hay lluvia
    if len(rain_only_values) >= 10:  # Necesitamos suficientes muestras
        # Definir categorías basadas en percentiles
        p25 = np.percentile(rain_only_values, 25)
        p50 = np.percentile(rain_only_values, 50)  # Mediana
        p75 = np.percentile(rain_only_values, 75)
        p90 = np.percentile(rain_only_values, 90)
        
        # Ajustar umbrales dinámicamente
        UMBRAL_LLUVIA_LIGERA = max(UMBRAL_LLUVIA_DETECTABLE, p25)
        UMBRAL_LLUVIA_MODERADA = max(UMBRAL_LLUVIA_LIGERA * 2, p50)
        UMBRAL_LLUVIA_FUERTE = max(UMBRAL_LLUVIA_MODERADA * 2, p75)
        UMBRAL_LLUVIA_EXTREMA = max(UMBRAL_LLUVIA_FUERTE * 1.5, p90)
        
        print(f"\nUMBRALES CALCULADOS AUTOMÁTICAMENTE:")
        print(f"  Lluvia Ligera: > {UMBRAL_LLUVIA_LIGERA:.3f} mm/h")
        print(f"  Lluvia Moderada: > {UMBRAL_LLUVIA_MODERADA:.3f} mm/h")
        print(f"  Lluvia Fuerte: > {UMBRAL_LLUVIA_FUERTE:.3f} mm/h")
        print(f"  Lluvia Extrema: > {UMBRAL_LLUVIA_EXTREMA:.3f} mm/h")
    else:
        print("\n⚠️  Pocas muestras con lluvia. Usando umbrales por defecto.")
        # Umbrales por defecto (escala típica)
        UMBRAL_LLUVIA_LIGERA = 0.5
        UMBRAL_LLUVIA_MODERADA = 2.0
        UMBRAL_LLUVIA_FUERTE = 5.0
        UMBRAL_LLUVIA_EXTREMA = 10.0
else:
    print("\n⚠️  NO HAY DATOS CON LLUVIA SIGNIFICATIVA EN EL HISTÓRICO")
    print("  Todos los valores son 0 o menores al umbral detectable")
    # Umbrales mínimos
    UMBRAL_LLUVIA_LIGERA = 0.5
    UMBRAL_LLUVIA_MODERADA = 2.0
    UMBRAL_LLUVIA_FUERTE = 5.0
    UMBRAL_LLUVIA_EXTREMA = 10.0

# Función de clasificación adaptativa
def classify_intensity_adaptive(mmh):
    if mmh <= UMBRAL_LLUVIA_DETECTABLE:
        return 'SIN LLUVIA'
    elif mmh <= UMBRAL_LLUVIA_LIGERA:
        return 'TRACE'  # Trazas
    elif mmh <= UMBRAL_LLUVIA_MODERADA:
        return 'LIGERA'
    elif mmh <= UMBRAL_LLUVIA_FUERTE:
        return 'MODERADA'
    elif mmh <= UMBRAL_LLUVIA_EXTREMA:
        return 'FUERTE'
    else:
        return 'EXTREMA'

# -------------------------
# 2) CONSTRUIR RUTA
# -------------------------
coords = list(zip(lats, lons))
dists = [0.0]
for i in range(1, len(coords)):
    dists.append(haversine_m(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1]))
s = np.cumsum(dists)

bearings = []
for i in range(len(coords)-1):
    bearings.append(math.radians(bearing_deg(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])))
bearings.append(bearings[-1])

route_table = pd.DataFrame({
    'location': locations, 
    'lat': lats, 
    'lon': lons, 
    's_m': s, 
    'bearing_rad': bearings
})

# -------------------------
# 3) AGRUPAR POR SEGMENTOS
# -------------------------
# Usar menos segmentos para lluvia (más variable espacialmente)
n_segments = min(10, len(locations))
loc_idx = np.arange(len(locations))
groups = np.array_split(loc_idx, n_segments)

segment_names = [f"seg_{i+1}" for i in range(len(groups))]
df_segments = pd.DataFrame(index=df_rain_all.index)

for name, inds in zip(segment_names, groups):
    locs_in = [locations[i] for i in inds]
    # Usar promedio de los puntos en el segmento
    df_segments[name] = df_rain_all[locs_in].mean(axis=1, skipna=True)

print(f"\nAgrupado en {n_segments} segmentos")
print(f"Forma de datos por segmentos: {df_segments.shape}")

# Rellenar NaN con interpolación limitada
df_segments_filled = df_segments.interpolate(limit=3).fillna(0)

# Suavizado simple (mediana para robustez ante outliers)
window_samples = max(3, int(60 / SAMPLING_MIN))  # Ventana de 60 minutos
df_segments_smooth = df_segments_filled.rolling(
    window=window_samples, min_periods=1, center=True
).median()

# -------------------------
# 4) ANÁLISIS DE PATRONES ESPACIALES
# -------------------------
print("\n" + "="*50)
print("ANÁLISIS DE PATRONES ESPACIALES ACTUALES")
print("="*50)

seg_positions = np.array([route_table['s_m'].values[list(g)].mean() for g in groups])
last_obs = df_segments_smooth.iloc[-1, :].values

print("\nESTADO ACTUAL POR SEGMENTO:")
for i, (seg_name, pos, value) in enumerate(zip(segment_names, seg_positions, last_obs)):
    categoria = classify_intensity_adaptive(value)
    print(f"  {seg_name} (pos={pos:.0f}m): {value:.3f} mm/h [{categoria}]")

# Distribución actual
categorias_actuales = [classify_intensity_adaptive(v) for v in last_obs]
categorias_unicas = set(categorias_actuales)
print(f"\nDistribución actual: {', '.join([f'{cat}: {categorias_actuales.count(cat)}' for cat in categorias_unicas])}")

# -------------------------
# 5) PRONÓSTICO POR ADVECCIÓN
# -------------------------
print("\n" + "="*50)
print("PRONÓSTICO POR ADVECCIÓN")
print("="*50)

H_seconds = HORIZON_MIN * 60
v_final = V_ADVECCION_FIJA

print(f"Horizonte: {HORIZON_MIN} minutos ({H_seconds} segundos)")
print(f"Velocidad de advección: {v_final:.2f} m/s")
print(f"Distancia advectada: {v_final * H_seconds:.0f} m")

def forecast_rain_adaptive(i, last_val, seg_positions, df_smooth, v, alpha=0.8):
    """Pronóstico adaptativo para lluvia"""
    s_up = seg_positions[i] - v * H_seconds
    
    # Si la posición upstream está antes del primer segmento
    if s_up <= seg_positions[0]:
        adv_val = df_smooth.iloc[-1, 0]
    else:
        # Encontrar los segmentos que enmarcan la posición upstream
        j = np.searchsorted(seg_positions, s_up) - 1
        j = max(0, min(j, len(seg_positions) - 2))
        
        s0, s1 = seg_positions[j], seg_positions[j+1]
        val0 = df_smooth.iloc[-1, j]
        val1 = df_smooth.iloc[-1, j+1]
        
        # Interpolación lineal
        frac = (s_up - s0) / (s1 - s0) if (s1 != s0) else 0.0
        adv_val = val0 + frac * (val1 - val0)
    
    # Si actualmente hay lluvia y upstream también, dar más peso a advección
    if last_val > UMBRAL_LLUVIA_DETECTABLE and adv_val > UMBRAL_LLUVIA_DETECTABLE:
        alpha_adaptive = 0.9  # Más peso a advección cuando hay sistemas activos
    else:
        alpha_adaptive = 0.7  # Menos peso cuando no hay sistemas claros
    
    forecast = alpha_adaptive * adv_val + (1 - alpha_adaptive) * last_val
    
    # No permitir valores negativos
    return max(0, forecast)

# Realizar pronósticos
forecasts = []
for i, last_val in enumerate(last_obs):
    forecast_val = forecast_rain_adaptive(i, last_val, seg_positions, df_segments_smooth, v_final)
    forecasts.append(forecast_val)

# -------------------------
# 6) BOOTSTRAP PARA INCERTIDUMBRE
# -------------------------
print("\nCalculando incertidumbre con bootstrap...")

def block_bootstrap_rain_adaptive(df_smooth, seg_positions, B=200, block_minutes=BLOCK_MIN):
    n = df_smooth.shape[0]
    block_size = max(1, int(block_minutes / SAMPLING_MIN))
    n_blocks = int(np.ceil(n / block_size))
    
    boot_forecasts = np.zeros((B, len(seg_positions)))
    rng = np.random.default_rng(12345)
    
    for b in range(B):
        # Remuestreo por bloques
        idx_blocks = rng.integers(0, n_blocks, size=n_blocks)
        indices = []
        for bl in idx_blocks:
            start = bl * block_size
            end = min(n, start + block_size)
            indices.extend(list(range(start, end)))
        idxs = np.array(indices)[:n]
        
        # Datos remuestreados
        M_boot = df_smooth.iloc[idxs, :].reset_index(drop=True)
        last_vals_boot = M_boot.iloc[-1, :].values
        
        # Pronóstico para cada segmento
        for i in range(len(seg_positions)):
            boot_forecasts[b, i] = forecast_rain_adaptive(
                i, last_vals_boot[i], seg_positions, M_boot, v_final
            )
    
    return boot_forecasts

# Ejecutar bootstrap
boot_forecasts = block_bootstrap_rain_adaptive(df_segments_smooth, seg_positions, B=B_BOOT, block_minutes=BLOCK_MIN)

# Calcular percentiles
p05 = np.percentile(boot_forecasts, 5, axis=0)
p50 = np.percentile(boot_forecasts, 50, axis=0)
p95 = np.percentile(boot_forecasts, 95, axis=0)

# -------------------------
# 7) RESULTADOS
# -------------------------
df_results = pd.DataFrame({
    'segment': segment_names,
    's_m': seg_positions,
    'last_obs_mmh': last_obs,
    'forecast_mmh': forecasts,
    'change_mmh': np.array(forecasts) - last_obs,
    'p05_mmh': p05,
    'p50_mmh': p50,
    'p95_mmh': p95,
    'uncertainty_mmh': p95 - p05
})

# Clasificar por intensidad usando umbrales adaptativos
df_results['current_category'] = df_results['last_obs_mmh'].apply(classify_intensity_adaptive)
df_results['forecast_category'] = df_results['p50_mmh'].apply(classify_intensity_adaptive)
df_results['worst_case_category'] = df_results['p95_mmh'].apply(classify_intensity_adaptive)

# Guardar resultados
output_file = "forecasts_lluvia_adaptativo_60min.csv"
df_results.to_csv(output_file, index=False, float_format='%.4f')
print(f"\n✓ Resultados guardados en: {output_file}")

# Guardar también los umbrales usados
umbrales_df = pd.DataFrame({
    'umbral': ['DETECTABLE', 'LIGERA', 'MODERADA', 'FUERTE', 'EXTREMA'],
    'valor_mmh': [UMBRAL_LLUVIA_DETECTABLE, UMBRAL_LLUVIA_LIGERA, 
                  UMBRAL_LLUVIA_MODERADA, UMBRAL_LLUVIA_FUERTE, UMBRAL_LLUVIA_EXTREMA]
})
umbrales_df.to_csv("umbrales_lluvia_calculados.csv", index=False)

# -------------------------
# 8) VISUALIZACIÓN
# -------------------------
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Gráfico 1: Distribución histórica de lluvia
ax1 = axes[0, 0]
if samples_with_rain > 0:
    ax1.hist(rain_only_values, bins=30, edgecolor='black', alpha=0.7)
    # Añadir líneas de umbrales
    colors = ['green', 'yellow', 'orange', 'red']
    umbrales = [UMBRAL_LLUVIA_LIGERA, UMBRAL_LLUVIA_MODERADA, 
                UMBRAL_LLUVIA_FUERTE, UMBRAL_LLUVIA_EXTREMA]
    labels = ['Ligera', 'Moderada', 'Fuerte', 'Extrema']
    
    for umbral, color, label in zip(umbrales, colors, labels):
        ax1.axvline(x=umbral, color=color, linestyle='--', alpha=0.7, 
                   label=f'{label}: {umbral:.3f} mm/h')
    
    ax1.set_xlabel('Intensidad de lluvia (mm/h)')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución histórica de lluvia (cuando llueve)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
else:
    ax1.text(0.5, 0.5, 'No hay datos históricos\ncon lluvia significativa', 
             ha='center', va='center', transform=ax1.transAxes, fontsize=12)
    ax1.set_title('Distribución histórica de lluvia')

# Gráfico 2: Pronóstico vs Observación
ax2 = axes[0, 1]
x_pos = np.arange(len(segment_names))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, df_results['last_obs_mmh'], width, 
                label='Observado', color='skyblue', alpha=0.7)
bars2 = ax2.bar(x_pos + width/2, df_results['p50_mmh'], width, 
                label='Pronóstico (p50)', color='lightgreen', alpha=0.7)

# Añadir barras de error para incertidumbre
ax2.errorbar(x_pos + width/2, df_results['p50_mmh'], 
             yerr=[df_results['p50_mmh'] - df_results['p05_mmh'], 
                   df_results['p95_mmh'] - df_results['p50_mmh']],
             fmt='none', ecolor='black', capsize=5, capthick=1, alpha=0.7)

ax2.set_xlabel('Segmento')
ax2.set_ylabel('Intensidad (mm/h)')
ax2.set_title('Lluvia: Observado vs Pronóstico (60 min)')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(segment_names, rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Gráfico 3: Cambios pronosticados
ax3 = axes[1, 0]
colors = ['red' if x < 0 else 'green' for x in df_results['change_mmh']]
bars_change = ax3.bar(df_results['segment'], df_results['change_mmh'], 
                      color=colors, alpha=0.7)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_xlabel('Segmento')
ax3.set_ylabel('Cambio (mm/h)')
ax3.set_title('Cambio Pronosticado (Pronóstico - Observado)')
ax3.set_xticklabels(df_results['segment'], rotation=45)
ax3.grid(True, alpha=0.3, axis='y')

# Añadir valores solo si cambio significativo
for bar, change in zip(bars_change, df_results['change_mmh']):
    height = bar.get_height()
    if abs(change) > 0.01:  # Solo mostrar cambios > 0.01 mm/h
        ax3.text(bar.get_x() + bar.get_width()/2., 
                height + (0.05 if height >= 0 else -0.1),
                f'{change:+.3f}', ha='center', 
                va='bottom' if height >= 0 else 'top', fontsize=8)

# Gráfico 4: Mapa de riesgo
ax4 = axes[1, 1]
# Crear matriz de colores basada en la categoría del peor caso
color_map = {
    'SIN LLUVIA': 'gray', 
    'TRACE': 'lightgray',
    'LIGERA': 'lightblue', 
    'MODERADA': 'orange', 
    'FUERTE': 'red',
    'EXTREMA': 'darkred'
}
colors_risk = [color_map[cat] for cat in df_results['worst_case_category']]

bars_risk = ax4.bar(df_results['segment'], df_results['p95_mmh'], 
                    color=colors_risk, alpha=0.7)
ax4.set_xlabel('Segmento')
ax4.set_ylabel('Peor caso (P95) mm/h')
ax4.set_title('Peor Escenario (Percentil 95)')
ax4.set_xticklabels(df_results['segment'], rotation=45)
ax4.grid(True, alpha=0.3, axis='y')

# Añadir líneas de umbrales
for umbral, color, label in zip(umbrales, colors, labels):
    ax4.axhline(y=umbral, color=color, linestyle='--', 
                alpha=0.5, linewidth=1)

# Gráfico 5: Distribución de categorías actual vs pronóstico
ax5 = axes[2, 0]
categorias_orden = ['SIN LLUVIA', 'TRACE', 'LIGERA', 'MODERADA', 'FUERTE', 'EXTREMA']
counts_current = [sum(df_results['current_category'] == cat) for cat in categorias_orden]
counts_forecast = [sum(df_results['forecast_category'] == cat) for cat in categorias_orden]

x = np.arange(len(categorias_orden))
width = 0.35

bars_curr = ax5.bar(x - width/2, counts_current, width, 
                    label='Actual', color='skyblue', alpha=0.7)
bars_fore = ax5.bar(x + width/2, counts_forecast, width, 
                    label='Pronóstico', color='lightgreen', alpha=0.7)

ax5.set_xlabel('Categoría de Lluvia')
ax5.set_ylabel('Número de Segmentos')
ax5.set_title('Distribución de Categorías (Actual vs Pronóstico)')
ax5.set_xticks(x)
ax5.set_xticklabels(categorias_orden, rotation=45, fontsize=9)
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Añadir valores
for bars in [bars_curr, bars_fore]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)

# Gráfico 6: Resumen de umbrales usados
ax6 = axes[2, 1]
umbrales_nombres = ['Detectable', 'Ligera', 'Moderada', 'Fuerte', 'Extrema']
umbrales_valores = [UMBRAL_LLUVIA_DETECTABLE, UMBRAL_LLUVIA_LIGERA, 
                    UMBRAL_LLUVIA_MODERADA, UMBRAL_LLUVIA_FUERTE, UMBRAL_LLUVIA_EXTREMA]

bars_umbrales = ax6.bar(umbrales_nombres, umbrales_valores, 
                        color=['gray', 'lightblue', 'orange', 'red', 'darkred'], alpha=0.7)
ax6.set_xlabel('Categoría')
ax6.set_ylabel('Umbral (mm/h)')
ax6.set_title('Umbrales de Clasificación Usados')
ax6.set_xticklabels(umbrales_nombres, rotation=45)
ax6.grid(True, alpha=0.3, axis='y')

# Añadir valores
for bar, valor in zip(bars_umbrales, umbrales_valores):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{valor:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plot_file = "forecast_lluvia_adaptativo_summary.png"
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"✓ Gráfico guardado en: {plot_file}")
plt.show()

# -------------------------
# 9) RESUMEN EJECUTIVO ADAPTATIVO
# -------------------------
print("\n" + "="*80)
print("RESUMEN EJECUTIVO - PRONÓSTICO ADAPTATIVO DE LLUVIA")
print("="*80)

print(f"\nANÁLISIS HISTÓRICO:")
print(f"  Total de muestras: {total_samples:,}")
print(f"  Muestras con lluvia: {samples_with_rain:,} ({percentage_with_rain:.1f}%)")
if samples_with_rain > 0:
    print(f"  Rango histórico: {np.min(rain_only_values):.3f} - {np.max(rain_only_values):.3f} mm/h")
    print(f"  Mediana histórica: {np.median(rain_only_values):.3f} mm/h")

print(f"\nUMBRALES CALCULADOS:")
print(f"  Detectable: > {UMBRAL_LLUVIA_DETECTABLE:.3f} mm/h")
print(f"  Lluvia Ligera: > {UMBRAL_LLUVIA_LIGERA:.3f} mm/h")
print(f"  Lluvia Moderada: > {UMBRAL_LLUVIA_MODERADA:.3f} mm/h")
print(f"  Lluvia Fuerte: > {UMBRAL_LLUVIA_FUERTE:.3f} mm/h")
print(f"  Lluvia Extrema: > {UMBRAL_LLUVIA_EXTREMA:.3f} mm/h")

print(f"\nESTADO ACTUAL:")
current_max = np.max(last_obs)
current_max_cat = classify_intensity_adaptive(current_max)
print(f"  Intensidad máxima actual: {current_max:.3f} mm/h [{current_max_cat}]")
print(f"  Segmentos con lluvia: {(last_obs > UMBRAL_LLUVIA_DETECTABLE).sum()}/{n_segments}")

print(f"\nPRONÓSTICO (60 min):")
forecast_max = np.max(forecasts)
forecast_max_cat = classify_intensity_adaptive(forecast_max)
print(f"  Intensidad máxima pronosticada: {forecast_max:.3f} mm/h [{forecast_max_cat}]")
print(f"  Segmentos con lluvia pronosticada: {(np.array(forecasts) > UMBRAL_LLUVIA_DETECTABLE).sum()}/{n_segments}")

print(f"\nINCERTIDUMBRE (P05-P95):")
print(f"  Rango de intensidad: {np.min(p05):.3f} a {np.max(p95):.3f} mm/h")
print(f"  Incertidumbre promedio: {np.mean(p95 - p05):.3f} mm/h")

print("\n" + "="*60)
print("RECOMENDACIONES PARA OPERACIÓN DE DRON")
print("="*60)

# Evaluar riesgo basado en el peor caso (P95)
max_p95 = np.max(p95)
max_p95_cat = classify_intensity_adaptive(max_p95)

if max_p95_cat in ['SIN LLUVIA', 'TRACE']:
    recomendacion = "✅ CONDICIONES ÓPTIMAS - Puede operar sin restricciones"
    nivel_riesgo = "MUY BAJO"
elif max_p95_cat == 'LIGERA':
    recomendacion = "✅ CONDICIONES ACEPTABLES - Puede operar con normalidad"
    nivel_riesgo = "BAJO"
elif max_p95_cat == 'MODERADA':
    recomendacion = "⚠️  PRECAUCIÓN - Lluvia moderada posible, monitorear condiciones"
    nivel_riesgo = "MODERADO"
elif max_p95_cat == 'FUERTE':
    recomendacion = "❌ ALTO RIESGO - Posible lluvia fuerte, considerar posponer vuelo"
    nivel_riesgo = "ALTO"
else:  # EXTREMA
    recomendacion = "❌ RIESGO EXTREMO - Posible lluvia extrema, NO VOLAR"
    nivel_riesgo = "EXTREMO"

print(f"\nNIVEL DE RIESGO BASADO EN PEOR ESCENARIO: {nivel_riesgo}")
print(f"CATEGORÍA MÁXIMA ESPERADA: {max_p95_cat} ({max_p95:.3f} mm/h)")
print(f"RECOMENDACIÓN: {recomendacion}")

# Segmentos problemáticos
problem_segments = df_results[df_results['worst_case_category'].isin(['MODERADA', 'FUERTE', 'EXTREMA'])]
if len(problem_segments) > 0:
    print(f"\nSEGMENTOS CON MAYOR RIESGO (categoría moderada o mayor):")
    for _, row in problem_segments.iterrows():
        print(f"  • {row['segment']}: {row['p95_mmh']:.3f} mm/h [{row['worst_case_category']}]")

print("\n" + "="*80)
print("Nota: Los umbrales se calcularon automáticamente basados en la")
print(f"distribución histórica de tus datos ({total_samples} muestras).")
print("="*80)