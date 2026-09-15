# pipeline_adveccion_temperatura.py
import numpy as np
import pandas as pd
import math
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# CONFIGURACIÓN / USUARIO - CAMBIOS PARA TEMPERATURA
# -------------------------
DATA_CSV = r"C:/Users/yanel/OneDrive/Desktop/thesis/data/datos_meteorologicos_rutaaeroprotec.csv"
SAMPLING_MIN = 10
HORIZON_MIN = 60
STEP_MINUTES = SAMPLING_MIN
SAMPLES_H = int(HORIZON_MIN / SAMPLING_MIN)

# PARÁMETROS AJUSTADOS PARA TEMPERATURA
RHO_MIN = 0.40        # umbral MÁS ALTO (temperatura tiene mayor correlación espacial)
FRAC_PAIRS_MIN = 0.6  # fracción MÁS ALTA
BLOCK_MIN = 90        # block bootstrap MÁS LARGO (90 min vs 60 min)
B_BOOT = 200

# USAR VELOCIDAD DE ADVECCIÓN YA CALCULADA DEL VIENTO
V_ADVECCION_FIJA = 3.19  # m/s (valor obtenido del análisis de viento)

# -------------------------
# Waypoints (MISMOS)
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
# FUNCIONES GEOMÉTRICAS (MISMAS)
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
# 1) LOAD DATA - CAMBIO CLAVE AQUÍ
# -------------------------
df = pd.read_csv(DATA_CSV)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

if 'location' not in df.columns:
    raise ValueError("Tu CSV debe contener columna 'location' con nombres de estaciones.")

# CAMBIO 1: Usar 'temp' en lugar de 'wind_speed' y 'wind_deg'
vars_needed = ['temp']  # ¡SOLO TEMPERATURA!

dfs = []
for loc in locations:
    sub = df[df['location']==loc][['timestamp'] + vars_needed].set_index('timestamp').rename(columns=lambda c: f"{loc}__{c}")
    dfs.append(sub)
    
df_wide = pd.concat(dfs, axis=1, join='inner').sort_index()
print("Forma de df_wide (temperatura):", df_wide.shape)
print("Columnas disponibles:", df_wide.columns.tolist()[:5])

# -------------------------
# 2) Construir ruta (MISMO)
# -------------------------
coords = list(zip(lats, lons))
dists = [0.0]
for i in range(1,len(coords)):
    dists.append(haversine_m(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1]))
s = np.cumsum(dists)

bearings = []
for i in range(len(coords)-1):
    bearings.append(math.radians(bearing_deg(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])))
bearings.append(bearings[-1])

route_table = pd.DataFrame({'location':locations, 'lat':lats, 'lon':lons, 's_m':s, 'bearing_rad':bearings})
print("Route table head:\n", route_table.head())

# -------------------------
# 3) CREAR MATRIZ DE TEMPERATURA - CAMBIO AQUÍ
# -------------------------
# Para temperatura no necesitamos convertir a u,v, solo extraer valores
temp_cols = [f"{loc}__temp" for loc in locations]
df_temp = df_wide[temp_cols].copy()
df_temp.columns = locations

# -------------------------
# 4) Aggregación por segmentos y suavizado temporal - CAMBIOS PARA TEMP
# -------------------------
n_segments = min(10, len(locations))
loc_idx = np.arange(len(locations))
groups = np.array_split(loc_idx, n_segments)

segment_names = [f"seg_{i+1}" for i in range(len(groups))]
df_segments = pd.DataFrame(index=df_temp.index)

for name, inds in zip(segment_names, groups):
    locs_in = [locations[i] for i in inds]
    df_segments[name] = df_temp[locs_in].mean(axis=1)

# CAMBIO 2: Suavizado con ventana MÁS LARGA para temperatura
window_samples = max(3, int(60 / SAMPLING_MIN))  # 60 min window (vs 30 min para viento)
df_segments_smooth = df_segments.rolling(window=window_samples, min_periods=1, center=True).mean()  # Usar mean en lugar de median

print("Segment matrix shape (temperatura):", df_segments_smooth.shape)
print("Temperatura promedio por segmento:", df_segments_smooth.iloc[-1,:].values)

# -------------------------
# 5) Intento 1: CCF entre segmentos (MISMO)
# -------------------------
max_lag_minutes = 180  # Aumentado para temperatura
max_lag_steps = int(max_lag_minutes / SAMPLING_MIN)

def cross_corr_lags(x, y, maxlag):
    lags = np.arange(-maxlag, maxlag+1)
    r = []
    for k in lags:
        if k < 0:
            a = x[:k]; b = y[-k:]
        elif k > 0:
            a = x[k:]; b = y[:-k]
        else:
            a = x; b = y
        if len(a) < 5:
            r.append(0.0)
        else:
            r.append(np.corrcoef(a, b)[0,1])
    return lags, np.array(r)

pair_results = []
for i in range(len(segment_names)-1):
    a = df_segments_smooth.iloc[:,i].values
    b = df_segments_smooth.iloc[:,i+1].values
    lags, r = cross_corr_lags(a, b, max_lag_steps)
    k_star = lags[np.nanargmax(r)]
    rho_max = np.nanmax(r)
    pair_results.append({'pair':(i,i+1), 'k_star':int(k_star), 'rho_max':float(rho_max)})
    
pairs_df = pd.DataFrame(pair_results)
print("Pairs CCF summary (temperatura):\n", pairs_df.head())

mask_good = (pairs_df['rho_max'].abs() > RHO_MIN) & (pairs_df['k_star'] != 0)
frac_good = mask_good.mean()
print(f"frac_good adjacent pairs: {frac_good:.2f} (threshold {FRAC_PAIRS_MIN})")

ccf_ok = frac_good >= FRAC_PAIRS_MIN

# CAMBIO 3: Usar velocidad fija en lugar de recalcular
v_final = V_ADVECCION_FIJA
method_used = "fixed_advection"

print("Para temperatura, usamos velocidad fija de:", v_final, "m/s")
print("Method selected:", method_used)

# -------------------------
# 6) Forecast advective nowcast - CAMBIOS PARA TEMPERATURA
# -------------------------
H_seconds = HORIZON_MIN * 60
seg_positions = np.array([route_table['s_m'].values[list(g)].mean() for g in groups])
last_obs = df_segments_smooth.iloc[-1,:].values

# AR coefficients para temperatura (generalmente cerca de 1.0)
seg_a = []
for col in df_segments_smooth.columns:
    ser = df_segments_smooth[col].dropna().values
    if len(ser) < 10:
        seg_a.append(np.nan)
    else:
        lr = LinearRegression().fit(ser[:-1].reshape(-1,1), ser[1:])
        seg_a.append(lr.coef_[0])
        
seg_a = np.array(seg_a)
# Para temperatura, AR(1) suele ser >0.9
seg_a[np.isnan(seg_a)] = 0.95  # Más alto que para viento

def advect_forecast_for_seg_temp(i, last_val, seg_positions, df_segments_smooth, v, alpha=0.4):
    # CAMBIO 4: alpha más alto (más peso a advección)
    s_up = seg_positions[i] - v * H_seconds
    
    if s_up <= seg_positions[0]:
        adv_val = df_segments_smooth.iloc[-1,0]
    else:
        j = np.searchsorted(seg_positions, s_up) - 1
        j = max(0, min(j, len(seg_positions)-2))
        s0, s1 = seg_positions[j], seg_positions[j+1]
        val0 = df_segments_smooth.iloc[-1,j]
        val1 = df_segments_smooth.iloc[-1,j+1]
        frac = (s_up - s0) / (s1 - s0) if (s1!=s0) else 0.0
        adv_val = val0 + frac*(val1 - val0)
    
    # AR(1) forecast
    y_ar = last_val
    a_loc = seg_a[i]
    for _ in range(SAMPLES_H):
        y_ar = a_loc * y_ar
    
    y_pers = last_val
    
    # CAMBIO 5: Pesos diferentes para temperatura
    # 40% AR(1), 20% persistencia, 40% advección
    return 0.4*y_ar + 0.2*y_pers + alpha*adv_val

forecasts = []
for i, last_val in enumerate(last_obs):
    f = advect_forecast_for_seg_temp(i, last_val, seg_positions, df_segments_smooth, v_final, alpha=0.4)
    forecasts.append(f)

df_fore = pd.DataFrame({
    'segment': segment_names, 
    's_m': seg_positions, 
    'last_obs': last_obs, 
    'forecast_60min': forecasts
})
print("\nPronóstico de temperatura por segmento:")
print(df_fore)

# -------------------------
# 7) Incertidumbre: block bootstrap - CON BLOQUES MÁS LARGOS
# -------------------------
def block_bootstrap_forecasts_temp(df_segments_smooth, seg_positions, B=200, block_minutes=BLOCK_MIN):
    n = df_segments_smooth.shape[0]
    block_size = max(1, int(block_minutes / SAMPLING_MIN))
    n_blocks = int(np.ceil(n / block_size))
    boot_forecasts = np.zeros((B, len(seg_positions)))
    rng = np.random.default_rng(12345)
    
    for b in range(B):
        idx_blocks = rng.integers(0, n_blocks, size=n_blocks)
        indices = []
        for bl in idx_blocks:
            start = bl*block_size
            end = min(n, start+block_size)
            indices.extend(list(range(start,end)))
        idxs = np.array(indices)[:n]
        M_boot = df_segments_smooth.iloc[idxs,:].reset_index(drop=True)
        
        # Usar velocidad fija
        v_boot = V_ADVECCION_FIJA
        
        last_vals_boot = M_boot.iloc[-1,:].values
        for i in range(len(seg_positions)):
            boot_forecasts[b,i] = advect_forecast_for_seg_temp(i, last_vals_boot[i], seg_positions, M_boot, v_boot, alpha=0.4)
            
    return boot_forecasts

print("Running bootstrap para temperatura...")
boot = block_bootstrap_forecasts_temp(df_segments_smooth, seg_positions, B=B_BOOT, block_minutes=BLOCK_MIN)

p05 = np.percentile(boot, 5, axis=0)
p50 = np.percentile(boot, 50, axis=0)
p95 = np.percentile(boot, 95, axis=0)

df_fore['p05'] = p05
df_fore['p50'] = p50
df_fore['p95'] = p95

# -------------------------
# 8) SAVE RESULTS & PLOTS - CAMBIAR NOMBRES
# -------------------------
df_fore.to_csv("forecasts_temperatura_60min_segments.csv", index=False)
print("Saved forecasts_temperatura_60min_segments.csv")

# Plot
plt.figure(figsize=(10,4))
plt.errorbar(df_fore['s_m'], df_fore['p50'], 
             yerr=[df_fore['p50']-df_fore['p05'], df_fore['p95']-df_fore['p50']], 
             fmt='o', label='forecast p50 (CI 5-95%)', capsize=5)
plt.scatter(df_fore['s_m'], df_fore['last_obs'], color='red', s=50, label='last obs')
plt.xlabel("Posición a lo largo de la ruta (m)")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title(f"Pronóstico de Temperatura - 60 min (v={v_final:.2f} m/s)")
plt.savefig("forecast_temperatura_segments_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved forecast_temperatura_segments_plot.png")

# -------------------------
# 9) RESUMEN PARA TEMPERATURA
# -------------------------
print("\n" + "="*50)
print("RESUMEN PRONÓSTICO DE TEMPERATURA")
print("="*50)
print(f"Velocidad de advección usada: {v_final:.2f} m/s")
print(f"Temperatura promedio actual: {np.mean(last_obs):.1f} °C")
print(f"Rango pronosticado (P05-P95): {np.min(p05):.1f} a {np.max(p95):.1f} °C")
print(f"Variación máxima esperada: {np.max(p95) - np.min(p05):.1f} °C")
print("="*50)