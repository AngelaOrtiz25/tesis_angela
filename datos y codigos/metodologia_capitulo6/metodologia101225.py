# pipeline_adveccion_nowcast.py
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
# CONFIGURACIÓN / USUARIO
# -------------------------
DATA_CSV = r"C:/Users/yanel/OneDrive/Desktop/thesis/data/datos_meteorologicos_rutaaeroprotec.csv"   # tu CSV con timestamp, location, wind_speed, wind_deg...
SAMPLING_MIN = 10
HORIZON_MIN = 60
STEP_MINUTES = SAMPLING_MIN
SAMPLES_H = int(HORIZON_MIN / SAMPLING_MIN)
RHO_MIN = 0.30        # umbral mínimo correlación para aceptar lag
FRAC_PAIRS_MIN = 0.5  # fracción de pares que deben pasar el umbral
BLOCK_MIN = 60        # block bootstrap (minutos)
B_BOOT = 200          # bootstrap replicates

# -------------------------
# Pega tus waypoints aquí
# (usa exactamente tus coordenadas reales)
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

# Nombre de ubicaciones esperadas en el CSV (ej. "Punto 1", "Punto 2", ...).
# Si tu columna 'location' usa otros nombres, adapta este vector o la
# correspondencia antes de correr.
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
    # bearing from point1 to point2, degrees from north clockwise
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlambda)
    br = math.degrees(math.atan2(x, y))
    return (br + 360) % 360

# -------------------------
# 1) LOAD DATA
# -------------------------
df = pd.read_csv(DATA_CSV)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Ensure location names match
if 'location' not in df.columns:
    raise ValueError("Tu CSV debe contener columna 'location' con nombres de estaciones.")

# Pivot to wide: rows = timestamps, columns = locations for wind_speed and wind_deg
# Keep only timestamps present for all locations by inner join (synchronization)
vars_needed = ['wind_speed','wind_deg']
dfs = []
for loc in locations:
    sub = df[df['location']==loc][['timestamp'] + vars_needed].set_index('timestamp').rename(columns=lambda c: f"{loc}__{c}")
    dfs.append(sub)
df_wide = pd.concat(dfs, axis=1, join='inner').sort_index()
print("Forma de df_wide:", df_wide.shape)

# -------------------------
# 2) Construir ruta curvilínea y s_i, bearing_i
# -------------------------
coords = list(zip(lats, lons))
# distances between consecutive waypoints
dists = [0.0]
for i in range(1,len(coords)):
    dists.append(haversine_m(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1]))
s = np.cumsum(dists)  # s positions at waypoints
# compute bearing per segment (assign to waypoint i the bearing to next)
bearings = []
for i in range(len(coords)-1):
    bearings.append(math.radians(bearing_deg(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])))
# for last point duplicate previous bearing
bearings.append(bearings[-1])

route_table = pd.DataFrame({'location':locations, 'lat':lats, 'lon':lons, 's_m':s, 'bearing_rad':bearings})
print("Route table head:\n", route_table.head())

# -------------------------
# 3) calcular u,v y proyección along/cross por timestamp y por punto
# -------------------------
# Build u/v wideframes
def wind_to_components(speed, deg_from):
    # convert direction FROM to TO
    wind_to = (deg_from + 180) % 360
    theta = np.deg2rad(wind_to)
    u = speed * np.sin(theta)  # east
    v = speed * np.cos(theta)  # north
    return u, v

# Build dataframes for u and v
u_cols = []
v_cols = []
for loc in locations:
    speed_col = f"{loc}__wind_speed"
    deg_col = f"{loc}__wind_deg"
    u_vals, v_vals = wind_to_components(df_wide[speed_col].values, df_wide[deg_col].values)
    df_wide[f"{loc}__u"] = u_vals
    df_wide[f"{loc}__v"] = v_vals
    u_cols.append(f"{loc}__u"); v_cols.append(f"{loc}__v")

# project into along and cross (along positive towards route end)
for loc in locations:
    beta = route_table.loc[route_table['location']==loc,'bearing_rad'].values[0]
    ucol = f"{loc}__u"; vcol = f"{loc}__v"
    df_wide[f"{loc}__along"] = df_wide[ucol] * math.sin(beta) + df_wide[vcol] * math.cos(beta)
    df_wide[f"{loc}__cross"] = -df_wide[ucol] * math.cos(beta) + df_wide[vcol] * math.sin(beta)

# create matrix along: rows time, cols segments (we'll later aggregate segments)
along_cols = [f"{loc}__along" for loc in locations]
df_along = df_wide[along_cols].copy()
df_along.columns = locations
# -------------------------
# 4) Aggregación por segmentos y suavizado temporal
# -------------------------
# choose number of segments; tradeoff: fewer segments -> larger spatial scale
n_segments = min(10, len(locations))  # ejemplo: 10 segmentos
# partition locations into contiguous groups
loc_idx = np.arange(len(locations))
groups = np.array_split(loc_idx, n_segments)

segment_names = [f"seg_{i+1}" for i in range(len(groups))]
df_segments = pd.DataFrame(index=df_along.index)

for name, inds in zip(segment_names, groups):
    locs_in = [locations[i] for i in inds]
    df_segments[name] = df_along[locs_in].mean(axis=1)  # promedio espacial
# smoothing temporal (rolling median)
window_samples = max(3, int(30 / SAMPLING_MIN))  # e.g., 30 min window
df_segments_smooth = df_segments.rolling(window=window_samples, min_periods=1, center=True).median()

print("Segment matrix shape:", df_segments_smooth.shape)

# -------------------------
# 5) Intento 1: CCF entre segmentos (pairwise adjacents)
# -------------------------
max_lag_minutes = 120
max_lag_steps = int(max_lag_minutes / SAMPLING_MIN)

def cross_corr_lags(x, y, maxlag):
    # return lags and correlation array (Pearson)
    lags = np.arange(-maxlag, maxlag+1)
    r = []
    for k in lags:
        if k < 0:
            a = x[:k]; b = y[-k:]
        elif k > 0:
            a = x[k:]; b = y[:-k]
        else:
            a = x; b = y
        # correlate only if length sufficient
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
print("Pairs CCF summary:\n", pairs_df.head())

# Criterio: aceptar CCF si fraction of adjacent pairs with |rho_max|>RHO_MIN and k_star != 0 >= FRAC_PAIRS_MIN
mask_good = (pairs_df['rho_max'].abs() > RHO_MIN) & (pairs_df['k_star'] != 0)
frac_good = mask_good.mean()
print(f"frac_good adjacent pairs: {frac_good:.2f} (threshold {FRAC_PAIRS_MIN})")

ccf_ok = frac_good >= FRAC_PAIRS_MIN

# If ok, compute t_ij from k_star and do WLS to get v
v_est_ccf = None
if ccf_ok:
    # build distances between adjacent segment centroids
    seg_s = []
    for inds in groups:
        # centroid s: mean s of locations in the segment
        seg_s.append(route_table['s_m'].values[list(inds)].mean())
    seg_s = np.array(seg_s)
    d_ij = []
    t_ij = []
    w_ij = []
    for r in pair_results:
        i,j = r['pair']
        d = abs(seg_s[j] - seg_s[i])
        t = abs(r['k_star'])*SAMPLING_MIN*60.0
        d_ij.append(d); t_ij.append(t); w_ij.append(abs(r['rho_max']))
    d_ij = np.array(d_ij); t_ij = np.array(t_ij); w_ij = np.array(w_ij)
    # WLS for t = m*d + c -> slope m
    W = np.diag(w_ij + 1e-6)
    X = np.vstack([d_ij, np.ones_like(d_ij)]).T
    try:
        beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ t_ij)
        m_hat, c_hat = beta[0], beta[1]
        if m_hat == 0:
            v_est_ccf = None
        else:
            v_est_ccf = 1.0 / m_hat  # in m/s
    except Exception as e:
        print("WLS falló:", e)
        v_est_ccf = None

    print("v_est_ccf (m/s):", v_est_ccf)

# -------------------------
# 6) Si CCF falla -> intento f-k (phase slope) sobre matrix seg x time
# -------------------------
v_est_fk = None
fk_ok = False
if not ccf_ok:
    print("CCF falló; intentando análisis f–k (phase slope)...")
    # build matrix: rows=segments, cols=time (we'll transpose so axis0=time)
    M = df_segments_smooth.values.T  # shape (n_segments, n_time)
    # detrend and window
    n_seg, n_time = M.shape
    # compute FFT in time for each segment (take limited freq band)
    fs = 60.0 / SAMPLING_MIN  # samples per hour (not necessary); better use Hz = 1/(SAMPLING_MIN*60)
    # use numpy fft for each segment
    # We will compute complex spectra and phase; choose a low-frequency band (periods 30-360 min)
    # convert to freq in Hz:
    dt_sec = SAMPLING_MIN * 60
    freqs = fftfreq(n_time, d=dt_sec)
    # positive freqs
    pos_idx = np.where(freqs>0)[0]
    # compute spectra
    S = np.fft.fft(M, axis=1)  # shape (n_seg, n_time)
    # pick a frequency with high coherence across segments
    coherences = []
    v_candidates = []
    for idx in pos_idx:
        # get phase as function of segment index
        phase = np.angle(S[:, idx])  # length n_seg
        # unwrap phase
        phase_un = np.unwrap(phase)
        # regress phase_un vs seg_s (centroid positions) to estimate slope
        seg_positions = np.array([route_table['s_m'].values[list(g)].mean() for g in groups])
        if np.ptp(phase_un) < 1e-6:
            continue
        lr = LinearRegression().fit(seg_positions.reshape(-1,1), phase_un)
        slope = lr.coef_[0]  # rad per meter
        omega = 2*np.pi*freqs[idx]  # rad/s
        if slope == 0:
            continue
        v_est = -omega / slope  # m/s
        v_candidates.append(v_est)
        # coherence heuristic: magnitude sum across segments
        mag = np.abs(S[:, idx]).mean()
        coherences.append((idx, v_est, mag))
    # pick robust median of candidates with positive magnitude
    if len(v_candidates) > 0:
        v_candidates = np.array(v_candidates)
        # filter reasonable values
        v_candidates = v_candidates[np.isfinite(v_candidates) & (np.abs(v_candidates) < 50)]
        if len(v_candidates)>0:
            v_est_fk = np.median(v_candidates)
            fk_ok = True
            print("v_est_fk (m/s):", v_est_fk)
        else:
            fk_ok = False
    else:
        fk_ok = False



# -------------------------
# 7) Si ambos fallan: fallback operativo (persistencia + AR1 + adv_corr)
# -------------------------
v_final = None
method_used = None

if ccf_ok and (v_est_ccf is not None):
    v_final = v_est_ccf
    method_used = "ccf"
elif fk_ok and (v_est_fk is not None):
    v_final = v_est_fk
    method_used = "fk"
else:
    method_used = "fallback"
    # estimate AR(1) per segment (mean across segments)
    a_vals = []
    for col in df_segments_smooth.columns:
        ser = df_segments_smooth[col].dropna().values
        if len(ser) < 10: continue
        X = ser[:-1].reshape(-1,1); yv = ser[1:]
        lr = LinearRegression().fit(X, yv)
        a_vals.append(lr.coef_[0])
    a_med = np.median(a_vals) if len(a_vals)>0 else 0.6
    # advective velocity estimate: use median of negative dy/dt scaled (regularized)
    # compute temporal derivative of spatial mean
    mean_series = df_segments_smooth.mean(axis=1).values
    mean_smooth = pd.Series(mean_series).rolling(max(3,int(30/SAMPLING_MIN)), center=True, min_periods=1).median().values
    dy_dt = np.gradient(mean_smooth, dt_sec)
    v_est_reg = -np.median(dy_dt) * 0.05  # very conservative scaling
    v_final = v_est_reg if np.isfinite(v_est_reg) else 0.0
    print("Fallback: a_med =", a_med, "v_est_reg (m/s) =", v_final)

print("Method selected:", method_used, "v_final (m/s):", v_final)

# -------------------------
# 8) Forecast advective nowcast 60 min for each segment
#   - advected_value: find s_up = s_seg - v_final * H_seconds, interpolate along segments
#   - blend with persistence and AR(1) if available
# -------------------------
H_seconds = HORIZON_MIN * 60
seg_positions = np.array([route_table['s_m'].values[list(g)].mean() for g in groups])
# last observed timeseries per segment
last_obs = df_segments_smooth.iloc[-1,:].values

# AR coefficients per segment (if possible), else use median a_med from fallback
seg_a = []
for col in df_segments_smooth.columns:
    ser = df_segments_smooth[col].dropna().values
    if len(ser) < 10:
        seg_a.append(np.nan)
    else:
        lr = LinearRegression().fit(ser[:-1].reshape(-1,1), ser[1:])
        seg_a.append(lr.coef_[0])
seg_a = np.array(seg_a)
seg_a[np.isnan(seg_a)] = a_med if 'a_med' in locals() else 0.6

def advect_forecast_for_seg(i, last_val, seg_positions, df_segments_smooth, v, alpha=0.2):
    # compute upstream position
    s_up = seg_positions[i] - v * H_seconds
    # if upstream outside domain -> fallback to persistence
    if s_up <= seg_positions[0]:
        adv_val = df_segments_smooth.iloc[-1,0]
    else:
        # find bracketing segments
        j = np.searchsorted(seg_positions, s_up) - 1
        j = max(0, min(j, len(seg_positions)-2))
        s0, s1 = seg_positions[j], seg_positions[j+1]
        val0 = df_segments_smooth.iloc[-1,j]
        val1 = df_segments_smooth.iloc[-1,j+1]
        # linear interpolation in space
        frac = (s_up - s0) / (s1 - s0) if (s1!=s0) else 0.0
        adv_val = val0 + frac*(val1 - val0)
    # AR(1) forecast (SAMPLES_H steps)
    y_ar = last_val
    a_loc = seg_a[i]
    for _ in range(SAMPLES_H):
        y_ar = a_loc * y_ar  # intercept forced 0 for simplicity
    y_pers = last_val
    # blend: weights (tunable)
    return 0.5*y_ar + 0.3*y_pers + alpha*adv_val

forecasts = []
for i, last_val in enumerate(last_obs):
    f = advect_forecast_for_seg(i, last_val, seg_positions, df_segments_smooth, v_final, alpha=0.2)
    forecasts.append(f)

df_fore = pd.DataFrame({'segment':segment_names, 's_m':seg_positions, 'last_obs':last_obs, 'forecast_60min':forecasts})
print(df_fore)

# -------------------------
# 9) Incertidumbre: block bootstrap on time (resample blocks), recompute v_final and forecasts
# -------------------------
def block_bootstrap_forecasts(df_segments_smooth, seg_positions, B=200, block_minutes=BLOCK_MIN):
    n = df_segments_smooth.shape[0]
    block_size = max(1, int(block_minutes / SAMPLING_MIN))
    n_blocks = int(np.ceil(n / block_size))
    boot_forecasts = np.zeros((B, len(seg_positions)))
    rng = np.random.default_rng(12345)
    for b in range(B):
        # sample blocks with replacement
        idx_blocks = rng.integers(0, n_blocks, size=n_blocks)
        indices = []
        for bl in idx_blocks:
            start = bl*block_size
            end = min(n, start+block_size)
            indices.extend(list(range(start,end)))
        idxs = np.array(indices)[:n]  # ensure length n
        M_boot = df_segments_smooth.iloc[idxs,:].reset_index(drop=True)
        # recompute simple advective estimator: median dy/dt scaled (fallback-style)
        mean_series_boot = M_boot.mean(axis=1).values
        mean_smooth_boot = pd.Series(mean_series_boot).rolling(max(3,int(30/SAMPLING_MIN)), center=True, min_periods=1).median().values
        dy_dt_boot = np.gradient(mean_smooth_boot, dt_sec)
        v_boot = -np.median(dy_dt_boot) * 0.05
        # compute forecast using same advect_forecast_for_seg (use last value as last of M_boot)
        last_vals_boot = M_boot.iloc[-1,:].values
        for i in range(len(seg_positions)):
            boot_forecasts[b,i] = advect_forecast_for_seg(i, last_vals_boot[i], seg_positions, M_boot, v_boot, alpha=0.2)
    return boot_forecasts

print("Running bootstrap (this may take a while)...")
boot = block_bootstrap_forecasts(df_segments_smooth, seg_positions, B=B_BOOT, block_minutes=BLOCK_MIN)
# compute percentiles
p05 = np.percentile(boot, 5, axis=0)
p50 = np.percentile(boot, 50, axis=0)
p95 = np.percentile(boot, 95, axis=0)

df_fore['p05'] = p05
df_fore['p50'] = p50
df_fore['p95'] = p95

# -------------------------
# 10) SAVE RESULTS & PLOTS
# -------------------------
df_fore.to_csv("forecasts_60min_segments.csv", index=False)
print("Saved forecasts_60min_segments.csv")

# Plot example: last obs vs forecast p50 with CI
plt.figure(figsize=(10,4))
plt.errorbar(df_fore['s_m'], df_fore['p50'], yerr=[df_fore['p50']-df_fore['p05'], df_fore['p95']-df_fore['p50']], fmt='o', label='forecast p50 (CI 5-95%)')
plt.scatter(df_fore['s_m'], df_fore['last_obs'], color='red', label='last obs')
plt.xlabel("s (m)")
plt.ylabel("along (m/s)")
plt.legend(); plt.grid(); plt.title(f"Forecast 60min by segment (method={method_used}, v={v_final:.2f} m/s)")
plt.savefig("forecast_segments_plot.png", dpi=150)
print("Saved forecast_segments_plot.png")

# -------------------------
# 11) DECISIÓN / CRITERIOS RESUMEN
# -------------------------
print("\n--- CRITERIOS Y DECISIONES ---")
print("CCF success:", ccf_ok, f"(frac_good={frac_good:.2f}, thresh={FRAC_PAIRS_MIN})")
print("f-k success:", fk_ok)
print("Method selected:", method_used)
print("Estimated advective velocity (m/s):", v_final)
print("Forecast file: forecasts_60min_segments.csv")