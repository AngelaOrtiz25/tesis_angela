# carga_datos.py
import pandas as pd
from config import DATA_CSV

def cargar_datos():
    """Carga datos una sola vez para todos"""
    df = pd.read_csv(DATA_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def extraer_variable(df, variable, locations):
    """Extrae una variable específica en formato wide"""
    dfs = []
    for loc in locations:
        sub = df[df['location']==loc][['timestamp', variable]]
        if not sub.empty:
            sub = sub.set_index('timestamp').rename(columns={variable: f"{loc}__{variable}"})
            dfs.append(sub)
    
    if dfs:
        df_wide = pd.concat(dfs, axis=1, join='inner').sort_index()
        return df_wide
    return None