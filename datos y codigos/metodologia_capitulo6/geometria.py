# geometria.py
import math
import numpy as np
import pandas as pd
from config import LATS, LONS

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

def crear_ruta(lats=None, lons=None, locations=None):
    """Crea la tabla de ruta una sola vez"""
    if lats is None:
        lats = LATS
    if lons is None:
        lons = LONS
    if locations is None:
        locations = [f"Punto {i}" for i in range(1, len(lats)+1)]
    
    coords = list(zip(lats, lons))
    dists = [0.0]
    for i in range(1, len(coords)):
        dists.append(haversine_m(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1]))
    s = np.cumsum(dists)
    
    bearings = []
    for i in range(len(coords)-1):
        bearings.append(math.radians(bearing_deg(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])))
    bearings.append(bearings[-1])
    
    return pd.DataFrame({
        'location': locations, 
        'lat': lats, 
        'lon': lons, 
        's_m': s, 
        'bearing_rad': bearings
    })

def crear_segmentos(route_table, n_segments=10):
    """Crea segmentos consistentes"""
    locations = route_table['location'].tolist()
    loc_idx = np.arange(len(locations))
    groups = np.array_split(loc_idx, n_segments)
    
    segment_names = [f"seg_{i+1}" for i in range(len(groups))]
    seg_positions = np.array([route_table['s_m'].values[list(g)].mean() for g in groups])
    
    return groups, segment_names, seg_positions