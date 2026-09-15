# pipeline_principal.py - VERSIÓN CON CODIFICACIÓN UTF-8
import time
from datetime import datetime
import subprocess
import re
import pandas as pd
import os

print("="*80)
print(" SISTEMA DE PRONÓSTICO METEOROLÓGICO INTEGRADO")
print(f" Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# -------------------------
# 1. CARGAR DATOS UNA VEZ
# -------------------------
print("\n [1/5] CARGANDO DATOS...")

try:
    DATA_CSV = r"C:/Users/yanel/OneDrive/Desktop/thesis/data/datos_meteorologicos_rutaaeroprotec.csv"
    
    df = pd.read_csv(DATA_CSV, encoding='utf-8')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"   ✅ {len(df)} registros de {df['location'].nunique()} ubicaciones")
    
except Exception as e:
    print(f"   ❌ Error cargando datos: {e}")
    exit()

# -------------------------
# 2. EJECUTAR ANÁLISIS DE VIENTO
# -------------------------
print("\n [2/5] EJECUTANDO ANÁLISIS DE VIENTO...")

try:
    inicio_viento = time.time()
    
    resultado = subprocess.run(
        ["python", "pipeline_adveccion_nowcast.py"],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    fin_viento = time.time()
    
    if resultado.returncode != 0:
        print(f"Error ejecutando análisis de viento:")
        print(f"      {resultado.stderr[:200]}")
        exit()
    
    print(f"Análisis de viento completado en {fin_viento - inicio_viento:.1f}s")
    
    # Extraer velocidad calculada
    v_adveccion = None
    for linea in resultado.stdout.split('\n'):
        if "v_final (m/s):" in linea:
            try:
                match = re.search(r'v_final \(m/s\):\s*([\d.]+)', linea)
                if match:
                    v_adveccion = float(match.group(1))
                    break
            except:
                continue
    
    if v_adveccion is None:
        try:
            df_viento = pd.read_csv("forecasts_60min_segments.csv", encoding='utf-8')
            v_adveccion = 3.19
            print(f"Usando valor por defecto: {v_adveccion} m/s")
        except:
            v_adveccion = 3.19
            print(f"Usando velocidad por defecto: {v_adveccion} m/s")
    else:
        print(f"Velocidad calculada: {v_adveccion:.2f} m/s")
    
    # Guardar velocidad
    with open("velocidad_adveccion_calculada.txt", "w", encoding='utf-8') as f:
        f.write(str(v_adveccion))
    
except Exception as e:
    print(f"Error en análisis de viento: {e}")
    v_adveccion = 3.19

# -------------------------
# 3. VERIFICAR ARCHIVOS EXISTENTES
# -------------------------
print("\n🔍 [3/5] BUSCANDO ARCHIVOS DE PRONÓSTICO...")

# Posibles nombres de archivos
posibles_scripts = [
    "temperatura.py", "humedad.py", "lluvia.py",
]

tus_scripts = []
for script in posibles_scripts:
    if os.path.exists(script):
        tus_scripts.append(script)
        print(f"   ✅ Encontrado: {script}")

# Si no encuentra nada, listar archivos disponibles
if not tus_scripts:
    print("   ⚠️  No se encontraron scripts con nombres estándar")
    print("   📁 Archivos .py en la carpeta:")
    todos_archivos = [f for f in os.listdir('.') if f.endswith('.py')]
    for archivo in todos_archivos:
        if archivo != "metodologia101225.py" and archivo != "pipeline_principal.py":
            print(f"      • {archivo}")
            tus_scripts.append(archivo)

if not tus_scripts:
    print("   ❌ No hay scripts para procesar")
    exit()

print(f"   📋 Scripts a procesar: {tus_scripts}")

# -------------------------
# 4. ACTUALIZAR VELOCIDAD EN SCRIPTS
# -------------------------
print("\n🔄 [4/5] ACTUALIZANDO VELOCIDAD EN SCRIPTS...")

for script in tus_scripts:
    try:
        # ABRIR CON UTF-8 (ESTO ES CLAVE)
        with open(script, 'r', encoding='utf-8', errors='ignore') as f:
            lineas = f.readlines()
        
        cambios = 0
        for i, linea in enumerate(lineas):
            # Buscar línea con V_ADVECCION_FIJA y un número
            if "V_ADVECCION_FIJA" in linea and "=" in linea:
                # Reemplazar el número
                nueva_linea = re.sub(
                    r'(V_ADVECCION_FIJA\s*=\s*)[\d.]+',
                    f'\\g<1>{v_adveccion}',
                    linea
                )
                if nueva_linea != linea:
                    lineas[i] = nueva_linea
                    cambios += 1
                    print(f"   ✅ {script}: línea {i+1} actualizada")
        
        if cambios > 0:
            # GUARDAR CON UTF-8
            with open(script, 'w', encoding='utf-8') as f:
                f.writelines(lineas)
            print(f"   💾 {script}: {cambios} líneas actualizadas")
        else:
            print(f"   ⚠️  {script}: No se encontró V_ADVECCION_FIJA")
            
    except UnicodeDecodeError:
        print(f"   ⚠️  {script}: Error de codificación, intentando latin-1...")
        try:
            # Intentar con otra codificación
            with open(script, 'r', encoding='latin-1') as f:
                lineas = f.readlines()
            
            cambios = 0
            for i, linea in enumerate(lineas):
                if "V_ADVECCION_FIJA" in linea and "=" in linea:
                    nueva_linea = re.sub(
                        r'(V_ADVECCION_FIJA\s*=\s*)[\d.]+',
                        f'\\g<1>{v_adveccion}',
                        linea
                    )
                    if nueva_linea != linea:
                        lineas[i] = nueva_linea
                        cambios += 1
            
            if cambios > 0:
                with open(script, 'w', encoding='utf-8') as f:
                    f.writelines(lineas)
                print(f"   💾 {script}: {cambios} líneas actualizadas (latin-1)")
        except Exception as e:
            print(f"   ❌ {script}: Error crítico: {e}")
            
    except FileNotFoundError:
        print(f"   ❌ {script}: Archivo no encontrado")
    except Exception as e:
        print(f"   ❌ {script}: Error: {e}")

# -------------------------
# 5. EJECUTAR SCRIPTS ACTUALIZADOS
# -------------------------
print("\n🚀 [5/5] EJECUTANDO SCRIPTS ACTUALIZADOS...")

for script in tus_scripts:
    print(f"\n   ▶ Ejecutando {script}...")
    inicio = time.time()
    
    try:
        resultado = subprocess.run(
            ["python", script],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=600
        )
        
        fin = time.time()
        
        if resultado.returncode == 0:
            print(f"     ✅ Completado en {fin - inicio:.1f}s")
            # Mostrar resumen
            lineas_salida = resultado.stdout.strip().split('\n')
            if len(lineas_salida) > 5:
                print(f"     📝 Últimas líneas:")
                for linea in lineas_salida[-3:]:
                    if linea.strip():
                        print(f"       {linea[:80]}")
        else:
            print(f"     ❌ Error en {script}")
            if resultado.stderr:
                print(f"       Error: {resultado.stderr[:100]}")
            
    except subprocess.TimeoutExpired:
        print(f"     ⏰ Timeout: {script} tardó más de 10 minutos")
    except Exception as e:
        print(f"     ❌ Error ejecutando {script}: {e}")

# -------------------------
# 6. EVALUAR CONDICIONES (OPCIONAL)
# -------------------------
print("\n📊 [6/6] EVALUANDO RESULTADOS...")

# Verificar archivos generados
archivos_esperados = [
    "forecasts_60min_segments.csv",
    "forecasts_temperatura_60min_segments.csv",
    "forecasts_humedad_60min_segments.csv",
    "forecasts_lluvia_adaptativo_60min.csv"
]

print("\n📁 ARCHIVOS GENERADOS:")
for archivo in archivos_esperados:
    if os.path.exists(archivo):
        tamaño = os.path.getsize(archivo)
        print(f"   ✅ {archivo} ({tamaño:,} bytes)")
    else:
        print(f"   ❌ {archivo} (no generado)")

# Intentar evaluación si existe el módulo
if os.path.exists("evaluacion_dron.py"):
    try:
        print("\n🚁 EVALUANDO CONDICIONES PARA DRON...")
        from evaluacion_dron import EvaluadorDron
        
        evaluador = EvaluadorDron()
        
        # Solo evaluar archivos que existen
        archivos_existentes = {}
        for nombre, archivo in {
            'viento': "forecasts_60min_segments.csv",
            'temperatura': "forecasts_temperatura_60min_segments.csv",
            'humedad': "forecasts_humedad_60min_segments.csv",
            'lluvia': "forecasts_lluvia_adaptativo_60min.csv"
        }.items():
            if os.path.exists(archivo):
                archivos_existentes[nombre] = archivo
        
        if len(archivos_existentes) > 0:
            resultados = evaluador.evaluar_condiciones(
                archivos_existentes.get('viento'),
                archivos_existentes.get('temperatura'),
                archivos_existentes.get('humedad'),
                archivos_existentes.get('lluvia')
            )
            evaluador.generar_reporte(resultados)
        else:
            print("   ⚠️  No hay archivos para evaluar")
            
    except ImportError:
        print("   ⚠️  Módulo evaluacion_dron.py no disponible")
    except Exception as e:
        print(f"   ⚠️  Error en evaluación: {e}")

# -------------------------
# FINAL
# -------------------------
print("\n" + "="*80)
print("✅ PROCESO COMPLETADO")
print("="*80)
print(f"\n📋 RESUMEN:")
print(f"   • Velocidad calculada: {v_adveccion:.2f} m/s")
print(f"   • Scripts procesados: {len(tus_scripts)}")
print(f"   • Tiempo total: {time.time() - inicio_viento:.1f}s")
print("\n🎯 Siguientes pasos:")
print("   1. Revisar archivos CSV generados")
print("   2. Verificar gráficos generados")
print("   3. Consultar reporte_operacion_dron.txt")
print("="*80)