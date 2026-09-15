# evaluacion_dron.py
import pandas as pd
import numpy as np
from config import DRON_ESPECIFICACIONES

class EvaluadorDron:
    """Evalúa si el dron puede volar basado en pronósticos"""
    
    def __init__(self, especificaciones=None):
        self.espec = especificaciones or DRON_ESPECIFICACIONES
    
    def evaluar_condiciones(self, pronosticos_viento, pronosticos_temp, 
                           pronosticos_humedad, pronosticos_lluvia):
        """Evalúa todas las condiciones"""
        
        resultados = {
            'viento': self._evaluar_viento(pronosticos_viento),
            'temperatura': self._evaluar_temperatura(pronosticos_temp),
            'humedad': self._evaluar_humedad(pronosticos_humedad),
            'lluvia': self._evaluar_lluvia(pronosticos_lluvia),
            'general': 'DESCONOCIDO'
        }
        
        # Decisión general
        if all(r['puede_volar'] for r in [resultados['viento'], resultados['temperatura'], 
                                         resultados['humedad'], resultados['lluvia']]):
            resultados['general'] = 'PUEDE_VOLAR'
        elif any(r['puede_volar'] == False for r in [resultados['viento'], resultados['temperatura'], 
                                                    resultados['humedad'], resultados['lluvia']]):
            resultados['general'] = 'NO_PUEDE_VOLAR'
        else:
            resultados['general'] = 'CONDICIONES_LIMITE'
        
        return resultados
    
    def _evaluar_viento(self, pronosticos):
        """Evalúa condiciones de viento"""
        # Leer CSV o usar DataFrame
        if isinstance(pronosticos, str):
            df = pd.read_csv(pronosticos)
        else:
            df = pronosticos
        
        # Peor caso (P95) de viento
        max_viento = df['p95'].max() if 'p95' in df.columns else df['forecast_60min'].max()
        
        puede_volar = max_viento <= self.espec['viento_maximo']
        
        return {
            'puede_volar': puede_volar,
            'valor_maximo': max_viento,
            'limite': self.espec['viento_maximo'],
            'recomendacion': '✅ Puede volar' if puede_volar else f'❌ Viento máximo excedido ({max_viento:.1f} m/s > {self.espec["viento_maximo"]} m/s)'
        }
    
    def _evaluar_temperatura(self, pronosticos):
        """Evalúa condiciones de temperatura"""
        if isinstance(pronosticos, str):
            df = pd.read_csv(pronosticos)
        else:
            df = pronosticos
        
        # Rango pronosticado (P05-P95)
        min_temp = df['p05'].min() if 'p05' in df.columns else df['forecast_60min'].min()
        max_temp = df['p95'].max() if 'p95' in df.columns else df['forecast_60min'].max()
        
        dentro_limites = (min_temp >= self.espec['temperatura_min'] and 
                         max_temp <= self.espec['temperatura_max'])
        
        return {
            'puede_volar': dentro_limites,
            'rango_pronosticado': f"{min_temp:.1f}°C a {max_temp:.1f}°C",
            'limites': f"{self.espec['temperatura_min']}°C a {self.espec['temperatura_max']}°C",
            'recomendacion': '✅ Temperatura dentro de límites' if dentro_limites else f'❌ Temperatura fuera de rango'
        }
    
    def _evaluar_humedad(self, pronosticos):
        """Evalúa condiciones de humedad"""
        if isinstance(pronosticos, str):
            df = pd.read_csv(pronosticos)
        else:
            df = pronosticos
        
        # Peor caso de humedad (P95)
        max_humedad = df['p95'].max() if 'p95' in df.columns else df['forecast_60min'].max()
        
        puede_volar = max_humedad <= self.espec['humedad_maxima']
        
        return {
            'puede_volar': puede_volar,
            'valor_maximo': max_humedad,
            'limite': self.espec['humedad_maxima'],
            'recomendacion': '✅ Humedad dentro de límites' if puede_volar else f'❌ Humedad muy alta ({max_humedad:.1f}% > {self.espec["humedad_maxima"]}%)'
        }
    
    def _evaluar_lluvia(self, pronosticos):
        """Evalúa condiciones de lluvia"""
        if isinstance(pronosticos, str):
            df = pd.read_csv(pronosticos)
        else:
            df = pronosticos
        
        # Peor caso de lluvia (P95)
        max_lluvia = df['p95_mmh'].max() if 'p95_mmh' in df.columns else df['forecast_mmh'].max()
        
        puede_volar = max_lluvia <= self.espec['lluvia_maxima']
        
        # Categoría de lluvia
        categoria = self._clasificar_lluvia(max_lluvia)
        
        return {
            'puede_volar': puede_volar,
            'valor_maximo': max_lluvia,
            'limite': self.espec['lluvia_maxima'],
            'categoria': categoria,
            'recomendacion': f'✅ Lluvia {categoria}' if puede_volar else f'❌ Lluvia {categoria} ({max_lluvia:.2f} mm/h > {self.espec["lluvia_maxima"]} mm/h)'
        }
    
    def _clasificar_lluvia(self, mmh):
        """Clasifica intensidad de lluvia"""
        if mmh < 0.1:
            return 'SIN LLUVIA'
        elif mmh < 0.5:
            return 'TRACE'
        elif mmh < 2.0:
            return 'LIGERA'
        elif mmh < 5.0:
            return 'MODERADA'
        elif mmh < 10.0:
            return 'FUERTE'
        else:
            return 'EXTREMA'
    
    def generar_reporte(self, resultados):
        """Genera reporte ejecutivo"""
        print("\n" + "="*80)
        print("📋 EVALUACIÓN DE CONDICIONES PARA OPERACIÓN DE DRON")
        print("="*80)
        print(f"\n🚁 ESPECIFICACIONES DEL DRON:")
        print(f"   • Viento máximo: {self.espec['viento_maximo']} m/s")
        print(f"   • Lluvia máxima: {self.espec['lluvia_maxima']} mm/h")
        print(f"   • Temperatura: {self.espec['temperatura_min']}°C a {self.espec['temperatura_max']}°C")
        print(f"   • Humedad máxima: {self.espec['humedad_maxima']}%")
        
        print("\n📊 RESULTADOS POR VARIABLE:")
        for var, res in resultados.items():
            if var != 'general':
                print(f"\n   {var.upper()}:")
                print(f"     {res['recomendacion']}")
                if 'valor_maximo' in res:
                    print(f"     Valor pronosticado: {res['valor_maximo']:.2f}")
                if 'rango_pronosticado' in res:
                    print(f"     Rango pronosticado: {res['rango_pronosticado']}")
        
        print("\n" + "="*80)
        print("🎯 DECISIÓN FINAL:")
        
        if resultados['general'] == 'PUEDE_VOLAR':
            print("✅ ✅ ✅ PUEDE VOLAR - Todas las condiciones dentro de límites")
        elif resultados['general'] == 'NO_PUEDE_VOLAR':
            print("❌ ❌ ❌ NO PUEDE VOLAR - Una o más condiciones exceden límites")
        else:
            print("⚠️ ⚠️ ⚠️ CONDICIONES EN LÍMITE - Evaluar riesgos adicionales")
        
        print("="*80)
def generar_reporte(self, resultados, archivo_salida="decision_dron.txt"):
    """Genera reporte y guarda decisión en archivo"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    contenido = f"""================================================================================
🚁 DECISIÓN DE OPERACIÓN DE DRON
🕐 Fecha y hora: {timestamp}
================================================================================

📊 CONDICIONES EVALUADAS:

   VIENTO:
     {'✅ Puede volar' if resultados['viento']['puede_volar'] else '❌ Viento excede límites'}
     Valor pronosticado: {resultados['viento']['valor_pronosticado']:.2f} m/s
     Límite: {resultados['viento']['limite']} m/s

   TEMPERATURA:
     {'✅ Temperatura dentro de límites' if resultados['temperatura']['puede_volar'] else '❌ Temperatura fuera de rango'}
     Rango pronosticado: {resultados['temperatura']['min_pronosticado']:.1f}°C a {resultados['temperatura']['max_pronosticado']:.1f}°C
     Rango permitido: {resultados['temperatura']['limite_min']}°C a {resultados['temperatura']['limite_max']}°C

   HUMEDAD:
     {'✅ Humedad dentro de límites' if resultados['humedad']['puede_volar'] else '❌ Humedad excede límites'}
     Valor pronosticado: {resultados['humedad']['valor_pronosticado']:.2f}%
     Límite: {resultados['humedad']['limite']}%

   LLUVIA:
     {'✅ Lluvia dentro de límites' if resultados['lluvia']['puede_volar'] else '❌ Lluvia excede límites'}
     Valor pronosticado: {resultados['lluvia']['valor_pronosticado']:.2f} mm/h
     Límite: {resultados['lluvia']['limite']} mm/h
     Categoría: {resultados['lluvia']['categoria']}

================================================================================
{"✅ ✅ ✅ PUEDE VOLAR - Todas las condiciones están dentro de límites" if resultados['decision_final'] else "❌ ❌ ❌ NO PUEDE VOLAR - Una o más condiciones exceden límites"}
================================================================================

📋 OBSERVACIONES:
{resultados.get('observaciones', 'Ninguna observación adicional.')}

📁 ARCHIVOS CONSULTADOS:
{chr(10).join(f'   • {arch}' for arch in resultados.get('archivos_consultados', []))}

-------------------------------------------------------------------------------
   🚁 Sistema de Pronóstico Meteorológico Integrado
   📅 Generado automáticamente
================================================================================
"""
    
    # Guardar en archivo
    with open(archivo_salida, "w", encoding="utf-8") as f:
        f.write(contenido)
    
    print(f"📄 Reporte guardado en: {archivo_salida}")
    
    return contenido