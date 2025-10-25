import os
import sys
from flask import Flask, request, jsonify
import requests
import pandas as pd
import numpy as np
import joblib # <- ¡La única librería que necesitamos para cargar!

# --- 1. Configuración Inicial ---
app = Flask(__name__)

# --- 2. Constantes y Configuración del Modelo ---
FEATURES = ['temperature_2m', 'precipitation', 'is_thunderstorm']

# --- 3. Cargar Modelo y Scaler (Mucho más simple) ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'modelo_rf.pkl') # <- El nuevo modelo
scaler_path = os.path.join(base_dir, 'scaler.pkl')

modelo = None
scaler = None

try:
    modelo = joblib.load(model_path)
    print(">>> Modelo 'modelo_rf.pkl' cargado exitosamente.", file=sys.stderr)
except Exception as e:
    print(f"!!! Error fatal al cargar 'modelo_rf.pkl'.", file=sys.stderr)
    print(f"Error: {e}", file=sys.stderr)

try:
    scaler = joblib.load(scaler_path)
    print(">>> Scaler 'scaler.pkl' cargado exitosamente.", file=sys.stderr)
except Exception as e:
    print(f"!!! Error fatal al cargar 'scaler.pkl'.", file=sys.stderr)
    print(f"Error: {e}", file=sys.stderr)

# --- 4. Definir Coordenadas (sin cambios) ---
coordenadas = {
    "Medellin": {"lat": 6.2518, "lon": -75.5636},
    "Ebejico": {"lat": 6.315, "lon": -75.766},
    "Heliconia": {"lat": 6.368, "lon": -75.736},
}

# --- 5. Funciones de Lógica (Simplificadas) ---

def obtener_y_preparar_datos(lat, lon):
    """
    Obtiene la HORA MÁS RECIENTE para hacer la predicción.
    """
    print(f"Obteniendo datos para: {lat}, {lon}", file=sys.stderr)
    
    # Pedimos el pronóstico, pero incluimos 1 día pasado
    # para asegurar que tenemos la última hora registrada.
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,precipitation,weather_code"
        f"&past_days=1" # <- Solo 1 día
        f"&forecast_days=1"
    )
    
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['hourly'])
    
    # Procesar
    df['is_thunderstorm'] = df['weather_code'].apply(lambda x: 1 if x in [95, 96, 99] else 0)
    df_features = df[FEATURES].dropna()
    
    # ¡¡CLAVE!! Tomamos solo la ÚLTIMA HORA DISPONIBLE
    df_final = df_features.tail(1)
    
    if scaler is None: raise Exception("El Scaler no está cargado.")
        
    # Escalar esa única fila
    datos_escalados = scaler.transform(df_final)
    
    # El modelo espera [n_samples, n_features], ej: [1, 3]
    return datos_escalados

def realizar_prediccion(datos_entrada):
    """
    Ejecuta el modelo Random Forest.
    """
    if modelo is None: raise Exception("El Modelo RF no está cargado.")

    # 1. Realizar la predicción
    # datos_entrada es (1, 3) [2D]
    # prediccion_escalada será (1, 3) [2D]
    prediccion_escalada = modelo.predict(datos_entrada)

    # 2. Invertir la escala
    # --- ¡AQUÍ ESTABA EL ERROR! ---
    # Quité los corchetes [ ] extra que envolvían a prediccion_escalada
    prediccion_desescalada = scaler.inverse_transform(prediccion_escalada)

    # 3. Formatear la salida
    resultado = prediccion_desescalada[0] # Tomamos la primera (y única) fila
    prediccion_final = {
        "temperatura_pred": round(float(resultado[0]), 2),
        "precipitacion_pred": round(float(resultado[1]), 2),
        "tormenta_pred": 1 if float(resultado[2]) > 0.5 else 0
    }

    return prediccion_final

# --- 6. El "Endpoint" de la API (sin cambios) ---

@app.route('/predecir', methods=['GET'])
def predecir():
    municipio_query = request.args.get('municipio')
    if not municipio_query:
        return jsonify({"error": "Debes proveer un parámetro 'municipio'."}), 400

    lugar = coordenadas.get(municipio_query.capitalize())
    if not lugar:
        return jsonify({"error": f"Municipio '{municipio_query}' no encontrado."}), 404

    try:
        datos_para_modelo = obtener_y_preparar_datos(lugar["lat"], lugar["lon"])
        prediccion = realizar_prediccion(datos_para_modelo)
        print(f"Predicción RF exitosa para {municipio_query}.", file=sys.stderr)
        return jsonify(prediccion)

    except Exception as e:
        print(f"!!! Error durante la predicción RF: {e}", file=sys.stderr)
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

@app.route('/')
def index():
    return "¡Bienvenido a la API de Predicción del Clima! (Versión Random Forest 🌳)"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
