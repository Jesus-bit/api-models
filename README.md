# Weather Models API 🌤️

Esta API permite consultar modelos de predicción climática. Ofrece datos sobre temperatura, humedad, velocidad del viento, y otras variables atmosféricas para diferentes ubicaciones geográficas. Ideal para aplicaciones de análisis y pronóstico climático.

## Características

- 📍 Consulta de datos climáticos por ubicación (ciudad, país, coordenadas).
- 📅 Predicciones a corto, medio y largo plazo.
- 📊 Variables atmosféricas disponibles: temperatura, humedad, velocidad del viento, presión, entre otros.
- 🔍 Búsqueda histórica de datos (si está disponible).

## Instalación

1. Clona este repositorio:
    ```bash
    git clone https://github.com/tu_usuario/weather-models-api.git
    cd weather-models-api
    ```

2. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

3. Configura las variables de entorno necesarias (por ejemplo, claves de API de servicios externos, si las usas).

4. Inicia el servidor de desarrollo:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

La API estará disponible en `http://localhost:8000`.
