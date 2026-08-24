# ⚾ Medidor de Ponches MLB

Aplicación para predecir strikeouts de lanzadores abridores de MLB y convertirlos en decisiones de apuestas Over/Under con valor esperado.

## 📌 Descripción

Este proyecto entrena modelos de machine learning sobre datos pitch-level de Statcast para estimar cuántos ponches realizará un pitcher en su próxima apertura. Luego, esas predicciones se transforman en recomendaciones de apuestas según la línea y cuota que fije la casa de apuestas.

La web permite al usuario ingresar la línea de strikeouts y la cuota actual, y el modelo responde si hay valor o no, mostrando:
- Probabilidad estimada de Over/Under.
- EV (valor esperado).
- Cuota mínima recomendada para que la apuesta sea rentable.

## 🌐 Web App

La web está construida como sitio estático y desplegada en Vercel.

### Secciones
- **Predicciones de hoy**: muestra todos los juegos con probabilidad ≥ 55% en alguna línea.
- **Historial**: resultados de apuestas desde el 24/08/2026.

### Cómo usar
1. Ve a la pestaña "Predicciones de hoy".
2. Para cada pitcher, ingresa:
   - **Línea de la casa** (ej. 7.5)
   - **Cuota americana** (ej. -110)
3. Haz clic en **Evaluar**.
4. El sistema te dirá si hay valor y te sugerirá una cuota mínima.

## 📂 Estructura del repositorio
├── web/ # Frontend estático (HTML, CSS, JS, data.json)
├── src/ # Código fuente del pipeline
├── scripts/ # Utilidades (generación de web/data.json)
├── reports/ # Predicciones diarias, ledger, métricas
├── models/ # Modelos entrenados (BNN, XGBoost, etc.)
├── data/ # Datos crudos y procesados
├── run_daily.py # Flujo diario completo
└── vercel.json # Configuración de despliegue

text

## 🚀 Flujo diario automatizado

El repositorio incluye un workflow de GitHub Actions (`daily.yml`) que se ejecuta tres veces al día en horario de Hermosillo (08:00, 12:45, 15:00).

### Qué hace cada corrida
1. Liquida las apuestas del día anterior contra resultados reales.
2. Predice la jornada actual usando el modelo BNN v5 + XGBoost.
3. Aplica calibración walk-forward con ventana de 30 días.
4. Genera `web/data.json` para la web.
5. Sube los cambios al repositorio.
6. Notifica por Discord (si está configurado).

## 🧠 Modelo

El modelo principal es una **red neuronal bayesiana con NB-Dropout** combinada con **XGBoost-Poisson**. La BNN produce una distribución de probabilidad de strikeouts mediante muestreo Monte Carlo, y luego se calibra para obtener probabilidades over/under confiables.

## ⚙️ Instalación local

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
▶️ Ejecutar localmente
bash
# Ejecutar flujo diario completo
python run_daily.py

# Generar web/data.json
python scripts\generate_site_data.py

# Probar la web localmente
cd web
python -m http.server 8000
Abrir http://localhost:8000.

🔧 Variables de entorno opcionales
ODDS_API_KEY: para obtener cuotas reales (si no se define, se asumen -110).

DISCORD_WEBHOOK_URL: para notificaciones.

RUN_DATE: para forzar una fecha específica (YYYY-MM-DD).

BANKROLL: bankroll para cálculo de stakes (default 1000).

📅 Historial
El historial muestra apuestas liquidadas desde el 24/08/2026. Los registros anteriores no se muestran.

👨‍💻 Autor
Construida por Francisco Noe Renteria Nevarez.

📄 Licencia
Este proyecto es solo con fines educativos y de análisis. No constituye consejo de apuestas.

text

## Paso 3: Guardar y subir

1. Guarda el archivo (`Ctrl+S`).
2. Cierra el bloc de notas.
3. Ejecuta:

```cmd
git add README.md
git commit -m "Actualizar README en español"
git push origin main
Vercel no necesita redeploy por cambios en README, pero GitHub lo reflejará.