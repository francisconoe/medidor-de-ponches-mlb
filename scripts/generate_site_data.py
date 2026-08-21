import glob
import json
import os
from pathlib import Path

import pandas as pd

def main():
    # Buscar archivos de predicción diaria en reports/predictions/
    prediction_files = glob.glob("reports/predictions/*.csv")
    if not prediction_files:
        print("No se encontraron archivos de predicción diaria. Ejecuta primero: python run_daily.py")
        return

    # Tomar el archivo más reciente por nombre (asumiendo que contienen fecha) o por modificación
    latest_file = max(prediction_files, key=os.path.getmtime)
    print(f"Usando archivo de predicciones: {latest_file}")

    try:
        df_pred = pd.read_csv(latest_file)
    except Exception as e:
        print(f"Error leyendo {latest_file}: {e}")
        return

    # Limpiar columnas innecesarias
    if "Unnamed: 0" in df_pred.columns:
        df_pred.drop(columns=["Unnamed: 0"], inplace=True)

    # Convertir a registros
    juegos = df_pred.to_dict(orient="records")

    # Obtener fecha de los juegos (asumimos todos misma fecha)
    if "game_date" in df_pred.columns and len(df_pred) > 0:
        fecha = str(df_pred.iloc[0]["game_date"])
    else:
        fecha = "Desconocida"

    # Leer historial del ledger si existe
    historial = []
    ledger_path = "reports/bets_ledger.csv"
    if Path(ledger_path).exists():
        try:
            df_ledger = pd.read_csv(ledger_path)
            if "Unnamed: 0" in df_ledger.columns:
                df_ledger.drop(columns=["Unnamed: 0"], inplace=True)
            # Tomar últimas 30 filas (las más recientes al final)
            historial = df_ledger.tail(30).to_dict(orient="records")
        except Exception as e:
            print(f"Error leyendo ledger: {e}")

    # Crear estructura final
    data = {
        "fecha": fecha,
        "juegos": juegos,
        "historial": historial
    }

    # Guardar en web/data.json
    output_dir = Path("web")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Datos guardados en {output_file}")
    print(f"Juegos: {len(juegos)} | Registros historial: {len(historial)}")

if __name__ == "__main__":
    main()