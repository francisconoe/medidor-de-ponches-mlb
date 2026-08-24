import glob
import json
import os
from pathlib import Path

import pandas as pd

def main():
    # Buscar archivos de predicción principales, no los agreement_slate
    pred_files = glob.glob("reports/predictions/realtime_v4_predictions_*.csv")
    if not pred_files:
        print("No se encontraron archivos de predicción diaria. Ejecuta primero: python run_daily.py")
        return

    # Ordenar por fecha en el nombre, no por modificación
    def fecha_desde_nombre(path):
        base = os.path.basename(path)
        try:
            fecha_str = base.replace("realtime_v4_predictions_", "").replace(".csv", "")
            return pd.to_datetime(fecha_str)
        except:
            return pd.Timestamp.min

    latest_file = max(pred_files, key=fecha_desde_nombre)
    print(f"Usando archivo de predicciones: {latest_file}")

    try:
        df_pred = pd.read_csv(latest_file)
    except Exception as e:
        print(f"Error leyendo {latest_file}: {e}")
        return

    if "Unnamed: 0" in df_pred.columns:
        df_pred.drop(columns=["Unnamed: 0"], inplace=True)

    # Filtrar juegos con probabilidad >= 0.55 en cualquier línea
    lineas_probs = ["prob_over_3.5", "prob_over_4.5", "prob_over_5.5", "prob_over_6.5"]
    if all(c in df_pred.columns for c in lineas_probs):
        mask = (
            (df_pred["prob_over_3.5"] >= 0.55) |
            (df_pred["prob_over_4.5"] >= 0.55) |
            (df_pred["prob_over_5.5"] >= 0.55) |
            (df_pred["prob_over_6.5"] >= 0.55) |
            ((1 - df_pred["prob_over_3.5"]) >= 0.55) |
            ((1 - df_pred["prob_over_4.5"]) >= 0.55) |
            ((1 - df_pred["prob_over_5.5"]) >= 0.55) |
            ((1 - df_pred["prob_over_6.5"]) >= 0.55)
        )
        df_pred = df_pred[mask]

    # Seleccionar columnas necesarias para la web
    columnas_juegos = [
        "pitcher_name", "team", "opponent", "line",
        "prob_over_3.5", "prob_over_4.5", "prob_over_5.5", "prob_over_6.5"
    ]
    columnas_existentes = [c for c in columnas_juegos if c in df_pred.columns]
    juegos = df_pred[columnas_existentes].to_dict(orient="records")

    if "game_date" in df_pred.columns and len(df_pred) > 0:
        fecha = str(df_pred.iloc[0]["game_date"])
    else:
        fecha = "Desconocida"

    # Historial desde el ledger, sin filtrar por fecha (la web ya filtrará)
    historial = []
    ledger_path = "reports/bets_ledger.csv"
    if Path(ledger_path).exists():
        try:
            df_ledger = pd.read_csv(ledger_path)
            if "Unnamed: 0" in df_ledger.columns:
                df_ledger.drop(columns=["Unnamed: 0"], inplace=True)
            # Convertir fecha a string
            if "game_date" in df_ledger.columns:
                df_ledger["game_date"] = pd.to_datetime(df_ledger["game_date"]).dt.strftime("%Y-%m-%d")
            # Tomar últimos 50 registros
            historial = df_ledger.tail(50).to_dict(orient="records")
        except Exception as e:
            print(f"Error leyendo ledger: {e}")

    data = {
        "fecha": fecha,
        "juegos": juegos,
        "historial": historial
    }

    output_dir = Path("web")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Datos guardados en {output_file}")
    print(f"Juegos: {len(juegos)} | Registros historial: {len(historial)}")

if __name__ == "__main__":
    main()