"""
Walk-forward calibration for BNN probabilities.

Simulates daily betting with a sliding window calibrator (Isotonic)
trained on the previous N days of actual outcomes.

Usage:
    python -m src.calibrate_walkforward
"""

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score

# Configuración
VENTANA_DIAS = 30
LINEAS = [3.5, 4.5, 5.5, 6.5]
PROB_MIN = 0.05
EV_MIN = 0.02   # apuesta solo si EV > 2%
ODDS_AMERICANO = -110

def prob_break_even(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def expected_value(prob, odds):
    if odds > 0:
        return prob * (odds / 100) - (1 - prob)
    else:
        return prob * (100 / -odds) - (1 - prob)

def main():
    df = pd.read_csv("reports/bnn_predictions.csv")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    # Resultados por línea
    for line in LINEAS:
        df[f"out_over_{line}"] = (df["strikeouts"] > line).astype(int)

    print(f"Total juegos en test: {len(df)}")
    print(f"Ventana deslizante: {VENTANA_DIAS} días\n")

    # Evaluación de estrategia base (usando prob_over directa)
    print("=== ESTRATEGIA BASE (prob_over original) ===")
    apuestas = 0
    ganadas = 0
    bankroll = 1000
    evs = []
    for idx, row in df.iterrows():
        for line in LINEAS:
            p = row[f"prob_over_{line}"]
            y = row[f"out_over_{line}"]
            # Elegir lado: over si p>0.5, under si p<0.5
            if p > 0.5:
                prob = p
                lado = "over"
                outcome = y
            else:
                prob = 1 - p
                lado = "under"
                outcome = 1 - y
            ev = expected_value(prob, ODDS_AMERICANO)
            if ev > EV_MIN and prob > PROB_MIN:
                apuestas += 1
                if outcome == 1:
                    ganadas += 1
                    bankroll += 100 * (100 / 110)  # apuesta fija de 100 unidades
                else:
                    bankroll -= 100
                evs.append(ev)
    roi = (bankroll - 1000) / 1000
    winrate = ganadas / apuestas if apuestas else 0
    print(f"Apuestas: {apuestas}, Win rate: {winrate:.2%}, ROI: {roi:.2%}, EV prom: {np.mean(evs):.3f}")

    # Calibración walk-forward
    print("\n=== CALIBRACIÓN WALK-FORWARD ===")
    # Para cada línea, recalibrar prob_over usando isotónica con datos de los últimos VENTANA_DIAS
    # Guardamos las nuevas probabilidades
    for line in LINEAS:
        df[f"cal_prob_over_{line}"] = np.nan

    # Recorrer cada juego en orden temporal
    for i, row in df.iterrows():
        fecha = row["game_date"]
        # Datos históricos hasta antes de este juego (excluyendo el actual)
        hist = df[(df["game_date"] < fecha) & (df["game_date"] >= fecha - pd.Timedelta(days=VENTANA_DIAS))]
        if len(hist) < 20:  # mínimo de muestras para calibrar
            continue
        for line in LINEAS:
            x_raw = hist[f"prob_over_{line}"].values
            y_raw = hist[f"out_over_{line}"].values
            # IsotonicRegression requiere al menos 2 puntos y no todos iguales
            if len(np.unique(x_raw)) < 2 or len(np.unique(y_raw)) < 2:
                continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(x_raw, y_raw)
            # Aplicar al juego actual
            p_cal = iso.predict([row[f"prob_over_{line}"]])[0]
            df.at[i, f"cal_prob_over_{line}"] = p_cal

    # Evaluar con calibración walk-forward
    apuestas = 0
    ganadas = 0
    bankroll = 1000
    evs = []
    for idx, row in df.iterrows():
        for line in LINEAS:
            p = row[f"cal_prob_over_{line}"]
            if pd.isna(p):
                continue
            y = row[f"out_over_{line}"]
            if p > 0.5:
                prob = p
                outcome = y
            else:
                prob = 1 - p
                outcome = 1 - y
            ev = expected_value(prob, ODDS_AMERICANO)
            if ev > EV_MIN and prob > PROB_MIN:
                apuestas += 1
                if outcome == 1:
                    ganadas += 1
                    bankroll += 100 * (100 / 110)
                else:
                    bankroll -= 100
                evs.append(ev)
    roi = (bankroll - 1000) / 1000
    winrate = ganadas / apuestas if apuestas else 0
    print(f"Apuestas: {apuestas}, Win rate: {winrate:.2%}, ROI: {roi:.2%}, EV prom: {np.mean(evs):.3f}")

    # También podemos mostrar la calibración ECE antes y después para una línea clave
    from src.train_bnn_v4 import ece
    print("\n=== ECE por línea (base vs calibrada) ===")
    for line in LINEAS:
        y = df[f"out_over_{line}"].values
        p_base = df[f"prob_over_{line}"].values
        p_cal = df[f"cal_prob_over_{line}"].values
        mask = ~np.isnan(p_cal)
        if mask.sum() > 0:
            ece_base = ece(p_base[mask], y[mask])
            ece_cal = ece(p_cal[mask], y[mask])
            print(f"Línea {line}: ECE base = {ece_base:.3f}, ECE walk-forward = {ece_cal:.3f}")

if __name__ == "__main__":
    main()