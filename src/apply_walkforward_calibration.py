"""
Aplica calibración walk-forward a un archivo de predicción diaria
y recalcula la decisión de apuesta basada en las nuevas probabilidades.

Uso:
    python -m src.apply_walkforward_calibration --fecha YYYY-MM-DD

Requisitos:
    - reports/settled_results.csv con columnas:
      game_pk, pitcher, game_date, strikeouts, prob_over_3.5, prob_over_4.5, prob_over_5.5, prob_over_6.5
    - reports/predictions/realtime_v4_predictions_YYYY-MM-DD.csv
      con columnas que incluyan prob_over_* y odds (o se asume -110)
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from pathlib import Path

PRED_DIR = "reports/predictions"
HIST_PATH = "reports/settled_results.csv"
VENTANA_DIAS = 30
MIN_JUEGOS = 20
EV_MIN = 0.05          # 5% de EV mínimo para apostar
PROB_MIN = 0.60        # probabilidad mínima del lado elegido (antes de acotar)
UNC_MAX = 0.12         # incertidumbre máxima permitida (si existe columna unc)
KELLY_FRACTION = 0.25  # 1/4 Kelly
STAKE_MAX = 0.01       # máximo 1% del bankroll por apuesta
PROB_LOWER = 0.05
PROB_UPPER = 0.85

def prob_break_even(odds_americano):
    """Probabilidad de break-even para cuota americana."""
    try:
        odds = float(odds_americano)
    except:
        odds = -110
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def expected_value(prob, odds_americano):
    """Valor esperado por unidad apostada."""
    try:
        odds = float(odds_americano)
    except:
        odds = -110
    if odds > 0:
        return prob * (odds / 100) - (1 - prob)
    else:
        return prob * (100 / -odds) - (1 - prob)

def kelly_fraction(prob, odds_americano):
    """Fracción de Kelly para una apuesta."""
    be = prob_break_even(odds_americano)
    if prob <= be:
        return 0.0
    try:
        odds = float(odds_americano)
    except:
        odds = -110
    if odds > 0:
        b = odds / 100
    else:
        b = 100 / -odds
    return (prob * b - (1 - prob)) / b

def limit_prob(p):
    """Limita la probabilidad al rango [0.05, 0.85]."""
    return max(PROB_LOWER, min(PROB_UPPER, p))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fecha", required=True, help="Fecha YYYY-MM-DD")
    args = parser.parse_args()
    fecha = args.fecha

    pred_path = f"{PRED_DIR}/realtime_v4_predictions_{fecha}.csv"
    if not Path(pred_path).exists():
        print(f"No se encontró {pred_path}")
        return

    # Cargar histórico
    if not Path(HIST_PATH).exists():
        print("No existe histórico de resultados. No se puede calibrar.")
        return
    hist = pd.read_csv(HIST_PATH)
    hist["game_date"] = pd.to_datetime(hist["game_date"])

    # Cargar predicción diaria
    pred = pd.read_csv(pred_path)

    # Asegurar que existan columnas prob_over_*
    lineas = [3.5, 4.5, 5.5, 6.5]
    if not all(f"prob_over_{l}" in pred.columns for l in lineas):
        print("El CSV diario no contiene prob_over_*, se omite la calibración.")
        return

    # Fecha del juego (asumimos todos misma fecha)
    pred["game_date"] = pd.to_datetime(fecha)

    # Para cada fila, recalibrar prob_over usando histórico hasta el día anterior
    for i, row in pred.iterrows():
        fecha_juego = row["game_date"]
        hist_reciente = hist[
            (hist["game_date"] < fecha_juego) &
            (hist["game_date"] >= fecha_juego - pd.Timedelta(days=VENTANA_DIAS))
        ]
        if len(hist_reciente) < MIN_JUEGOS:
            continue
        for l in lineas:
            x = hist_reciente[f"prob_over_{l}"].values
            y = (hist_reciente["strikeouts"] > l).astype(int).values
            if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
                continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(x, y)
            p_cal = iso.predict([row[f"prob_over_{l}"]])[0]
            pred.at[i, f"prob_over_{l}"] = limit_prob(p_cal)   # acotar over

    # Recalcular decisiones con probabilidades calibradas
    bankroll = float(os.environ.get("BANKROLL", "1000"))
    for i, row in pred.iterrows():
        # Filtro por incertidumbre
        if "unc" in row and not pd.isna(row["unc"]):
            if float(row["unc"]) > UNC_MAX:
                pred.at[i, "PLAY"] = "— pass —"
                pred.at[i, "stake"] = 0.0
                pred.at[i, "call"] = ""
                pred.at[i, "line"] = ""
                pred.at[i, "model_prob"] = np.nan
                pred.at[i, "EV"] = 0.0
                pred.at[i, "edge"] = 0.0
                continue

        # Buscar la mejor jugada entre líneas y lados
        candidatos = []
        for line in lineas:
            p_over = pred.at[i, f"prob_over_{line}"]
            # Over
            if p_over > PROB_MIN:
                ev = expected_value(p_over, row["odds"])
                if ev > EV_MIN:
                    candidatos.append(("over", line, p_over, ev))
            # Under
            p_under = 1 - p_over
            if p_under > PROB_MIN:
                ev = expected_value(p_under, row["odds"])
                if ev > EV_MIN:
                    candidatos.append(("under", line, p_under, ev))

        if not candidatos:
            pred.at[i, "PLAY"] = "— pass —"
            pred.at[i, "stake"] = 0.0
            continue

        # Elegir la de mayor EV
        mejor = max(candidatos, key=lambda x: x[3])
        lado, line, prob_cruda, ev_cruda = mejor

        # Acotar la probabilidad final del lado seleccionado
        prob_final = limit_prob(prob_cruda)
        ev_final = expected_value(prob_final, row["odds"])

        # Si tras acotar el EV ya no supera el mínimo, pasar
        if ev_final <= EV_MIN or prob_final <= prob_break_even(row["odds"]):
            pred.at[i, "PLAY"] = "— pass —"
            pred.at[i, "stake"] = 0.0
            continue

        # Calcular stake con Kelly fraccional, tope 1%
        frac = kelly_fraction(prob_final, row["odds"]) * KELLY_FRACTION
        stake = bankroll * min(frac, STAKE_MAX)
        stake = max(0.0, stake)

        pred.at[i, "line"] = line
        pred.at[i, "call"] = f"{lado} {line}"
        pred.at[i, "model_prob"] = prob_final
        pred.at[i, "fair_prob"] = prob_break_even(row["odds"])
        pred.at[i, "edge"] = prob_final - pred.at[i, "fair_prob"]
        pred.at[i, "EV"] = ev_final
        pred.at[i, "stake"] = stake
        pred.at[i, "PLAY"] = f"{lado} {line}" if stake > 0 else "— pass —"

    # Guardar CSV sobrescrito
    pred.to_csv(pred_path, index=False)
    print(f"Calibración aplicada y decisiones recalculadas en {pred_path}")

if __name__ == "__main__":
    main()