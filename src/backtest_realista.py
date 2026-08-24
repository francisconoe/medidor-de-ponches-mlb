"""
Backtest realista: una apuesta por juego, Kelly fraccional, comparación
de calibración base vs walk-forward.

Uso:
    python -m src.backtest_realista
"""

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression

# Parámetros
BANKROLL_INICIAL = 1000
LINEAS = [3.5, 4.5, 5.5, 6.5]
VENTANA_DIAS = 30
MIN_JUEGOS_CALIBRAR = 20
EV_MIN = 0.02
PROB_MIN = 0.05
KELLY_FRACTION = 0.25   # 1/4 Kelly
STAKE_MAX = 0.02        # máximo 2% del bankroll por apuesta
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

def kelly_fraction(prob, odds):
    break_even = prob_break_even(odds)
    if prob <= break_even:
        return 0.0
    if odds > 0:
        b = odds / 100
    else:
        b = 100 / -odds
    return (prob * b - (1 - prob)) / b

def elegir_apuesta(row, calibrada=False):
    """Devuelve (lado, línea, prob) de la apuesta con mayor EV, o None si ninguna supera umbral."""
    mejor = None
    for line in LINEAS:
        if calibrada:
            p = row[f"cal_prob_over_{line}"]
            if pd.isna(p):
                continue
        else:
            p = row[f"prob_over_{line}"]
        # Lado over
        if p > 0.5:
            prob = p
            lado = "over"
        else:
            prob = 1 - p
            lado = "under"
        ev = expected_value(prob, ODDS_AMERICANO)
        if ev > EV_MIN and prob > PROB_MIN:
            if mejor is None or ev > mejor[3]:
                mejor = (lado, line, prob, ev)
    return mejor

def run_backtest(df, calibrada=False):
    bankroll = BANKROLL_INICIAL
    apuestas = 0
    ganadas = 0
    retornos = []
    for idx, row in df.iterrows():
        ap = elegir_apuesta(row, calibrada)
        if ap is None:
            continue
        lado, line, prob, ev = ap
        # Determinar resultado
        outcome = row["strikeouts"]
        if lado == "over":
            gano = outcome > line
        else:
            gano = outcome <= line
        # Kelly fraccional
        frac = kelly_fraction(prob, ODDS_AMERICANO) * KELLY_FRACTION
        stake = bankroll * min(frac, STAKE_MAX)
        if stake <= 0:
            continue
        apuestas += 1
        if gano:
            ganadas += 1
            profit = stake * (100 / 110)   # -110 paga 100/110 por unidad
            bankroll += profit
        else:
            bankroll -= stake
        retornos.append(bankroll)
    roi = (bankroll - BANKROLL_INICIAL) / BANKROLL_INICIAL
    winrate = ganadas / apuestas if apuestas else 0
    # Sharpe aproximado con retornos por apuesta
    retornos_arr = np.array(retornos)
    if len(retornos_arr) > 1:
        before = np.concatenate([[BANKROLL_INICIAL], retornos_arr[:-1]])
        returns_per_bet = (retornos_arr - before) / before
        sharpe = np.mean(returns_per_bet) / (np.std(returns_per_bet) + 1e-9) * np.sqrt(len(returns_per_bet))
    else:
        sharpe = 0
    max_drawdown = 0
    peak = BANKROLL_INICIAL
    for b in retornos:
        if b > peak:
            peak = b
        dd = (peak - b) / peak
        if dd > max_drawdown:
            max_drawdown = dd
    return {
        "apuestas": apuestas,
        "winrate": winrate,
        "bankroll": bankroll,
        "roi": roi,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown
    }

def main():
    df = pd.read_csv("reports/bnn_predictions.csv")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    # Añadir columnas de resultado por línea
    for line in LINEAS:
        df[f"out_over_{line}"] = (df["strikeouts"] > line).astype(int)

    # Calibración walk-forward: recalcular prob_over para cada juego usando datos de los últimos 30 días
    for line in LINEAS:
        df[f"cal_prob_over_{line}"] = np.nan

    for i, row in df.iterrows():
        fecha = row["game_date"]
        hist = df[(df["game_date"] < fecha) & (df["game_date"] >= fecha - pd.Timedelta(days=VENTANA_DIAS))]
        if len(hist) < MIN_JUEGOS_CALIBRAR:
            continue
        for line in LINEAS:
            x_raw = hist[f"prob_over_{line}"].values
            y_raw = hist[f"out_over_{line}"].values
            if len(np.unique(x_raw)) < 2 or len(np.unique(y_raw)) < 2:
                continue
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(x_raw, y_raw)
            p_cal = iso.predict([row[f"prob_over_{line}"]])[0]
            df.at[i, f"cal_prob_over_{line}"] = p_cal

    print("=== BACKTEST REALISTA: ESTRATEGIA BASE (prob original) ===")
    res_base = run_backtest(df, calibrada=False)
    for k, v in res_base.items():
        if k in ["winrate", "roi", "max_drawdown"]:
            print(f"{k}: {v:.2%}" if isinstance(v, float) else f"{k}: {v}")
        else:
            print(f"{k}: {v}")

    print("\n=== BACKTEST REALISTA: CALIBRACIÓN WALK-FORWARD ===")
    res_wf = run_backtest(df, calibrada=True)
    for k, v in res_wf.items():
        if k in ["winrate", "roi", "max_drawdown"]:
            print(f"{k}: {v:.2%}" if isinstance(v, float) else f"{k}: {v}")
        else:
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()