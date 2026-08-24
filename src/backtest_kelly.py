import pandas as pd
import numpy as np
from scipy.stats import norm

# Configuración
BANKROLL_INICIAL = 1000
LINEAS = [3.5, 4.5, 5.5, 6.5]
UMBRAL_PROB = 0.55           # percentil 10% debe superar este valor
COEF_VARIACION_MAX = 0.3     # sigma_p / p máximo para apostar con stake completo
KELLY_FRACTION = 0.25        # fracción de Kelly (1/4 Kelly)

def prob_break_even(odds_americano):
    """Probabilidad de break-even para cuota americana."""
    if odds_americano > 0:
        return 100 / (odds_americano + 100)
    else:
        return -odds_americano / (-odds_americano + 100)

def expected_value(prob, odds_americano):
    """Valor esperado por unidad apostada."""
    if odds_americano > 0:
        return prob * (odds_americano / 100) - (1 - prob)
    else:
        return prob * (100 / -odds_americano) - (1 - prob)

def kelly_fraction(prob, odds_americano):
    """Fracción de Kelly para una apuesta (sin ajuste)."""
    break_even = prob_break_even(odds_americano)
    if prob <= break_even:
        return 0.0
    if odds_americano > 0:
        b = odds_americano / 100
    else:
        b = 100 / -odds_americano
    return (prob * b - (1 - prob)) / b

def kelly_adaptativo(prob, sigma_p, odds_americano):
    """Kelly fraccional con ajuste por incertidumbre."""
    if sigma_p == 0:
        return kelly_fraction(prob, odds_americano) * KELLY_FRACTION
    cv = sigma_p / prob if prob > 0 else 1e9
    # Si el coeficiente de variación es alto, reducimos más
    if cv > COEF_VARIACION_MAX:
        factor = 0.5   # Reducir a la mitad
    else:
        factor = 1.0
    return kelly_fraction(prob, odds_americano) * KELLY_FRACTION * factor

def simular_estrategia(df, nombre, stake_fn, umbral_confianza=False):
    """
    Simula una estrategia de apuestas.
    df: DataFrame con columnas: game_pk, pitcher, strikeouts, y columnas de probabilidad y sigma.
    stake_fn: función que recibe (prob, sigma, odds) y devuelve fracción a apostar.
    umbral_confianza: si True, solo apuesta si percentil 10% de p > UMBRAL_PROB.
    """
    bankroll = BANKROLL_INICIAL
    apuestas = 0
    ganadas = 0
    retornos = []
    for idx, row in df.iterrows():
        # Elegir la línea más cercana a la predicción de BNN (o usar la línea de la fila si existe)
        pred = row["pred_bnn_mean"]
        # Seleccionar línea más cercana entre 3.5, 4.5, 5.5, 6.5
        linea = min(LINEAS, key=lambda l: abs(pred - l))
        prob_col = f"prob_over_{linea}"
        prob = row[prob_col]
        sigma = row["pred_bnn_std"]

        # Lado: over o under
        if prob >= 0.5:
            lado = "over"
            p = prob
        else:
            lado = "under"
            p = 1 - prob

        # Umbral de confianza: percentil 10% > UMBRAL_PROB
        if umbral_confianza:
            # Asumimos distribución beta aproximada; percentil 10% ~ p - 1.28*sigma
            p10 = p - 1.28 * sigma
            if p10 <= UMBRAL_PROB:
                continue

        # Odds asumidas -110
        odds = -110
        # Calcular fracción de stake
        frac = stake_fn(p, sigma, odds)
        if frac <= 0:
            continue

        stake = bankroll * min(frac, 0.05)  # Limitamos a 5% del bankroll por apuesta
        apuestas += 1

        # Resultado
        if lado == "over":
            gano = row["strikeouts"] > linea
        else:
            gano = row["strikeouts"] <= linea

        if gano:
            ganadas += 1
            profit = stake * (100 / 110)  # -110 paga 100/110 por unidad
            bankroll += profit
        else:
            bankroll -= stake

        retornos.append(bankroll)

    roi = (bankroll - BANKROLL_INICIAL) / BANKROLL_INICIAL
    sharpe = np.mean(np.diff(retornos)) / (np.std(np.diff(retornos)) + 1e-9) if len(retornos) > 1 else 0
    max_drawdown = 0
    peak = BANKROLL_INICIAL
    for b in retornos:
        if b > peak:
            peak = b
        dd = (peak - b) / peak
        if dd > max_drawdown:
            max_drawdown = dd
    winrate = ganadas / apuestas if apuestas > 0 else 0

    print(f"\nEstrategia: {nombre}")
    print(f"  Apuestas realizadas: {apuestas}")
    print(f"  Win rate: {winrate:.2%}")
    print(f"  Bankroll final: ${bankroll:.2f}")
    print(f"  ROI: {roi:.2%}")
    print(f"  Sharpe (aprox): {sharpe:.2f}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")

def main():
    # Cargar predicciones
    bnn = pd.read_csv("reports/bnn_predictions.csv")
    xgb = pd.read_csv("reports/xgb_predictions.csv")
    nn = pd.read_csv("reports/nn_predictions.csv")
    # Nos basamos en BNN porque tiene probabilidades y sigma
    # Podríamos unir con xgb/nn para comparar, pero para este backtest usamos BNN puro
    df = bnn.copy()
    # Asegurar que tenemos columnas necesarias
    if "pred_bnn_std" not in df.columns:
        print("Falta columna pred_bnn_std")
        return

    # Simular estrategia Kelly puro (sin ajuste de incertidumbre)
    simular_estrategia(df, "Kelly puro", lambda p, sigma, odds: kelly_fraction(p, odds) * KELLY_FRACTION)
    # Simular Kelly adaptativo con filtro de incertidumbre
    simular_estrategia(df, "Kelly adaptativo", kelly_adaptativo)
    # Simular Kelly adaptativo + umbral de confianza
    simular_estrategia(df, "Kelly adaptativo + umbral confianza", kelly_adaptativo, umbral_confianza=True)

if __name__ == "__main__":
    main()