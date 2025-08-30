from binance.client import Client
import pandas as pd
import time
import numpy as np
from datetime import datetime, timedelta
import ta  # pour les indicateurs techniques

# === Configuration API Binance (public OK) ===
API_KEY = ''
API_SECRET = ''
client = Client(API_KEY, API_SECRET)

# === Fonction 1 : calcul du buffer historique requis ===
def compute_buffered_start_date(start_date, interval):
    dt = pd.to_datetime(start_date)

    if interval == '1h':
        buffer = timedelta(hours=60)
    elif interval == '4h':
        buffer = timedelta(hours=4 * 60)
    elif interval == '1d':
        buffer = timedelta(days=60)
    else:
        raise ValueError("Interval non pris en charge.")

    buffered_date = dt - buffer
    return buffered_date.strftime('%Y-%m-%d')

# === Fonction 2 : téléchargement OHLCV brut depuis Binance ===
def download_ohlcv_binance(symbol='BTCUSDT', interval='1h', start_str='2025-03-01', end_str='2025-08-01'):
    print(f"📥 Téléchargement de {symbol} en {interval} de {start_str} à {end_str}")
    df = pd.DataFrame(client.get_historical_klines(symbol, interval, start_str, end_str))
    df = df.iloc[:, :6]
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    df = df.astype(float)
    return df

# === Fonction 3 : ajout des indicateurs techniques ===
def add_technical_indicators(df):
    df = df.copy()

    # Momentum: open - close
    df['momentum'] = df['open'] - df['close']

    # Range: high - low
    df['range'] = df['high'] - df['low']
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    # Features dérivées du MACD
    df['macd_histogram'] = df['MACD'] - df['MACD_signal']
    df['macd_divergence'] = df['macd_histogram'].abs()
    df['macd_slope'] = df['MACD'].diff()

    # Ichimoku (optionnel mais souvent utile)
    ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
    df['tenkan_sen'] = ichimoku.ichimoku_conversion_line()
    df['kijun_sen'] = ichimoku.ichimoku_base_line()
    df['senkou_span_a'] = ichimoku.ichimoku_a()
    df['senkou_span_b'] = ichimoku.ichimoku_b()
    
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bollinger_mavg'] = bollinger.bollinger_mavg()
    df['bollinger_hband'] = bollinger.bollinger_hband()
    df['bollinger_lband'] = bollinger.bollinger_lband()
    df['bollinger_width'] = df['bollinger_hband'] - df['bollinger_lband']  # utile !

    return df

# === Fonction 4 : pipeline complet pour un timeframe donné ===
def prepare_and_save_timeframe(symbol, interval, true_start_str, end_str, out_file):
    # Calcul du buffer
    buffered_start_str = compute_buffered_start_date(true_start_str, interval)
    
    # Téléchargement + indicateurs
    df = download_ohlcv_binance(symbol, interval, buffered_start_str, end_str)
    df = add_technical_indicators(df)
    
    # Suppression du buffer
    df = df[df.index >= pd.to_datetime(true_start_str)].copy()
    df['timeframe'] = interval
    
    # Sauvegarde
    df.reset_index(inplace=True)
    df.to_csv(out_file, index=False)
    print(f"✅ Fichier sauvegardé : {out_file} ({len(df)} lignes)")

# === Appel des 3 timeframes ===
prepare_and_save_timeframe('BTCUSDT', '1h', '2025-04-01', '2025-08-01', 'btc_h1_test.csv')
time.sleep(1)
prepare_and_save_timeframe('BTCUSDT', '4h', '2025-04-01', '2025-08-01', 'btc_h4_test.csv')
time.sleep(1)
prepare_and_save_timeframe('BTCUSDT', '1d', '2025-04-01', '2025-08-01', 'btc_1d_test.csv')
