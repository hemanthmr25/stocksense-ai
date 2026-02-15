import ta
import numpy as np
import pandas as pd

def add_features(df):
    # 🔥 FORCE 1D arrays (THIS FIXES THE ERROR)
    close = pd.Series(df['Close'].values.flatten(), index=df.index)
    open_ = pd.Series(df['Open'].values.flatten(), index=df.index)
    high  = pd.Series(df['High'].values.flatten(), index=df.index)
    low   = pd.Series(df['Low'].values.flatten(), index=df.index)

    # Technical indicators
    df['sma_20'] = ta.trend.sma_indicator(close, window=20)
    df['sma_50'] = ta.trend.sma_indicator(close, window=50)
    df['rsi'] = ta.momentum.rsi(close, window=14)
    df['macd'] = ta.trend.macd_diff(close)

    df['atr'] = ta.volatility.average_true_range(
        high=high,
        low=low,
        close=close,
        window=14
    )

    # 🔥 VECTORISED HAMMER PATTERN (SAFE)
    body = (close - open_).abs()
    lower_wick = np.minimum(close, open_) - low
    df['hammer'] = ((lower_wick > 2 * body) & (body > 0)).astype(int)

    df.dropna(inplace=True)
    return df
