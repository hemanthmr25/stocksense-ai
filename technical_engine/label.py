def create_labels(df, horizon=5):
    df['future_return'] = df['Close'].shift(-horizon) / df['Close'] - 1

    def label(x):
        if x > 0.02:
            return 1    # BUY
        elif x < -0.02:
            return -1   # SELL
        else:
            return 0    # HOLD

    df['signal'] = df['future_return'].apply(label)
    return df.dropna()
